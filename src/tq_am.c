#include "postgres.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "access/amapi.h"
#include "access/genam.h"
#include "access/reloptions.h"
#include "access/tableam.h"
#include "nodes/tidbitmap.h"
#include "catalog/index.h"
#include "catalog/namespace.h"
#include "catalog/pg_opfamily.h"
#include "catalog/storage.h"
#include "commands/vacuum.h"
#include "nodes/pathnodes.h"
#include "optimizer/optimizer.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "storage/itemptr.h"
#include "utils/rel.h"
#include "utils/syscache.h"

#include "src/tq_am_routine.h"
#include "src/tq_bitmap_filter.h"
#include "src/tq_codec_prod.h"
#include "src/tq_guc.h"
#include "src/tq_options.h"
#include "src/tq_page.h"
#include "src/tq_pgvector_compat.h"
#include "src/tq_query_tuning.h"
#include "src/tq_reloptions.h"
#include "src/tq_router.h"
#include "src/tq_scan.h"
#include "src/tq_simd_avx2.h"
#include "src/tq_transform.h"
#include "src/tq_wal.h"

PG_FUNCTION_INFO_V1(turboquanthandler);

typedef struct TqBuildState
{
	Relation	index_relation;
	TqOptionConfig option_config;
	TqMetaPageFields meta_fields;
	TqTransformState transform_state;
	TqRouterModel router_model;
	TqProdCodecConfig prod_config;
	TqProdPackedLayout prod_layout;
	uint32_t	source_dimension;
	uint32_t	dimension;
	uint16_t	lane_count;
	TqDistanceKind distance_kind;
	TqVectorInputKind input_kind;
	bool		initialized;
	float	   *source_values;
	float	   *transformed_values;
	float	   *collected_vectors;
	TqTid	   *collected_tids;
	uint8_t	   *packed_code;
	size_t		vector_count;
	size_t		vector_capacity;
	double		index_tuples;
} TqBuildState;

typedef struct TqScanOpaque
{
	TqTransformState transform_state;
	TqProdLut	lut;
	TqCandidateHeap candidates;
	TqProdCodecConfig prod_config;
	float	   *query_values;
	uint32_t	query_dimension;
	TqDistanceKind distance_kind;
	TqVectorInputKind input_kind;
	bool		normalized;
	bool		prepared;
} TqScanOpaque;

typedef struct TqVacuumSummary
{
	double		live_tuples;
	double		tuples_removed;
	BlockNumber reclaimable_pages;
} TqVacuumSummary;

typedef struct TqBoundedPageCandidate
{
	BlockNumber block_number;
	float		optimistic_distance;
} TqBoundedPageCandidate;

static void tq_load_option_config(Relation index_relation, TqOptionConfig *config);
static void tq_reset_scan_opaque(TqScanOpaque *opaque);
static void tq_buildstate_reset(TqBuildState *state);
static bool tq_read_meta_page(Relation index_relation,
							  TqMetaPageFields *fields,
							  char *errmsg,
							  size_t errmsg_len);
static void tq_write_meta_page(Relation index_relation, const TqMetaPageFields *fields);
static Buffer tq_lock_meta_page_buffer(Relation index_relation);
static bool tq_read_meta_page_buffer(Buffer buffer,
									 TqMetaPageFields *fields,
									 char *errmsg,
									 size_t errmsg_len);
static void tq_write_meta_page_buffer(Relation index_relation,
									  Buffer buffer,
									  const TqMetaPageFields *fields);
static void tq_build_callback(Relation index, ItemPointer tid, Datum *values,
							  bool *isnull, bool tupleIsAlive, void *stateptr);
static void tq_buildstate_initialize(TqBuildState *state, uint32_t dimension);
static void tq_buildstate_collect_vector(TqBuildState *state, const float *values,
										 ItemPointer tid);
static void tq_buildstate_flush(TqBuildState *state);
static void tq_write_centroid_pages(Relation index_relation,
									const TqRouterModel *model,
									uint32_t *root_block);
static void tq_write_directory_pages(Relation index_relation,
									 const TqListDirEntry *entries,
									 uint32_t list_count,
									 uint32_t *root_block);
static void tq_write_batch_pages(TqBuildState *state,
								 const uint32_t *list_assignments,
								 uint32_t list_count,
								 TqListDirEntry *entries);
static void tq_update_batch_page_next_block(Relation index_relation,
											BlockNumber block_number,
											BlockNumber next_block);
static bool tq_recompute_batch_page_summary(const TqProdCodecConfig *config,
											const void *page,
											size_t page_size,
											TqBatchPageSummary *summary,
											char *errmsg,
											size_t errmsg_len);
static bool tq_refresh_batch_page_summary(Relation index_relation,
										  Buffer buffer,
										  const TqProdCodecConfig *config,
										  char *errmsg,
										  size_t errmsg_len);
static bool tq_load_router_model(Relation index_relation,
								 const TqMetaPageFields *meta_fields,
								 TqRouterModel *model,
								 char *errmsg,
								 size_t errmsg_len);
static bool tq_read_list_directory_entry(Relation index_relation,
										 BlockNumber root_block,
										 uint32_t list_id,
										 TqListDirEntry *entry,
										 char *errmsg,
										 size_t errmsg_len);
static bool tq_write_list_directory_entry(Relation index_relation,
										  BlockNumber root_block,
										  uint32_t list_id,
										  const TqListDirEntry *entry,
										  char *errmsg,
										  size_t errmsg_len);
static BlockNumber tq_find_reusable_batch_block(Relation index_relation,
												 BlockNumber head_block,
												 BlockNumber tail_block,
												 char *errmsg,
												 size_t errmsg_len);
static BlockNumber tq_find_detached_free_batch_block(Relation index_relation,
													 char *errmsg,
													 size_t errmsg_len);
static void tq_truncate_reusable_tail_pages(Relation index_relation,
											 const TqMetaPageFields *meta_fields,
											 char *errmsg,
											 size_t errmsg_len);
static void tq_append_packed_tuple(Relation index_relation,
								   const TqMetaPageFields *meta_fields,
								   const TqProdCodecConfig *prod_config,
								   uint32_t list_id,
								   const TqTid *heap_tid,
								   const uint8_t *packed_code,
								   size_t packed_code_len);
static void tq_summarize_index(Relation index_relation,
							   const TqMetaPageFields *meta_fields,
							   IndexBulkDeleteCallback callback,
							   void *callback_state,
							   bool apply_deletes,
							   TqVacuumSummary *summary);
static bool tq_scan_batch_block(Relation index_relation,
								BlockNumber block_number,
								TqScanOpaque *opaque,
								char *errmsg,
								size_t errmsg_len);
static bool tq_collect_bounded_pages_for_chain(Relation index_relation,
											   BlockNumber head_block,
											   TqScanOpaque *opaque,
											   TqBoundedPageCandidate *pages,
											   size_t max_pages,
											   size_t *page_count,
											   char *errmsg,
											   size_t errmsg_len);
static bool tq_count_batch_pages_for_chain(Relation index_relation,
										   BlockNumber head_block,
										   size_t *page_count,
										   char *errmsg,
										   size_t errmsg_len);
static bool tq_scan_prepare(IndexScanDesc scan, TqScanOpaque *opaque);
static int64 tq_scan_all_live_tids_to_bitmap(Relation index_relation,
											 TIDBitmap *tbm,
											 char *errmsg,
											 size_t errmsg_len);
static void *tq_page_payload(Page page);
static size_t tq_page_payload_size(Page page);
static size_t tq_relation_payload_size(void);
static TqDistanceKind tq_distance_kind_from_index(Relation index_relation);
static TqVectorInputKind tq_input_kind_from_index(Relation index_relation);

static void *
tq_page_payload(Page page)
{
	return PageGetContents(page);
}

static size_t
tq_page_payload_size(Page page)
{
	return (size_t) (PageGetPageSize(page) - SizeOfPageHeaderData);
}

static size_t
tq_relation_payload_size(void)
{
	return (size_t) (BLCKSZ - SizeOfPageHeaderData);
}

static TqDistanceKind
tq_distance_kind_from_index(Relation index_relation)
{
	HeapTuple	tuple;
	Form_pg_opfamily form;
	char	   *name;
	TqDistanceKind distance_kind = TQ_DISTANCE_COSINE;

	tuple = SearchSysCache1(OPFAMILYOID,
							ObjectIdGetDatum(index_relation->rd_opfamily[0]));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "turboquant could not resolve operator family %u",
			 index_relation->rd_opfamily[0]);

	form = (Form_pg_opfamily) GETSTRUCT(tuple);
	name = NameStr(form->opfname);

	if (strstr(name, "_ip_") != NULL)
		distance_kind = TQ_DISTANCE_IP;
	else if (strstr(name, "_l2_") != NULL)
		distance_kind = TQ_DISTANCE_L2;
	else if (strstr(name, "_cosine_") != NULL)
		distance_kind = TQ_DISTANCE_COSINE;
	else
	{
		ReleaseSysCache(tuple);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant does not recognize operator family \"%s\"", name)));
	}

	ReleaseSysCache(tuple);
	return distance_kind;
}

static TqVectorInputKind
tq_input_kind_from_index(Relation index_relation)
{
	TqVectorInputKind input_kind;
	char		error_buf[256];

	memset(&input_kind, 0, sizeof(input_kind));
	if (!tq_vector_input_kind_from_typid(index_relation->rd_opcintype[0],
										 &input_kind,
										 error_buf,
										 sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	return input_kind;
}

static void
tq_load_option_config(Relation index_relation, TqOptionConfig *config)
{
	TqAmOptions *options = (TqAmOptions *) index_relation->rd_options;

	memset(config, 0, sizeof(*config));
	config->bits = 4;
	config->lists = 0;
	config->router_samples = 256;
	config->router_iterations = 8;
	config->router_restarts = 3;
	config->router_seed = 20260327;
	config->normalized = true;
	config->transform_name = "hadamard";
	config->lanes_name = "auto";

	if (options == NULL)
		return;

	config->bits = options->bits;
	config->lists = options->lists;
	config->router_samples = options->router_samples;
	config->router_iterations = options->router_iterations;
	config->router_restarts = options->router_restarts == 0 ? 3 : options->router_restarts;
	config->router_seed = options->router_seed;
	config->normalized = options->normalized;
	config->transform_name = GET_STRING_RELOPTION(options, transform_offset);
	config->lanes_name = GET_STRING_RELOPTION(options, lanes_offset);
}

static void
tq_reset_scan_opaque(TqScanOpaque *opaque)
{
	if (opaque == NULL)
		return;

	tq_transform_reset(&opaque->transform_state);
	tq_prod_lut_reset(&opaque->lut);
	tq_candidate_heap_reset(&opaque->candidates);
	if (opaque->query_values != NULL)
		pfree(opaque->query_values);
	memset(&opaque->prod_config, 0, sizeof(opaque->prod_config));
	opaque->query_values = NULL;
	opaque->query_dimension = 0;
	opaque->prepared = false;
}

static void
tq_buildstate_reset(TqBuildState *state)
{
	if (state == NULL)
		return;

	tq_transform_reset(&state->transform_state);
	tq_router_reset(&state->router_model);
	if (state->source_values != NULL)
		pfree(state->source_values);
	if (state->transformed_values != NULL)
		pfree(state->transformed_values);
	if (state->collected_vectors != NULL)
		pfree(state->collected_vectors);
	if (state->collected_tids != NULL)
		pfree(state->collected_tids);
	if (state->packed_code != NULL)
		pfree(state->packed_code);
	memset(state, 0, sizeof(*state));
}

static bool
tq_read_meta_page(Relation index_relation,
				  TqMetaPageFields *fields,
				  char *errmsg,
				  size_t errmsg_len)
{
	Buffer		buffer;
	bool		ok = false;

	buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
	ok = tq_meta_page_read(tq_page_payload(BufferGetPage(buffer)),
						   tq_page_payload_size(BufferGetPage(buffer)),
						   fields,
						   errmsg,
						   errmsg_len);
	ReleaseBuffer(buffer);
	return ok;
}

static void
tq_write_meta_page(Relation index_relation, const TqMetaPageFields *fields)
{
	Buffer		buffer;
	buffer = tq_lock_meta_page_buffer(index_relation);
	tq_write_meta_page_buffer(index_relation, buffer, fields);
	LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buffer);
}

static Buffer
tq_lock_meta_page_buffer(Relation index_relation)
{
	Buffer		buffer;

	if (RelationGetNumberOfBlocks(index_relation) == 0)
		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, P_NEW, RBM_NORMAL, NULL);
	else
		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);

	LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
	return buffer;
}

static bool
tq_read_meta_page_buffer(Buffer buffer,
						 TqMetaPageFields *fields,
						 char *errmsg,
						 size_t errmsg_len)
{
	return tq_meta_page_read(tq_page_payload(BufferGetPage(buffer)),
							 tq_page_payload_size(BufferGetPage(buffer)),
							 fields,
							 errmsg,
							 errmsg_len);
}

static void
tq_write_meta_page_buffer(Relation index_relation,
						  Buffer buffer,
						  const TqMetaPageFields *fields)
{
	char		error_buf[256];

	if (!tq_wal_init_meta_page(index_relation, buffer, fields,
							   error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);
}

static void
tq_buildstate_initialize(TqBuildState *state, uint32_t dimension)
{
	TqLaneConfig lane_config;
	TqTransformConfig transform_config;
	TqTransformMetadata transform_metadata;
	int			resolved_lane_count = 0;
	char		error_buf[256];

	memset(&lane_config, 0, sizeof(lane_config));
	memset(&transform_config, 0, sizeof(transform_config));
	memset(&transform_metadata, 0, sizeof(transform_metadata));

	state->source_dimension = dimension;

	transform_config.kind = TQ_TRANSFORM_HADAMARD;
	transform_config.dimension = dimension;
	transform_config.seed = UINT64_C(0);

	if (!tq_transform_metadata_init(&transform_config, &transform_metadata,
									error_buf, sizeof(error_buf))
		|| !tq_transform_prepare_metadata(&transform_metadata, &state->transform_state,
										  error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	state->dimension = transform_metadata.output_dimension;
	state->prod_config.dimension = state->dimension;
	state->prod_config.bits = (uint8_t) state->option_config.bits;

	if (!tq_prod_packed_layout(&state->prod_config, &state->prod_layout,
							   error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	lane_config.block_size = BLCKSZ;
	lane_config.dimension = (int) state->dimension;
	lane_config.bits = state->option_config.bits;
	lane_config.codec = TQ_CODEC_PROD;
	lane_config.normalized = state->option_config.normalized;
	lane_config.page_header_bytes = TQ_PAGE_HEADER_BYTES;
	lane_config.special_space_bytes = TQ_PAGE_SPECIAL_BYTES;
	lane_config.reserve_bytes = TQ_PAGE_RESERVED_BYTES;
	lane_config.tid_bytes = TQ_TID_BYTES;

	if (!tq_resolve_lane_count(&lane_config, &resolved_lane_count,
							   error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);
	if (resolved_lane_count <= 0 || resolved_lane_count > UINT16_MAX)
		elog(ERROR, "turboquant resolved lane count %d is out of range", resolved_lane_count);
	state->lane_count = (uint16_t) resolved_lane_count;

	state->source_values = (float *) palloc(sizeof(float) * (size_t) state->source_dimension);
	state->transformed_values = (float *) palloc(sizeof(float) * (size_t) state->dimension);
	state->packed_code = (uint8_t *) palloc(state->prod_layout.total_bytes);

	memset(&state->meta_fields, 0, sizeof(state->meta_fields));
	state->meta_fields.dimension = state->source_dimension;
	state->meta_fields.transform_output_dimension = state->dimension;
	state->meta_fields.codec = TQ_CODEC_PROD;
	state->meta_fields.distance = state->distance_kind;
	state->meta_fields.bits = state->option_config.bits;
	state->meta_fields.lane_count = state->lane_count;
	state->meta_fields.transform = TQ_TRANSFORM_HADAMARD;
	state->meta_fields.transform_version = transform_metadata.contract_version;
	state->meta_fields.normalized = state->option_config.normalized;
	state->meta_fields.list_count = state->option_config.lists > 0
		? (uint32_t) state->option_config.lists
		: 0;
	state->meta_fields.directory_root_block = TQ_INVALID_BLOCK_NUMBER;
	state->meta_fields.centroid_root_block = TQ_INVALID_BLOCK_NUMBER;
	state->meta_fields.transform_seed = transform_metadata.seed;
	state->meta_fields.router_seed = (uint32_t) state->option_config.router_seed;
	state->meta_fields.router_sample_count = (uint32_t) state->option_config.router_samples;
	state->meta_fields.router_max_iterations = (uint32_t) state->option_config.router_iterations;
	state->meta_fields.router_completed_iterations = 0;
	state->meta_fields.router_trained_vector_count = 0;
	state->meta_fields.router_algorithm = TQ_ROUTER_ALGORITHM_FIRST_K;
	state->meta_fields.router_restart_count = (uint32_t) state->option_config.router_restarts;
	state->meta_fields.router_selected_restart = 0;
	state->meta_fields.router_mean_distortion = 0.0f;
	state->meta_fields.router_max_list_over_avg = 0.0f;
	state->meta_fields.router_coeff_var = 0.0f;
	state->meta_fields.router_balance_penalty = 0.0f;
	state->meta_fields.router_selection_score = 0.0f;
	state->initialized = true;
}

static void
tq_buildstate_collect_vector(TqBuildState *state, const float *values, ItemPointer tid)
{
	size_t		required_count = state->vector_count + 1;
	size_t		new_capacity = 0;
	TqTid		page_tid;

	if (!tq_transform_apply(&state->transform_state, values, state->transformed_values,
							state->dimension, NULL, 0))
		elog(ERROR, "turboquant failed to transform build vector");

	if (required_count > state->vector_capacity)
	{
		new_capacity = state->vector_capacity == 0 ? 32 : (state->vector_capacity * 2);
		while (new_capacity < required_count)
			new_capacity *= 2;

		if (state->collected_vectors == NULL)
			state->collected_vectors = (float *) palloc(sizeof(float) * new_capacity * (size_t) state->dimension);
		else
			state->collected_vectors = (float *) repalloc(state->collected_vectors,
														  sizeof(float) * new_capacity * (size_t) state->dimension);

		if (state->collected_tids == NULL)
			state->collected_tids = (TqTid *) palloc(sizeof(TqTid) * new_capacity);
		else
			state->collected_tids = (TqTid *) repalloc(state->collected_tids,
													   sizeof(TqTid) * new_capacity);

		state->vector_capacity = new_capacity;
	}

	page_tid.block_number = ItemPointerGetBlockNumber(tid);
	page_tid.offset_number = ItemPointerGetOffsetNumber(tid);

	memcpy(state->collected_vectors + (state->vector_count * (size_t) state->dimension),
		   state->transformed_values,
		   sizeof(float) * (size_t) state->dimension);
	state->collected_tids[state->vector_count] = page_tid;
	state->vector_count = required_count;
	state->index_tuples += 1.0;
}

static void
tq_update_batch_page_next_block(Relation index_relation,
								BlockNumber block_number,
								BlockNumber next_block)
{
	Buffer		buffer;
	char		error_buf[256];

	buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
	LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
	if (!tq_wal_set_batch_next_block(index_relation, buffer, next_block,
									 error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);
	LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buffer);
}

static float
tq_vector_l2_distance(const float *left, const float *right, uint32_t dimension)
{
	float sum = 0.0f;
	uint32_t i = 0;

	for (i = 0; i < dimension; i++)
	{
		float diff = left[i] - right[i];

		sum += diff * diff;
	}

	return sqrtf(sum);
}

static int
tq_bounded_page_candidate_compare(const void *left, const void *right)
{
	const TqBoundedPageCandidate *lhs = (const TqBoundedPageCandidate *) left;
	const TqBoundedPageCandidate *rhs = (const TqBoundedPageCandidate *) right;

	if (lhs->optimistic_distance < rhs->optimistic_distance)
		return -1;
	if (lhs->optimistic_distance > rhs->optimistic_distance)
		return 1;
	if (lhs->block_number < rhs->block_number)
		return -1;
	if (lhs->block_number > rhs->block_number)
		return 1;
	return 0;
}

static bool
tq_recompute_batch_page_summary(const TqProdCodecConfig *config,
								const void *page,
								size_t page_size,
								TqBatchPageSummary *summary,
								char *errmsg,
								size_t errmsg_len)
{
	TqBatchPageHeaderView header;
	TqBatchPageSummary computed;
	uint16_t lane = 0;
	uint8_t *rep_code = NULL;
	uint8_t *lane_code = NULL;
	float *rep_vector = NULL;
	float *lane_vector = NULL;
	bool ok = false;

	if (config == NULL || page == NULL || summary == NULL)
	{
		snprintf(errmsg, errmsg_len,
				 "invalid turboquant batch summary: codec, page, and summary output must be non-null");
		return false;
	}

	memset(&header, 0, sizeof(header));
	memset(&computed, 0, sizeof(computed));
	computed.representative_lane = TQ_BATCH_PAGE_NO_REPRESENTATIVE;
	computed.residual_radius = 0.0f;

	if (!tq_batch_page_read_header(page, page_size, &header, errmsg, errmsg_len))
		return false;

	if (header.live_count == 0
		|| !tq_batch_page_next_live_lane(page, page_size, -1, &lane, errmsg, errmsg_len))
	{
		*summary = computed;
		return true;
	}

	rep_code = (uint8_t *) palloc((Size) header.code_bytes);
	lane_code = (uint8_t *) palloc((Size) header.code_bytes);
	rep_vector = (float *) palloc(sizeof(float) * (Size) config->dimension);
	lane_vector = (float *) palloc(sizeof(float) * (Size) config->dimension);

	ok = tq_batch_page_get_code(page, page_size, lane, rep_code, header.code_bytes,
								errmsg, errmsg_len)
		&& tq_prod_decode(config, rep_code, header.code_bytes, rep_vector,
						  config->dimension, errmsg, errmsg_len);
	if (!ok)
		goto done;

	computed.representative_lane = lane;
	computed.residual_radius = 0.0f;

	do
	{
		float distance = 0.0f;

		if (lane == computed.representative_lane)
			continue;

		ok = tq_batch_page_get_code(page, page_size, lane, lane_code, header.code_bytes,
									errmsg, errmsg_len)
			&& tq_prod_decode(config, lane_code, header.code_bytes, lane_vector,
							  config->dimension, errmsg, errmsg_len);
		if (!ok)
			goto done;

		distance = tq_vector_l2_distance(rep_vector, lane_vector, config->dimension);
		if (distance > computed.residual_radius)
			computed.residual_radius = distance;
	}
	while (tq_batch_page_next_live_lane(page, page_size, (int) lane, &lane, errmsg, errmsg_len));

	*summary = computed;
	ok = true;

done:
	if (rep_code != NULL)
		pfree(rep_code);
	if (lane_code != NULL)
		pfree(lane_code);
	if (rep_vector != NULL)
		pfree(rep_vector);
	if (lane_vector != NULL)
		pfree(lane_vector);
	return ok;
}

static bool
tq_refresh_batch_page_summary(Relation index_relation,
							  Buffer buffer,
							  const TqProdCodecConfig *config,
							  char *errmsg,
							  size_t errmsg_len)
{
	TqBatchPageSummary summary;
	Page page = BufferGetPage(buffer);

	memset(&summary, 0, sizeof(summary));
	if (!tq_recompute_batch_page_summary(config,
										 tq_page_payload(page),
										 tq_page_payload_size(page),
										 &summary,
										 errmsg,
										 errmsg_len))
		return false;

	return tq_wal_set_batch_summary(index_relation, buffer, &summary, errmsg, errmsg_len);
}

static void
tq_write_centroid_pages(Relation index_relation,
						const TqRouterModel *model,
						uint32_t *root_block)
{
	size_t		payload_size = tq_relation_payload_size();
	uint16_t	page_capacity = 0;
	uint32_t	page_count = 0;
	BlockNumber *blocks = NULL;
	uint32_t	page_index = 0;
	char		error_buf[256];

	*root_block = TQ_INVALID_BLOCK_NUMBER;
	if (model == NULL || model->list_count == 0)
		return;

	page_capacity = tq_centroid_page_capacity(payload_size, model->dimension);
	if (page_capacity == 0)
		elog(ERROR, "turboquant centroid page cannot fit dimension %u", model->dimension);

	page_count = (uint32_t) (((size_t) model->list_count + (size_t) page_capacity - 1)
							 / (size_t) page_capacity);
	blocks = (BlockNumber *) palloc(sizeof(BlockNumber) * (size_t) page_count);

	for (page_index = 0; page_index < page_count; page_index++)
	{
		Buffer		buffer;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, P_NEW, RBM_NORMAL, NULL);
		LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
		blocks[page_index] = BufferGetBlockNumber(buffer);
		if (!tq_wal_init_centroid_page(index_relation, buffer, model->dimension,
									   page_capacity, TQ_INVALID_BLOCK_NUMBER,
									   error_buf, sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);
		LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
		ReleaseBuffer(buffer);
	}

	*root_block = blocks[0];

	for (page_index = 0; page_index < page_count; page_index++)
	{
		Buffer		buffer;
		uint32_t	start_list = page_index * (uint32_t) page_capacity;
		uint32_t	remaining = model->list_count - start_list;
		uint16_t	write_count = remaining < (uint32_t) page_capacity
			? (uint16_t) remaining
			: page_capacity;
		uint16_t	local_index = 0;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, blocks[page_index], RBM_NORMAL, NULL);
		LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
		if (!tq_wal_init_centroid_page(index_relation, buffer,
									   model->dimension, page_capacity,
									   page_index + 1 < page_count ? blocks[page_index + 1] : TQ_INVALID_BLOCK_NUMBER,
									   error_buf, sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);

		for (local_index = 0; local_index < write_count; local_index++)
		{
			const float *centroid = model->centroids
				+ ((size_t) (start_list + local_index) * (size_t) model->dimension);

			if (!tq_wal_set_centroid(index_relation, buffer, local_index,
									 centroid, model->dimension,
									 error_buf, sizeof(error_buf)))
				elog(ERROR, "%s", error_buf);
		}
		LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
		ReleaseBuffer(buffer);
	}

	pfree(blocks);
}

static void
tq_write_directory_pages(Relation index_relation,
						 const TqListDirEntry *entries,
						 uint32_t list_count,
						 uint32_t *root_block)
{
	size_t		payload_size = tq_relation_payload_size();
	uint16_t	page_capacity = 0;
	uint32_t	page_count = 0;
	BlockNumber *blocks = NULL;
	uint32_t	page_index = 0;
	char		error_buf[256];

	*root_block = TQ_INVALID_BLOCK_NUMBER;
	if (entries == NULL || list_count == 0)
		return;

	page_capacity = tq_list_dir_page_capacity(payload_size);
	if (page_capacity == 0)
		elog(ERROR, "turboquant list directory page capacity is zero");

	page_count = (uint32_t) (((size_t) list_count + (size_t) page_capacity - 1)
							 / (size_t) page_capacity);
	blocks = (BlockNumber *) palloc(sizeof(BlockNumber) * (size_t) page_count);

	for (page_index = 0; page_index < page_count; page_index++)
	{
		Buffer		buffer;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, P_NEW, RBM_NORMAL, NULL);
		LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
		blocks[page_index] = BufferGetBlockNumber(buffer);
		if (!tq_wal_init_list_dir_page(index_relation, buffer, page_capacity,
									   TQ_INVALID_BLOCK_NUMBER,
									   error_buf, sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);
		LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
		ReleaseBuffer(buffer);
	}

	*root_block = blocks[0];

	for (page_index = 0; page_index < page_count; page_index++)
	{
		Buffer		buffer;
		uint32_t	start_list = page_index * (uint32_t) page_capacity;
		uint32_t	remaining = list_count - start_list;
		uint16_t	write_count = remaining < (uint32_t) page_capacity
			? (uint16_t) remaining
			: page_capacity;
		uint16_t	local_index = 0;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, blocks[page_index], RBM_NORMAL, NULL);
		LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
		if (!tq_wal_init_list_dir_page(index_relation, buffer,
									   page_capacity,
									   page_index + 1 < page_count ? blocks[page_index + 1] : TQ_INVALID_BLOCK_NUMBER,
									   error_buf, sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);

		for (local_index = 0; local_index < write_count; local_index++)
		{
			if (!tq_wal_set_list_dir_entry(index_relation, buffer, local_index,
										   &entries[start_list + local_index],
										   error_buf, sizeof(error_buf)))
				elog(ERROR, "%s", error_buf);
		}
		LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
		ReleaseBuffer(buffer);
	}

	pfree(blocks);
}

static void
tq_write_batch_pages(TqBuildState *state,
					 const uint32_t *list_assignments,
					 uint32_t list_count,
					 TqListDirEntry *entries)
{
	uint32_t	effective_list_count = list_count == 0 ? 1 : list_count;
	uint32_t	list_id = 0;
	char		error_buf[256];

	for (list_id = 0; list_id < effective_list_count; list_id++)
	{
		Buffer		current_buffer = InvalidBuffer;
		uint16_t	lanes_used = 0;
		size_t		vector_index = 0;

		for (vector_index = 0; vector_index < state->vector_count; vector_index++)
		{
			const float *vector = state->collected_vectors
				+ (vector_index * (size_t) state->dimension);
			TqTid	   *tid = &state->collected_tids[vector_index];
			uint32_t	target_list = list_assignments != NULL ? list_assignments[vector_index] : 0;

			if (target_list != list_id)
				continue;

			if (!BufferIsValid(current_buffer) || lanes_used >= state->lane_count)
			{
				TqBatchPageParams params;
				Buffer		new_buffer;
				BlockNumber new_block;

				memset(&params, 0, sizeof(params));
				params.lane_count = state->lane_count;
				params.code_bytes = (uint32_t) state->prod_layout.total_bytes;
				params.list_id = list_id;
				params.next_block = TQ_INVALID_BLOCK_NUMBER;

				new_buffer = ReadBufferExtended(state->index_relation, MAIN_FORKNUM, P_NEW, RBM_NORMAL, NULL);
				LockBuffer(new_buffer, BUFFER_LOCK_EXCLUSIVE);
				new_block = BufferGetBlockNumber(new_buffer);

				if (!tq_wal_init_batch_page(state->index_relation, new_buffer,
										   &params, error_buf, sizeof(error_buf)))
					elog(ERROR, "%s", error_buf);
				LockBuffer(new_buffer, BUFFER_LOCK_UNLOCK);

				if (entries != NULL)
				{
					if (entries[list_id].head_block == TQ_INVALID_BLOCK_NUMBER)
						entries[list_id].head_block = new_block;
					if (entries[list_id].tail_block != TQ_INVALID_BLOCK_NUMBER)
						tq_update_batch_page_next_block(state->index_relation,
														entries[list_id].tail_block,
														new_block);
					entries[list_id].tail_block = new_block;
				}

				if (BufferIsValid(current_buffer))
					ReleaseBuffer(current_buffer);

				current_buffer = new_buffer;
				lanes_used = 0;
			}

			if (!tq_prod_encode(&state->prod_config, vector, state->packed_code,
								state->prod_layout.total_bytes, error_buf, sizeof(error_buf)))
				elog(ERROR, "%s", error_buf);

			LockBuffer(current_buffer, BUFFER_LOCK_EXCLUSIVE);
			if (!tq_wal_append_batch_code(state->index_relation, current_buffer,
										 tid, state->packed_code,
										 state->prod_layout.total_bytes,
										 &lanes_used, error_buf, sizeof(error_buf))
				|| !tq_refresh_batch_page_summary(state->index_relation,
												 current_buffer,
												 &state->prod_config,
												 error_buf,
												 sizeof(error_buf)))
				elog(ERROR, "%s", error_buf);
			LockBuffer(current_buffer, BUFFER_LOCK_UNLOCK);

			if (entries != NULL)
			{
				entries[list_id].live_count += 1;
				entries[list_id].free_lane_hint = (uint16_t) (lanes_used + 1);
			}
			lanes_used += 1;
		}

		if (BufferIsValid(current_buffer))
			ReleaseBuffer(current_buffer);
	}
}

static void
tq_buildstate_flush(TqBuildState *state)
{
	uint32_t	list_count = state->meta_fields.list_count;

	if (!state->initialized || state->vector_count == 0)
		return;

	if (list_count == 0)
	{
		tq_write_batch_pages(state, NULL, 0, NULL);
		return;
	}

	{
		uint32_t   *list_assignments = (uint32_t *) palloc(sizeof(uint32_t) * state->vector_count);
		TqListDirEntry *entries = (TqListDirEntry *) palloc0(sizeof(TqListDirEntry) * (size_t) list_count);
		TqRouterTrainingConfig router_config;
		size_t		vector_index = 0;
		char		error_buf[256];

		memset(&router_config, 0, sizeof(router_config));
		router_config.seed = (uint32_t) state->option_config.router_seed;
		router_config.sample_count = (uint32_t) state->option_config.router_samples;
		router_config.max_iterations = (uint32_t) state->option_config.router_iterations;
		router_config.restart_count = (uint32_t) state->option_config.router_restarts;

		if (!tq_router_train_kmeans(state->collected_vectors, state->vector_count,
									state->dimension, list_count, &router_config,
									&state->router_model, error_buf, sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);

		state->meta_fields.router_seed = state->router_model.metadata.seed;
		state->meta_fields.router_sample_count = state->router_model.metadata.sample_count;
		state->meta_fields.router_max_iterations = state->router_model.metadata.max_iterations;
		state->meta_fields.router_completed_iterations = state->router_model.metadata.completed_iterations;
		state->meta_fields.router_trained_vector_count = state->router_model.metadata.trained_vector_count;
		state->meta_fields.router_algorithm = state->router_model.metadata.algorithm;
		state->meta_fields.router_restart_count = state->router_model.metadata.restart_count;
		state->meta_fields.router_selected_restart = state->router_model.metadata.selected_restart;
		state->meta_fields.router_mean_distortion = state->router_model.metadata.mean_distortion;
		state->meta_fields.router_max_list_over_avg = state->router_model.metadata.max_list_over_avg;
		state->meta_fields.router_coeff_var = state->router_model.metadata.coeff_var;
		state->meta_fields.router_balance_penalty = state->router_model.metadata.balance_penalty;
		state->meta_fields.router_selection_score = state->router_model.metadata.selection_score;

		for (vector_index = 0; vector_index < state->vector_count; vector_index++)
		{
			const float *vector = state->collected_vectors
				+ (vector_index * (size_t) state->dimension);

			if (!tq_router_assign_best(&state->router_model, vector,
									   &list_assignments[vector_index], NULL,
									   error_buf, sizeof(error_buf)))
				elog(ERROR, "%s", error_buf);
		}

		for (vector_index = 0; vector_index < list_count; vector_index++)
		{
			entries[vector_index].list_id = (uint32_t) vector_index;
			entries[vector_index].head_block = TQ_INVALID_BLOCK_NUMBER;
			entries[vector_index].tail_block = TQ_INVALID_BLOCK_NUMBER;
		}

		tq_write_centroid_pages(state->index_relation, &state->router_model,
								&state->meta_fields.centroid_root_block);
		tq_write_batch_pages(state, list_assignments, list_count, entries);
		tq_write_directory_pages(state->index_relation, entries, list_count,
								 &state->meta_fields.directory_root_block);

		pfree(entries);
		pfree(list_assignments);
	}
}

static void
tq_build_callback(Relation index, ItemPointer tid, Datum *values,
				  bool *isnull, bool tupleIsAlive, void *stateptr)
{
	TqBuildState *state = (TqBuildState *) stateptr;
	uint32_t	dimension = 0;
	char		error_buf[256];

	(void) index;

	if (!tupleIsAlive || isnull[0])
		return;

	if (!state->initialized)
	{
		if (!tq_vector_dimension_from_datum_typed(values[0], state->input_kind,
												  &dimension,
												  error_buf, sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);
		tq_buildstate_initialize(state, dimension);
	}

	if (!tq_vector_copy_from_datum_typed(values[0], state->input_kind,
										 state->source_values, state->source_dimension,
										 &dimension, error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	if (dimension != state->source_dimension)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turboquant build requires consistent vector dimensions")));

	tq_buildstate_collect_vector(state, state->source_values, tid);
}

static IndexBuildResult *
tq_ambuild(Relation heap_relation, Relation index_relation, IndexInfo *index_info)
{
	IndexBuildResult *result;
	TqBuildState build_state;
	double		heap_tuples = 0.0;

	memset(&build_state, 0, sizeof(build_state));
	build_state.index_relation = index_relation;
	build_state.distance_kind = tq_distance_kind_from_index(index_relation);
	build_state.input_kind = tq_input_kind_from_index(index_relation);
	tq_load_option_config(index_relation, &build_state.option_config);
	tq_write_meta_page(index_relation, &build_state.meta_fields);

	heap_tuples = table_index_build_scan(heap_relation, index_relation, index_info,
										 false, true, tq_build_callback, &build_state, NULL);

	tq_buildstate_flush(&build_state);
	tq_write_meta_page(index_relation, &build_state.meta_fields);

	result = (IndexBuildResult *) palloc0(sizeof(IndexBuildResult));
	result->heap_tuples = heap_tuples;
	result->index_tuples = build_state.index_tuples;

	tq_buildstate_reset(&build_state);

	return result;
}

static void
tq_ambuildempty(Relation index_relation)
{
	TqMetaPageFields meta_fields;

	memset(&meta_fields, 0, sizeof(meta_fields));
	meta_fields.codec = TQ_CODEC_PROD;
	meta_fields.distance = tq_distance_kind_from_index(index_relation);
	meta_fields.directory_root_block = TQ_INVALID_BLOCK_NUMBER;
	meta_fields.centroid_root_block = TQ_INVALID_BLOCK_NUMBER;
	meta_fields.transform = TQ_TRANSFORM_HADAMARD;
	meta_fields.transform_version = TQ_TRANSFORM_CONTRACT_VERSION;
	meta_fields.normalized = true;
	meta_fields.router_seed = 20260327;
	meta_fields.router_sample_count = 256;
	meta_fields.router_max_iterations = 8;
	meta_fields.router_completed_iterations = 0;
	meta_fields.router_trained_vector_count = 0;
	meta_fields.router_algorithm = TQ_ROUTER_ALGORITHM_FIRST_K;
	meta_fields.router_restart_count = 3;
	meta_fields.router_selected_restart = 0;
	meta_fields.router_mean_distortion = 0.0f;
	meta_fields.router_max_list_over_avg = 0.0f;
	meta_fields.router_coeff_var = 0.0f;
	meta_fields.router_balance_penalty = 0.0f;
	meta_fields.router_selection_score = 0.0f;
	tq_write_meta_page(index_relation, &meta_fields);
}

static bool
tq_aminsert(Relation index_relation, Datum *values, bool *isnull,
			ItemPointer heap_tid, Relation heap_relation,
			IndexUniqueCheck check_unique, bool index_unchanged,
			IndexInfo *index_info)
{
	Buffer		meta_buffer = InvalidBuffer;
	TqMetaPageFields meta_fields;
	TqOptionConfig option_config;
	TqTransformConfig transform_config;
	TqTransformMetadata transform_metadata;
	TqTransformState transform_state;
	TqProdCodecConfig prod_config;
	TqProdPackedLayout prod_layout;
	TqRouterModel router_model;
	TqTid		page_tid;
	float	   *source_values = NULL;
	float	   *transformed_values = NULL;
	uint8_t	   *packed_code = NULL;
	uint32_t	dimension = 0;
	uint32_t	list_id = 0;
	int			resolved_lane_count = 0;
	TqLaneConfig lane_config;
	TqDistanceKind distance_kind;
	TqVectorInputKind input_kind;
	bool		needs_meta_write = false;
	char		error_buf[256];

	(void) heap_relation;
	(void) check_unique;
	(void) index_unchanged;
	(void) index_info;

	if (isnull[0])
		return false;

	memset(&meta_fields, 0, sizeof(meta_fields));
	memset(&option_config, 0, sizeof(option_config));
	memset(&transform_config, 0, sizeof(transform_config));
	memset(&transform_metadata, 0, sizeof(transform_metadata));
	memset(&transform_state, 0, sizeof(transform_state));
	memset(&prod_config, 0, sizeof(prod_config));
	memset(&prod_layout, 0, sizeof(prod_layout));
	memset(&router_model, 0, sizeof(router_model));
	memset(&page_tid, 0, sizeof(page_tid));
	memset(&lane_config, 0, sizeof(lane_config));
	memset(&distance_kind, 0, sizeof(distance_kind));
	memset(&input_kind, 0, sizeof(input_kind));

	distance_kind = tq_distance_kind_from_index(index_relation);
	input_kind = tq_input_kind_from_index(index_relation);

	meta_buffer = tq_lock_meta_page_buffer(index_relation);
	if (!tq_read_meta_page_buffer(meta_buffer, &meta_fields, error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	if (meta_fields.dimension == 0)
	{
		tq_load_option_config(index_relation, &option_config);
		if (!tq_vector_dimension_from_datum_typed(values[0], input_kind,
												  &dimension,
												  error_buf, sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);

		transform_config.kind = TQ_TRANSFORM_HADAMARD;
		transform_config.dimension = dimension;
		transform_config.seed = UINT64_C(0);
		if (!tq_transform_metadata_init(&transform_config, &transform_metadata,
										error_buf, sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);

		prod_config.dimension = transform_metadata.output_dimension;
		prod_config.bits = (uint8_t) option_config.bits;
		if (!tq_prod_packed_layout(&prod_config, &prod_layout, error_buf, sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);

		lane_config.block_size = BLCKSZ;
		lane_config.dimension = (int) transform_metadata.output_dimension;
		lane_config.bits = option_config.bits;
		lane_config.codec = TQ_CODEC_PROD;
		lane_config.normalized = option_config.normalized;
		lane_config.page_header_bytes = TQ_PAGE_HEADER_BYTES;
		lane_config.special_space_bytes = TQ_PAGE_SPECIAL_BYTES;
		lane_config.reserve_bytes = TQ_PAGE_RESERVED_BYTES;
		lane_config.tid_bytes = TQ_TID_BYTES;
		if (!tq_resolve_lane_count(&lane_config, &resolved_lane_count,
								   error_buf, sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);

		meta_fields.dimension = dimension;
		meta_fields.transform_output_dimension = transform_metadata.output_dimension;
		meta_fields.codec = TQ_CODEC_PROD;
		meta_fields.distance = distance_kind;
		meta_fields.bits = (uint16_t) option_config.bits;
		meta_fields.lane_count = (uint16_t) resolved_lane_count;
		meta_fields.transform = TQ_TRANSFORM_HADAMARD;
		meta_fields.transform_version = transform_metadata.contract_version;
		meta_fields.normalized = option_config.normalized;
		meta_fields.list_count = option_config.lists > 0 ? (uint32_t) option_config.lists : 0;
		meta_fields.directory_root_block = TQ_INVALID_BLOCK_NUMBER;
		meta_fields.centroid_root_block = TQ_INVALID_BLOCK_NUMBER;
		meta_fields.transform_seed = transform_metadata.seed;
		meta_fields.router_seed = (uint32_t) option_config.router_seed;
		meta_fields.router_sample_count = (uint32_t) option_config.router_samples;
		meta_fields.router_max_iterations = (uint32_t) option_config.router_iterations;
		meta_fields.router_completed_iterations = 0;
		meta_fields.router_trained_vector_count = 0;
		meta_fields.router_algorithm = TQ_ROUTER_ALGORITHM_FIRST_K;
		meta_fields.router_restart_count = (uint32_t) option_config.router_restarts;
		meta_fields.router_selected_restart = 0;
		meta_fields.router_mean_distortion = 0.0f;
		meta_fields.router_max_list_over_avg = 0.0f;
		meta_fields.router_coeff_var = 0.0f;
		meta_fields.router_balance_penalty = 0.0f;
		meta_fields.router_selection_score = 0.0f;
		needs_meta_write = true;
	}
	else
	{
		dimension = meta_fields.dimension;
		prod_config.dimension = meta_fields.transform_output_dimension;
		prod_config.bits = (uint8_t) meta_fields.bits;
		if (!tq_prod_packed_layout(&prod_config, &prod_layout, error_buf, sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);
	}

	source_values = (float *) palloc(sizeof(float) * (size_t) dimension);
	transformed_values = (float *) palloc(sizeof(float) * (size_t) prod_config.dimension);
	packed_code = (uint8_t *) palloc(prod_layout.total_bytes);

	if (!tq_vector_copy_from_datum_typed(values[0], input_kind,
										 source_values, dimension, &dimension,
										 error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);
	if (dimension != meta_fields.dimension)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turboquant insert vector dimension does not match index dimension")));

	transform_config.kind = meta_fields.transform;
	transform_config.dimension = meta_fields.dimension;
	transform_config.seed = meta_fields.transform_seed;
	if (!tq_transform_metadata_init(&transform_config, &transform_metadata,
									error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);
	transform_metadata.output_dimension = meta_fields.transform_output_dimension;
	transform_metadata.contract_version = meta_fields.transform_version;
	if (!tq_transform_prepare_metadata(&transform_metadata, &transform_state,
							  error_buf, sizeof(error_buf))
		|| !tq_transform_apply(&transform_state, source_values, transformed_values,
							   prod_config.dimension, error_buf, sizeof(error_buf))
		|| !tq_prod_encode(&prod_config, transformed_values, packed_code,
						   prod_layout.total_bytes, error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	if (meta_fields.list_count > 0)
	{
		if (meta_fields.centroid_root_block == TQ_INVALID_BLOCK_NUMBER
			|| meta_fields.directory_root_block == TQ_INVALID_BLOCK_NUMBER)
		{
			TqListDirEntry *entries;
			TqRouterTrainingConfig router_config;
			uint32_t	index = 0;

			entries = (TqListDirEntry *) palloc0(sizeof(TqListDirEntry) * (size_t) meta_fields.list_count);
			for (index = 0; index < meta_fields.list_count; index++)
			{
				entries[index].list_id = index;
				entries[index].head_block = TQ_INVALID_BLOCK_NUMBER;
				entries[index].tail_block = TQ_INVALID_BLOCK_NUMBER;
			}

			memset(&router_config, 0, sizeof(router_config));
			router_config.seed = meta_fields.router_seed;
			router_config.sample_count = meta_fields.router_sample_count;
			router_config.max_iterations = meta_fields.router_max_iterations;
			router_config.restart_count = meta_fields.router_restart_count;

			if (!tq_router_train_kmeans(transformed_values, 1, meta_fields.transform_output_dimension,
										meta_fields.list_count, &router_config,
										&router_model, error_buf, sizeof(error_buf)))
				elog(ERROR, "%s", error_buf);
			meta_fields.router_seed = router_model.metadata.seed;
			meta_fields.router_sample_count = router_model.metadata.sample_count;
			meta_fields.router_max_iterations = router_model.metadata.max_iterations;
			meta_fields.router_completed_iterations = router_model.metadata.completed_iterations;
			meta_fields.router_trained_vector_count = router_model.metadata.trained_vector_count;
			meta_fields.router_algorithm = router_model.metadata.algorithm;
			meta_fields.router_restart_count = router_model.metadata.restart_count;
			meta_fields.router_selected_restart = router_model.metadata.selected_restart;
			meta_fields.router_mean_distortion = router_model.metadata.mean_distortion;
			meta_fields.router_max_list_over_avg = router_model.metadata.max_list_over_avg;
			meta_fields.router_coeff_var = router_model.metadata.coeff_var;
			meta_fields.router_balance_penalty = router_model.metadata.balance_penalty;
			meta_fields.router_selection_score = router_model.metadata.selection_score;
			tq_write_centroid_pages(index_relation, &router_model, &meta_fields.centroid_root_block);
			tq_write_directory_pages(index_relation, entries, meta_fields.list_count,
									 &meta_fields.directory_root_block);
			tq_router_reset(&router_model);
			pfree(entries);
			needs_meta_write = true;
		}

		if (!tq_load_router_model(index_relation, &meta_fields, &router_model,
								  error_buf, sizeof(error_buf))
			|| !tq_router_assign_best(&router_model, transformed_values,
									  &list_id, NULL, error_buf, sizeof(error_buf)))
		{
			tq_router_reset(&router_model);
			elog(ERROR, "%s", error_buf);
		}
		tq_router_reset(&router_model);
	}

	page_tid.block_number = ItemPointerGetBlockNumber(heap_tid);
	page_tid.offset_number = ItemPointerGetOffsetNumber(heap_tid);

	if (needs_meta_write)
		tq_write_meta_page_buffer(index_relation, meta_buffer, &meta_fields);

	tq_append_packed_tuple(index_relation, &meta_fields, &prod_config, list_id,
						   &page_tid, packed_code, prod_layout.total_bytes);
	LockBuffer(meta_buffer, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(meta_buffer);

	tq_transform_reset(&transform_state);
	pfree(source_values);
	pfree(transformed_values);
	pfree(packed_code);
	return false;
}

static IndexBulkDeleteResult *
tq_ambulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
				IndexBulkDeleteCallback callback, void *callback_state)
{
	TqMetaPageFields meta_fields;
	TqVacuumSummary summary;
	char		error_buf[256];

	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));

	memset(&summary, 0, sizeof(summary));
	memset(&meta_fields, 0, sizeof(meta_fields));
	if (!tq_read_meta_page(info->index, &meta_fields, error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	tq_summarize_index(info->index, &meta_fields, callback, callback_state, true, &summary);

	stats->num_pages = RelationGetNumberOfBlocks(info->index);
	stats->estimated_count = false;
	stats->num_index_tuples = summary.live_tuples;
	stats->tuples_removed += summary.tuples_removed;
	stats->pages_newly_deleted = 0;
	stats->pages_deleted = summary.reclaimable_pages;
	stats->pages_free = summary.reclaimable_pages;
	return stats;
}

static IndexBulkDeleteResult *
tq_amvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	TqMetaPageFields meta_fields;
	TqVacuumSummary summary;
	char		error_buf[256];

	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));

	memset(&summary, 0, sizeof(summary));
	memset(&meta_fields, 0, sizeof(meta_fields));
	if (!tq_read_meta_page(info->index, &meta_fields, error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	/*
	 * v1 keeps append-only payload pages and dead bitmaps. REINDEX is the
	 * explicit refresh path when fragmentation or router drift needs more than
	 * dead-row cleanup can provide.
	 */
	tq_summarize_index(info->index, &meta_fields, NULL, NULL, false, &summary);

	stats->num_pages = RelationGetNumberOfBlocks(info->index);
	stats->estimated_count = false;
	stats->num_index_tuples = summary.live_tuples;
	stats->pages_deleted = summary.reclaimable_pages;
	stats->pages_free = summary.reclaimable_pages;
	return stats;
}

static void
tq_amcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
				  Cost *index_startup_cost, Cost *index_total_cost,
				  Selectivity *index_selectivity, double *index_correlation,
				  double *index_pages)
{
	(void) loop_count;

	if (path->indexorderbys != NIL)
	{
		Relation	relation = NULL;
		TqOptionConfig config;
		TqPlannerCostEstimate estimate;
		double		output_rows = path->path.rows;

		(void) root;
		memset(&config, 0, sizeof(config));
		memset(&estimate, 0, sizeof(estimate));

		relation = RelationIdGetRelation(path->indexinfo->indexoid);
		if (relation != NULL)
		{
			tq_load_option_config(relation, &config);
			RelationClose(relation);
		}
		else
			tq_load_option_config(NULL, &config);

		if (output_rows <= 0.0 || !isfinite(output_rows))
			output_rows = clamp_row_est(path->indexinfo->tuples * 0.1);

		if (tq_estimate_ordered_scan_cost(path->indexinfo->pages,
										  path->indexinfo->tuples,
										  output_rows,
										  config.lists,
										  tq_guc_probes,
										  tq_guc_oversample_factor,
										  tq_guc_max_visited_codes,
										  tq_guc_max_visited_pages,
										  cpu_index_tuple_cost,
										  cpu_operator_cost,
										  random_page_cost,
										  cpu_tuple_cost,
										  &estimate))
		{
			*index_startup_cost = estimate.startup_cost;
			*index_total_cost = estimate.total_cost;
			*index_selectivity = estimate.selectivity * estimate.scanned_fraction;
			*index_correlation = 0.0;
			*index_pages = estimate.pages_fetched;
			return;
		}
	}

	{
		double		tuples = Max(path->indexinfo->tuples, 1.0);
		double		output_rows = path->path.rows;
		double		selectivity;
		double		pages_fetched;

		(void) root;
		if (output_rows <= 0.0 || !isfinite(output_rows))
			output_rows = clamp_row_est(tuples * 0.25);
		output_rows = Min(output_rows, tuples);
		selectivity = output_rows / tuples;
		pages_fetched = Max(1.0, ceil(path->indexinfo->pages * Max(selectivity, 0.05)));

		*index_startup_cost = cpu_operator_cost;
		*index_total_cost = *index_startup_cost
			+ (pages_fetched * random_page_cost)
			+ (tuples * cpu_index_tuple_cost)
			+ (output_rows * cpu_operator_cost);
		*index_selectivity = selectivity;
		*index_correlation = 0.0;
		*index_pages = pages_fetched;
	}
}

static bytea *
tq_amoptions(Datum reloptions, bool validate)
{
	return tq_reloptions(reloptions, validate);
}

static bool
tq_amvalidate(Oid opclass_oid)
{
	(void) opclass_oid;
	return true;
}

static IndexScanDesc
tq_ambeginscan(Relation index_relation, int nkeys, int norderbys)
{
	IndexScanDesc scan = RelationGetIndexScan(index_relation, nkeys, norderbys);

	scan->opaque = palloc0(sizeof(TqScanOpaque));
	if (norderbys > 0)
	{
		scan->xs_orderbyvals = (Datum *) palloc0(sizeof(Datum) * norderbys);
		scan->xs_orderbynulls = (bool *) palloc0(sizeof(bool) * norderbys);
	}

	return scan;
}

static void
tq_amrescan(IndexScanDesc scan, ScanKey keys, int nkeys,
			ScanKey orderbys, int norderbys)
{
	TqScanOpaque *opaque = (TqScanOpaque *) scan->opaque;

	if (keys != NULL && nkeys > 0 && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys,
				sizeof(ScanKeyData) * (size_t) scan->numberOfKeys);

	if (orderbys != NULL && norderbys > 0 && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys,
				sizeof(ScanKeyData) * (size_t) scan->numberOfOrderBys);

	tq_reset_scan_opaque(opaque);
	scan->xs_recheck = false;
	scan->xs_recheckorderby = false;
}

static bool
tq_load_router_model(Relation index_relation,
					 const TqMetaPageFields *meta_fields,
					 TqRouterModel *model,
					 char *errmsg,
					 size_t errmsg_len)
{
	BlockNumber block_number = meta_fields->centroid_root_block;
	uint32_t	loaded = 0;

	if (meta_fields->list_count == 0
		|| meta_fields->centroid_root_block == TQ_INVALID_BLOCK_NUMBER)
	{
		(void) snprintf(errmsg, errmsg_len,
						"invalid turboquant router storage: centroid root block is missing");
		return false;
	}

	memset(model, 0, sizeof(*model));
	model->dimension = meta_fields->transform_output_dimension;
	model->list_count = meta_fields->list_count;
	model->centroids = (float *) calloc((size_t) meta_fields->list_count
										* (size_t) meta_fields->transform_output_dimension,
										sizeof(float));
	if (model->centroids == NULL)
	{
		(void) snprintf(errmsg, errmsg_len,
						"invalid turboquant router storage: out of memory");
		return false;
	}

	while (block_number != TQ_INVALID_BLOCK_NUMBER && loaded < meta_fields->list_count)
	{
		Buffer		buffer;
		Page		page;
		TqCentroidPageHeaderView header;
		uint16_t	index = 0;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
		page = BufferGetPage(buffer);
		memset(&header, 0, sizeof(header));
		if (!tq_centroid_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
										  &header, errmsg, errmsg_len))
		{
			ReleaseBuffer(buffer);
			tq_router_reset(model);
			return false;
		}

		if (header.dimension != meta_fields->transform_output_dimension)
		{
			ReleaseBuffer(buffer);
			tq_router_reset(model);
			(void) snprintf(errmsg, errmsg_len,
							"invalid turboquant router storage: centroid dimension does not match meta page");
			return false;
		}

		for (index = 0; index < header.centroid_count && loaded < meta_fields->list_count; index++)
		{
			float	   *dst = model->centroids
				+ ((size_t) loaded * (size_t) meta_fields->transform_output_dimension);

			if (!tq_centroid_page_get_centroid(tq_page_payload(page), tq_page_payload_size(page),
											   index, dst, meta_fields->transform_output_dimension,
											   errmsg, errmsg_len))
			{
				ReleaseBuffer(buffer);
				tq_router_reset(model);
				return false;
			}
			loaded += 1;
		}

			block_number = header.next_block;
			ReleaseBuffer(buffer);
	}

	if (loaded != meta_fields->list_count)
	{
		tq_router_reset(model);
		(void) snprintf(errmsg, errmsg_len,
						"invalid turboquant router storage: expected %u centroids, loaded %u",
						meta_fields->list_count, loaded);
		return false;
	}

	return true;
}

static bool
tq_read_list_directory_entry(Relation index_relation,
							 BlockNumber root_block,
							 uint32_t list_id,
							 TqListDirEntry *entry,
							 char *errmsg,
							 size_t errmsg_len)
{
	BlockNumber block_number = root_block;
	uint32_t	base_list_id = 0;

	while (block_number != TQ_INVALID_BLOCK_NUMBER)
	{
		Buffer		buffer;
		Page		page;
		TqListDirPageHeaderView header;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
		page = BufferGetPage(buffer);
		memset(&header, 0, sizeof(header));
		if (!tq_list_dir_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
										  &header, errmsg, errmsg_len))
		{
			ReleaseBuffer(buffer);
			return false;
		}

		if (list_id < base_list_id + (uint32_t) header.entry_count)
		{
			uint16_t	local_index = (uint16_t) (list_id - base_list_id);
			bool		ok = tq_list_dir_page_get_entry(tq_page_payload(page),
													 tq_page_payload_size(page),
													 local_index,
													 entry,
													 errmsg,
													 errmsg_len);

			ReleaseBuffer(buffer);
			if (ok && entry->list_id != list_id)
			{
				(void) snprintf(errmsg, errmsg_len,
								"invalid turboquant list directory: expected list %u, found %u",
								list_id, entry->list_id);
				return false;
			}
			return ok;
		}

		base_list_id += header.entry_count;
		block_number = header.next_block;
		ReleaseBuffer(buffer);
	}

	(void) snprintf(errmsg, errmsg_len,
					"invalid turboquant list directory: list %u was not found",
					list_id);
	return false;
}

static bool
tq_write_list_directory_entry(Relation index_relation,
							  BlockNumber root_block,
							  uint32_t list_id,
							  const TqListDirEntry *entry,
							  char *errmsg,
							  size_t errmsg_len)
{
	BlockNumber block_number = root_block;
	uint32_t	base_list_id = 0;

	while (block_number != TQ_INVALID_BLOCK_NUMBER)
	{
		Buffer		buffer;
		Page		page;
		TqListDirPageHeaderView header;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
		LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buffer);
		memset(&header, 0, sizeof(header));
		if (!tq_list_dir_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
										  &header, errmsg, errmsg_len))
		{
			LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
			ReleaseBuffer(buffer);
			return false;
		}

		if (list_id < base_list_id + (uint32_t) header.entry_count)
		{
			uint16_t	local_index = (uint16_t) (list_id - base_list_id);
			bool		ok = tq_wal_set_list_dir_entry(index_relation, buffer,
													local_index, entry,
													errmsg, errmsg_len);
			LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
			ReleaseBuffer(buffer);
			return ok;
		}

		base_list_id += header.entry_count;
		block_number = header.next_block;
		LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
		ReleaseBuffer(buffer);
	}

	(void) snprintf(errmsg, errmsg_len,
					"invalid turboquant list directory: list %u was not found for update",
					list_id);
	return false;
}

static BlockNumber
tq_find_reusable_batch_block(Relation index_relation,
							 BlockNumber head_block,
							 BlockNumber tail_block,
							 char *errmsg,
							 size_t errmsg_len)
{
	BlockNumber block_number = TQ_INVALID_BLOCK_NUMBER;

	if (head_block != TQ_INVALID_BLOCK_NUMBER)
		block_number = head_block;
	else if (tail_block != TQ_INVALID_BLOCK_NUMBER)
		block_number = tail_block;

	while (block_number != TQ_INVALID_BLOCK_NUMBER)
	{
		Buffer		buffer;
		Page		page;
		TqBatchPageHeaderView header;
		bool		has_capacity = false;
		BlockNumber next_block = TQ_INVALID_BLOCK_NUMBER;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
		page = BufferGetPage(buffer);
		memset(&header, 0, sizeof(header));
		if (!tq_batch_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
									   &header, errmsg, errmsg_len)
			|| !tq_batch_page_has_capacity(tq_page_payload(page), tq_page_payload_size(page),
										   &has_capacity, errmsg, errmsg_len))
		{
			ReleaseBuffer(buffer);
			return TQ_INVALID_BLOCK_NUMBER;
		}

		next_block = header.next_block;
		ReleaseBuffer(buffer);

		if (has_capacity)
			return block_number;

		if (head_block == TQ_INVALID_BLOCK_NUMBER)
		{
			if (block_number <= 1)
				break;
			block_number--;
		}
		else
			block_number = next_block;
	}

	return TQ_INVALID_BLOCK_NUMBER;
}

static BlockNumber
tq_find_detached_free_batch_block(Relation index_relation,
								  char *errmsg,
								  size_t errmsg_len)
{
	BlockNumber block_number = 1;
	BlockNumber nblocks = RelationGetNumberOfBlocks(index_relation);

	for (block_number = 1; block_number < nblocks; block_number++)
	{
		Buffer		buffer;
		Page		page;
		TqPageKind page_kind;
		TqBatchPageHeaderView header;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
		page = BufferGetPage(buffer);
		if (!tq_page_read_kind(tq_page_payload(page), tq_page_payload_size(page),
							   &page_kind, errmsg, errmsg_len))
		{
			ReleaseBuffer(buffer);
			return TQ_INVALID_BLOCK_NUMBER;
		}
		if (page_kind != TQ_PAGE_KIND_BATCH)
		{
			ReleaseBuffer(buffer);
			continue;
		}
		memset(&header, 0, sizeof(header));
		if (!tq_batch_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
									   &header, errmsg, errmsg_len))
		{
			ReleaseBuffer(buffer);
			return TQ_INVALID_BLOCK_NUMBER;
		}
		ReleaseBuffer(buffer);

		if (header.list_id == TQ_DETACHED_FREE_LIST_ID
			&& header.occupied_count == 0
			&& header.live_count == 0
			&& header.next_block == TQ_INVALID_BLOCK_NUMBER)
			return block_number;
	}

	return TQ_INVALID_BLOCK_NUMBER;
}

static void
tq_truncate_reusable_tail_pages(Relation index_relation,
								  const TqMetaPageFields *meta_fields,
								  char *errmsg,
								  size_t errmsg_len)
{
	BlockNumber nblocks = RelationGetNumberOfBlocks(index_relation);
	BlockNumber new_nblocks = nblocks;

	while (new_nblocks > 1)
	{
		Buffer		buffer;
		Page		page;
		TqPageKind page_kind;
		TqBatchPageHeaderView header;
		bool		is_reusable_tail = false;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, new_nblocks - 1, RBM_NORMAL, NULL);
		page = BufferGetPage(buffer);
		if (!tq_page_read_kind(tq_page_payload(page), tq_page_payload_size(page),
							   &page_kind, errmsg, errmsg_len))
		{
			ReleaseBuffer(buffer);
			elog(ERROR, "%s", errmsg);
		}
		if (page_kind != TQ_PAGE_KIND_BATCH)
		{
			ReleaseBuffer(buffer);
			break;
		}
		memset(&header, 0, sizeof(header));
		if (!tq_batch_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
									   &header, errmsg, errmsg_len))
		{
			ReleaseBuffer(buffer);
			elog(ERROR, "%s", errmsg);
		}
		ReleaseBuffer(buffer);

		is_reusable_tail = header.occupied_count == 0
			&& header.live_count == 0
			&& header.next_block == TQ_INVALID_BLOCK_NUMBER
			&& (meta_fields->list_count == 0 || header.list_id == TQ_DETACHED_FREE_LIST_ID);

		if (!is_reusable_tail)
			break;

		new_nblocks--;
	}

	if (new_nblocks < nblocks)
		RelationTruncate(index_relation, new_nblocks);
}

static void
tq_append_packed_tuple(Relation index_relation,
					   const TqMetaPageFields *meta_fields,
					   const TqProdCodecConfig *prod_config,
					   uint32_t list_id,
					   const TqTid *heap_tid,
					   const uint8_t *packed_code,
					   size_t packed_code_len)
{
	BlockNumber head_block = TQ_INVALID_BLOCK_NUMBER;
	BlockNumber tail_block = TQ_INVALID_BLOCK_NUMBER;
	BlockNumber reusable_block = TQ_INVALID_BLOCK_NUMBER;
	Buffer		buffer = InvalidBuffer;
	uint16_t	lane_index = 0;
	char		error_buf[256];
	TqListDirEntry entry;
	bool		have_entry = false;

	memset(&entry, 0, sizeof(entry));

	if (meta_fields->list_count == 0)
	{
		if (RelationGetNumberOfBlocks(index_relation) > 1)
		{
			head_block = 1;
			tail_block = RelationGetNumberOfBlocks(index_relation) - 1;
		}
	}
	else
	{
		if (!tq_read_list_directory_entry(index_relation,
										  meta_fields->directory_root_block,
										  list_id,
										  &entry,
										  error_buf,
										  sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);
		head_block = entry.head_block;
		tail_block = entry.tail_block;
		have_entry = true;
	}

	reusable_block = tq_find_reusable_batch_block(index_relation,
												  head_block,
												  tail_block,
												  error_buf,
												  sizeof(error_buf));

	if (reusable_block != TQ_INVALID_BLOCK_NUMBER)
		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, reusable_block, RBM_NORMAL, NULL);

	if (!BufferIsValid(buffer) && have_entry)
	{
		BlockNumber detached_block = tq_find_detached_free_batch_block(index_relation,
																		error_buf,
																		sizeof(error_buf));
		if (detached_block != TQ_INVALID_BLOCK_NUMBER)
		{
			TqBatchPageParams params;

			memset(&params, 0, sizeof(params));
			params.lane_count = meta_fields->lane_count;
			params.code_bytes = (uint32_t) packed_code_len;
			params.list_id = list_id;
			params.next_block = TQ_INVALID_BLOCK_NUMBER;

			buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, detached_block, RBM_NORMAL, NULL);
			LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
			if (!tq_wal_init_batch_page(index_relation, buffer, &params,
									   error_buf, sizeof(error_buf)))
			{
				LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
				ReleaseBuffer(buffer);
				elog(ERROR, "%s", error_buf);
			}
			LockBuffer(buffer, BUFFER_LOCK_UNLOCK);

			if (entry.head_block == TQ_INVALID_BLOCK_NUMBER)
				entry.head_block = detached_block;
			if (entry.tail_block != TQ_INVALID_BLOCK_NUMBER)
				tq_update_batch_page_next_block(index_relation, entry.tail_block, detached_block);
			entry.tail_block = detached_block;
		}
	}

	if (!BufferIsValid(buffer))
	{
		TqBatchPageParams params;
		BlockNumber new_block;

		memset(&params, 0, sizeof(params));
		params.lane_count = meta_fields->lane_count;
		params.code_bytes = (uint32_t) packed_code_len;
		params.list_id = list_id;
		params.next_block = TQ_INVALID_BLOCK_NUMBER;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, P_NEW, RBM_NORMAL, NULL);
		LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
		new_block = BufferGetBlockNumber(buffer);

		if (!tq_wal_init_batch_page(index_relation, buffer, &params,
								   error_buf, sizeof(error_buf)))
		{
			LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
			ReleaseBuffer(buffer);
			elog(ERROR, "%s", error_buf);
		}
		LockBuffer(buffer, BUFFER_LOCK_UNLOCK);

		if (tail_block != TQ_INVALID_BLOCK_NUMBER)
			tq_update_batch_page_next_block(index_relation, tail_block, new_block);

		tail_block = new_block;

		if (have_entry)
		{
			if (entry.head_block == TQ_INVALID_BLOCK_NUMBER)
				entry.head_block = new_block;
			entry.tail_block = new_block;
		}
	}

	LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
	if (!tq_wal_append_batch_code(index_relation, buffer, heap_tid,
								  packed_code, packed_code_len, &lane_index,
								  error_buf, sizeof(error_buf))
		|| !tq_refresh_batch_page_summary(index_relation,
										  buffer,
										  prod_config,
										  error_buf,
										  sizeof(error_buf)))
	{
		LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
		ReleaseBuffer(buffer);
		elog(ERROR, "%s", error_buf);
	}
	LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buffer);

	if (have_entry)
	{
		entry.live_count += 1;
		entry.free_lane_hint = 0;
		if (!tq_write_list_directory_entry(index_relation,
										   meta_fields->directory_root_block,
										   list_id,
										   &entry,
										   error_buf,
										   sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);
	}
}

static void
tq_summarize_index(Relation index_relation,
				   const TqMetaPageFields *meta_fields,
				   IndexBulkDeleteCallback callback,
				   void *callback_state,
				   bool apply_deletes,
				   TqVacuumSummary *summary)
{
	char		error_buf[256];

	memset(summary, 0, sizeof(*summary));

	if (meta_fields->dimension == 0)
		return;

	if (meta_fields->list_count == 0)
	{
		BlockNumber block_number;
		BlockNumber nblocks = RelationGetNumberOfBlocks(index_relation);

		for (block_number = 1; block_number < nblocks; block_number++)
		{
			Buffer		buffer;
			Page		page;
			TqBatchPageHeaderView header;
			uint16_t	lane = 0;
			bool		should_reclaim = false;

			buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
			LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
			page = BufferGetPage(buffer);
			memset(&header, 0, sizeof(header));
			if (!tq_batch_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
										   &header, error_buf, sizeof(error_buf)))
			{
				LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
				ReleaseBuffer(buffer);
				elog(ERROR, "%s", error_buf);
			}

			if (tq_batch_page_next_live_lane(tq_page_payload(page), tq_page_payload_size(page),
											 -1, &lane, error_buf, sizeof(error_buf)))
			{
				do
				{
					TqTid		tid;
					ItemPointerData itemptr;

					memset(&tid, 0, sizeof(tid));
					if (!tq_batch_page_get_tid(tq_page_payload(page), tq_page_payload_size(page),
											   lane, &tid, error_buf, sizeof(error_buf)))
					{
						LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
						ReleaseBuffer(buffer);
						elog(ERROR, "%s", error_buf);
					}

					ItemPointerSet(&itemptr, tid.block_number, tid.offset_number);
					if (apply_deletes && callback != NULL && callback(&itemptr, callback_state))
					{
						if (!tq_wal_mark_batch_dead(index_relation, buffer, lane,
													error_buf, sizeof(error_buf)))
						{
							LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
							ReleaseBuffer(buffer);
							elog(ERROR, "%s", error_buf);
						}
						summary->tuples_removed += 1.0;
					}
				}
				while (tq_batch_page_next_live_lane(tq_page_payload(page), tq_page_payload_size(page),
													(int) lane, &lane, error_buf, sizeof(error_buf)));
			}

			if (!tq_batch_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
										   &header, error_buf, sizeof(error_buf))
				|| !tq_batch_page_should_reclaim(tq_page_payload(page), tq_page_payload_size(page),
												 &should_reclaim, error_buf, sizeof(error_buf)))
			{
				LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
				ReleaseBuffer(buffer);
				elog(ERROR, "%s", error_buf);
			}

			if (header.live_count != header.occupied_count)
			{
				if (!tq_wal_compact_batch_page(index_relation, buffer,
											   error_buf, sizeof(error_buf))
					|| !tq_batch_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
												  &header, error_buf, sizeof(error_buf))
					|| !tq_batch_page_should_reclaim(tq_page_payload(page), tq_page_payload_size(page),
													 &should_reclaim, error_buf, sizeof(error_buf)))
				{
					LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
					ReleaseBuffer(buffer);
					elog(ERROR, "%s", error_buf);
				}
			}

			summary->live_tuples += header.live_count;
			if (should_reclaim)
				summary->reclaimable_pages += 1;
			LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
			ReleaseBuffer(buffer);
		}
	}
	else
	{
		uint32_t	list_id = 0;

		for (list_id = 0; list_id < meta_fields->list_count; list_id++)
		{
			TqListDirEntry entry;
			BlockNumber block_number;
			BlockNumber prev_block = TQ_INVALID_BLOCK_NUMBER;
			double		list_live = 0.0;

			memset(&entry, 0, sizeof(entry));
			if (!tq_read_list_directory_entry(index_relation, meta_fields->directory_root_block,
											  list_id, &entry, error_buf, sizeof(error_buf)))
				elog(ERROR, "%s", error_buf);

			block_number = entry.head_block;
			while (block_number != TQ_INVALID_BLOCK_NUMBER)
			{
				Buffer		buffer;
				Page		page;
				TqBatchPageHeaderView header;
				TqBatchPageParams params;
				BlockNumber next_block = TQ_INVALID_BLOCK_NUMBER;
				uint16_t	lane = 0;
				bool		should_reclaim = false;

				buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
				LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
				page = BufferGetPage(buffer);
				memset(&header, 0, sizeof(header));
				if (!tq_batch_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
											   &header, error_buf, sizeof(error_buf)))
				{
					LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
					ReleaseBuffer(buffer);
					elog(ERROR, "%s", error_buf);
				}

				if (tq_batch_page_next_live_lane(tq_page_payload(page), tq_page_payload_size(page),
												 -1, &lane, error_buf, sizeof(error_buf)))
				{
					do
					{
						TqTid		tid;
						ItemPointerData itemptr;

						memset(&tid, 0, sizeof(tid));
						if (!tq_batch_page_get_tid(tq_page_payload(page), tq_page_payload_size(page),
												   lane, &tid, error_buf, sizeof(error_buf)))
						{
							LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
							ReleaseBuffer(buffer);
							elog(ERROR, "%s", error_buf);
						}

						ItemPointerSet(&itemptr, tid.block_number, tid.offset_number);
						if (apply_deletes && callback != NULL && callback(&itemptr, callback_state))
						{
							if (!tq_wal_mark_batch_dead(index_relation, buffer, lane,
														error_buf, sizeof(error_buf)))
							{
								LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
								ReleaseBuffer(buffer);
								elog(ERROR, "%s", error_buf);
							}
							summary->tuples_removed += 1.0;
						}
					}
					while (tq_batch_page_next_live_lane(tq_page_payload(page), tq_page_payload_size(page),
														(int) lane, &lane, error_buf, sizeof(error_buf)));
				}

				if (!tq_batch_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
											   &header, error_buf, sizeof(error_buf))
					|| !tq_batch_page_should_reclaim(tq_page_payload(page), tq_page_payload_size(page),
													 &should_reclaim, error_buf, sizeof(error_buf)))
				{
					LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
					ReleaseBuffer(buffer);
					elog(ERROR, "%s", error_buf);
				}

				if (header.live_count != header.occupied_count)
				{
					TqProdCodecConfig prod_config;

					memset(&prod_config, 0, sizeof(prod_config));
					prod_config.dimension = meta_fields->transform_output_dimension;
					prod_config.bits = (uint8_t) meta_fields->bits;

					if (!tq_wal_compact_batch_page(index_relation, buffer,
												   error_buf, sizeof(error_buf))
						|| !tq_refresh_batch_page_summary(index_relation, buffer,
														  &prod_config,
														  error_buf, sizeof(error_buf))
						|| !tq_batch_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
													  &header, error_buf, sizeof(error_buf))
						|| !tq_batch_page_should_reclaim(tq_page_payload(page), tq_page_payload_size(page),
														 &should_reclaim, error_buf, sizeof(error_buf)))
					{
						LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
						ReleaseBuffer(buffer);
						elog(ERROR, "%s", error_buf);
					}
				}

				next_block = header.next_block;
				if (header.live_count == 0)
				{
					memset(&params, 0, sizeof(params));
					params.lane_count = header.lane_count;
					params.code_bytes = header.code_bytes;
					params.list_id = TQ_DETACHED_FREE_LIST_ID;
					params.next_block = TQ_INVALID_BLOCK_NUMBER;

					if (prev_block != TQ_INVALID_BLOCK_NUMBER)
						tq_update_batch_page_next_block(index_relation, prev_block, next_block);
					else
						entry.head_block = next_block;

					if (entry.tail_block == block_number)
						entry.tail_block = prev_block;

					if (!tq_wal_init_batch_page(index_relation, buffer, &params,
											   error_buf, sizeof(error_buf)))
					{
						LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
						ReleaseBuffer(buffer);
						elog(ERROR, "%s", error_buf);
					}

					block_number = next_block;
					LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
					ReleaseBuffer(buffer);
					continue;
				}

				list_live += header.live_count;
				if (should_reclaim)
					summary->reclaimable_pages += 1;
				prev_block = block_number;
				block_number = next_block;
				LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
				ReleaseBuffer(buffer);
			}

			entry.live_count = (uint32_t) list_live;
			entry.dead_count = 0;
			entry.free_lane_hint = 0;
			if (!tq_write_list_directory_entry(index_relation, meta_fields->directory_root_block,
											   list_id, &entry, error_buf, sizeof(error_buf)))
				elog(ERROR, "%s", error_buf);
			summary->live_tuples += list_live;
		}
	}

	tq_truncate_reusable_tail_pages(index_relation, meta_fields,
									 error_buf, sizeof(error_buf));
}

static bool
tq_scan_batch_block(Relation index_relation,
					BlockNumber block_number,
					TqScanOpaque *opaque,
					char *errmsg,
					size_t errmsg_len)
{
	Buffer buffer;
	Page page;

	buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
	page = BufferGetPage(buffer);
	if (!tq_batch_page_scan_prod(tq_page_payload(page),
								 tq_page_payload_size(page),
								 &opaque->prod_config,
								 opaque->normalized,
								 opaque->distance_kind,
								 &opaque->lut,
								 opaque->query_values,
								 opaque->query_dimension,
								 &opaque->candidates,
								 errmsg,
								 errmsg_len))
	{
		ReleaseBuffer(buffer);
		return false;
	}

	ReleaseBuffer(buffer);
	return true;
}

static bool
tq_collect_bounded_pages_for_chain(Relation index_relation,
								   BlockNumber head_block,
								   TqScanOpaque *opaque,
								   TqBoundedPageCandidate *pages,
								   size_t max_pages,
								   size_t *page_count,
								   char *errmsg,
								   size_t errmsg_len)
{
	BlockNumber block_number = head_block;

	while (block_number != TQ_INVALID_BLOCK_NUMBER)
	{
		Buffer		buffer;
		Page		page;
		TqBatchPageHeaderView header;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
		page = BufferGetPage(buffer);
		memset(&header, 0, sizeof(header));
		if (!tq_batch_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
									   &header, errmsg, errmsg_len))
		{
			ReleaseBuffer(buffer);
			return false;
		}

		if (header.live_count > 0)
		{
			if (*page_count >= max_pages)
			{
				snprintf(errmsg, errmsg_len,
						 "invalid turboquant scan: bounded page array exceeded capacity");
				ReleaseBuffer(buffer);
				return false;
			}

			pages[*page_count].block_number = block_number;
			if (!tq_scan_page_optimistic_distance_bound(&opaque->prod_config,
													   &opaque->lut,
													   tq_page_payload(page),
													   tq_page_payload_size(page),
													   opaque->normalized,
													   opaque->distance_kind,
													   opaque->query_values,
													   opaque->query_dimension,
													   &pages[*page_count].optimistic_distance,
													   errmsg,
													   errmsg_len))
			{
				ReleaseBuffer(buffer);
				return false;
			}
			*page_count += 1;
		}

		block_number = header.next_block;
		ReleaseBuffer(buffer);
	}

	return true;
}

static bool
tq_count_batch_pages_for_chain(Relation index_relation,
							   BlockNumber head_block,
							   size_t *page_count,
							   char *errmsg,
							   size_t errmsg_len)
{
	BlockNumber block_number = head_block;

	if (page_count == NULL)
	{
		snprintf(errmsg, errmsg_len,
				 "invalid turboquant scan: page count output must be non-null");
		return false;
	}

	*page_count = 0;
	while (block_number != TQ_INVALID_BLOCK_NUMBER)
	{
		Buffer buffer;
		Page page;
		TqBatchPageHeaderView header;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
		page = BufferGetPage(buffer);
		memset(&header, 0, sizeof(header));
		if (!tq_batch_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
									   &header, errmsg, errmsg_len))
		{
			ReleaseBuffer(buffer);
			return false;
		}

		*page_count += 1;
		block_number = header.next_block;
		ReleaseBuffer(buffer);
	}

	return true;
}

static bool
tq_scan_prepare(IndexScanDesc scan, TqScanOpaque *opaque)
{
	Buffer		buffer;
	TqMetaPageFields meta_fields;
	TqTransformConfig transform_config;
	TqTransformMetadata transform_metadata;
	float	   *query_values;
	float	   *transformed_query;
	uint32_t	query_dimension = 0;
	size_t		total_live = 0;
	char		error_buf[256];

	if (scan->numberOfOrderBys != 1)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant scans require exactly one ORDER BY expression")));

	buffer = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
	memset(&meta_fields, 0, sizeof(meta_fields));
	if (!tq_meta_page_read(tq_page_payload(BufferGetPage(buffer)),
						   tq_page_payload_size(BufferGetPage(buffer)),
						   &meta_fields,
						   error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);
	ReleaseBuffer(buffer);

	tq_scan_stats_begin(meta_fields.list_count == 0 ? TQ_SCAN_MODE_FLAT : TQ_SCAN_MODE_IVF,
						(size_t) tq_guc_probes);

	if (meta_fields.codec != TQ_CODEC_PROD
		|| meta_fields.dimension == 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant v1 scans require a populated prod-codec index")));

	query_values = (float *) palloc(sizeof(float) * (size_t) meta_fields.dimension);
	transformed_query = (float *) palloc(sizeof(float) * (size_t) meta_fields.transform_output_dimension);
	memset(&transform_config, 0, sizeof(transform_config));
	memset(&transform_metadata, 0, sizeof(transform_metadata));

	opaque->distance_kind = meta_fields.distance;
	opaque->input_kind = tq_input_kind_from_index(scan->indexRelation);

	if (!tq_vector_copy_from_datum_typed(scan->orderByData[0].sk_argument,
										 opaque->input_kind,
										 query_values,
										 meta_fields.dimension,
										 &query_dimension,
										 error_buf,
										 sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	if (query_dimension != meta_fields.dimension)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turboquant query vector dimension does not match index dimension")));

	opaque->prod_config.dimension = meta_fields.transform_output_dimension;
	opaque->prod_config.bits = (uint8_t) meta_fields.bits;

	transform_config.kind = meta_fields.transform;
	transform_config.dimension = meta_fields.dimension;
	transform_config.seed = meta_fields.transform_seed;

	if (!tq_transform_metadata_init(&transform_config, &transform_metadata,
									error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);
	transform_metadata.output_dimension = meta_fields.transform_output_dimension;
	transform_metadata.contract_version = meta_fields.transform_version;

	if (!tq_transform_prepare_metadata(&transform_metadata, &opaque->transform_state,
							  error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	if (!tq_transform_apply(&opaque->transform_state, query_values, transformed_query,
							meta_fields.transform_output_dimension,
							error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	opaque->query_values = (float *) palloc(sizeof(float) * (size_t) meta_fields.transform_output_dimension);
	memcpy(opaque->query_values, transformed_query,
		   sizeof(float) * (size_t) meta_fields.transform_output_dimension);
	opaque->query_dimension = meta_fields.transform_output_dimension;
	opaque->normalized = meta_fields.normalized;

	if (!tq_prod_lut_build(&opaque->prod_config, transformed_query, &opaque->lut,
						   error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	pfree(query_values);
	pfree(transformed_query);

	if (meta_fields.list_count == 0)
	{
		BlockNumber block_number;
		BlockNumber nblocks = RelationGetNumberOfBlocks(scan->indexRelation);

		if (!tq_candidate_heap_init(&opaque->candidates,
									tq_streaming_candidate_capacity(tq_guc_probes,
																 tq_guc_oversample_factor)))
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("turboquant could not allocate scan candidate heap")));

		for (block_number = 1; block_number < nblocks; block_number++)
		{
			TqBatchPageHeaderView header;

			buffer = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
			memset(&header, 0, sizeof(header));
			if (!tq_batch_page_read_header(tq_page_payload(BufferGetPage(buffer)),
										   tq_page_payload_size(BufferGetPage(buffer)),
										   &header,
										   error_buf,
										   sizeof(error_buf)))
			{
				ReleaseBuffer(buffer);
				elog(ERROR, "%s", error_buf);
			}
			tq_scan_stats_add_selected_live((size_t) header.live_count);
			if (!tq_batch_page_scan_prod(tq_page_payload(BufferGetPage(buffer)),
										tq_page_payload_size(BufferGetPage(buffer)),
										&opaque->prod_config,
										opaque->normalized,
										opaque->distance_kind,
										&opaque->lut,
										opaque->query_values,
										opaque->query_dimension,
										&opaque->candidates,
										error_buf,
										sizeof(error_buf)))
				elog(ERROR, "%s", error_buf);
			ReleaseBuffer(buffer);
		}
		tq_scan_stats_set_candidate_heap_metrics(opaque->candidates.capacity,
												 opaque->candidates.count);
	}
	else
	{
		TqRouterModel router_model;
		TqProbeBudgetResult probe_budget;
		uint32_t   *selected_lists;
		uint32_t   *ranked_live_counts = NULL;
		uint32_t   *ranked_page_counts = NULL;
		TqBoundedPageCandidate *page_candidates = NULL;
		uint32_t	selected_count = 0;
		uint32_t	effective_count = 0;
		uint32_t	index = 0;
		size_t		page_count = 0;
		size_t		max_pages = 0;

		memset(&router_model, 0, sizeof(router_model));
		if (!tq_load_router_model(scan->indexRelation, &meta_fields, &router_model,
								  error_buf, sizeof(error_buf)))
			elog(ERROR, "%s", error_buf);

		selected_lists = (uint32_t *) palloc(sizeof(uint32_t) * (size_t) meta_fields.list_count);
		if (!tq_router_select_probes(&router_model, opaque->query_values,
									 (uint32_t) tq_guc_probes,
									 selected_lists,
									 meta_fields.list_count,
									 &selected_count,
									 error_buf, sizeof(error_buf)))
		{
			tq_router_reset(&router_model);
			elog(ERROR, "%s", error_buf);
		}

		ranked_live_counts = (uint32_t *) palloc0(sizeof(uint32_t) * (size_t) selected_count);
		ranked_page_counts = (uint32_t *) palloc0(sizeof(uint32_t) * (size_t) selected_count);

		for (index = 0; index < selected_count; index++)
		{
			TqListDirEntry entry;
			size_t		list_page_count = 0;

			memset(&entry, 0, sizeof(entry));
			if (!tq_read_list_directory_entry(scan->indexRelation,
											  meta_fields.directory_root_block,
											  selected_lists[index],
											  &entry,
											  error_buf, sizeof(error_buf))
				|| !tq_count_batch_pages_for_chain(scan->indexRelation,
												   entry.head_block,
												   &list_page_count,
											  error_buf, sizeof(error_buf)))
			{
				tq_router_reset(&router_model);
				pfree(ranked_live_counts);
				pfree(ranked_page_counts);
				pfree(selected_lists);
				elog(ERROR, "%s", error_buf);
			}

			ranked_live_counts[index] = entry.live_count;
			ranked_page_counts[index] = (uint32_t) list_page_count;
		}

		memset(&probe_budget, 0, sizeof(probe_budget));
		if (!tq_choose_probe_budget(ranked_live_counts,
									ranked_page_counts,
									(size_t) selected_count,
									tq_guc_probes,
									tq_guc_max_visited_codes,
									tq_guc_max_visited_pages,
									&probe_budget,
									error_buf,
									sizeof(error_buf)))
		{
			tq_router_reset(&router_model);
			pfree(ranked_live_counts);
			pfree(ranked_page_counts);
			pfree(selected_lists);
			elog(ERROR, "%s", error_buf);
		}
		effective_count = (uint32_t) probe_budget.effective_probe_count;
		tq_scan_stats_set_probe_budget(probe_budget.nominal_probe_count,
									   probe_budget.effective_probe_count,
									   probe_budget.max_visited_codes,
									   probe_budget.max_visited_pages);

		for (index = 0; index < effective_count; index++)
		{
			total_live += ranked_live_counts[index];
			tq_scan_stats_record_selected_list((size_t) ranked_live_counts[index]);
		}

		if (total_live > 0
			&& !tq_candidate_heap_init(&opaque->candidates,
									   tq_scan_candidate_capacity(total_live,
																 tq_guc_probes,
																 tq_guc_oversample_factor)))
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("turboquant could not allocate scan candidate heap")));

		max_pages = (size_t) Max((BlockNumber) 1, RelationGetNumberOfBlocks(scan->indexRelation));
		page_candidates = (TqBoundedPageCandidate *) palloc0(sizeof(TqBoundedPageCandidate) * max_pages);

		for (index = 0; index < effective_count; index++)
		{
			TqListDirEntry entry;

			memset(&entry, 0, sizeof(entry));
			if (!tq_read_list_directory_entry(scan->indexRelation,
											  meta_fields.directory_root_block,
											  selected_lists[index],
											  &entry,
											  error_buf, sizeof(error_buf))
				|| !tq_collect_bounded_pages_for_chain(scan->indexRelation,
													   entry.head_block,
													   opaque,
													   page_candidates,
													   max_pages,
													   &page_count,
													   error_buf,
													   sizeof(error_buf)))
			{
				tq_router_reset(&router_model);
				pfree(ranked_live_counts);
				pfree(ranked_page_counts);
				pfree(page_candidates);
				pfree(selected_lists);
				elog(ERROR, "%s", error_buf);
			}
		}

		qsort(page_candidates, page_count, sizeof(page_candidates[0]),
			  tq_bounded_page_candidate_compare);

		for (index = 0; index < page_count; index++)
		{
			bool should_prune = false;

			if (!tq_scan_should_prune_page(&opaque->candidates,
										   page_candidates[index].optimistic_distance,
										   &should_prune,
										   error_buf,
										   sizeof(error_buf)))
			{
				tq_router_reset(&router_model);
				pfree(ranked_live_counts);
				pfree(ranked_page_counts);
				pfree(page_candidates);
				pfree(selected_lists);
				elog(ERROR, "%s", error_buf);
			}

			if (should_prune)
			{
				tq_scan_stats_add_page_prunes(page_count - (size_t) index);
				tq_scan_stats_add_early_stops(1);
				break;
			}

			if (!tq_scan_batch_block(scan->indexRelation,
									 page_candidates[index].block_number,
									 opaque,
									 error_buf,
									 sizeof(error_buf)))
			{
				tq_router_reset(&router_model);
				pfree(ranked_live_counts);
				pfree(ranked_page_counts);
				pfree(page_candidates);
				pfree(selected_lists);
				elog(ERROR, "%s", error_buf);
			}
		}

		tq_scan_stats_set_candidate_heap_metrics(opaque->candidates.capacity,
												 opaque->candidates.count);

		tq_router_reset(&router_model);
		pfree(ranked_live_counts);
		pfree(ranked_page_counts);
		pfree(page_candidates);
		pfree(selected_lists);
	}

	opaque->prepared = true;
	return true;
}

static int64
tq_scan_all_live_tids_to_bitmap(Relation index_relation,
								TIDBitmap *tbm,
								char *errmsg,
								size_t errmsg_len)
{
	BlockNumber nblocks = RelationGetNumberOfBlocks(index_relation);
	BlockNumber block_number;
	int64		match_count = 0;

	for (block_number = 1; block_number < nblocks; block_number++)
	{
		Buffer		buffer;
		Page		page;
		TqPageKind kind;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
		page = BufferGetPage(buffer);
		memset(&kind, 0, sizeof(kind));
		if (!tq_page_read_kind(tq_page_payload(page), tq_page_payload_size(page),
							   &kind, errmsg, errmsg_len))
		{
			ReleaseBuffer(buffer);
			return -1;
		}

		if (kind == TQ_PAGE_KIND_BATCH)
		{
			TqBatchPageHeaderView header;

			memset(&header, 0, sizeof(header));
			if (!tq_batch_page_read_header(tq_page_payload(page), tq_page_payload_size(page),
										   &header, errmsg, errmsg_len))
			{
				ReleaseBuffer(buffer);
				return -1;
			}

			tq_scan_stats_record_page_visit();
			tq_scan_stats_add_selected_live((size_t) header.live_count);

			if (header.live_count > 0)
			{
				uint16_t	lane = 0;

				if (tq_batch_page_next_live_lane(tq_page_payload(page), tq_page_payload_size(page),
												 -1, &lane, errmsg, errmsg_len))
				{
					do
					{
						TqTid tid;
						ItemPointerData itemptr;

						memset(&tid, 0, sizeof(tid));
						if (!tq_batch_page_get_tid(tq_page_payload(page), tq_page_payload_size(page),
												   lane, &tid, errmsg, errmsg_len))
						{
							ReleaseBuffer(buffer);
							return -1;
						}

						ItemPointerSet(&itemptr, tid.block_number, tid.offset_number);
						tbm_add_tuples(tbm, &itemptr, 1, true);
						match_count++;
						tq_scan_stats_record_code_visit(false);
					}
					while (tq_batch_page_next_live_lane(tq_page_payload(page), tq_page_payload_size(page),
														(int) lane, &lane, errmsg, errmsg_len));
				}
			}
		}

		ReleaseBuffer(buffer);
	}

	return match_count;
}

static bool
tq_amgettuple(IndexScanDesc scan, ScanDirection direction)
{
	TqScanOpaque *opaque = (TqScanOpaque *) scan->opaque;
	TqCandidateEntry entry;

	if (direction != ForwardScanDirection)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant scans support forward direction only")));

	if (!opaque->prepared)
		tq_scan_prepare(scan, opaque);

	memset(&entry, 0, sizeof(entry));
	if (!tq_candidate_heap_pop_best(&opaque->candidates, &entry))
		return false;

	ItemPointerSet(&scan->xs_heaptid, entry.tid.block_number, entry.tid.offset_number);
	scan->xs_recheck = false;
	scan->xs_recheckorderby = false;
	scan->xs_heap_continue = false;

	if (scan->xs_orderbyvals != NULL && scan->numberOfOrderBys > 0)
	{
		scan->xs_orderbyvals[0] = Float8GetDatum((double) entry.score);
		scan->xs_orderbynulls[0] = false;
	}

	return true;
}

static int64
tq_amgetbitmap(IndexScanDesc scan, TIDBitmap *tbm)
{
	TqScanOpaque *opaque = (TqScanOpaque *) scan->opaque;
	TqMetaPageFields meta_fields;
	float	   *query_values = NULL;
	uint32_t	query_dimension = 0;
	double		threshold = 0.0;
	char		error_buf[256];
	int64		match_count = 0;

	if (tbm == NULL)
		return 0;

	if (scan->numberOfKeys != 1)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant bitmap scans require exactly one bitmap filter condition")));

	if (scan->numberOfOrderBys != 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant bitmap scans do not support ORDER BY operators")));

	if (scan->keyData[0].sk_flags & SK_ISNULL)
		return 0;

	memset(&meta_fields, 0, sizeof(meta_fields));
	memset(error_buf, 0, sizeof(error_buf));
	tq_scan_stats_begin(TQ_SCAN_MODE_BITMAP, 0);
	tq_scan_stats_set_score_mode(TQ_SCAN_SCORE_MODE_BITMAP_FILTER);
	if (!tq_read_meta_page(scan->indexRelation, &meta_fields, error_buf, sizeof(error_buf)))
		elog(ERROR, "%s", error_buf);

	if (meta_fields.distance != TQ_DISTANCE_COSINE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant bitmap scans currently support cosine opclasses only")));

	if (!tq_bitmap_filter_parse(DatumGetByteaPP(scan->keyData[0].sk_argument),
								TQ_DISTANCE_COSINE,
								&query_values,
								&query_dimension,
								&threshold,
								error_buf,
								sizeof(error_buf)))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));

	pfree(query_values);
	if (query_dimension != meta_fields.dimension)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turboquant bitmap filter dimension does not match index dimension")));

	(void) threshold;
	tq_reset_scan_opaque(opaque);
	scan->xs_recheck = true;
	scan->xs_recheckorderby = false;
	match_count = tq_scan_all_live_tids_to_bitmap(scan->indexRelation, tbm,
												  error_buf, sizeof(error_buf));
	if (match_count < 0)
		elog(ERROR, "%s", error_buf);
	tq_scan_stats_set_candidate_heap_metrics(0, 0);
	return match_count;
}

static void
tq_amendscan(IndexScanDesc scan)
{
	TqScanOpaque *opaque = (TqScanOpaque *) scan->opaque;

	tq_reset_scan_opaque(opaque);
	if (scan->numberOfOrderBys > 0 && scan->xs_orderbyvals != NULL)
		pfree(scan->xs_orderbyvals);
	if (scan->numberOfOrderBys > 0 && scan->xs_orderbynulls != NULL)
		pfree(scan->xs_orderbynulls);
	if (opaque != NULL)
		pfree(opaque);
	scan->opaque = NULL;
}

Datum
turboquanthandler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	(void) fcinfo;

	tq_init_amroutine(amroutine);

	amroutine->ambuild = tq_ambuild;
	amroutine->ambuildempty = tq_ambuildempty;
	amroutine->aminsert = tq_aminsert;
	amroutine->ambulkdelete = tq_ambulkdelete;
	amroutine->amvacuumcleanup = tq_amvacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = tq_amcostestimate;
	amroutine->amoptions = tq_amoptions;
	amroutine->amproperty = NULL;
	amroutine->ambuildphasename = NULL;
	amroutine->amvalidate = tq_amvalidate;
	amroutine->amadjustmembers = NULL;
	amroutine->ambeginscan = tq_ambeginscan;
	amroutine->amrescan = tq_amrescan;
	amroutine->amgettuple = tq_amgettuple;
	amroutine->amgetbitmap = tq_amgetbitmap;
	amroutine->amendscan = tq_amendscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan = NULL;
	amroutine->amparallelrescan = NULL;

	PG_RETURN_POINTER(amroutine);
}
