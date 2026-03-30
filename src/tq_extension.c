#include "postgres.h"

#include <math.h>

#include "access/htup_details.h"
#include "access/tableam.h"
#include "catalog/index.h"
#include "catalog/pg_am.h"
#include "catalog/pg_index.h"
#include "catalog/pg_opclass.h"
#include "catalog/pg_type_d.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/relcache.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/syscache.h"

#include "src/tq_page.h"
#include "src/tq_reloptions.h"
#include "src/tq_router.h"
#include "src/tq_scan.h"
#include "src/tq_simd_avx2.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(tq_smoke);
PG_FUNCTION_INFO_V1(tq_debug_validate_reloptions);
PG_FUNCTION_INFO_V1(tq_debug_router_metadata);
PG_FUNCTION_INFO_V1(tq_debug_transform_metadata);
PG_FUNCTION_INFO_V1(tq_index_metadata_core);
PG_FUNCTION_INFO_V1(tq_last_scan_stats_core);
PG_FUNCTION_INFO_V1(tq_last_shadow_decode_candidate_tids_core);
PG_FUNCTION_INFO_V1(tq_runtime_simd_features_core);

typedef struct TqListAggregate
{
	uint64_t	live_count;
	uint64_t	dead_count;
	uint32_t	batch_page_count;
	uint32_t	head_block;
	uint32_t	tail_block;
} TqListAggregate;

static const char *tq_distance_kind_name(TqDistanceKind distance_kind);
static const char *tq_codec_kind_name(TqCodecKind codec_kind);
static const char *tq_transform_kind_name(TqTransformKind transform_kind);
static const char *tq_router_algorithm_name(TqRouterAlgorithmKind algorithm_kind);
static const char *tq_quantizer_family_name(uint16_t version);
static const char *tq_residual_sketch_kind_name(uint16_t version);
static const char *tq_estimator_mode_name(uint16_t version);
static const char *tq_page_summary_mode_name(const TqMetaPageFields *meta_fields);
static void tq_json_append_string(StringInfo buf, const char *value);
static int tq_u64_compare(const void *left, const void *right);
static bool tq_read_meta_fields(Relation index_relation,
								TqMetaPageFields *meta_fields,
								char *errmsg,
								size_t errmsg_len);
static bool tq_read_list_directory_entry(Relation index_relation,
										 BlockNumber root_block,
										 uint32_t list_id,
										 TqListDirEntry *entry,
										 char *errmsg,
										 size_t errmsg_len);
static bool tq_collect_batch_chain_stats(Relation index_relation,
										 BlockNumber head_block,
										 TqListAggregate *aggregate,
										 uint32_t *reclaimable_pages,
										 char *errmsg,
										 size_t errmsg_len);
static uint32_t tq_count_centroid_pages(Relation index_relation,
										 BlockNumber root_block,
										 char *errmsg,
										 size_t errmsg_len);
static bool tq_append_list_metadata_json(StringInfo buf,
										 Relation index_relation,
										 const TqMetaPageFields *meta_fields,
										 uint64_t *total_live_count,
										 uint64_t *total_dead_count,
										 uint32_t *batch_page_count,
										 uint32_t *reclaimable_pages,
										 char *errmsg,
										 size_t errmsg_len);

static const char *
tq_distance_kind_name(TqDistanceKind distance_kind)
{
	switch (distance_kind)
	{
		case TQ_DISTANCE_COSINE:
			return "cosine";
		case TQ_DISTANCE_IP:
			return "ip";
		case TQ_DISTANCE_L2:
			return "l2";
	}

	return "unknown";
}

static const char *
tq_codec_kind_name(TqCodecKind codec_kind)
{
	switch (codec_kind)
	{
		case TQ_CODEC_MSE:
			return "mse";
		case TQ_CODEC_PROD:
			return "prod";
	}

	return "unknown";
}

static const char *
tq_transform_kind_name(TqTransformKind transform_kind)
{
	switch (transform_kind)
	{
		case TQ_TRANSFORM_HADAMARD:
			return "hadamard";
	}

	return "unknown";
}

static const char *
tq_router_algorithm_name(TqRouterAlgorithmKind algorithm_kind)
{
	switch (algorithm_kind)
	{
		case TQ_ROUTER_ALGORITHM_FIRST_K:
			return "first_k";
		case TQ_ROUTER_ALGORITHM_KMEANS:
			return "kmeans";
	}

	return "unknown";
}

static const char *
tq_quantizer_family_name(uint16_t version)
{
	if (version == TQ_QUANTIZER_VERSION)
		return "beta_lloyd_max";
	return "unknown";
}

static const char *
tq_residual_sketch_kind_name(uint16_t version)
{
	if (version == TQ_RESIDUAL_SKETCH_VERSION)
		return "1bit_qjl";
	return "unknown";
}

static const char *
tq_estimator_mode_name(uint16_t version)
{
	if (version == TQ_ESTIMATOR_VERSION)
		return "qprod_unbiased_ip";
	return "unknown";
}

static const char *
tq_page_summary_mode_name(const TqMetaPageFields *meta_fields)
{
	if (meta_fields == NULL || meta_fields->list_count == 0)
		return "disabled";

	if (meta_fields->normalized
		&& meta_fields->codec == TQ_CODEC_PROD
		&& (meta_fields->distance == TQ_DISTANCE_COSINE
			|| meta_fields->distance == TQ_DISTANCE_IP))
		return "safe_summary_pruning";

	if (meta_fields->list_count > 0)
		return "ordering_only";
	return "disabled";
}

static void
tq_json_append_string(StringInfo buf, const char *value)
{
	const unsigned char *ptr = (const unsigned char *) value;

	appendStringInfoChar(buf, '"');
	if (value != NULL)
	{
		while (*ptr != '\0')
		{
			switch (*ptr)
			{
				case '\\':
				case '"':
					appendStringInfoChar(buf, '\\');
					appendStringInfoChar(buf, (char) *ptr);
					break;
				case '\n':
					appendStringInfoString(buf, "\\n");
					break;
				case '\r':
					appendStringInfoString(buf, "\\r");
					break;
				case '\t':
					appendStringInfoString(buf, "\\t");
					break;
				default:
					appendStringInfoChar(buf, (char) *ptr);
					break;
			}
			ptr++;
		}
	}
	appendStringInfoChar(buf, '"');
}

static int
tq_u64_compare(const void *left, const void *right)
{
	const uint64_t *lhs = (const uint64_t *) left;
	const uint64_t *rhs = (const uint64_t *) right;

	if (*lhs < *rhs)
		return -1;
	if (*lhs > *rhs)
		return 1;
	return 0;
}

static bool
tq_read_meta_fields(Relation index_relation,
					   TqMetaPageFields *meta_fields,
					   char *errmsg,
					   size_t errmsg_len)
{
	Buffer		buffer;
	Page		page;
	bool		ok;

	buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
	page = BufferGetPage(buffer);
	ok = tq_meta_page_read(PageGetContents(page),
						   (size_t) (PageGetPageSize(page) - SizeOfPageHeaderData),
						   meta_fields,
						   errmsg,
						   errmsg_len);
	ReleaseBuffer(buffer);
	return ok;
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
		if (!tq_list_dir_page_read_header(PageGetContents(page),
										  (size_t) (PageGetPageSize(page) - SizeOfPageHeaderData),
										  &header,
										  errmsg,
										  errmsg_len))
		{
			ReleaseBuffer(buffer);
			return false;
		}

		if (list_id < base_list_id + (uint32_t) header.entry_count)
		{
			uint16_t	local_index = (uint16_t) (list_id - base_list_id);
			bool		ok = tq_list_dir_page_get_entry(PageGetContents(page),
													 (size_t) (PageGetPageSize(page) - SizeOfPageHeaderData),
													 local_index,
													 entry,
													 errmsg,
													 errmsg_len);
			ReleaseBuffer(buffer);
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
tq_collect_batch_chain_stats(Relation index_relation,
							 BlockNumber head_block,
							 TqListAggregate *aggregate,
							 uint32_t *reclaimable_pages,
							 char *errmsg,
							 size_t errmsg_len)
{
	BlockNumber block_number = head_block;

	memset(aggregate, 0, sizeof(*aggregate));
	aggregate->head_block = head_block;
	aggregate->tail_block = TQ_INVALID_BLOCK_NUMBER;

	while (block_number != TQ_INVALID_BLOCK_NUMBER)
	{
		Buffer		buffer;
		Page		page;
		TqBatchPageHeaderView header;
		bool		should_reclaim = false;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
		page = BufferGetPage(buffer);
		memset(&header, 0, sizeof(header));
		if (!tq_batch_page_read_header(PageGetContents(page),
									   (size_t) (PageGetPageSize(page) - SizeOfPageHeaderData),
									   &header,
									   errmsg,
									   errmsg_len)
			|| !tq_batch_page_should_reclaim(PageGetContents(page),
											 (size_t) (PageGetPageSize(page) - SizeOfPageHeaderData),
											 &should_reclaim,
											 errmsg,
											 errmsg_len))
		{
			ReleaseBuffer(buffer);
			return false;
		}

		aggregate->live_count += header.live_count;
		aggregate->dead_count += (uint64_t) (header.occupied_count - header.live_count);
		aggregate->batch_page_count += 1;
		aggregate->tail_block = block_number;
		if (should_reclaim)
			*reclaimable_pages += 1;
		block_number = header.next_block;
		ReleaseBuffer(buffer);
	}

	return true;
}

static uint32_t
tq_count_centroid_pages(Relation index_relation,
						  BlockNumber root_block,
						  char *errmsg,
						  size_t errmsg_len)
{
	BlockNumber block_number = root_block;
	uint32_t	page_count = 0;

	while (block_number != TQ_INVALID_BLOCK_NUMBER)
	{
		Buffer		buffer;
		Page		page;
		TqCentroidPageHeaderView header;

		buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
		page = BufferGetPage(buffer);
		memset(&header, 0, sizeof(header));
		if (!tq_centroid_page_read_header(PageGetContents(page),
										  (size_t) (PageGetPageSize(page) - SizeOfPageHeaderData),
										  &header,
										  errmsg,
										  errmsg_len))
		{
			ReleaseBuffer(buffer);
			return UINT32_MAX;
		}

		page_count += 1;
		block_number = header.next_block;
		ReleaseBuffer(buffer);
	}

	return page_count;
}

static bool
tq_append_list_metadata_json(StringInfo buf,
								Relation index_relation,
								const TqMetaPageFields *meta_fields,
								uint64_t *total_live_count,
								uint64_t *total_dead_count,
								uint32_t *batch_page_count,
								uint32_t *reclaimable_pages,
								char *errmsg,
								size_t errmsg_len)
{
	uint32_t	list_id = 0;
	uint64_t	min_live_count = 0;
	uint64_t	max_live_count = 0;
	double		avg_live_count = 0.0;
	double		coeff_var = 0.0;
	double		max_list_over_avg = 0.0;
	uint64_t	p95_live_count = 0;
	uint64_t   *live_counts = NULL;

	appendStringInfoString(buf, "\"lists\":[");

	if (meta_fields->list_count == 0)
	{
		BlockNumber block_number;
		BlockNumber nblocks = RelationGetNumberOfBlocks(index_relation);

		for (block_number = 1; block_number < nblocks; block_number++)
		{
			Buffer		buffer;
			Page		page;
			TqBatchPageHeaderView header;
			bool		should_reclaim = false;

			buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, block_number, RBM_NORMAL, NULL);
			page = BufferGetPage(buffer);
			memset(&header, 0, sizeof(header));
			if (!tq_batch_page_read_header(PageGetContents(page),
									   (size_t) (PageGetPageSize(page) - SizeOfPageHeaderData),
									   &header,
									   errmsg,
									   errmsg_len)
				|| !tq_batch_page_should_reclaim(PageGetContents(page),
											 (size_t) (PageGetPageSize(page) - SizeOfPageHeaderData),
											 &should_reclaim,
											 errmsg,
											 errmsg_len))
			{
				ReleaseBuffer(buffer);
				return false;
			}

			*total_live_count += header.live_count;
			*total_dead_count += (uint64_t) (header.occupied_count - header.live_count);
			*batch_page_count += 1;
			if (should_reclaim)
				*reclaimable_pages += 1;
			ReleaseBuffer(buffer);
		}

		appendStringInfoString(buf, "],");
		appendStringInfo(buf,
						 "\"list_distribution\":{\"min_live_count\":0,\"max_live_count\":0,\"avg_live_count\":0.00,"
						 "\"avg_list_size\":0.00,\"max_list_size\":0,\"p95_list_size\":0,\"coeff_var\":0.0000,"
						 "\"max_list_over_avg\":0.0000}");
		return true;
	}

	live_counts = (uint64_t *) palloc0(sizeof(uint64_t) * (size_t) meta_fields->list_count);

	for (list_id = 0; list_id < meta_fields->list_count; list_id++)
	{
		TqListDirEntry entry;
		TqListAggregate aggregate;

		memset(&entry, 0, sizeof(entry));
		memset(&aggregate, 0, sizeof(aggregate));
		if (!tq_read_list_directory_entry(index_relation,
										  meta_fields->directory_root_block,
										  list_id,
										  &entry,
										  errmsg,
										  errmsg_len))
			return false;

		if (!tq_collect_batch_chain_stats(index_relation,
										 entry.head_block,
										 &aggregate,
										 reclaimable_pages,
										 errmsg,
										 errmsg_len))
			return false;

		*total_live_count += aggregate.live_count;
		*total_dead_count += aggregate.dead_count;
		*batch_page_count += entry.batch_page_count;
		if (list_id == 0 || aggregate.live_count < min_live_count)
			min_live_count = aggregate.live_count;
		if (list_id == 0 || aggregate.live_count > max_live_count)
			max_live_count = aggregate.live_count;
		live_counts[list_id] = aggregate.live_count;
		avg_live_count += (double) aggregate.live_count;

		if (list_id > 0)
			appendStringInfoChar(buf, ',');
		appendStringInfo(buf,
						 "{\"list_id\":%u,\"head_block\":%u,\"tail_block\":%u,\"live_count\":%llu,\"dead_count\":%llu,\"batch_page_count\":%u}",
						 list_id,
						 entry.head_block,
						 entry.tail_block,
						 (unsigned long long) aggregate.live_count,
						 (unsigned long long) aggregate.dead_count,
						 entry.batch_page_count);
	}

	avg_live_count /= (double) meta_fields->list_count;
	if (avg_live_count > 0.0)
	{
		double		variance = 0.0;

		for (list_id = 0; list_id < meta_fields->list_count; list_id++)
		{
			double		centered = (double) live_counts[list_id] - avg_live_count;

			variance += centered * centered;
		}
		variance /= (double) meta_fields->list_count;
		coeff_var = sqrt(variance) / avg_live_count;
		max_list_over_avg = (double) max_live_count / avg_live_count;
	}

	qsort(live_counts, meta_fields->list_count, sizeof(uint64_t), tq_u64_compare);
	p95_live_count = live_counts[((meta_fields->list_count * 95u) + 99u) / 100u - 1u];
	pfree(live_counts);

	appendStringInfoString(buf, "],");
	appendStringInfo(buf,
					 "\"list_distribution\":{\"min_live_count\":%llu,\"max_live_count\":%llu,\"avg_live_count\":%.2f,"
					 "\"avg_list_size\":%.2f,\"max_list_size\":%llu,\"p95_list_size\":%llu,\"coeff_var\":%.4f,"
					 "\"max_list_over_avg\":%.4f}",
					 (unsigned long long) min_live_count,
					 (unsigned long long) max_live_count,
					 avg_live_count,
					 avg_live_count,
					 (unsigned long long) max_live_count,
					 (unsigned long long) p95_live_count,
					 coeff_var,
					 max_list_over_avg);
	return true;
}

Datum
tq_smoke(PG_FUNCTION_ARGS)
{
	(void) fcinfo;
	PG_RETURN_TEXT_P(cstring_to_text("pg_turboquant"));
}

Datum
tq_debug_validate_reloptions(PG_FUNCTION_ARGS)
{
	ArrayType  *reloptions = PG_GETARG_ARRAYTYPE_P(0);
	bytea	   *parsed_options;

	(void) fcinfo;

	parsed_options = tq_reloptions(PointerGetDatum(reloptions), true);
	if (parsed_options != NULL)
		pfree(parsed_options);

	PG_RETURN_TEXT_P(cstring_to_text("ok"));
}

Datum
tq_debug_router_metadata(PG_FUNCTION_ARGS)
{
	Oid			index_oid = PG_GETARG_OID(0);
	Relation	index_relation;
	Buffer		buffer;
	TqMetaPageFields meta_fields;
	char		error_buf[256];
	char		result[512];

	(void) fcinfo;

	index_relation = RelationIdGetRelation(index_oid);
	if (index_relation == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("relation with OID %u could not be opened", index_oid)));
	buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
	memset(&meta_fields, 0, sizeof(meta_fields));
	if (!tq_meta_page_read(PageGetContents(BufferGetPage(buffer)),
						   (size_t) (PageGetPageSize(BufferGetPage(buffer)) - SizeOfPageHeaderData),
						   &meta_fields,
						   error_buf,
						   sizeof(error_buf)))
	{
		ReleaseBuffer(buffer);
		RelationClose(index_relation);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));
	}

	ReleaseBuffer(buffer);
	RelationClose(index_relation);

	snprintf(result,
			 sizeof(result),
			 "seed=%u sample_count=%u trained_vector_count=%u max_iterations=%u completed_iterations=%u "
			 "restart_count=%u selected_restart=%u balance_penalty=%.4f selection_score=%.4f",
			 meta_fields.router_seed,
			 meta_fields.router_sample_count,
			 meta_fields.router_trained_vector_count,
			 meta_fields.router_max_iterations,
			 meta_fields.router_completed_iterations,
			 meta_fields.router_restart_count,
			 meta_fields.router_selected_restart,
			 meta_fields.router_balance_penalty,
			 meta_fields.router_selection_score);
	PG_RETURN_TEXT_P(cstring_to_text(result));
}

Datum
tq_debug_transform_metadata(PG_FUNCTION_ARGS)
{
	Oid			index_oid = PG_GETARG_OID(0);
	Relation	index_relation;
	Buffer		buffer;
	TqMetaPageFields meta_fields;
	char		error_buf[256];
	char		result[256];

	(void) fcinfo;

	index_relation = RelationIdGetRelation(index_oid);
	if (index_relation == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("relation with OID %u could not be opened", index_oid)));
	buffer = ReadBufferExtended(index_relation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
	memset(&meta_fields, 0, sizeof(meta_fields));
	if (!tq_meta_page_read(PageGetContents(BufferGetPage(buffer)),
						   (size_t) (PageGetPageSize(BufferGetPage(buffer)) - SizeOfPageHeaderData),
						   &meta_fields,
						   error_buf,
						   sizeof(error_buf)))
	{
		ReleaseBuffer(buffer);
		RelationClose(index_relation);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));
	}

	ReleaseBuffer(buffer);
	RelationClose(index_relation);

	snprintf(result,
			 sizeof(result),
			 "transform_version=%u input_dimension=%u output_dimension=%u seed=%llu",
			 (unsigned int) meta_fields.transform_version,
			 meta_fields.dimension,
			 meta_fields.transform_output_dimension,
			 (unsigned long long) meta_fields.transform_seed);
	PG_RETURN_TEXT_P(cstring_to_text(result));
}

Datum
tq_index_metadata_core(PG_FUNCTION_ARGS)
{
	Oid			index_oid = PG_GETARG_OID(0);
	Relation	index_relation;
	TqMetaPageFields meta_fields;
	uint64_t	total_live_count = 0;
	uint64_t	total_dead_count = 0;
	uint32_t	reclaimable_pages = 0;
	uint32_t	batch_page_count = 0;
	uint32_t	centroid_page_count = 0;
	char		error_buf[256];
	StringInfoData buf;

	(void) fcinfo;

	index_relation = RelationIdGetRelation(index_oid);
	if (index_relation == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("relation with OID %u could not be opened", index_oid)));

	memset(&meta_fields, 0, sizeof(meta_fields));
	if (!tq_read_meta_fields(index_relation, &meta_fields, error_buf, sizeof(error_buf)))
	{
		RelationClose(index_relation);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));
	}

	if (meta_fields.centroid_root_block != TQ_INVALID_BLOCK_NUMBER)
	{
		centroid_page_count = tq_count_centroid_pages(index_relation,
													 meta_fields.centroid_root_block,
													 error_buf,
													 sizeof(error_buf));
		if (centroid_page_count == UINT32_MAX)
		{
			RelationClose(index_relation);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("%s", error_buf)));
		}
	}

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '{');
	appendStringInfo(&buf, "\"index_oid\":%u,", index_oid);
	appendStringInfo(&buf, "\"heap_relation_oid\":%u,", index_relation->rd_index->indrelid);
	appendStringInfo(&buf, "\"format_version\":%u,", (unsigned int) TQ_PAGE_FORMAT_VERSION);
	appendStringInfoString(&buf, "\"metric\":");
	tq_json_append_string(&buf, tq_distance_kind_name(meta_fields.distance));
	appendStringInfoChar(&buf, ',');
	appendStringInfoString(&buf, "\"codec\":");
	tq_json_append_string(&buf, tq_codec_kind_name(meta_fields.codec));
	appendStringInfoChar(&buf, ',');
	appendStringInfo(&buf, "\"algorithm_version\":%u,", (unsigned int) meta_fields.algorithm_version);
	appendStringInfoString(&buf, "\"faithful_fast_path\":");
	appendStringInfoString(&buf,
						   (meta_fields.normalized
							&& (meta_fields.distance == TQ_DISTANCE_COSINE
								|| meta_fields.distance == TQ_DISTANCE_IP))
						   ? "true"
						   : "false");
	appendStringInfoChar(&buf, ',');
	appendStringInfoString(&buf, "\"compatibility_fallback_only\":");
	appendStringInfoString(&buf,
						   (meta_fields.normalized
							&& (meta_fields.distance == TQ_DISTANCE_COSINE
								|| meta_fields.distance == TQ_DISTANCE_IP))
						   ? "false"
						   : "true");
	appendStringInfoChar(&buf, ',');
	appendStringInfoString(&buf, "\"page_summary\":{");
	appendStringInfoString(&buf, "\"mode\":");
	tq_json_append_string(&buf, tq_page_summary_mode_name(&meta_fields));
	appendStringInfo(&buf,
					 ",\"safe_pruning\":%s},",
					 (meta_fields.list_count > 0
					  && meta_fields.normalized
					  && meta_fields.codec == TQ_CODEC_PROD
					  && (meta_fields.distance == TQ_DISTANCE_COSINE
						  || meta_fields.distance == TQ_DISTANCE_IP))
					 ? "true" : "false");
	appendStringInfoString(&buf, "\"normalized\":");
	appendStringInfoString(&buf, meta_fields.normalized ? "true" : "false");
	appendStringInfo(&buf,
					 ",\"bits\":%u,\"lane_count\":%u,\"list_count\":%u,\"directory_root_block\":%u,\"centroid_root_block\":%u,",
					 (unsigned int) meta_fields.bits,
					 (unsigned int) meta_fields.lane_count,
					 meta_fields.list_count,
					 meta_fields.directory_root_block,
					 meta_fields.centroid_root_block);
	appendStringInfoString(&buf, "\"transform\":{");
	appendStringInfoString(&buf, "\"kind\":");
	tq_json_append_string(&buf, tq_transform_kind_name(meta_fields.transform));
	appendStringInfo(&buf,
					 ",\"version\":%u,\"input_dimension\":%u,\"output_dimension\":%u,\"seed\":%llu},",
					 (unsigned int) meta_fields.transform_version,
					 meta_fields.dimension,
					 meta_fields.transform_output_dimension,
					 (unsigned long long) meta_fields.transform_seed);
	appendStringInfoString(&buf, "\"quantizer\":{");
	appendStringInfoString(&buf, "\"family\":");
	tq_json_append_string(&buf, tq_quantizer_family_name(meta_fields.quantizer_version));
	appendStringInfo(&buf, ",\"version\":%u},", (unsigned int) meta_fields.quantizer_version);
	appendStringInfoString(&buf, "\"residual_sketch\":{");
	appendStringInfoString(&buf, "\"kind\":");
	tq_json_append_string(&buf, tq_residual_sketch_kind_name(meta_fields.residual_sketch_version));
	appendStringInfo(&buf,
					 ",\"version\":%u,\"bits_per_dimension\":%u,\"projected_dimension\":%u,\"bit_budget\":%u},",
					 (unsigned int) meta_fields.residual_sketch_version,
					 (unsigned int) meta_fields.residual_bits_per_dimension,
					 (unsigned int) meta_fields.residual_sketch_dimension,
					 (unsigned int) (meta_fields.residual_sketch_dimension
									 * meta_fields.residual_bits_per_dimension));
	appendStringInfoString(&buf, "\"estimator\":{");
	appendStringInfoString(&buf, "\"mode\":");
	tq_json_append_string(&buf, tq_estimator_mode_name(meta_fields.estimator_version));
	appendStringInfo(&buf, ",\"version\":%u},", (unsigned int) meta_fields.estimator_version);
	appendStringInfoString(&buf, "\"router\":{");
	appendStringInfoString(&buf, "\"algorithm\":");
	tq_json_append_string(&buf, tq_router_algorithm_name(meta_fields.router_algorithm));
	appendStringInfo(&buf,
					 ",\"seed\":%u,\"sample_count\":%u,\"max_iterations\":%u,\"completed_iterations\":%u,"
					 "\"trained_vector_count\":%u,\"restart_count\":%u,\"selected_restart\":%u,"
					 "\"mean_distortion\":%.6f,\"max_list_over_avg\":%.4f,\"coeff_var\":%.4f,"
					 "\"balance_penalty\":%.4f,\"selection_score\":%.6f,"
					 "\"balance_weights\":{\"max_list_over_avg\":%.2f,\"coeff_var\":%.2f}},",
					 meta_fields.router_seed,
					 meta_fields.router_sample_count,
					 meta_fields.router_max_iterations,
					 meta_fields.router_completed_iterations,
					 meta_fields.router_trained_vector_count,
					 meta_fields.router_restart_count,
					 meta_fields.router_selected_restart,
					 meta_fields.router_mean_distortion,
					 meta_fields.router_max_list_over_avg,
					 meta_fields.router_coeff_var,
					 meta_fields.router_balance_penalty,
					 meta_fields.router_selection_score,
					 TQ_ROUTER_MAX_LIST_WEIGHT,
					 TQ_ROUTER_COEFF_VAR_WEIGHT);
	if (!tq_append_list_metadata_json(&buf,
									  index_relation,
									  &meta_fields,
									  &total_live_count,
									  &total_dead_count,
									  &batch_page_count,
									  &reclaimable_pages,
									  error_buf,
									  sizeof(error_buf)))
	{
		RelationClose(index_relation);
		pfree(buf.data);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));
	}
	appendStringInfo(&buf,
					 ",\"live_count\":%llu,\"dead_count\":%llu,\"reclaimable_pages\":%u,\"batch_page_count\":%u,\"centroid_page_count\":%u}",
					 (unsigned long long) total_live_count,
					 (unsigned long long) total_dead_count,
					 reclaimable_pages,
					 batch_page_count,
					 centroid_page_count);

	RelationClose(index_relation);
	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}

Datum
tq_last_scan_stats_core(PG_FUNCTION_ARGS)
{
	TqScanStats stats;
	char		json[2048];

	memset(&stats, 0, sizeof(stats));
	memset(json, 0, sizeof(json));
	tq_scan_stats_snapshot(&stats);

	if (!tq_scan_stats_serialize_json(&stats, json, sizeof(json)))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turboquant could not serialize last scan stats")));

	PG_RETURN_TEXT_P(cstring_to_text(json));
}

Datum
tq_last_shadow_decode_candidate_tids_core(PG_FUNCTION_ARGS)
{
	ArrayType  *result = NULL;
	Datum	   *values = NULL;
	TqTid	   *tids = NULL;
	size_t		count = 0;
	size_t		index = 0;

	(void) fcinfo;

	if (!tq_scan_stats_copy_shadow_decode_tids(NULL, 0, &count))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turboquant could not inspect shadow decode candidate tids")));

	if (count == 0)
		PG_RETURN_ARRAYTYPE_P(construct_empty_array(TEXTOID));

	tids = (TqTid *) palloc(sizeof(TqTid) * count);
	values = (Datum *) palloc(sizeof(Datum) * count);

	if (!tq_scan_stats_copy_shadow_decode_tids(tids, count, &count))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turboquant could not copy shadow decode candidate tids")));

	for (index = 0; index < count; index++)
		values[index] = CStringGetTextDatum(
			psprintf("(%u,%u)",
					 tids[index].block_number,
					 (unsigned int) tids[index].offset_number));

	result = construct_array(values, (int) count, TEXTOID, -1, false, 'i');
	PG_RETURN_ARRAYTYPE_P(result);
}

Datum
tq_runtime_simd_features_core(PG_FUNCTION_ARGS)
{
	StringInfoData buf;

	(void) fcinfo;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '{');
	appendStringInfo(&buf,
					 "\"preferred_kernel\":\"%s\","
					 "\"force_disabled\":%s,"
					 "\"compiled\":{"
					 "\"scalar\":true,"
					 "\"avx2\":%s,"
					 "\"avx512\":%s,"
					 "\"neon\":%s},"
					 "\"runtime_available\":{"
					 "\"scalar\":true,"
					 "\"avx2\":%s,"
					 "\"avx512\":%s,"
					 "\"neon\":%s}}",
					 tq_prod_score_kernel_name(tq_prod_score_preferred_kernel()),
					 "false",
					 tq_simd_avx2_compile_available() ? "true" : "false",
					 tq_simd_avx512_compile_available() ? "true" : "false",
					 tq_simd_neon_compile_available() ? "true" : "false",
					 tq_simd_avx2_runtime_available() ? "true" : "false",
					 tq_simd_avx512_runtime_available() ? "true" : "false",
					 tq_simd_neon_runtime_available() ? "true" : "false");

	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}
