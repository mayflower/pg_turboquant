#include "postgres.h"

#include <string.h>

#include "src/tq_wal.h"

#define TQ_WAL_META_PAGE_HEADER_BYTES 122
#define TQ_WAL_LIST_DIR_PAGE_HEADER_BYTES 16
#define TQ_WAL_LIST_DIR_ENTRY_BYTES 32
#define TQ_WAL_CENTROID_PAGE_HEADER_BYTES 20
#define TQ_WAL_BATCH_SUMMARY_PAGE_HEADER_BYTES 20

typedef bool (*TqWalMutator) (Page page, void *arg, char *errmsg, size_t errmsg_len);

typedef struct TqWalMetaInitArgs
{
	const TqMetaPageFields *fields;
} TqWalMetaInitArgs;

typedef struct TqWalListDirInitArgs
{
	uint16_t	entry_capacity;
	uint32_t	next_block;
} TqWalListDirInitArgs;

typedef struct TqWalListDirEntryArgs
{
	uint16_t	index;
	const TqListDirEntry *entry;
} TqWalListDirEntryArgs;

typedef struct TqWalCentroidInitArgs
{
	uint32_t	dimension;
	uint16_t	centroid_capacity;
	uint32_t	next_block;
} TqWalCentroidInitArgs;

typedef struct TqWalCentroidSetArgs
{
	uint16_t	index;
	const float *centroid;
	uint32_t	dimension;
} TqWalCentroidSetArgs;

typedef struct TqWalBatchInitArgs
{
	const TqBatchPageParams *params;
} TqWalBatchInitArgs;

typedef struct TqWalBatchSummaryPageInitArgs
{
	uint32_t	code_bytes;
	uint16_t	entry_capacity;
	uint32_t	next_block;
} TqWalBatchSummaryPageInitArgs;

typedef struct TqWalBatchSummaryPageNextBlockArgs
{
	uint32_t	next_block;
} TqWalBatchSummaryPageNextBlockArgs;

typedef struct TqWalBatchSummaryPageEntryArgs
{
	uint16_t	index;
	uint32_t	block_number;
	const TqBatchPageSummary *summary;
	const uint8_t *representative_code;
	size_t		code_len;
} TqWalBatchSummaryPageEntryArgs;

typedef struct TqWalBatchNextBlockArgs
{
	uint32_t	next_block;
} TqWalBatchNextBlockArgs;

typedef struct TqWalBatchSummaryArgs
{
	const TqBatchPageSummary *summary;
} TqWalBatchSummaryArgs;

typedef struct TqWalBatchAppendArgs
{
	const TqTid *tid;
	const int32_t *filter_value;
	const uint8_t *packed_code;
	size_t		packed_code_len;
	uint16_t   *lane_index;
} TqWalBatchAppendArgs;

typedef struct TqWalBatchSoaAppendArgs
{
	const TqTid *tid;
	const uint8_t *nibbles;
	uint32_t	dimension;
	float		gamma;
	uint16_t   *lane_index;
} TqWalBatchSoaAppendArgs;

typedef struct TqWalBatchRepCodeArgs
{
	const uint8_t *code;
	size_t		code_len;
} TqWalBatchRepCodeArgs;

typedef struct TqWalBatchDeadArgs
{
	uint16_t	lane_index;
} TqWalBatchDeadArgs;

typedef struct TqWalBatchCompactArgs
{
	int			unused;
} TqWalBatchCompactArgs;

static void *
tq_wal_payload(Page page)
{
	return PageGetContents(page);
}

static size_t
tq_wal_payload_size(Page page)
{
	return (size_t) (PageGetPageSize(page) - SizeOfPageHeaderData);
}

static bool
tq_wal_sync_page_bounds(Page page, char *errmsg, size_t errmsg_len)
{
	PageHeader	header = (PageHeader) page;
	TqPageKind	kind;
	size_t		used_bytes = 0;
	void	   *payload = tq_wal_payload(page);
	size_t		payload_size = tq_wal_payload_size(page);

	if (!tq_page_read_kind(payload, payload_size, &kind, errmsg, errmsg_len))
		return false;

	switch (kind)
	{
		case TQ_PAGE_KIND_META:
			used_bytes = TQ_WAL_META_PAGE_HEADER_BYTES;
			break;
		case TQ_PAGE_KIND_LIST_DIRECTORY:
		{
			TqListDirPageHeaderView list_header;

			memset(&list_header, 0, sizeof(list_header));
			if (!tq_list_dir_page_read_header(payload, payload_size,
											  &list_header, errmsg, errmsg_len))
				return false;
			used_bytes = TQ_WAL_LIST_DIR_PAGE_HEADER_BYTES
				+ ((size_t) list_header.entry_count * (size_t) TQ_WAL_LIST_DIR_ENTRY_BYTES);
			break;
		}
		case TQ_PAGE_KIND_BATCH:
			if (!tq_batch_page_used_bytes(payload, payload_size,
										  &used_bytes, errmsg, errmsg_len))
				return false;
			break;
		case TQ_PAGE_KIND_CENTROID:
		{
			TqCentroidPageHeaderView centroid_header;

			memset(&centroid_header, 0, sizeof(centroid_header));
			if (!tq_centroid_page_read_header(payload, payload_size,
											  &centroid_header, errmsg, errmsg_len))
				return false;
			used_bytes = TQ_WAL_CENTROID_PAGE_HEADER_BYTES
				+ ((size_t) centroid_header.centroid_count
				   * (size_t) centroid_header.dimension
				   * sizeof(float));
			break;
		}
		case TQ_PAGE_KIND_BATCH_SUMMARY:
		{
			TqBatchSummaryPageHeaderView summary_header;

			memset(&summary_header, 0, sizeof(summary_header));
			if (!tq_batch_summary_page_read_header(payload, payload_size,
												  &summary_header, errmsg, errmsg_len))
				return false;
			used_bytes = tq_batch_summary_page_required_bytes(summary_header.entry_count,
															 summary_header.code_bytes);
			break;
		}
		default:
			snprintf(errmsg, errmsg_len,
					 "invalid turboquant page: unsupported kind %u for wal bounds",
					 (unsigned int) kind);
			return false;
	}

	header->pd_lower = (LocationIndex) (SizeOfPageHeaderData + used_bytes);
	header->pd_upper = BLCKSZ;
	return true;
}

static bool
tq_wal_apply(Relation relation,
			 Buffer buffer,
			 int flags,
			 TqWalMutator mutator,
			 void *arg,
			 char *errmsg,
			 size_t errmsg_len)
{
	GenericXLogState *state;
	Page		page;

	if (relation == NULL || !BufferIsValid(buffer) || mutator == NULL)
		return false;

	state = GenericXLogStart(relation);
	page = GenericXLogRegisterBuffer(state, buffer, flags);

	if (!mutator(page, arg, errmsg, errmsg_len))
	{
		GenericXLogAbort(state);
		return false;
	}

	GenericXLogFinish(state);
	return true;
}

static bool
tq_wal_mutate_meta_init(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalMetaInitArgs *args = (const TqWalMetaInitArgs *) arg;

	PageInit(page, BLCKSZ, 0);
	if (!tq_meta_page_init(tq_wal_payload(page), tq_wal_payload_size(page),
						   args->fields, errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_list_dir_init(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalListDirInitArgs *args = (const TqWalListDirInitArgs *) arg;

	PageInit(page, BLCKSZ, 0);
	if (!tq_list_dir_page_init(tq_wal_payload(page), tq_wal_payload_size(page),
							   args->entry_capacity, args->next_block,
							   errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_list_dir_set(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalListDirEntryArgs *args = (const TqWalListDirEntryArgs *) arg;

	if (!tq_list_dir_page_set_entry(tq_wal_payload(page), tq_wal_payload_size(page),
									args->index, args->entry,
									errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_centroid_init(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalCentroidInitArgs *args = (const TqWalCentroidInitArgs *) arg;

	PageInit(page, BLCKSZ, 0);
	if (!tq_centroid_page_init(tq_wal_payload(page), tq_wal_payload_size(page),
							   args->dimension, args->centroid_capacity,
							   args->next_block, errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_centroid_set(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalCentroidSetArgs *args = (const TqWalCentroidSetArgs *) arg;

	if (!tq_centroid_page_set_centroid(tq_wal_payload(page), tq_wal_payload_size(page),
									   args->index, args->centroid, args->dimension,
									   errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_batch_summary_page_init(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalBatchSummaryPageInitArgs *args = (const TqWalBatchSummaryPageInitArgs *) arg;

	PageInit(page, BLCKSZ, 0);
	if (!tq_batch_summary_page_init(tq_wal_payload(page), tq_wal_payload_size(page),
									args->code_bytes, args->entry_capacity, args->next_block,
									errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_batch_summary_page_next_block(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalBatchSummaryPageNextBlockArgs *args = (const TqWalBatchSummaryPageNextBlockArgs *) arg;

	if (!tq_batch_summary_page_set_next_block(tq_wal_payload(page), tq_wal_payload_size(page),
											  args->next_block, errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_batch_summary_page_entry(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalBatchSummaryPageEntryArgs *args = (const TqWalBatchSummaryPageEntryArgs *) arg;

	if (!tq_batch_summary_page_set_entry(tq_wal_payload(page), tq_wal_payload_size(page),
										 args->index, args->block_number, args->summary,
										 args->representative_code, args->code_len,
										 errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_batch_init(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalBatchInitArgs *args = (const TqWalBatchInitArgs *) arg;

	PageInit(page, BLCKSZ, 0);
	if (!tq_batch_page_init(tq_wal_payload(page), tq_wal_payload_size(page),
							args->params, errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_batch_next_block(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalBatchNextBlockArgs *args = (const TqWalBatchNextBlockArgs *) arg;

	if (!tq_batch_page_set_next_block(tq_wal_payload(page), tq_wal_payload_size(page),
									  args->next_block, errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_batch_append(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalBatchAppendArgs *args = (const TqWalBatchAppendArgs *) arg;
	uint16_t	lane_index = 0;

	if (!tq_batch_page_append_lane(tq_wal_payload(page), tq_wal_payload_size(page),
								   args->tid, &lane_index, errmsg, errmsg_len)
		|| (args->filter_value != NULL
			&& !tq_batch_page_set_filter_int4(tq_wal_payload(page), tq_wal_payload_size(page),
											  lane_index, *args->filter_value,
											  errmsg, errmsg_len))
		|| !tq_batch_page_set_code(tq_wal_payload(page), tq_wal_payload_size(page),
								   lane_index, args->packed_code, args->packed_code_len,
								   errmsg, errmsg_len))
		return false;

	if (args->lane_index != NULL)
		*args->lane_index = lane_index;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_batch_soa_append(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalBatchSoaAppendArgs *args = (const TqWalBatchSoaAppendArgs *) arg;
	uint16_t	lane_index = 0;

	if (!tq_batch_page_append_lane(tq_wal_payload(page), tq_wal_payload_size(page),
								   args->tid, &lane_index, errmsg, errmsg_len)
		|| !tq_batch_page_set_nibble_and_gamma(tq_wal_payload(page), tq_wal_payload_size(page),
											   lane_index, args->nibbles, args->dimension,
											   args->gamma, errmsg, errmsg_len))
		return false;

	if (args->lane_index != NULL)
		*args->lane_index = lane_index;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_batch_representative_code(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalBatchRepCodeArgs *args = (const TqWalBatchRepCodeArgs *) arg;

	if (!tq_batch_page_set_representative_code(tq_wal_payload(page), tq_wal_payload_size(page),
											   args->code, args->code_len,
											   errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_batch_summary(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalBatchSummaryArgs *args = (const TqWalBatchSummaryArgs *) arg;

	if (!tq_batch_page_set_summary(tq_wal_payload(page), tq_wal_payload_size(page),
								   args->summary, errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_batch_dead(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	const TqWalBatchDeadArgs *args = (const TqWalBatchDeadArgs *) arg;

	if (!tq_batch_page_mark_dead(tq_wal_payload(page), tq_wal_payload_size(page),
								 args->lane_index, errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

static bool
tq_wal_mutate_batch_compact(Page page, void *arg, char *errmsg, size_t errmsg_len)
{
	(void) arg;

	if (!tq_batch_page_compact(tq_wal_payload(page), tq_wal_payload_size(page),
							   errmsg, errmsg_len))
		return false;

	return tq_wal_sync_page_bounds(page, errmsg, errmsg_len);
}

bool
tq_wal_init_meta_page(Relation relation,
					  Buffer buffer,
					  const TqMetaPageFields *fields,
					  char *errmsg,
					  size_t errmsg_len)
{
	TqWalMetaInitArgs args;

	args.fields = fields;
	return tq_wal_apply(relation, buffer, GENERIC_XLOG_FULL_IMAGE,
						tq_wal_mutate_meta_init, &args, errmsg, errmsg_len);
}

bool
tq_wal_init_list_dir_page(Relation relation,
						  Buffer buffer,
						  uint16_t entry_capacity,
						  uint32_t next_block,
						  char *errmsg,
						  size_t errmsg_len)
{
	TqWalListDirInitArgs args;

	args.entry_capacity = entry_capacity;
	args.next_block = next_block;
	return tq_wal_apply(relation, buffer, GENERIC_XLOG_FULL_IMAGE,
						tq_wal_mutate_list_dir_init, &args, errmsg, errmsg_len);
}

bool
tq_wal_set_list_dir_entry(Relation relation,
						  Buffer buffer,
						  uint16_t index,
						  const TqListDirEntry *entry,
						  char *errmsg,
						  size_t errmsg_len)
{
	TqWalListDirEntryArgs args;

	args.index = index;
	args.entry = entry;
	return tq_wal_apply(relation, buffer, 0,
						tq_wal_mutate_list_dir_set, &args, errmsg, errmsg_len);
}

bool
tq_wal_init_centroid_page(Relation relation,
						  Buffer buffer,
						  uint32_t dimension,
						  uint16_t centroid_capacity,
						  uint32_t next_block,
						  char *errmsg,
						  size_t errmsg_len)
{
	TqWalCentroidInitArgs args;

	args.dimension = dimension;
	args.centroid_capacity = centroid_capacity;
	args.next_block = next_block;
	return tq_wal_apply(relation, buffer, GENERIC_XLOG_FULL_IMAGE,
						tq_wal_mutate_centroid_init, &args, errmsg, errmsg_len);
}

bool
tq_wal_set_centroid(Relation relation,
					Buffer buffer,
					uint16_t index,
					const float *centroid,
					uint32_t dimension,
					char *errmsg,
					size_t errmsg_len)
{
	TqWalCentroidSetArgs args;

	args.index = index;
	args.centroid = centroid;
	args.dimension = dimension;
	return tq_wal_apply(relation, buffer, 0,
						tq_wal_mutate_centroid_set, &args, errmsg, errmsg_len);
}

bool
tq_wal_init_batch_summary_page(Relation relation,
							   Buffer buffer,
							   uint32_t code_bytes,
							   uint16_t entry_capacity,
							   uint32_t next_block,
							   char *errmsg,
							   size_t errmsg_len)
{
	TqWalBatchSummaryPageInitArgs args;

	args.code_bytes = code_bytes;
	args.entry_capacity = entry_capacity;
	args.next_block = next_block;
	return tq_wal_apply(relation, buffer, GENERIC_XLOG_FULL_IMAGE,
						tq_wal_mutate_batch_summary_page_init, &args, errmsg, errmsg_len);
}

bool
tq_wal_set_batch_summary_next_block(Relation relation,
									Buffer buffer,
									uint32_t next_block,
									char *errmsg,
									size_t errmsg_len)
{
	TqWalBatchSummaryPageNextBlockArgs args;

	args.next_block = next_block;
	return tq_wal_apply(relation, buffer, 0,
						tq_wal_mutate_batch_summary_page_next_block, &args, errmsg, errmsg_len);
}

bool
tq_wal_set_batch_summary_entry(Relation relation,
							   Buffer buffer,
							   uint16_t index,
							   uint32_t block_number,
							   const TqBatchPageSummary *summary,
							   const uint8_t *representative_code,
							   size_t code_len,
							   char *errmsg,
							   size_t errmsg_len)
{
	TqWalBatchSummaryPageEntryArgs args;

	memset(&args, 0, sizeof(args));
	args.index = index;
	args.block_number = block_number;
	args.summary = summary;
	args.representative_code = representative_code;
	args.code_len = code_len;
	return tq_wal_apply(relation, buffer, 0,
						tq_wal_mutate_batch_summary_page_entry, &args, errmsg, errmsg_len);
}

bool
tq_wal_init_batch_page(Relation relation,
					   Buffer buffer,
					   const TqBatchPageParams *params,
					   char *errmsg,
					   size_t errmsg_len)
{
	TqWalBatchInitArgs args;

	args.params = params;
	return tq_wal_apply(relation, buffer, GENERIC_XLOG_FULL_IMAGE,
						tq_wal_mutate_batch_init, &args, errmsg, errmsg_len);
}

bool
tq_wal_set_batch_next_block(Relation relation,
							Buffer buffer,
							uint32_t next_block,
							char *errmsg,
							size_t errmsg_len)
{
	TqWalBatchNextBlockArgs args;

	args.next_block = next_block;
	return tq_wal_apply(relation, buffer, 0,
						tq_wal_mutate_batch_next_block, &args, errmsg, errmsg_len);
}

bool
tq_wal_set_batch_summary(Relation relation,
						 Buffer buffer,
						 const TqBatchPageSummary *summary,
						 char *errmsg,
						 size_t errmsg_len)
{
	TqWalBatchSummaryArgs args;

	args.summary = summary;
	return tq_wal_apply(relation, buffer, 0,
						tq_wal_mutate_batch_summary, &args, errmsg, errmsg_len);
}

bool
tq_wal_append_batch_code(Relation relation,
						 Buffer buffer,
						 const TqTid *tid,
						 const int32_t *filter_value,
						 const uint8_t *packed_code,
						 size_t packed_code_len,
						 uint16_t *lane_index,
						 char *errmsg,
						 size_t errmsg_len)
{
	TqWalBatchAppendArgs args;

	memset(&args, 0, sizeof(args));
	args.tid = tid;
	args.filter_value = filter_value;
	args.packed_code = packed_code;
	args.packed_code_len = packed_code_len;
	args.lane_index = lane_index;
	return tq_wal_apply(relation, buffer, 0,
						tq_wal_mutate_batch_append, &args, errmsg, errmsg_len);
}

bool
tq_wal_append_batch_soa(Relation relation,
						 Buffer buffer,
						 const TqTid *tid,
						 const uint8_t *nibbles,
						 uint32_t dimension,
						 float gamma,
						 uint16_t *lane_index,
						 char *errmsg,
						 size_t errmsg_len)
{
	TqWalBatchSoaAppendArgs args;

	memset(&args, 0, sizeof(args));
	args.tid = tid;
	args.nibbles = nibbles;
	args.dimension = dimension;
	args.gamma = gamma;
	args.lane_index = lane_index;
	return tq_wal_apply(relation, buffer, 0,
						tq_wal_mutate_batch_soa_append, &args, errmsg, errmsg_len);
}

bool
tq_wal_set_batch_representative_code(Relation relation,
									  Buffer buffer,
									  const uint8_t *code,
									  size_t code_len,
									  char *errmsg,
									  size_t errmsg_len)
{
	TqWalBatchRepCodeArgs args;

	memset(&args, 0, sizeof(args));
	args.code = code;
	args.code_len = code_len;
	return tq_wal_apply(relation, buffer, 0,
						tq_wal_mutate_batch_representative_code, &args, errmsg, errmsg_len);
}

bool
tq_wal_mark_batch_dead(Relation relation,
					   Buffer buffer,
					   uint16_t lane_index,
					   char *errmsg,
					   size_t errmsg_len)
{
	TqWalBatchDeadArgs args;

	args.lane_index = lane_index;
	return tq_wal_apply(relation, buffer, 0,
						tq_wal_mutate_batch_dead, &args, errmsg, errmsg_len);
}

bool
tq_wal_compact_batch_page(Relation relation,
						  Buffer buffer,
						  char *errmsg,
						  size_t errmsg_len)
{
	TqWalBatchCompactArgs args;

	memset(&args, 0, sizeof(args));
	return tq_wal_apply(relation, buffer, 0,
						tq_wal_mutate_batch_compact, &args, errmsg, errmsg_len);
}
