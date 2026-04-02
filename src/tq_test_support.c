#include "postgres.h"

#include <string.h>

#include "fmgr.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/rel.h"

#include "src/tq_page.h"

PG_FUNCTION_INFO_V1(tq_test_corrupt_meta_magic);
PG_FUNCTION_INFO_V1(tq_test_corrupt_meta_format_version);
PG_FUNCTION_INFO_V1(tq_test_corrupt_meta_transform_contract);
PG_FUNCTION_INFO_V1(tq_test_corrupt_first_list_head_to_directory_root);
PG_FUNCTION_INFO_V1(tq_test_corrupt_first_batch_occupied_count);

#define TQ_TEST_META_MAGIC_OFFSET 0
#define TQ_TEST_META_VERSION_OFFSET 8
#define TQ_TEST_META_TRANSFORM_OUTPUT_DIMENSION_OFFSET 16
#define TQ_TEST_BATCH_LANE_COUNT_OFFSET 8
#define TQ_TEST_BATCH_OCCUPIED_COUNT_OFFSET 10

static void
tq_test_write_u16(uint8_t *dst, size_t offset, uint16_t value)
{
	dst[offset] = (uint8_t) (value & 0xFFu);
	dst[offset + 1] = (uint8_t) ((value >> 8) & 0xFFu);
}

static void
tq_test_write_u32(uint8_t *dst, size_t offset, uint32_t value)
{
	dst[offset] = (uint8_t) (value & 0xFFu);
	dst[offset + 1] = (uint8_t) ((value >> 8) & 0xFFu);
	dst[offset + 2] = (uint8_t) ((value >> 16) & 0xFFu);
	dst[offset + 3] = (uint8_t) ((value >> 24) & 0xFFu);
}

static Relation
tq_test_open_index_relation(Oid index_oid)
{
	Relation	relation = RelationIdGetRelation(index_oid);

	if (relation == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("relation with OID %u could not be opened", index_oid)));

	if (relation->rd_rel == NULL || relation->rd_rel->relkind != RELKIND_INDEX)
	{
		RelationClose(relation);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("relation with OID %u is not an index", index_oid)));
	}

	return relation;
}

static void
tq_test_read_meta(Relation relation, TqMetaPageFields *meta_fields)
{
	Buffer		buffer;
	Page		page;
	char		error_buf[256];

	memset(meta_fields, 0, sizeof(*meta_fields));
	memset(error_buf, 0, sizeof(error_buf));

	buffer = ReadBufferExtended(relation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
	page = BufferGetPage(buffer);

	if (!tq_meta_page_read(PageGetContents(page),
						   (size_t) (PageGetPageSize(page) - SizeOfPageHeaderData),
						   meta_fields,
						   error_buf,
						   sizeof(error_buf)))
	{
		ReleaseBuffer(buffer);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));
	}

	ReleaseBuffer(buffer);
}

static TqListDirEntry
tq_test_read_first_list_entry(Relation relation, const TqMetaPageFields *meta_fields)
{
	Buffer			buffer;
	Page			page;
	TqListDirEntry	entry;
	char			error_buf[256];

	if (meta_fields->list_count == 0 || meta_fields->directory_root_block == TQ_INVALID_BLOCK_NUMBER)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("turboquant index does not have IVF list-directory metadata")));

	memset(&entry, 0, sizeof(entry));
	memset(error_buf, 0, sizeof(error_buf));

	buffer = ReadBufferExtended(relation, MAIN_FORKNUM,
								meta_fields->directory_root_block, RBM_NORMAL, NULL);
	page = BufferGetPage(buffer);

	if (!tq_list_dir_page_get_entry(PageGetContents(page),
									(size_t) (PageGetPageSize(page) - SizeOfPageHeaderData),
									0,
									&entry,
									error_buf,
									sizeof(error_buf)))
	{
		ReleaseBuffer(buffer);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));
	}

	ReleaseBuffer(buffer);
	return entry;
}

Datum
tq_test_corrupt_meta_magic(PG_FUNCTION_ARGS)
{
	Oid			index_oid = PG_GETARG_OID(0);
	Relation	relation = tq_test_open_index_relation(index_oid);
	Buffer		buffer = ReadBufferExtended(relation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
	Page		page = BufferGetPage(buffer);
	uint8_t    *payload = (uint8_t *) PageGetContents(page);

	LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
	tq_test_write_u32(payload, TQ_TEST_META_MAGIC_OFFSET, 0);
	MarkBufferDirty(buffer);
	LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buffer);
	RelationClose(relation);
	PG_RETURN_VOID();
}

Datum
tq_test_corrupt_meta_format_version(PG_FUNCTION_ARGS)
{
	Oid			index_oid = PG_GETARG_OID(0);
	int32		format_version = PG_GETARG_INT32(1);
	Relation	relation = tq_test_open_index_relation(index_oid);
	Buffer		buffer = ReadBufferExtended(relation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
	Page		page = BufferGetPage(buffer);
	uint8_t    *payload = (uint8_t *) PageGetContents(page);

	LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
	tq_test_write_u32(payload, TQ_TEST_META_VERSION_OFFSET, (uint32_t) format_version);
	MarkBufferDirty(buffer);
	LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buffer);
	RelationClose(relation);
	PG_RETURN_VOID();
}

Datum
tq_test_corrupt_meta_transform_contract(PG_FUNCTION_ARGS)
{
	Oid					index_oid = PG_GETARG_OID(0);
	Relation			relation = tq_test_open_index_relation(index_oid);
	TqMetaPageFields	meta_fields;
	Buffer				buffer;
	Page				page;
	uint8_t			   *payload;

	tq_test_read_meta(relation, &meta_fields);

	buffer = ReadBufferExtended(relation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
	page = BufferGetPage(buffer);
	payload = (uint8_t *) PageGetContents(page);

	LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
	tq_test_write_u32(payload,
					  TQ_TEST_META_TRANSFORM_OUTPUT_DIMENSION_OFFSET,
					  meta_fields.transform_output_dimension + 3);
	MarkBufferDirty(buffer);
	LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buffer);
	RelationClose(relation);
	PG_RETURN_VOID();
}

Datum
tq_test_corrupt_first_list_head_to_directory_root(PG_FUNCTION_ARGS)
{
	Oid					index_oid = PG_GETARG_OID(0);
	Relation			relation = tq_test_open_index_relation(index_oid);
	TqMetaPageFields	meta_fields;
	Buffer				buffer;
	Page				page;
	TqListDirEntry		entry;
	char				error_buf[256];

	tq_test_read_meta(relation, &meta_fields);
	entry = tq_test_read_first_list_entry(relation, &meta_fields);
	entry.head_block = meta_fields.directory_root_block;

	memset(error_buf, 0, sizeof(error_buf));
	buffer = ReadBufferExtended(relation, MAIN_FORKNUM,
								meta_fields.directory_root_block, RBM_NORMAL, NULL);
	page = BufferGetPage(buffer);

	LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
	if (!tq_list_dir_page_set_entry(PageGetContents(page),
									(size_t) (PageGetPageSize(page) - SizeOfPageHeaderData),
									0,
									&entry,
									error_buf,
									sizeof(error_buf)))
	{
		LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
		ReleaseBuffer(buffer);
		RelationClose(relation);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", error_buf)));
	}

	MarkBufferDirty(buffer);
	LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buffer);
	RelationClose(relation);
	PG_RETURN_VOID();
}

Datum
tq_test_corrupt_first_batch_occupied_count(PG_FUNCTION_ARGS)
{
	Oid					index_oid = PG_GETARG_OID(0);
	Relation			relation = tq_test_open_index_relation(index_oid);
	TqMetaPageFields	meta_fields;
	TqListDirEntry		entry;
	Buffer				buffer;
	Page				page;
	uint8_t			   *payload;
	uint16_t			lane_count = 0;

	tq_test_read_meta(relation, &meta_fields);
	entry = tq_test_read_first_list_entry(relation, &meta_fields);
	if (entry.head_block == TQ_INVALID_BLOCK_NUMBER)
	{
		RelationClose(relation);
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("turboquant index does not have a reachable batch page for list 0")));
	}

	buffer = ReadBufferExtended(relation, MAIN_FORKNUM, entry.head_block, RBM_NORMAL, NULL);
	page = BufferGetPage(buffer);
	payload = (uint8_t *) PageGetContents(page);
	lane_count = (uint16_t) payload[TQ_TEST_BATCH_LANE_COUNT_OFFSET]
		| (uint16_t) ((uint16_t) payload[TQ_TEST_BATCH_LANE_COUNT_OFFSET + 1] << 8);

	LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
	tq_test_write_u16(payload, TQ_TEST_BATCH_OCCUPIED_COUNT_OFFSET, (uint16_t) (lane_count + 1));
	MarkBufferDirty(buffer);
	LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buffer);
	RelationClose(relation);
	PG_RETURN_VOID();
}
