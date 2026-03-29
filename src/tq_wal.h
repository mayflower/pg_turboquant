#ifndef TQ_WAL_H
#define TQ_WAL_H

#include "postgres.h"

#include <stddef.h>
#include <stdint.h>

#include "access/generic_xlog.h"
#include "storage/bufmgr.h"

#include "src/tq_page.h"

extern bool tq_wal_init_meta_page(Relation relation,
								  Buffer buffer,
								  const TqMetaPageFields *fields,
								  char *errmsg,
								  size_t errmsg_len);
extern bool tq_wal_init_list_dir_page(Relation relation,
									  Buffer buffer,
									  uint16_t entry_capacity,
									  uint32_t next_block,
									  char *errmsg,
									  size_t errmsg_len);
extern bool tq_wal_set_list_dir_entry(Relation relation,
									  Buffer buffer,
									  uint16_t index,
									  const TqListDirEntry *entry,
									  char *errmsg,
									  size_t errmsg_len);
extern bool tq_wal_init_centroid_page(Relation relation,
									  Buffer buffer,
									  uint32_t dimension,
									  uint16_t centroid_capacity,
									  uint32_t next_block,
									  char *errmsg,
									  size_t errmsg_len);
extern bool tq_wal_set_centroid(Relation relation,
								Buffer buffer,
								uint16_t index,
								const float *centroid,
								uint32_t dimension,
								char *errmsg,
								size_t errmsg_len);
extern bool tq_wal_init_batch_page(Relation relation,
								   Buffer buffer,
								   const TqBatchPageParams *params,
								   char *errmsg,
								   size_t errmsg_len);
extern bool tq_wal_set_batch_next_block(Relation relation,
										Buffer buffer,
										uint32_t next_block,
										char *errmsg,
										size_t errmsg_len);
extern bool tq_wal_append_batch_code(Relation relation,
									 Buffer buffer,
									 const TqTid *tid,
									 const uint8_t *packed_code,
									 size_t packed_code_len,
									 uint16_t *lane_index,
									 char *errmsg,
									 size_t errmsg_len);
extern bool tq_wal_mark_batch_dead(Relation relation,
								   Buffer buffer,
								   uint16_t lane_index,
								   char *errmsg,
								   size_t errmsg_len);
extern bool tq_wal_compact_batch_page(Relation relation,
									  Buffer buffer,
									  char *errmsg,
									  size_t errmsg_len);

#endif
