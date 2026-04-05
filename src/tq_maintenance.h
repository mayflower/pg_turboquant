#ifndef TQ_MAINTENANCE_H
#define TQ_MAINTENANCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/rel.h"

extern bool tq_merge_delta_relation(Relation index_relation,
									uint64_t *merged_delta_count,
									uint32_t *rewritten_list_count,
									uint32_t *recycled_delta_page_count,
									char *errmsg,
									size_t errmsg_len);
extern bool tq_compact_index_relation(Relation index_relation,
									  char *errmsg,
									  size_t errmsg_len);

#endif
