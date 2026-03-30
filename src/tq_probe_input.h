#ifndef TQ_PROBE_INPUT_H
#define TQ_PROBE_INPUT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/tq_page.h"

typedef bool (*TqProbePageCountFallback) (const TqListDirEntry *entry,
										  uint32_t *page_count,
										  void *context,
										  char *errmsg,
										  size_t errmsg_len);

extern bool tq_build_probe_budget_inputs(const TqListDirEntry *selected_entries,
										 size_t selected_count,
										 uint32_t *ranked_live_counts,
										 uint32_t *ranked_page_counts,
										 TqProbePageCountFallback fallback,
										 void *fallback_context,
										 char *errmsg,
										 size_t errmsg_len);

#endif
