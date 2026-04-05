#include "src/tq_probe_input.h"

#include <stdio.h>

#ifdef TQ_UNIT_TEST
#undef snprintf
#endif

bool
tq_build_probe_budget_inputs(const TqListDirEntry *selected_entries,
							 size_t selected_count,
							 uint32_t *ranked_live_counts,
							 uint32_t *ranked_page_counts,
							 TqProbePageCountFallback fallback,
							 void *fallback_context,
							 char *errmsg,
							 size_t errmsg_len)
{
	size_t		index = 0;

	if (selected_count > 0
		&& (selected_entries == NULL || ranked_live_counts == NULL || ranked_page_counts == NULL))
	{
		if (errmsg_len > 0)
			snprintf(errmsg, errmsg_len,
					 "invalid turboquant probe inputs: entries and outputs must be non-null");
		return false;
	}

	for (index = 0; index < selected_count; index++)
	{
		const TqListDirEntry *entry = &selected_entries[index];

		ranked_live_counts[index] = entry->live_count;
		if (entry->batch_page_count > 0 || entry->head_block == TQ_INVALID_BLOCK_NUMBER)
		{
			ranked_page_counts[index] = entry->batch_page_count;
			continue;
		}

		if (fallback == NULL)
		{
			if (errmsg_len > 0)
				snprintf(errmsg, errmsg_len,
						 "invalid turboquant probe inputs: page count metadata is missing for list %u",
						 entry->list_id);
			return false;
		}

		if (!fallback(entry,
					  &ranked_page_counts[index],
					  fallback_context,
					  errmsg,
					  errmsg_len))
			return false;
	}

	return true;
}
