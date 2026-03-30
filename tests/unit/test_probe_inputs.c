#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "src/tq_page.h"
#include "src/tq_probe_input.h"

typedef struct ProbeInputFallbackState
{
	int			call_count;
	uint32_t	last_list_id;
	uint32_t	last_head_block;
	uint32_t	returned_page_count;
} ProbeInputFallbackState;

static bool
test_probe_input_fallback(const TqListDirEntry *entry,
						  uint32_t *page_count,
						  void *context,
						  char *errmsg,
						  size_t errmsg_len)
{
	ProbeInputFallbackState *state = (ProbeInputFallbackState *) context;

	(void) errmsg;
	(void) errmsg_len;

	state->call_count += 1;
	state->last_list_id = entry->list_id;
	state->last_head_block = entry->head_block;
	*page_count = state->returned_page_count;
	return true;
}

static void
test_metadata_page_counts_avoid_fallback(void)
{
	TqListDirEntry entries[] = {
		{.list_id = 0, .head_block = 11, .tail_block = 13, .live_count = 32, .dead_count = 0, .batch_page_count = 3, .free_lane_hint = 0},
		{.list_id = 1, .head_block = 21, .tail_block = 24, .live_count = 12, .dead_count = 1, .batch_page_count = 4, .free_lane_hint = 0},
	};
	TqListDirEntry selected_entries[] = {entries[1], entries[0]};
	uint32_t live_counts[2];
	uint32_t page_counts[2];
	ProbeInputFallbackState state;
	char errmsg[256];

	memset(live_counts, 0, sizeof(live_counts));
	memset(page_counts, 0, sizeof(page_counts));
	memset(&state, 0, sizeof(state));

	assert(tq_build_probe_budget_inputs(selected_entries,
										2,
										live_counts,
										page_counts,
										test_probe_input_fallback,
										&state,
										errmsg,
										sizeof(errmsg)));
	assert(state.call_count == 0);
	assert(live_counts[0] == 12);
	assert(live_counts[1] == 32);
	assert(page_counts[0] == 4);
	assert(page_counts[1] == 3);
}

static void
test_missing_page_counts_use_fallback(void)
{
	TqListDirEntry entries[] = {
		{.list_id = 0, .head_block = 41, .tail_block = 44, .live_count = 18, .dead_count = 0, .batch_page_count = 0, .free_lane_hint = 0},
	};
	uint32_t live_counts[1];
	uint32_t page_counts[1];
	ProbeInputFallbackState state;
	char errmsg[256];

	memset(live_counts, 0, sizeof(live_counts));
	memset(page_counts, 0, sizeof(page_counts));
	memset(&state, 0, sizeof(state));
	state.returned_page_count = 6;

	assert(tq_build_probe_budget_inputs(entries,
										1,
										live_counts,
										page_counts,
										test_probe_input_fallback,
										&state,
										errmsg,
										sizeof(errmsg)));
	assert(state.call_count == 1);
	assert(state.last_list_id == 0);
	assert(state.last_head_block == 41);
	assert(live_counts[0] == 18);
	assert(page_counts[0] == 6);
}

int
main(void)
{
	test_metadata_page_counts_avoid_fallback();
	test_missing_page_counts_use_fallback();
	return 0;
}
