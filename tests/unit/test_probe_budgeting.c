#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "src/tq_query_tuning.h"

static void
test_small_lists_keep_nominal_probes(void)
{
	uint32_t live_counts[] = {3, 4, 2, 1};
	uint32_t page_counts[] = {1, 1, 1, 1};
	TqProbeBudgetResult result;
	char errmsg[256];

	memset(&result, 0, sizeof(result));
	assert(tq_choose_probe_budget(live_counts,
								  page_counts,
								  4,
								  3,
								  32,
								  0,
								  &result,
								  errmsg,
								  sizeof(errmsg)));
	assert(result.nominal_probe_count == 3);
	assert(result.effective_probe_count == 3);
	assert(result.selected_live_count == 9);
	assert(result.selected_page_count == 3);
}

static void
test_large_lists_reduce_effective_probes(void)
{
	uint32_t live_counts[] = {80, 60, 5, 4};
	uint32_t page_counts[] = {10, 7, 1, 1};
	TqProbeBudgetResult result;
	char errmsg[256];

	memset(&result, 0, sizeof(result));
	assert(tq_choose_probe_budget(live_counts,
								  page_counts,
								  4,
								  4,
								  64,
								  0,
								  &result,
								  errmsg,
								  sizeof(errmsg)));
	assert(result.nominal_probe_count == 4);
	assert(result.effective_probe_count == 1);
	assert(result.selected_live_count == 80);
	assert(result.selected_page_count == 10);
}

static void
test_selected_live_respects_budget_after_first_probe(void)
{
	uint32_t live_counts[] = {70, 20, 18, 3};
	uint32_t page_counts[] = {8, 2, 2, 1};
	TqProbeBudgetResult result;
	char errmsg[256];

	memset(&result, 0, sizeof(result));
	assert(tq_choose_probe_budget(live_counts,
								  page_counts,
								  4,
								  4,
								  75,
								  0,
								  &result,
								  errmsg,
								  sizeof(errmsg)));
	assert(result.effective_probe_count == 1);
	assert(result.selected_live_count <= 75 || result.effective_probe_count == 1);
}

static void
test_router_order_is_preserved(void)
{
	uint32_t live_counts[] = {12, 5, 40, 3};
	uint32_t page_counts[] = {2, 1, 6, 1};
	TqProbeBudgetResult result;
	char errmsg[256];

	memset(&result, 0, sizeof(result));
	assert(tq_choose_probe_budget(live_counts,
								  page_counts,
								  4,
								  4,
								  20,
								  0,
								  &result,
								  errmsg,
								  sizeof(errmsg)));
	assert(result.effective_probe_count == 2);
	assert(result.selected_live_count == 17);
	assert(result.selected_page_count == 3);
}

static void
test_flat_mode_disables_adaptive_budgeting(void)
{
	assert(!tq_adaptive_probe_budget_enabled(0, 128, 0));
	assert(!tq_adaptive_probe_budget_enabled(0, 0, 16));
	assert(tq_adaptive_probe_budget_enabled(8, 128, 0));
}

int
main(void)
{
	test_small_lists_keep_nominal_probes();
	test_large_lists_reduce_effective_probes();
	test_selected_live_respects_budget_after_first_probe();
	test_router_order_is_preserved();
	test_flat_mode_disables_adaptive_budgeting();
	return 0;
}
