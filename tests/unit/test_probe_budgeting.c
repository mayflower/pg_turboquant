#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "src/tq_query_tuning.h"

static void
assert_selected_indexes(const size_t *selected_indexes,
						  size_t selected_count,
						  size_t expected_first,
						  size_t expected_second,
						  size_t expected_third)
{
	assert(selected_count == 3);
	assert(selected_indexes[0] == expected_first);
	assert(selected_indexes[1] == expected_second);
	assert(selected_indexes[2] == expected_third);
}

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

static void
test_near_exhaustive_scan_activates_on_live_fraction_threshold(void)
{
	assert(tq_should_use_near_exhaustive_scan(70, 100, 2, 10));
	assert(!tq_should_use_near_exhaustive_scan(69, 100, 2, 10));
}

static void
test_near_exhaustive_scan_activates_on_page_fraction_threshold(void)
{
	assert(tq_should_use_near_exhaustive_scan(6, 20, 7, 10));
	assert(!tq_should_use_near_exhaustive_scan(6, 20, 6, 10));
}

static void
test_near_exhaustive_scan_requires_nonzero_totals(void)
{
	assert(!tq_should_use_near_exhaustive_scan(0, 0, 0, 0));
	assert(!tq_should_use_near_exhaustive_scan(8, 0, 0, 0));
	assert(!tq_should_use_near_exhaustive_scan(0, 0, 8, 0));
}

static void
test_cost_aware_selector_prefers_cheaper_combo_under_budget(void)
{
	double scores[] = {100.0, 99.6, 99.5, 99.4};
	uint32_t live_counts[] = {120, 24, 24, 24};
	uint32_t page_counts[] = {16, 3, 3, 3};
	size_t selected_indexes[4] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
	size_t selected_count = 0;
	TqProbeBudgetResult result;
	char errmsg[256];

	memset(&result, 0, sizeof(result));
	assert(tq_select_cost_aware_probes(scores,
									   live_counts,
									   page_counts,
									   4,
									   3,
									   72,
									   0,
									   selected_indexes,
									   4,
									   &selected_count,
									   &result,
									   errmsg,
									   sizeof(errmsg)));
	assert_selected_indexes(selected_indexes, selected_count, 1, 2, 3);
	assert(result.nominal_probe_count == 3);
	assert(result.effective_probe_count == 3);
	assert(result.selected_live_count == 72);
	assert(result.selected_page_count == 9);
}

static void
test_cost_aware_selector_keeps_closest_centroids_without_budget(void)
{
	double scores[] = {100.0, 99.0, 98.0, 97.0};
	uint32_t live_counts[] = {120, 24, 24, 24};
	uint32_t page_counts[] = {16, 3, 3, 3};
	size_t selected_indexes[4] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
	size_t selected_count = 0;
	TqProbeBudgetResult result;
	char errmsg[256];

	memset(&result, 0, sizeof(result));
	assert(tq_select_cost_aware_probes(scores,
									   live_counts,
									   page_counts,
									   4,
									   3,
									   0,
									   0,
									   selected_indexes,
									   4,
									   &selected_count,
									   &result,
									   errmsg,
									   sizeof(errmsg)));
	assert_selected_indexes(selected_indexes, selected_count, 0, 1, 2);
	assert(result.nominal_probe_count == 3);
	assert(result.effective_probe_count == 3);
	assert(result.selected_live_count == 168);
	assert(result.selected_page_count == 22);
}

static void
test_cost_aware_selector_is_deterministic_for_equal_scores_and_costs(void)
{
	double scores[] = {10.0, 10.0, 10.0, 10.0};
	uint32_t live_counts[] = {20, 20, 20, 20};
	uint32_t page_counts[] = {2, 2, 2, 2};
	size_t selected_indexes[4] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
	size_t selected_count = 0;
	TqProbeBudgetResult result;
	char errmsg[256];

	memset(&result, 0, sizeof(result));
	assert(tq_select_cost_aware_probes(scores,
									   live_counts,
									   page_counts,
									   4,
									   2,
									   40,
									   4,
									   selected_indexes,
									   4,
									   &selected_count,
									   &result,
									   errmsg,
									   sizeof(errmsg)));
	assert(selected_count == 2);
	assert(selected_indexes[0] == 0);
	assert(selected_indexes[1] == 1);
	assert(result.selected_live_count == 40);
	assert(result.selected_page_count == 4);
}

int
main(void)
{
	test_small_lists_keep_nominal_probes();
	test_large_lists_reduce_effective_probes();
	test_selected_live_respects_budget_after_first_probe();
	test_router_order_is_preserved();
	test_flat_mode_disables_adaptive_budgeting();
	test_near_exhaustive_scan_activates_on_live_fraction_threshold();
	test_near_exhaustive_scan_activates_on_page_fraction_threshold();
	test_near_exhaustive_scan_requires_nonzero_totals();
	test_cost_aware_selector_prefers_cheaper_combo_under_budget();
	test_cost_aware_selector_keeps_closest_centroids_without_budget();
	test_cost_aware_selector_is_deterministic_for_equal_scores_and_costs();
	return 0;
}
