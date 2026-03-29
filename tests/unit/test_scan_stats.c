#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "src/tq_scan.h"

static void
test_stats_initialize_to_zero_and_empty(void)
{
	TqScanStats stats;
	char json[1024];

	memset(&stats, 0, sizeof(stats));
	memset(json, 0, sizeof(json));

	tq_scan_stats_reset_last();
	tq_scan_stats_snapshot(&stats);

	assert(stats.mode == TQ_SCAN_MODE_NONE);
	assert(stats.score_mode == TQ_SCAN_SCORE_MODE_NONE);
	assert(stats.configured_probe_count == 0);
	assert(stats.selected_list_count == 0);
	assert(stats.selected_live_count == 0);
	assert(stats.visited_page_count == 0);
	assert(stats.visited_code_count == 0);
	assert(stats.retained_candidate_count == 0);
	assert(stats.candidate_heap_capacity == 0);
	assert(stats.candidate_heap_count == 0);
	assert(stats.decoded_vector_count == 0);
	assert(stats.page_prune_count == 0);
	assert(stats.early_stop_count == 0);
	assert(tq_scan_stats_serialize_json(&stats, json, sizeof(json)));
	assert(strstr(json, "\"mode\":\"none\"") != NULL);
	assert(strstr(json, "\"score_mode\":\"none\"") != NULL);
	assert(strstr(json, "\"visited_code_count\":0") != NULL);
	assert(strstr(json, "\"candidate_heap_count\":0") != NULL);
}

static void
test_stats_reset_between_scans(void)
{
	TqScanStats stats;

	memset(&stats, 0, sizeof(stats));

	tq_scan_stats_begin(TQ_SCAN_MODE_IVF, 4);
	tq_scan_stats_set_score_mode(TQ_SCAN_SCORE_MODE_DECODE);
	tq_scan_stats_record_selected_list(5);
	tq_scan_stats_record_page_visit();
	tq_scan_stats_record_code_visit(true);
	tq_scan_stats_set_candidate_heap_metrics(8, 1);

	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 2);
	tq_scan_stats_snapshot(&stats);

	assert(stats.mode == TQ_SCAN_MODE_FLAT);
	assert(stats.score_mode == TQ_SCAN_SCORE_MODE_NONE);
	assert(stats.configured_probe_count == 2);
	assert(stats.selected_list_count == 0);
	assert(stats.selected_live_count == 0);
	assert(stats.visited_page_count == 0);
	assert(stats.visited_code_count == 0);
	assert(stats.retained_candidate_count == 0);
	assert(stats.candidate_heap_capacity == 0);
	assert(stats.candidate_heap_count == 0);
}

static void
test_visited_code_count_bounds_retained_candidates(void)
{
	TqScanStats stats;

	memset(&stats, 0, sizeof(stats));

	tq_scan_stats_begin(TQ_SCAN_MODE_IVF, 3);
	tq_scan_stats_set_score_mode(TQ_SCAN_SCORE_MODE_DECODE);
	tq_scan_stats_record_selected_list(7);
	tq_scan_stats_record_selected_list(4);
	tq_scan_stats_record_code_visit(true);
	tq_scan_stats_record_code_visit(true);
	tq_scan_stats_record_code_visit(true);
	tq_scan_stats_record_code_visit(true);
	tq_scan_stats_record_code_visit(true);
	tq_scan_stats_set_candidate_heap_metrics(6, 3);
	tq_scan_stats_snapshot(&stats);

	assert(stats.visited_code_count >= stats.retained_candidate_count);
	assert(stats.selected_list_count <= stats.configured_probe_count);
	assert(stats.selected_live_count == 11);
	assert(stats.decoded_vector_count == stats.visited_code_count);
}

static void
test_serialization_exposes_stable_field_names(void)
{
	TqScanStats stats;
	char json[1024];

	memset(&stats, 0, sizeof(stats));
	memset(json, 0, sizeof(json));

	tq_scan_stats_begin(TQ_SCAN_MODE_BITMAP, 0);
	tq_scan_stats_set_score_mode(TQ_SCAN_SCORE_MODE_BITMAP_FILTER);
	tq_scan_stats_record_page_visit();
	tq_scan_stats_record_page_visit();
	tq_scan_stats_record_code_visit(false);
	tq_scan_stats_record_code_visit(false);
	tq_scan_stats_set_candidate_heap_metrics(0, 0);
	tq_scan_stats_snapshot(&stats);

	assert(tq_scan_stats_serialize_json(&stats, json, sizeof(json)));
	assert(strstr(json, "\"selected_list_count\"") != NULL);
	assert(strstr(json, "\"selected_live_count\"") != NULL);
	assert(strstr(json, "\"visited_page_count\"") != NULL);
	assert(strstr(json, "\"visited_code_count\"") != NULL);
	assert(strstr(json, "\"retained_candidate_count\"") != NULL);
	assert(strstr(json, "\"candidate_heap_capacity\"") != NULL);
	assert(strstr(json, "\"candidate_heap_count\"") != NULL);
	assert(strstr(json, "\"decoded_vector_count\"") != NULL);
	assert(strstr(json, "\"page_prune_count\"") != NULL);
	assert(strstr(json, "\"early_stop_count\"") != NULL);
}

int
main(void)
{
	test_stats_initialize_to_zero_and_empty();
	test_stats_reset_between_scans();
	test_visited_code_count_bounds_retained_candidates();
	test_serialization_exposes_stable_field_names();
	return 0;
}
