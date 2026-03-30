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
	assert(stats.selected_page_count == 0);
	assert(stats.visited_page_count == 0);
	assert(stats.visited_code_count == 0);
	assert(stats.retained_candidate_count == 0);
	assert(stats.candidate_heap_capacity == 0);
	assert(stats.candidate_heap_count == 0);
	assert(stats.candidate_heap_insert_count == 0);
	assert(stats.candidate_heap_replace_count == 0);
	assert(stats.candidate_heap_reject_count == 0);
	assert(stats.shadow_decoded_vector_count == 0);
	assert(stats.shadow_decode_candidate_count == 0);
	assert(stats.shadow_decode_overlap_count == 0);
	assert(stats.shadow_decode_primary_only_count == 0);
	assert(stats.shadow_decode_only_count == 0);
	assert(stats.decoded_vector_count == 0);
	assert(stats.bound_data_page_reads == 0);
	assert(stats.page_prune_count == 0);
	assert(stats.early_stop_count == 0);
	assert(tq_scan_stats_serialize_json(&stats, json, sizeof(json)));
	assert(strstr(json, "\"mode\":\"none\"") != NULL);
	assert(strstr(json, "\"score_mode\":\"none\"") != NULL);
	assert(strstr(json, "\"visited_code_count\":0") != NULL);
	assert(strstr(json, "\"bound_data_page_reads\":0") != NULL);
	assert(strstr(json, "\"candidate_heap_count\":0") != NULL);
	assert(strstr(json, "\"candidate_heap_insert_count\":0") != NULL);
	assert(strstr(json, "\"candidate_heap_replace_count\":0") != NULL);
	assert(strstr(json, "\"candidate_heap_reject_count\":0") != NULL);
	assert(strstr(json, "\"shadow_decoded_vector_count\":0") != NULL);
	assert(strstr(json, "\"shadow_decode_candidate_count\":0") != NULL);
	assert(strstr(json, "\"shadow_decode_overlap_count\":0") != NULL);
	assert(strstr(json, "\"shadow_decode_primary_only_count\":0") != NULL);
	assert(strstr(json, "\"shadow_decode_only_count\":0") != NULL);
}

static void
test_stats_reset_between_scans(void)
{
	TqScanStats stats;

	memset(&stats, 0, sizeof(stats));

	tq_scan_stats_begin(TQ_SCAN_MODE_IVF, 4);
	tq_scan_stats_set_score_mode(TQ_SCAN_SCORE_MODE_DECODE);
	tq_scan_stats_record_selected_list(5, 2);
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
	assert(stats.selected_page_count == 0);
	assert(stats.visited_page_count == 0);
	assert(stats.visited_code_count == 0);
	assert(stats.retained_candidate_count == 0);
	assert(stats.candidate_heap_capacity == 0);
	assert(stats.candidate_heap_count == 0);
	assert(stats.candidate_heap_insert_count == 0);
	assert(stats.candidate_heap_replace_count == 0);
	assert(stats.candidate_heap_reject_count == 0);
	assert(stats.shadow_decoded_vector_count == 0);
	assert(stats.shadow_decode_candidate_count == 0);
	assert(stats.shadow_decode_overlap_count == 0);
	assert(stats.shadow_decode_primary_only_count == 0);
	assert(stats.shadow_decode_only_count == 0);
}

static void
test_visited_code_count_bounds_retained_candidates(void)
{
	TqScanStats stats;

	memset(&stats, 0, sizeof(stats));

	tq_scan_stats_begin(TQ_SCAN_MODE_IVF, 3);
	tq_scan_stats_set_score_mode(TQ_SCAN_SCORE_MODE_DECODE);
	tq_scan_stats_record_selected_list(7, 3);
	tq_scan_stats_record_selected_list(4, 2);
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
	assert(stats.selected_page_count == 5);
	assert(stats.decoded_vector_count == stats.visited_code_count);
}

static void
test_candidate_heap_event_counters_track_insert_replace_and_reject(void)
{
	TqCandidateHeap heap;
	TqScanStats stats;

	memset(&heap, 0, sizeof(heap));
	memset(&stats, 0, sizeof(stats));

	tq_scan_stats_begin(TQ_SCAN_MODE_IVF, 4);
	assert(tq_candidate_heap_init(&heap, 2));
	assert(tq_candidate_heap_push(&heap, 5.0f, 1, 1));
	assert(tq_candidate_heap_push(&heap, 3.0f, 1, 2));
	assert(tq_candidate_heap_push(&heap, 4.0f, 1, 3));
	assert(tq_candidate_heap_push(&heap, 6.0f, 1, 4));
	tq_scan_stats_set_candidate_heap_metrics(heap.capacity, heap.count);
	tq_scan_stats_snapshot(&stats);

	assert(stats.candidate_heap_insert_count == 2);
	assert(stats.candidate_heap_replace_count == 1);
	assert(stats.candidate_heap_reject_count == 1);
	assert(stats.candidate_heap_count == 2);
	assert(stats.retained_candidate_count == 2);

	tq_candidate_heap_reset(&heap);
}

static void
test_shadow_decode_metrics_track_heap_overlap(void)
{
	TqCandidateHeap primary;
	TqCandidateHeap shadow;
	TqScanStats stats;
	TqTid copied[4];
	size_t copied_count = 0;

	memset(&primary, 0, sizeof(primary));
	memset(&shadow, 0, sizeof(shadow));
	memset(&stats, 0, sizeof(stats));
	memset(copied, 0, sizeof(copied));

	tq_scan_stats_begin(TQ_SCAN_MODE_IVF, 4);
	assert(tq_candidate_heap_init(&primary, 3));
	assert(tq_candidate_heap_init(&shadow, 3));

	assert(tq_candidate_heap_push(&primary, 1.0f, 1, 1));
	assert(tq_candidate_heap_push(&primary, 2.0f, 1, 2));
	assert(tq_candidate_heap_push(&primary, 3.0f, 1, 3));

	assert(tq_candidate_heap_push(&shadow, 2.0f, 1, 2));
	assert(tq_candidate_heap_push(&shadow, 3.0f, 1, 3));
	assert(tq_candidate_heap_push(&shadow, 4.0f, 1, 4));

	tq_scan_stats_record_shadow_decoded_vector();
	tq_scan_stats_record_shadow_decoded_vector();
	tq_scan_stats_record_shadow_decoded_vector();
	tq_scan_stats_set_shadow_decode_metrics(&primary, &shadow);
	tq_scan_stats_snapshot(&stats);

	assert(stats.shadow_decoded_vector_count == 3);
	assert(stats.shadow_decode_candidate_count == 3);
	assert(stats.shadow_decode_overlap_count == 2);
	assert(stats.shadow_decode_primary_only_count == 1);
	assert(stats.shadow_decode_only_count == 1);
	assert(tq_scan_stats_copy_shadow_decode_tids(copied, 4, &copied_count));
	assert(copied_count == 3);
	assert((copied[0].block_number == 1 && copied[0].offset_number >= 2 && copied[0].offset_number <= 4)
		   || (copied[1].block_number == 1 && copied[1].offset_number >= 2 && copied[1].offset_number <= 4)
		   || (copied[2].block_number == 1 && copied[2].offset_number >= 2 && copied[2].offset_number <= 4));
	assert((copied[0].offset_number == 2) || (copied[1].offset_number == 2) || (copied[2].offset_number == 2));
	assert((copied[0].offset_number == 3) || (copied[1].offset_number == 3) || (copied[2].offset_number == 3));
	assert((copied[0].offset_number == 4) || (copied[1].offset_number == 4) || (copied[2].offset_number == 4));

	tq_candidate_heap_reset(&primary);
	tq_candidate_heap_reset(&shadow);
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
	assert(strstr(json, "\"selected_page_count\"") != NULL);
	assert(strstr(json, "\"visited_page_count\"") != NULL);
	assert(strstr(json, "\"visited_code_count\"") != NULL);
	assert(strstr(json, "\"retained_candidate_count\"") != NULL);
	assert(strstr(json, "\"candidate_heap_capacity\"") != NULL);
	assert(strstr(json, "\"candidate_heap_count\"") != NULL);
	assert(strstr(json, "\"candidate_heap_insert_count\"") != NULL);
	assert(strstr(json, "\"candidate_heap_replace_count\"") != NULL);
	assert(strstr(json, "\"candidate_heap_reject_count\"") != NULL);
	assert(strstr(json, "\"shadow_decoded_vector_count\"") != NULL);
	assert(strstr(json, "\"shadow_decode_candidate_count\"") != NULL);
	assert(strstr(json, "\"shadow_decode_overlap_count\"") != NULL);
	assert(strstr(json, "\"shadow_decode_primary_only_count\"") != NULL);
	assert(strstr(json, "\"shadow_decode_only_count\"") != NULL);
	assert(strstr(json, "\"decoded_vector_count\"") != NULL);
	assert(strstr(json, "\"bound_data_page_reads\"") != NULL);
	assert(strstr(json, "\"page_prune_count\"") != NULL);
	assert(strstr(json, "\"early_stop_count\"") != NULL);
}

int
main(void)
{
	test_stats_initialize_to_zero_and_empty();
	test_stats_reset_between_scans();
	test_visited_code_count_bounds_retained_candidates();
	test_candidate_heap_event_counters_track_insert_replace_and_reject();
	test_shadow_decode_metrics_track_heap_overlap();
	test_serialization_exposes_stable_field_names();
	return 0;
}
