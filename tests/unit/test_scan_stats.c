#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "src/tq_scan.h"
#include "src/tq_simd_avx2.h"

static void
test_stats_initialize_to_zero_and_empty(void)
{
	TqScanStats stats;
	char json[4096];

	memset(&stats, 0, sizeof(stats));
	memset(json, 0, sizeof(json));

	tq_scan_stats_reset_last();
	tq_scan_stats_snapshot(&stats);

	assert(stats.mode == TQ_SCAN_MODE_NONE);
	assert(stats.score_mode == TQ_SCAN_SCORE_MODE_NONE);
	assert(stats.scan_orchestration == TQ_SCAN_ORCHESTRATION_NONE);
	assert(stats.configured_probe_count == 0);
	assert(!stats.near_exhaustive_crossover);
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
	assert(stats.local_candidate_heap_insert_count == 0);
	assert(stats.local_candidate_heap_replace_count == 0);
	assert(stats.local_candidate_heap_reject_count == 0);
	assert(stats.local_candidate_merge_count == 0);
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
	assert(strstr(json, "\"scan_orchestration\":\"none\"") != NULL);
	assert(strstr(json, "\"near_exhaustive_crossover\":false") != NULL);
	assert(strstr(json, "\"visited_code_count\":0") != NULL);
	assert(strstr(json, "\"bound_data_page_reads\":0") != NULL);
	assert(strstr(json, "\"candidate_heap_count\":0") != NULL);
	assert(strstr(json, "\"candidate_heap_insert_count\":0") != NULL);
	assert(strstr(json, "\"candidate_heap_replace_count\":0") != NULL);
	assert(strstr(json, "\"candidate_heap_reject_count\":0") != NULL);
	assert(strstr(json, "\"local_candidate_heap_insert_count\":0") != NULL);
	assert(strstr(json, "\"local_candidate_heap_replace_count\":0") != NULL);
	assert(strstr(json, "\"local_candidate_heap_reject_count\":0") != NULL);
	assert(strstr(json, "\"local_candidate_merge_count\":0") != NULL);
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
	assert(stats.scan_orchestration == TQ_SCAN_ORCHESTRATION_NONE);
	assert(stats.configured_probe_count == 2);
	assert(!stats.near_exhaustive_crossover);
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
	assert(stats.local_candidate_heap_insert_count == 0);
	assert(stats.local_candidate_heap_replace_count == 0);
	assert(stats.local_candidate_heap_reject_count == 0);
	assert(stats.local_candidate_merge_count == 0);
	assert(stats.shadow_decoded_vector_count == 0);
	assert(stats.shadow_decode_candidate_count == 0);
	assert(stats.shadow_decode_overlap_count == 0);
	assert(stats.shadow_decode_primary_only_count == 0);
	assert(stats.shadow_decode_only_count == 0);
}

static void
test_scan_orchestration_tracks_near_exhaustive_crossover(void)
{
	TqScanStats stats;
	char json[4096];

	memset(&stats, 0, sizeof(stats));
	memset(json, 0, sizeof(json));

	tq_scan_stats_begin(TQ_SCAN_MODE_IVF, 8);
	tq_scan_stats_set_scan_orchestration(TQ_SCAN_ORCHESTRATION_IVF_NEAR_EXHAUSTIVE, true);
	tq_scan_stats_snapshot(&stats);

	assert(stats.scan_orchestration == TQ_SCAN_ORCHESTRATION_IVF_NEAR_EXHAUSTIVE);
	assert(stats.near_exhaustive_crossover);
	assert(tq_scan_stats_serialize_json(&stats, json, sizeof(json)));
	assert(strstr(json, "\"scan_orchestration\":\"ivf_near_exhaustive\"") != NULL);
	assert(strstr(json, "\"near_exhaustive_crossover\":true") != NULL);
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
	assert(tq_candidate_heap_push(&heap, 5.0f, 1, 1, NULL, 0));
	assert(tq_candidate_heap_push(&heap, 3.0f, 1, 2, NULL, 0));
	assert(tq_candidate_heap_push(&heap, 4.0f, 1, 3, NULL, 0));
	assert(tq_candidate_heap_push(&heap, 6.0f, 1, 4, NULL, 0));
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
test_local_candidate_heap_metrics_are_serialized_separately(void)
{
	TqScanStats stats;
	char json[4096];

	memset(&stats, 0, sizeof(stats));
	memset(json, 0, sizeof(json));

	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1);
	tq_scan_stats_record_local_candidate_heap_insert();
	tq_scan_stats_record_local_candidate_heap_insert();
	tq_scan_stats_record_local_candidate_heap_replace();
	tq_scan_stats_record_local_candidate_heap_reject();
	tq_scan_stats_record_local_candidate_merge();
	tq_scan_stats_record_local_candidate_merge();
	tq_scan_stats_snapshot(&stats);

	assert(stats.local_candidate_heap_insert_count == 2);
	assert(stats.local_candidate_heap_replace_count == 1);
	assert(stats.local_candidate_heap_reject_count == 1);
	assert(stats.local_candidate_merge_count == 2);
	assert(stats.candidate_heap_insert_count == 0);
	assert(stats.candidate_heap_replace_count == 0);
	assert(stats.candidate_heap_reject_count == 0);
	assert(tq_scan_stats_serialize_json(&stats, json, sizeof(json)));
	assert(strstr(json, "\"local_candidate_heap_insert_count\":2") != NULL);
	assert(strstr(json, "\"local_candidate_heap_replace_count\":1") != NULL);
	assert(strstr(json, "\"local_candidate_heap_reject_count\":1") != NULL);
	assert(strstr(json, "\"local_candidate_merge_count\":2") != NULL);
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

	assert(tq_candidate_heap_push(&primary, 1.0f, 1, 1, NULL, 0));
	assert(tq_candidate_heap_push(&primary, 2.0f, 1, 2, NULL, 0));
	assert(tq_candidate_heap_push(&primary, 3.0f, 1, 3, NULL, 0));

	assert(tq_candidate_heap_push(&shadow, 2.0f, 1, 2, NULL, 0));
	assert(tq_candidate_heap_push(&shadow, 3.0f, 1, 3, NULL, 0));
	assert(tq_candidate_heap_push(&shadow, 4.0f, 1, 4, NULL, 0));

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
	char json[4096];

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
	assert(strstr(json, "\"scan_orchestration\"") != NULL);
	assert(strstr(json, "\"near_exhaustive_crossover\"") != NULL);
	assert(strstr(json, "\"visited_page_count\"") != NULL);
	assert(strstr(json, "\"visited_code_count\"") != NULL);
	assert(strstr(json, "\"retained_candidate_count\"") != NULL);
	assert(strstr(json, "\"candidate_heap_capacity\"") != NULL);
	assert(strstr(json, "\"candidate_heap_count\"") != NULL);
	assert(strstr(json, "\"candidate_heap_insert_count\"") != NULL);
	assert(strstr(json, "\"candidate_heap_replace_count\"") != NULL);
	assert(strstr(json, "\"candidate_heap_reject_count\"") != NULL);
	assert(strstr(json, "\"local_candidate_heap_insert_count\"") != NULL);
	assert(strstr(json, "\"local_candidate_heap_replace_count\"") != NULL);
	assert(strstr(json, "\"local_candidate_heap_reject_count\"") != NULL);
	assert(strstr(json, "\"local_candidate_merge_count\"") != NULL);
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

static void
test_router_selection_method_is_serialized(void)
{
	TqScanStats stats;
	char json[8192];

	memset(&stats, 0, sizeof(stats));
	memset(json, 0, sizeof(json));

	tq_scan_stats_begin(TQ_SCAN_MODE_IVF, 4);
	tq_scan_stats_set_router_selection_method(TQ_ROUTER_SELECTION_PARTIAL);
	tq_scan_stats_snapshot(&stats);

	assert(stats.router_selection_method == TQ_ROUTER_SELECTION_PARTIAL);
	assert(tq_scan_stats_serialize_json(&stats, json, sizeof(json)));
	assert(strstr(json, "\"router_selection_method\":\"partial\"") != NULL);

	tq_scan_stats_begin(TQ_SCAN_MODE_IVF, 8);
	tq_scan_stats_set_router_selection_method(TQ_ROUTER_SELECTION_FULL_SORT);
	tq_scan_stats_snapshot(&stats);

	assert(stats.router_selection_method == TQ_ROUTER_SELECTION_FULL_SORT);
	assert(tq_scan_stats_serialize_json(&stats, json, sizeof(json)));
	assert(strstr(json, "\"router_selection_method\":\"full_sort\"") != NULL);

	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1);
	tq_scan_stats_snapshot(&stats);

	assert(stats.router_selection_method == TQ_ROUTER_SELECTION_NONE);
	assert(tq_scan_stats_serialize_json(&stats, json, sizeof(json)));
	assert(strstr(json, "\"router_selection_method\":\"none\"") != NULL);
}

static void
test_block16_select_top_m_returns_best_candidates(void)
{
	/* Scores: higher = better. Select top-3 from 8 candidates */
	float scores[8] = {0.1f, 0.9f, 0.3f, 0.7f, 0.5f, 0.2f, 0.8f, 0.4f};
	uint32_t selected[8];
	uint32_t count = 0;
	uint32_t i = 0;
	bool has_1 = false;
	bool has_6 = false;
	bool has_3 = false;

	memset(selected, 0, sizeof(selected));

	/* Top-3 should be indices 1 (0.9), 6 (0.8), 3 (0.7) */
	count = tq_block16_select_top_m(scores, 8, 3, selected);
	assert(count == 3);

	for (i = 0; i < count; i++)
	{
		if (selected[i] == 1) has_1 = true;
		if (selected[i] == 6) has_6 = true;
		if (selected[i] == 3) has_3 = true;
	}
	assert(has_1 && has_6 && has_3);

	/* Top-M >= candidate_count returns all */
	count = tq_block16_select_top_m(scores, 8, 10, selected);
	assert(count == 8);

	/* Top-1 should be index 1 (score 0.9) */
	count = tq_block16_select_top_m(scores, 8, 1, selected);
	assert(count == 1);
	assert(selected[0] == 1);
}

static void
test_block_local_selection_stats_are_serialized(void)
{
	TqScanStats stats;
	char json[8192];

	memset(&stats, 0, sizeof(stats));
	memset(json, 0, sizeof(json));

	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1);
	tq_scan_stats_record_block_local_selection(16, 8);
	tq_scan_stats_record_block_local_selection(12, 5);
	tq_scan_stats_snapshot(&stats);

	assert(stats.block_local_scored_count == 28);
	assert(stats.block_local_survivor_count == 13);
	assert(stats.block_local_rejected_count == 15);

	assert(tq_scan_stats_serialize_json(&stats, json, sizeof(json)));
	assert(strstr(json, "\"block_local_scored_count\":28") != NULL);
	assert(strstr(json, "\"block_local_survivor_count\":13") != NULL);
	assert(strstr(json, "\"block_local_rejected_count\":15") != NULL);
}

static void
test_speed_path_label_enums_return_stable_strings(void)
{
	assert(strcmp(tq_lookup_style_name(TQ_LOOKUP_STYLE_SCALAR_LOOP), "scalar_loop") == 0);
	assert(strcmp(tq_lookup_style_name(TQ_LOOKUP_STYLE_FLOAT_GATHER), "float_gather") == 0);
	assert(strcmp(tq_lookup_style_name(TQ_LOOKUP_STYLE_LUT16_SCALAR), "lut16_scalar") == 0);
	assert(strcmp(tq_lookup_style_name(TQ_LOOKUP_STYLE_LUT16_AVX2), "lut16_avx2") == 0);
	assert(strcmp(tq_lookup_style_name(TQ_LOOKUP_STYLE_LUT16_NEON), "lut16_neon") == 0);
	assert(strcmp(tq_lookup_style_name(TQ_LOOKUP_STYLE_LUT16_AVX512), "lut16_avx512") == 0);

	assert(strcmp(tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_SCALAR), "float32_scalar") == 0);
	assert(strcmp(tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_VECTOR), "float32_vector") == 0);
	assert(strcmp(tq_gamma_path_name(TQ_GAMMA_PATH_FP16_VECTOR), "fp16_vector") == 0);

	assert(strcmp(tq_qjl_path_name(TQ_QJL_PATH_FLOAT), "float") == 0);
	assert(strcmp(tq_qjl_path_name(TQ_QJL_PATH_INT16_QUANTIZED), "int16_quantized") == 0);
	assert(strcmp(tq_qjl_path_name(TQ_QJL_PATH_LUT16_QUANTIZED), "lut16_quantized") == 0);

	assert(tq_lookup_style_for_kernel(TQ_PROD_SCORE_SCALAR) == TQ_LOOKUP_STYLE_SCALAR_LOOP);
	assert(tq_lookup_style_for_kernel(TQ_PROD_SCORE_AVX2) == TQ_LOOKUP_STYLE_FLOAT_GATHER);
	assert(tq_lookup_style_for_kernel(TQ_PROD_SCORE_NEON) == TQ_LOOKUP_STYLE_FLOAT_GATHER);

	assert(tq_qjl_path_for_kernel(TQ_PROD_SCORE_SCALAR, false) == TQ_QJL_PATH_FLOAT);
	assert(tq_qjl_path_for_kernel(TQ_PROD_SCORE_SCALAR, true) == TQ_QJL_PATH_FLOAT);
	assert(tq_qjl_path_for_kernel(TQ_PROD_SCORE_AVX2, true) == TQ_QJL_PATH_INT16_QUANTIZED);
	assert(tq_qjl_path_for_kernel(TQ_PROD_SCORE_NEON, true) == TQ_QJL_PATH_INT16_QUANTIZED);
	assert(tq_qjl_path_for_kernel(TQ_PROD_SCORE_AVX2, false) == TQ_QJL_PATH_FLOAT);
}

int
main(void)
{
	test_stats_initialize_to_zero_and_empty();
	test_stats_reset_between_scans();
	test_scan_orchestration_tracks_near_exhaustive_crossover();
	test_visited_code_count_bounds_retained_candidates();
	test_candidate_heap_event_counters_track_insert_replace_and_reject();
	test_local_candidate_heap_metrics_are_serialized_separately();
	test_shadow_decode_metrics_track_heap_overlap();
	test_serialization_exposes_stable_field_names();
	test_speed_path_label_enums_return_stable_strings();
	test_router_selection_method_is_serialized();
	test_block16_select_top_m_returns_best_candidates();
	test_block_local_selection_stats_are_serialized();
	return 0;
}
