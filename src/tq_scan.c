#include "src/tq_scan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/tq_guc.h"
#include "src/tq_simd_avx2.h"

static void
tq_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg_len == 0)
		return;

	snprintf(errmsg, errmsg_len, "%s", message);
}

static TqScanStats tq_last_scan_stats;
static TqTid *tq_last_shadow_decode_tids = NULL;
static size_t tq_last_shadow_decode_tid_count = 0;
static size_t tq_last_shadow_decode_tid_capacity = 0;
typedef enum TqCandidateHeapMetricsMode
{
	TQ_CANDIDATE_HEAP_METRICS_NONE = 0,
	TQ_CANDIDATE_HEAP_METRICS_GLOBAL,
	TQ_CANDIDATE_HEAP_METRICS_LOCAL
} TqCandidateHeapMetricsMode;
static int tq_candidate_compare_best_qsort(const void *left, const void *right);
static int tq_candidate_tid_compare_qsort(const void *left, const void *right);
static int tq_tid_compare(const TqTid *left, const TqTid *right);
static bool tq_candidate_tid_is_selected(const TqTid *candidate_tids,
										 size_t candidate_tid_count,
										 const TqTid *tid);
static bool tq_scan_scratch_ensure_decoded_capacity(TqScanScratch *scratch,
													size_t required_capacity,
													char *errmsg,
													size_t errmsg_len);
static void tq_scan_stats_record_heap_insert(TqCandidateHeapMetricsMode metrics_mode);
static void tq_scan_stats_record_heap_replace(TqCandidateHeapMetricsMode metrics_mode);
static void tq_scan_stats_record_heap_reject(TqCandidateHeapMetricsMode metrics_mode);
static bool tq_candidate_heap_push_internal(TqCandidateHeap *heap,
											 float score,
											 uint32_t block_number,
											 uint16_t offset_number,
											 TqCandidateHeapMetricsMode metrics_mode);
static bool tq_candidate_heap_merge_into_global(TqCandidateHeap *global_heap,
												 const TqCandidateHeap *local_heap);
static const char *tq_page_bound_mode_name(TqPageBoundMode mode);
static const char *tq_scan_orchestration_name(TqScanOrchestration scan_orchestration);
static TqPageBoundMode tq_page_bound_mode_merge(TqPageBoundMode current,
												TqPageBoundMode incoming);

static const char *
tq_scan_mode_name(TqScanMode mode)
{
	switch (mode)
	{
		case TQ_SCAN_MODE_FLAT:
			return "flat";
		case TQ_SCAN_MODE_IVF:
			return "ivf";
		case TQ_SCAN_MODE_BITMAP:
			return "bitmap";
		case TQ_SCAN_MODE_NONE:
		default:
			return "none";
	}
}

static const char *
tq_scan_score_mode_name(TqScanScoreMode mode)
{
	switch (mode)
	{
		case TQ_SCAN_SCORE_MODE_DECODE:
			return "decode";
		case TQ_SCAN_SCORE_MODE_DECODE_RESCORE:
			return "decode_rescore";
		case TQ_SCAN_SCORE_MODE_CODE_DOMAIN:
			return "code_domain";
		case TQ_SCAN_SCORE_MODE_BITMAP_FILTER:
			return "bitmap_filter";
		case TQ_SCAN_SCORE_MODE_NONE:
		default:
			return "none";
	}
}

static const char *
tq_scan_score_kernel_name(const TqScanStats *stats)
{
	if (stats == NULL)
		return "none";

	if (stats->score_mode == TQ_SCAN_SCORE_MODE_NONE
		|| stats->score_mode == TQ_SCAN_SCORE_MODE_BITMAP_FILTER)
		return "none";

	return tq_prod_score_kernel_name(stats->score_kernel);
}

static const char *
tq_scan_orchestration_name(TqScanOrchestration scan_orchestration)
{
	switch (scan_orchestration)
	{
		case TQ_SCAN_ORCHESTRATION_FLAT_STREAMING:
			return "flat_streaming";
		case TQ_SCAN_ORCHESTRATION_FLAT_BOUNDED_PAGES:
			return "flat_bounded_pages";
		case TQ_SCAN_ORCHESTRATION_IVF_BOUNDED_PAGES:
			return "ivf_bounded_pages";
		case TQ_SCAN_ORCHESTRATION_IVF_NEAR_EXHAUSTIVE:
			return "ivf_near_exhaustive";
		case TQ_SCAN_ORCHESTRATION_BITMAP_FILTER:
			return "bitmap_filter";
		case TQ_SCAN_ORCHESTRATION_NONE:
		default:
			return "none";
	}
}

static const char *
tq_page_bound_mode_name(TqPageBoundMode mode)
{
	switch (mode)
	{
		case TQ_PAGE_BOUND_MODE_NONE:
			return "none";
		case TQ_PAGE_BOUND_MODE_DISABLED:
			return "disabled";
		case TQ_PAGE_BOUND_MODE_SAFE_SUMMARY_PRUNING:
			return "safe_summary_pruning";
		case TQ_PAGE_BOUND_MODE_ORDERING_ONLY_SUMMARY:
			return "ordering_only_summary";
		case TQ_PAGE_BOUND_MODE_DATA_PAGE_FALLBACK:
			return "data_page_fallback";
		case TQ_PAGE_BOUND_MODE_MIXED:
			return "mixed";
	}

	return "unknown";
}

static const char *
tq_router_selection_method_name(TqRouterSelectionMethod method)
{
	switch (method)
	{
		case TQ_ROUTER_SELECTION_NONE:
			return "none";
		case TQ_ROUTER_SELECTION_PARTIAL:
			return "partial";
		case TQ_ROUTER_SELECTION_FULL_SORT:
			return "full_sort";
	}

	return "unknown";
}

static TqPageBoundMode
tq_page_bound_mode_merge(TqPageBoundMode current, TqPageBoundMode incoming)
{
	if (incoming == TQ_PAGE_BOUND_MODE_NONE)
		return current;
	if (current == TQ_PAGE_BOUND_MODE_NONE
		|| current == TQ_PAGE_BOUND_MODE_DISABLED)
		return incoming;
	if (incoming == TQ_PAGE_BOUND_MODE_DISABLED)
		return current;
	if (current == incoming)
		return current;
	return TQ_PAGE_BOUND_MODE_MIXED;
}

static void
tq_candidate_swap(TqCandidateEntry *left, TqCandidateEntry *right)
{
	TqCandidateEntry tmp = *left;

	*left = *right;
	*right = tmp;
}

static int
tq_candidate_compare_worst(const TqCandidateEntry *left, const TqCandidateEntry *right)
{
	if (left->score > right->score)
		return 1;
	if (left->score < right->score)
		return -1;
	if (left->tid.block_number > right->tid.block_number)
		return 1;
	if (left->tid.block_number < right->tid.block_number)
		return -1;
	if (left->tid.offset_number > right->tid.offset_number)
		return 1;
	if (left->tid.offset_number < right->tid.offset_number)
		return -1;
	return 0;
}

static bool
tq_candidate_tid_equal(const TqCandidateEntry *left, const TqCandidateEntry *right)
{
	return left != NULL
		&& right != NULL
		&& left->tid.block_number == right->tid.block_number
		&& left->tid.offset_number == right->tid.offset_number;
}

static bool
tq_scan_stats_ensure_shadow_decode_tid_capacity(size_t required_capacity)
{
	TqTid	   *resized = NULL;

	if (required_capacity <= tq_last_shadow_decode_tid_capacity)
		return true;

	resized = (TqTid *) realloc(tq_last_shadow_decode_tids,
								required_capacity * sizeof(TqTid));
	if (resized == NULL)
		return false;

	tq_last_shadow_decode_tids = resized;
	tq_last_shadow_decode_tid_capacity = required_capacity;
	return true;
}

static void
tq_scan_stats_store_shadow_decode_tids(const TqCandidateHeap *shadow)
{
	TqCandidateEntry *ordered_entries = NULL;
	size_t		index = 0;

	tq_last_shadow_decode_tid_count = 0;

	if (shadow == NULL || shadow->entries == NULL || shadow->count == 0)
		return;

	if (!tq_scan_stats_ensure_shadow_decode_tid_capacity(shadow->count))
		return;

	ordered_entries = (TqCandidateEntry *) malloc(sizeof(TqCandidateEntry) * shadow->count);
	if (ordered_entries == NULL)
		return;

	memcpy(ordered_entries,
		   shadow->entries,
		   sizeof(TqCandidateEntry) * shadow->count);
	qsort(ordered_entries,
		  shadow->count,
		  sizeof(TqCandidateEntry),
		  tq_candidate_compare_best_qsort);

	for (index = 0; index < shadow->count; index++)
		tq_last_shadow_decode_tids[index] = ordered_entries[index].tid;

	tq_last_shadow_decode_tid_count = shadow->count;
	free(ordered_entries);
}

bool
tq_scan_active_uses_prod_code_domain(bool normalized, TqDistanceKind distance)
{
	if (!normalized)
		return false;

	switch (distance)
	{
		case TQ_DISTANCE_COSINE:
		case TQ_DISTANCE_IP:
			return true;
		default:
			return false;
	}
}

static bool
tq_scan_can_use_prod_code_domain(bool normalized, TqDistanceKind distance)
{
	return tq_scan_active_uses_prod_code_domain(normalized, distance);
}

static bool
tq_scan_scratch_ensure_decoded_capacity(TqScanScratch *scratch,
										size_t required_capacity,
										char *errmsg,
										size_t errmsg_len)
{
	float *resized = NULL;

	if (scratch == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: scratch state must be non-null");
		return false;
	}

	if (required_capacity <= scratch->decoded_capacity)
	{
		scratch->decoded_buffer_reuses += 1;
		tq_scan_stats_record_decoded_buffer_reuses(1);
		return true;
	}

	resized = (float *) realloc(scratch->decoded_values,
								sizeof(float) * required_capacity);
	if (resized == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: out of memory");
		return false;
	}

	scratch->decoded_values = resized;
	scratch->decoded_capacity = required_capacity;
	scratch->scratch_allocations += 1;
	tq_scan_stats_record_scratch_allocations(1);
	return true;
}

static float
tq_dot_product_scalar(const float *left, const float *right, size_t len)
{
	float		sum = 0.0f;
	size_t		i = 0;

	for (i = 0; i < len; i++)
		sum += left[i] * right[i];

	return sum;
}

static float
tq_norm_squared_scalar(const float *values, size_t len)
{
	float		sum = 0.0f;
	size_t		i = 0;

	for (i = 0; i < len; i++)
		sum += values[i] * values[i];

	return sum;
}

static bool
tq_metric_distance_from_decoded_vector(TqDistanceKind distance,
									   const float *query_values,
									   size_t query_len,
									   const float *decoded_values,
									   size_t decoded_len,
									   float query_norm_squared,
									   float *distance_value,
									   char *errmsg,
									   size_t errmsg_len)
{
	float		dot_product = 0.0f;

	if (query_values == NULL || decoded_values == NULL || distance_value == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: decoded metric scorer requires query, decoded vector, and output");
		return false;
	}

	if (query_len != decoded_len)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: decoded metric scorer requires matching vector dimensions");
		return false;
	}

	dot_product = tq_dot_product_scalar(query_values, decoded_values, query_len);

	switch (distance)
	{
		case TQ_DISTANCE_COSINE:
			{
				float	decoded_norm_squared = tq_norm_squared_scalar(decoded_values, decoded_len);
				float	denominator = sqrtf(query_norm_squared * decoded_norm_squared);
				float	cosine_similarity = 0.0f;

				if (query_norm_squared <= 0.0f || decoded_norm_squared <= 0.0f)
				{
					*distance_value = 1.0f;
					return true;
				}

				cosine_similarity = dot_product / denominator;
				if (cosine_similarity > 1.0f)
					cosine_similarity = 1.0f;
				else if (cosine_similarity < -1.0f)
					cosine_similarity = -1.0f;

				*distance_value = 1.0f - cosine_similarity;
				return true;
			}
		case TQ_DISTANCE_IP:
			*distance_value = -dot_product;
			return true;
		case TQ_DISTANCE_L2:
			{
				float	decoded_norm_squared = tq_norm_squared_scalar(decoded_values, decoded_len);

				*distance_value = query_norm_squared + decoded_norm_squared - (2.0f * dot_product);
				if (*distance_value < 0.0f && *distance_value > -1e-6f)
					*distance_value = 0.0f;
				return true;
			}
		default:
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant scan: unsupported distance kind");
			return false;
	}
}

static void
tq_candidate_sift_up(TqCandidateHeap *heap, size_t index)
{
	while (index > 0)
	{
		size_t		parent = (index - 1) / 2;

		if (tq_candidate_compare_worst(&heap->entries[index], &heap->entries[parent]) <= 0)
			break;

		tq_candidate_swap(&heap->entries[index], &heap->entries[parent]);
		index = parent;
	}
}

static void
tq_candidate_sift_down(TqCandidateHeap *heap, size_t index)
{
	for (;;)
	{
		size_t		left = (index * 2) + 1;
		size_t		right = left + 1;
		size_t		worst = index;

		if (left < heap->count
			&& tq_candidate_compare_worst(&heap->entries[left], &heap->entries[worst]) > 0)
			worst = left;

		if (right < heap->count
			&& tq_candidate_compare_worst(&heap->entries[right], &heap->entries[worst]) > 0)
			worst = right;

		if (worst == index)
			break;

		tq_candidate_swap(&heap->entries[index], &heap->entries[worst]);
		index = worst;
	}
}

static int
tq_candidate_compare_best_qsort(const void *left, const void *right)
{
	const TqCandidateEntry *lhs = (const TqCandidateEntry *) left;
	const TqCandidateEntry *rhs = (const TqCandidateEntry *) right;

	if (lhs->score < rhs->score)
		return -1;
	if (lhs->score > rhs->score)
		return 1;
	if (lhs->tid.block_number < rhs->tid.block_number)
		return -1;
	if (lhs->tid.block_number > rhs->tid.block_number)
		return 1;
	if (lhs->tid.offset_number < rhs->tid.offset_number)
		return -1;
	if (lhs->tid.offset_number > rhs->tid.offset_number)
		return 1;
	return 0;
}

static int
tq_candidate_tid_compare_qsort(const void *left, const void *right)
{
	const TqTid *lhs = (const TqTid *) left;
	const TqTid *rhs = (const TqTid *) right;

	return tq_tid_compare(lhs, rhs);
}

static int
tq_tid_compare(const TqTid *left, const TqTid *right)
{
	if (left->block_number < right->block_number)
		return -1;
	if (left->block_number > right->block_number)
		return 1;
	if (left->offset_number < right->offset_number)
		return -1;
	if (left->offset_number > right->offset_number)
		return 1;
	return 0;
}

static bool
tq_candidate_tid_is_selected(const TqTid *candidate_tids,
							 size_t candidate_tid_count,
							 const TqTid *tid)
{
	size_t left = 0;
	size_t right = candidate_tid_count;

	if (candidate_tids == NULL || tid == NULL || candidate_tid_count == 0)
		return false;

	while (left < right)
	{
		size_t mid = left + ((right - left) / 2);
		int cmp = tq_tid_compare(&candidate_tids[mid], tid);

		if (cmp == 0)
			return true;
		if (cmp < 0)
			left = mid + 1;
		else
			right = mid;
	}

	return false;
}

void
tq_scan_stats_reset_last(void)
{
	memset(&tq_last_scan_stats, 0, sizeof(tq_last_scan_stats));
	tq_last_scan_stats.mode = TQ_SCAN_MODE_NONE;
	tq_last_scan_stats.score_mode = TQ_SCAN_SCORE_MODE_NONE;
	tq_last_scan_stats.score_kernel = TQ_PROD_SCORE_SCALAR;
	tq_last_scan_stats.page_bound_mode = TQ_PAGE_BOUND_MODE_DISABLED;
	tq_last_scan_stats.scan_orchestration = TQ_SCAN_ORCHESTRATION_NONE;
	tq_last_scan_stats.faithful_fast_path = false;
	tq_last_scan_stats.compatibility_fallback = false;
	tq_last_scan_stats.safe_pruning_enabled = false;
	tq_last_scan_stats.near_exhaustive_crossover = false;
	tq_last_shadow_decode_tid_count = 0;
}

void
tq_scan_stats_begin(TqScanMode mode, size_t configured_probe_count)
{
	tq_scan_stats_reset_last();
	tq_last_scan_stats.mode = mode;
	tq_last_scan_stats.configured_probe_count = configured_probe_count;
}

void
tq_scan_stats_set_score_mode(TqScanScoreMode score_mode)
{
	tq_last_scan_stats.score_mode = score_mode;
}

void
tq_scan_stats_set_score_kernel(TqProdScoreKernel score_kernel)
{
	tq_last_scan_stats.score_kernel = score_kernel;
}

void
tq_scan_stats_set_path_flags(bool faithful_fast_path,
							 bool compatibility_fallback)
{
	tq_last_scan_stats.faithful_fast_path = faithful_fast_path;
	tq_last_scan_stats.compatibility_fallback = compatibility_fallback;
}

void
tq_scan_stats_set_scan_orchestration(TqScanOrchestration scan_orchestration,
									 bool near_exhaustive_crossover)
{
	tq_last_scan_stats.scan_orchestration = scan_orchestration;
	tq_last_scan_stats.near_exhaustive_crossover = near_exhaustive_crossover;
}

void
tq_scan_stats_record_page_bound_mode(TqPageBoundMode page_bound_mode,
									 bool safe_pruning_enabled)
{
	tq_last_scan_stats.page_bound_mode =
		tq_page_bound_mode_merge(tq_last_scan_stats.page_bound_mode,
								 page_bound_mode);
	tq_last_scan_stats.safe_pruning_enabled =
		tq_last_scan_stats.safe_pruning_enabled || safe_pruning_enabled;
}

void
tq_scan_stats_set_router_selection_method(TqRouterSelectionMethod method)
{
	tq_last_scan_stats.router_selection_method = method;
}

void
tq_scan_stats_reset_candidate_heap_metrics(void)
{
	tq_last_scan_stats.retained_candidate_count = 0;
	tq_last_scan_stats.candidate_heap_capacity = 0;
	tq_last_scan_stats.candidate_heap_count = 0;
	tq_last_scan_stats.candidate_heap_insert_count = 0;
	tq_last_scan_stats.candidate_heap_replace_count = 0;
	tq_last_scan_stats.candidate_heap_reject_count = 0;
	tq_last_scan_stats.local_candidate_heap_insert_count = 0;
	tq_last_scan_stats.local_candidate_heap_replace_count = 0;
	tq_last_scan_stats.local_candidate_heap_reject_count = 0;
	tq_last_scan_stats.local_candidate_merge_count = 0;
}

void
tq_scan_stats_set_probe_budget(size_t nominal_probe_count,
							   size_t effective_probe_count,
							   size_t max_visited_codes,
							   size_t max_visited_pages)
{
	tq_last_scan_stats.nominal_probe_count = nominal_probe_count;
	tq_last_scan_stats.effective_probe_count = effective_probe_count;
	tq_last_scan_stats.max_visited_codes = max_visited_codes;
	tq_last_scan_stats.max_visited_pages = max_visited_pages;
}

void
tq_scan_stats_record_selected_list(size_t live_count, size_t page_count)
{
	tq_last_scan_stats.selected_list_count += 1;
	tq_last_scan_stats.selected_live_count += live_count;
	tq_last_scan_stats.selected_page_count += page_count;
}

void
tq_scan_stats_add_selected_live(size_t live_count)
{
	tq_last_scan_stats.selected_live_count += live_count;
}

void
tq_scan_stats_record_page_visit(void)
{
	tq_last_scan_stats.visited_page_count += 1;
}

void
tq_scan_stats_record_code_visit(bool decoded_vector)
{
	tq_last_scan_stats.visited_code_count += 1;
	if (decoded_vector)
		tq_last_scan_stats.decoded_vector_count += 1;
}

void
tq_scan_stats_record_bound_data_page_read(void)
{
	tq_last_scan_stats.bound_data_page_reads += 1;
}

void
tq_scan_stats_add_page_prunes(size_t count)
{
	tq_last_scan_stats.page_prune_count += count;
}

void
tq_scan_stats_add_early_stops(size_t count)
{
	tq_last_scan_stats.early_stop_count += count;
}

void
tq_scan_stats_record_candidate_heap_insert(void)
{
	tq_last_scan_stats.candidate_heap_insert_count += 1;
}

void
tq_scan_stats_record_candidate_heap_replace(void)
{
	tq_last_scan_stats.candidate_heap_replace_count += 1;
}

void
tq_scan_stats_record_candidate_heap_reject(void)
{
	tq_last_scan_stats.candidate_heap_reject_count += 1;
}

void
tq_scan_stats_record_local_candidate_heap_insert(void)
{
	tq_last_scan_stats.local_candidate_heap_insert_count += 1;
}

void
tq_scan_stats_record_local_candidate_heap_replace(void)
{
	tq_last_scan_stats.local_candidate_heap_replace_count += 1;
}

void
tq_scan_stats_record_local_candidate_heap_reject(void)
{
	tq_last_scan_stats.local_candidate_heap_reject_count += 1;
}

void
tq_scan_stats_record_local_candidate_merge(void)
{
	tq_last_scan_stats.local_candidate_merge_count += 1;
}

static void
tq_scan_stats_record_heap_insert(TqCandidateHeapMetricsMode metrics_mode)
{
	switch (metrics_mode)
	{
		case TQ_CANDIDATE_HEAP_METRICS_GLOBAL:
			tq_scan_stats_record_candidate_heap_insert();
			break;
		case TQ_CANDIDATE_HEAP_METRICS_LOCAL:
			tq_scan_stats_record_local_candidate_heap_insert();
			break;
		case TQ_CANDIDATE_HEAP_METRICS_NONE:
		default:
			break;
	}
}

static void
tq_scan_stats_record_heap_replace(TqCandidateHeapMetricsMode metrics_mode)
{
	switch (metrics_mode)
	{
		case TQ_CANDIDATE_HEAP_METRICS_GLOBAL:
			tq_scan_stats_record_candidate_heap_replace();
			break;
		case TQ_CANDIDATE_HEAP_METRICS_LOCAL:
			tq_scan_stats_record_local_candidate_heap_replace();
			break;
		case TQ_CANDIDATE_HEAP_METRICS_NONE:
		default:
			break;
	}
}

static void
tq_scan_stats_record_heap_reject(TqCandidateHeapMetricsMode metrics_mode)
{
	switch (metrics_mode)
	{
		case TQ_CANDIDATE_HEAP_METRICS_GLOBAL:
			tq_scan_stats_record_candidate_heap_reject();
			break;
		case TQ_CANDIDATE_HEAP_METRICS_LOCAL:
			tq_scan_stats_record_local_candidate_heap_reject();
			break;
		case TQ_CANDIDATE_HEAP_METRICS_NONE:
		default:
			break;
	}
}

void
tq_scan_stats_record_shadow_decoded_vector(void)
{
	tq_last_scan_stats.shadow_decoded_vector_count += 1;
}

void
tq_scan_stats_record_scratch_allocations(size_t count)
{
	tq_last_scan_stats.scratch_allocations += count;
}

void
tq_scan_stats_record_decoded_buffer_reuses(size_t count)
{
	tq_last_scan_stats.decoded_buffer_reuses += count;
}

void
tq_scan_stats_record_code_view_uses(size_t count)
{
	tq_last_scan_stats.code_view_uses += count;
}

void
tq_scan_stats_record_code_copy_uses(size_t count)
{
	tq_last_scan_stats.code_copy_uses += count;
}

void
tq_scan_stats_record_block_local_selection(size_t scored, size_t survivors)
{
	if (survivors > scored)
		survivors = scored;

	tq_last_scan_stats.block_local_scored_count += scored;
	tq_last_scan_stats.block_local_survivor_count += survivors;
	tq_last_scan_stats.block_local_rejected_count += (scored - survivors);
}

uint32_t
tq_block16_select_top_m(const float *scores,
						uint32_t candidate_count,
						uint32_t top_m,
						uint32_t *selected_indices)
{
	/*
	 * Select the top_m candidates with the highest scores (lowest distance)
	 * from a block of up to 16 candidates. Since candidate_count <= 16,
	 * a simple insertion into a sorted buffer is efficient.
	 *
	 * Distance is 1-score for cosine, so higher score = better = lower distance.
	 * We select the top_m highest scores.
	 */
	uint32_t	selected = 0;
	float		threshold = -1e30f;
	uint32_t	c = 0;

	if (scores == NULL || selected_indices == NULL
		|| candidate_count == 0 || top_m == 0)
		return 0;

	if (top_m >= candidate_count)
	{
		for (c = 0; c < candidate_count; c++)
			selected_indices[c] = c;
		return candidate_count;
	}

	/* Simple selection: maintain sorted buffer of top_m best scores */
	for (c = 0; c < candidate_count; c++)
	{
		if (selected < top_m)
		{
			/* Insert into sorted buffer */
			uint32_t pos = selected;

			while (pos > 0 && scores[c] > scores[selected_indices[pos - 1]])
			{
				selected_indices[pos] = selected_indices[pos - 1];
				pos--;
			}
			selected_indices[pos] = c;
			selected++;
			if (selected == top_m)
				threshold = scores[selected_indices[selected - 1]];
		}
		else if (scores[c] > threshold)
		{
			/* Replace worst in buffer */
			uint32_t pos = selected - 1;

			while (pos > 0 && scores[c] > scores[selected_indices[pos - 1]])
			{
				selected_indices[pos] = selected_indices[pos - 1];
				pos--;
			}
			selected_indices[pos] = c;
			threshold = scores[selected_indices[selected - 1]];
		}
	}

	return selected;
}

void
tq_scan_stats_record_decoded_vector_only(void)
{
	tq_last_scan_stats.decoded_vector_count += 1;
}

void
tq_scan_stats_set_candidate_heap_metrics(size_t capacity, size_t count)
{
	tq_last_scan_stats.candidate_heap_capacity = capacity;
	tq_last_scan_stats.candidate_heap_count = count;
	tq_last_scan_stats.retained_candidate_count = count;
}

void
tq_scan_stats_set_shadow_decode_metrics(const TqCandidateHeap *primary,
										const TqCandidateHeap *shadow)
{
	size_t overlap_count = 0;
	size_t primary_index = 0;

	if (shadow == NULL || shadow->entries == NULL || shadow->capacity == 0)
	{
		tq_last_scan_stats.shadow_decode_candidate_count = 0;
		tq_last_scan_stats.shadow_decode_overlap_count = 0;
		tq_last_scan_stats.shadow_decode_primary_only_count = 0;
		tq_last_scan_stats.shadow_decode_only_count = 0;
		tq_last_shadow_decode_tid_count = 0;
		return;
	}

	tq_last_scan_stats.shadow_decode_candidate_count = shadow->count;

	if (primary != NULL && primary->entries != NULL && primary->capacity > 0)
	{
		for (primary_index = 0; primary_index < primary->count; primary_index++)
		{
			size_t shadow_index = 0;

			for (shadow_index = 0; shadow_index < shadow->count; shadow_index++)
			{
				if (tq_candidate_tid_equal(&primary->entries[primary_index], &shadow->entries[shadow_index]))
				{
					overlap_count += 1;
					break;
				}
			}
		}
	}

	tq_last_scan_stats.shadow_decode_overlap_count = overlap_count;
	tq_last_scan_stats.shadow_decode_primary_only_count =
		(primary == NULL || primary->count < overlap_count) ? 0 : (primary->count - overlap_count);
	tq_last_scan_stats.shadow_decode_only_count =
		(shadow->count < overlap_count) ? 0 : (shadow->count - overlap_count);
	tq_scan_stats_store_shadow_decode_tids(shadow);
}

bool
tq_scan_stats_copy_shadow_decode_tids(TqTid *dest, size_t capacity, size_t *count)
{
	if (count != NULL)
		*count = tq_last_shadow_decode_tid_count;

	if (dest == NULL)
		return true;

	if (tq_last_shadow_decode_tid_count == 0)
		return true;

	if (capacity < tq_last_shadow_decode_tid_count)
		return false;

	memcpy(dest,
		   tq_last_shadow_decode_tids,
		   sizeof(TqTid) * tq_last_shadow_decode_tid_count);
	return true;
}

bool
tq_candidate_heap_copy_sorted_tids(const TqCandidateHeap *heap,
								   TqTid *dest,
								   size_t capacity,
								   size_t *count)
{
	TqTid *ordered_tids = NULL;
	size_t index = 0;

	if (count != NULL)
		*count = (heap == NULL) ? 0 : heap->count;

	if (heap == NULL || heap->entries == NULL || heap->count == 0)
		return true;

	if (dest == NULL)
		return true;

	if (capacity < heap->count)
		return false;

	ordered_tids = (TqTid *) malloc(sizeof(TqTid) * heap->count);
	if (ordered_tids == NULL)
		return false;

	for (index = 0; index < heap->count; index++)
		ordered_tids[index] = heap->entries[index].tid;

	qsort(ordered_tids,
		  heap->count,
		  sizeof(TqTid),
		  tq_candidate_tid_compare_qsort);
	memcpy(dest, ordered_tids, sizeof(TqTid) * heap->count);
	free(ordered_tids);
	return true;
}

void
tq_scan_stats_snapshot(TqScanStats *stats)
{
	if (stats == NULL)
		return;

	*stats = tq_last_scan_stats;
}

void
tq_scan_scratch_reset(TqScanScratch *scratch)
{
	if (scratch == NULL)
		return;

	free(scratch->decoded_values);
	if (scratch->block16_set_initialized)
		tq_scratch_block16_set_reset(&scratch->block16_set);
	memset(scratch, 0, sizeof(*scratch));
}

bool
tq_scratch_block16_set_init(TqScratchBlock16Set *set,
							uint32_t dimension,
							uint32_t max_candidates,
							char *errmsg,
							size_t errmsg_len)
{
	uint32_t	max_blocks = 0;

	if (set == NULL || dimension == 0 || max_candidates == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid block16 set init: set, dimension, and max_candidates must be non-null/non-zero");
		return false;
	}

	memset(set, 0, sizeof(*set));
	max_blocks = (max_candidates + TQ_BLOCK16_MAX_CANDIDATES - 1u) / TQ_BLOCK16_MAX_CANDIDATES;

	set->blocks = (TqScratchBlock16 *) calloc(max_blocks, sizeof(TqScratchBlock16));
	set->nibble_storage = (uint8_t *) calloc((size_t) max_blocks * TQ_BLOCK16_MAX_CANDIDATES * (size_t) dimension, sizeof(uint8_t));
	set->gamma_storage = (float *) calloc((size_t) max_blocks * TQ_BLOCK16_MAX_CANDIDATES, sizeof(float));
	set->tid_storage = (TqTid *) calloc((size_t) max_blocks * TQ_BLOCK16_MAX_CANDIDATES, sizeof(TqTid));

	if (set->blocks == NULL || set->nibble_storage == NULL
		|| set->gamma_storage == NULL || set->tid_storage == NULL)
	{
		tq_scratch_block16_set_reset(set);
		tq_set_error(errmsg, errmsg_len,
					 "invalid block16 set init: out of memory");
		return false;
	}

	set->block_capacity = max_blocks;
	set->dimension = dimension;
	return true;
}

void
tq_scratch_block16_set_reset(TqScratchBlock16Set *set)
{
	if (set == NULL)
		return;

	free(set->blocks);
	free(set->nibble_storage);
	free(set->gamma_storage);
	free(set->tid_storage);
	memset(set, 0, sizeof(*set));
}

bool
tq_batch_page_transpose_block16(const void *page,
								size_t page_size,
								const TqProdCodecConfig *config,
								TqScratchBlock16Set *set,
								char *errmsg,
								size_t errmsg_len)
{
	TqBatchPageHeaderView header;
	uint16_t	lane = 0;
	uint32_t	candidate_index = 0;
	uint32_t	block_index = 0;

	if (page == NULL || config == NULL || set == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid transpose: page, config, and set must be non-null");
		return false;
	}

	if (!tq_prod_lut16_is_supported(config, errmsg, errmsg_len))
		return false;

	memset(&header, 0, sizeof(header));
	if (!tq_batch_page_read_header(page, page_size, &header, errmsg, errmsg_len))
		return false;

	set->block_count = 0;
	set->total_candidates = 0;
	candidate_index = 0;

	if (!tq_batch_page_next_live_lane(page, page_size, -1, &lane, errmsg, errmsg_len))
		return true;

	/*
	 * SoA fast path: when the page stores nibbles in dimension-major
	 * layout and all lanes are live with lane_count == 16, we can copy
	 * directly from the page nibble block without per-lane extraction.
	 */
	if (tq_batch_page_is_soa(page, page_size)
		&& header.live_count == header.occupied_count
		&& header.occupied_count == TQ_BLOCK16_MAX_CANDIDATES)
	{
		const uint8_t *page_nibbles = NULL;
		const float *page_gammas = NULL;
		uint32_t	soa_dim = 0;
		uint16_t	soa_lane_count = 0;

		if (!tq_batch_page_get_nibble_ptr(page, page_size, &page_nibbles,
										  &soa_dim, &soa_lane_count,
										  errmsg, errmsg_len))
			return false;

		if (!tq_batch_page_get_gamma_ptr(page, page_size, &page_gammas,
										 errmsg, errmsg_len))
			return false;

		if (soa_lane_count == TQ_BLOCK16_MAX_CANDIDATES
			&& soa_dim == config->dimension)
		{
			size_t		pair_cols = ((size_t) soa_lane_count + 1u) / 2u;
			uint32_t	d;

			if (set->block_capacity < 1)
			{
				tq_set_error(errmsg, errmsg_len,
							 "invalid transpose: block set capacity too small for SoA direct copy");
				return false;
			}

			/*
			 * Unpack 4-bit packed nibbles from the page into full-byte
			 * dimension-major layout for the SIMD kernel.  Each page byte
			 * holds two candidates: low nibble = even, high = odd.
			 * For 16 lanes (pair_cols=8), load 8 packed bytes and split
			 * into 16 full bytes using mask+shift.
			 */
			if (pair_cols == 8u)
			{
				/*
				 * Fast path for 16 lanes: load 8 bytes, produce 16
				 * nibble bytes via interleaved low/high extraction.
				 */
				for (d = 0; d < soa_dim; d++)
				{
					const uint8_t *src = page_nibbles + (size_t) d * 8u;
					uint8_t		  *dst = set->nibble_storage + (size_t) d * 16u;
					size_t		   p;

					for (p = 0; p < 8u; p++)
					{
						dst[p * 2]     = src[p] & 0x0Fu;
						dst[p * 2 + 1] = (src[p] >> 4u) & 0x0Fu;
					}
				}
			}
			else
			{
				for (d = 0; d < soa_dim; d++)
				{
					const uint8_t *src = page_nibbles + (size_t) d * pair_cols;
					uint8_t		  *dst = set->nibble_storage + (size_t) d * TQ_BLOCK16_MAX_CANDIDATES;
					size_t		   p;

					for (p = 0; p < pair_cols; p++)
					{
						dst[p * 2]     = src[p] & 0x0Fu;
						dst[p * 2 + 1] = (src[p] >> 4u) & 0x0Fu;
					}
				}
			}
			memcpy(set->gamma_storage, page_gammas,
				   sizeof(float) * TQ_BLOCK16_MAX_CANDIDATES);

			/* Collect TIDs */
			for (candidate_index = 0; candidate_index < TQ_BLOCK16_MAX_CANDIDATES; candidate_index++)
			{
				TqTid	tid;

				memset(&tid, 0, sizeof(tid));
				if (!tq_batch_page_get_tid(page, page_size, (uint16_t) candidate_index,
										   &tid, errmsg, errmsg_len))
					return false;
				set->tid_storage[candidate_index] = tid;
			}

			set->total_candidates = TQ_BLOCK16_MAX_CANDIDATES;
			set->block_count = 1;
			set->blocks[0].nibbles = set->nibble_storage;
			set->blocks[0].gammas = set->gamma_storage;
			set->blocks[0].tids = set->tid_storage;
			set->blocks[0].count = TQ_BLOCK16_MAX_CANDIDATES;
			set->blocks[0].dimension = config->dimension;
			return true;
		}
	}

	do
	{
		const uint8_t *code = NULL;
		size_t		code_len = 0;
		TqTid		tid;
		float		gamma = 0.0f;

		memset(&tid, 0, sizeof(tid));

		block_index = candidate_index / TQ_BLOCK16_MAX_CANDIDATES;

		if (block_index >= set->block_capacity)
		{
			tq_set_error(errmsg, errmsg_len,
						 "invalid transpose: too many candidates for block set capacity");
			return false;
		}

		if (!tq_batch_page_get_tid(page, page_size, lane, &tid, errmsg, errmsg_len))
			return false;

		if (tq_batch_page_is_soa(page, page_size))
		{
			/*
			 * SoA page with partial occupancy or dead lanes: extract
			 * this lane's nibbles from the dimension-major block and
			 * scatter into the scratch block's candidate-major layout.
			 */
			const uint8_t *page_nibbles = NULL;
			const float *page_gammas = NULL;
			uint32_t	soa_dim = 0;
			uint16_t	soa_lane_count = 0;
			uint32_t	slot = candidate_index % TQ_BLOCK16_MAX_CANDIDATES;
			size_t		block_base = (size_t) block_index * (size_t) TQ_BLOCK16_MAX_CANDIDATES
										* (size_t) config->dimension;
			uint32_t	d;

			if (!tq_batch_page_get_nibble_ptr(page, page_size, &page_nibbles,
											  &soa_dim, &soa_lane_count,
											  errmsg, errmsg_len))
				return false;
			if (!tq_batch_page_get_gamma_ptr(page, page_size, &page_gammas,
											 errmsg, errmsg_len))
				return false;

			for (d = 0; d < config->dimension; d++)
			{
				set->nibble_storage[block_base + (size_t) d * TQ_BLOCK16_MAX_CANDIDATES + slot]
					= page_nibbles[(size_t) d * soa_lane_count + lane];
			}
			gamma = page_gammas[lane];
		}
		else
		{
			if (!tq_batch_page_code_view(page, page_size, lane, &code, &code_len, errmsg, errmsg_len))
				return false;

			if (!tq_prod_read_gamma(config, code, code_len, &gamma, errmsg, errmsg_len))
				return false;

			/*
			 * Fast extract + scatter: extract 8 nibbles at a time from the
			 * packed code and write directly to dimension-major positions.
			 * Avoids the per-bit tq_unpack_bits and the intermediate buffer.
			 *
			 * For bits=4: each group of 8 dimensions uses 3 bytes of idx
			 * (24 bits = 8 x 3-bit codes) and 1 byte of signs (8 x 1-bit).
			 * Since LUT16 requires dim % 8 == 0, groups are byte-aligned.
			 */
			{
				uint32_t	slot = candidate_index % TQ_BLOCK16_MAX_CANDIDATES;
				size_t		block_base = (size_t) block_index * (size_t) TQ_BLOCK16_MAX_CANDIDATES
											* (size_t) config->dimension;
				const uint8_t *idx_base = code;
				const uint8_t *sign_base = code + (size_t) config->dimension * 3u / 8u;
				uint32_t	d;

				for (d = 0; d < config->dimension; d += 8u)
				{
					uint32_t	chunk = (uint32_t) idx_base[0]
									| ((uint32_t) idx_base[1] << 8u)
									| ((uint32_t) idx_base[2] << 16u);
					uint8_t		signs = sign_base[0];
					uint8_t    *dst = set->nibble_storage + block_base
									+ (size_t) d * TQ_BLOCK16_MAX_CANDIDATES + slot;

					dst[0 * TQ_BLOCK16_MAX_CANDIDATES] = (uint8_t) (((signs & 0x01u) << 3u) | ((chunk >> 0u) & 7u));
					dst[1 * TQ_BLOCK16_MAX_CANDIDATES] = (uint8_t) (((signs & 0x02u) << 2u) | ((chunk >> 3u) & 7u));
					dst[2 * TQ_BLOCK16_MAX_CANDIDATES] = (uint8_t) (((signs & 0x04u) << 1u) | ((chunk >> 6u) & 7u));
					dst[3 * TQ_BLOCK16_MAX_CANDIDATES] = (uint8_t) (((signs & 0x08u)      ) | ((chunk >> 9u) & 7u));
					dst[4 * TQ_BLOCK16_MAX_CANDIDATES] = (uint8_t) (((signs & 0x10u) >> 1u) | ((chunk >> 12u) & 7u));
					dst[5 * TQ_BLOCK16_MAX_CANDIDATES] = (uint8_t) (((signs & 0x20u) >> 2u) | ((chunk >> 15u) & 7u));
					dst[6 * TQ_BLOCK16_MAX_CANDIDATES] = (uint8_t) (((signs & 0x40u) >> 3u) | ((chunk >> 18u) & 7u));
					dst[7 * TQ_BLOCK16_MAX_CANDIDATES] = (uint8_t) (((signs & 0x80u) >> 4u) | ((chunk >> 21u) & 7u));

					idx_base += 3;
					sign_base += 1;
				}
			}
		}

		set->gamma_storage[candidate_index] = gamma;
		set->tid_storage[candidate_index] = tid;
		candidate_index++;
	} while (tq_batch_page_next_live_lane(page, page_size, (int) lane, &lane, errmsg, errmsg_len));

	/* Wire up blocks */
	set->total_candidates = candidate_index;
	set->block_count = (candidate_index + TQ_BLOCK16_MAX_CANDIDATES - 1u) / TQ_BLOCK16_MAX_CANDIDATES;

	for (block_index = 0; block_index < set->block_count; block_index++)
	{
		uint32_t	start = block_index * TQ_BLOCK16_MAX_CANDIDATES;
		uint32_t	count = candidate_index - start;
		size_t		block_base = (size_t) block_index * (size_t) TQ_BLOCK16_MAX_CANDIDATES
									* (size_t) config->dimension;

		if (count > TQ_BLOCK16_MAX_CANDIDATES)
			count = TQ_BLOCK16_MAX_CANDIDATES;

		/* Zero-pad unused candidate slots in partial blocks */
		if (count < TQ_BLOCK16_MAX_CANDIDATES)
		{
			uint32_t	d;
			for (d = 0; d < config->dimension; d++)
			{
				uint32_t	c;
				for (c = count; c < TQ_BLOCK16_MAX_CANDIDATES; c++)
					set->nibble_storage[block_base + (size_t) d * TQ_BLOCK16_MAX_CANDIDATES + c] = 0;
			}
		}

		set->blocks[block_index].nibbles = set->nibble_storage + block_base;
		set->blocks[block_index].gammas = set->gamma_storage + start;
		set->blocks[block_index].tids = set->tid_storage + start;
		set->blocks[block_index].count = count;
		set->blocks[block_index].dimension = config->dimension;
	}

	return true;
}

bool
tq_scan_stats_serialize_json(const TqScanStats *stats, char *buffer, size_t buffer_len)
{
	int written = 0;

	if (stats == NULL || buffer == NULL || buffer_len == 0)
		return false;

	written = snprintf(
		buffer,
		buffer_len,
		"{\"mode\":\"%s\",\"score_mode\":\"%s\",\"score_kernel\":\"%s\",\"page_bound_mode\":\"%s\","
		"\"scan_orchestration\":\"%s\",\"router_selection_method\":\"%s\","
		"\"faithful_fast_path\":%s,\"compatibility_fallback\":%s,"
		"\"safe_pruning_enabled\":%s,\"near_exhaustive_crossover\":%s,"
		"\"configured_probe_count\":%zu,"
		"\"nominal_probe_count\":%zu,\"effective_probe_count\":%zu,"
		"\"max_visited_codes\":%zu,\"max_visited_pages\":%zu,"
		"\"selected_list_count\":%zu,\"selected_live_count\":%zu,\"selected_page_count\":%zu,"
		"\"visited_page_count\":%zu,\"visited_code_count\":%zu,"
		"\"retained_candidate_count\":%zu,\"candidate_heap_capacity\":%zu,"
		"\"candidate_heap_count\":%zu,\"candidate_heap_insert_count\":%zu,"
		"\"candidate_heap_replace_count\":%zu,\"candidate_heap_reject_count\":%zu,"
		"\"local_candidate_heap_insert_count\":%zu,"
		"\"local_candidate_heap_replace_count\":%zu,"
		"\"local_candidate_heap_reject_count\":%zu,"
		"\"local_candidate_merge_count\":%zu,"
		"\"shadow_decoded_vector_count\":%zu,\"shadow_decode_candidate_count\":%zu,"
		"\"shadow_decode_overlap_count\":%zu,\"shadow_decode_primary_only_count\":%zu,"
		"\"shadow_decode_only_count\":%zu,"
		"\"decoded_vector_count\":%zu,"
		"\"bound_data_page_reads\":%zu,"
		"\"page_prune_count\":%zu,\"early_stop_count\":%zu,"
		"\"scratch_allocations\":%zu,\"decoded_buffer_reuses\":%zu,"
		"\"code_view_uses\":%zu,\"code_copy_uses\":%zu,"
		"\"block_local_scored_count\":%zu,\"block_local_survivor_count\":%zu,"
		"\"block_local_rejected_count\":%zu}",
		tq_scan_mode_name(stats->mode),
		tq_scan_score_mode_name(stats->score_mode),
		tq_scan_score_kernel_name(stats),
		tq_page_bound_mode_name(stats->page_bound_mode),
		tq_scan_orchestration_name(stats->scan_orchestration),
		tq_router_selection_method_name(stats->router_selection_method),
		stats->faithful_fast_path ? "true" : "false",
		stats->compatibility_fallback ? "true" : "false",
		stats->safe_pruning_enabled ? "true" : "false",
		stats->near_exhaustive_crossover ? "true" : "false",
		stats->configured_probe_count,
		stats->nominal_probe_count,
		stats->effective_probe_count,
		stats->max_visited_codes,
		stats->max_visited_pages,
		stats->selected_list_count,
		stats->selected_live_count,
		stats->selected_page_count,
		stats->visited_page_count,
		stats->visited_code_count,
		stats->retained_candidate_count,
		stats->candidate_heap_capacity,
		stats->candidate_heap_count,
		stats->candidate_heap_insert_count,
		stats->candidate_heap_replace_count,
		stats->candidate_heap_reject_count,
		stats->local_candidate_heap_insert_count,
		stats->local_candidate_heap_replace_count,
		stats->local_candidate_heap_reject_count,
		stats->local_candidate_merge_count,
		stats->shadow_decoded_vector_count,
		stats->shadow_decode_candidate_count,
		stats->shadow_decode_overlap_count,
		stats->shadow_decode_primary_only_count,
		stats->shadow_decode_only_count,
		stats->decoded_vector_count,
		stats->bound_data_page_reads,
		stats->page_prune_count,
		stats->early_stop_count,
		stats->scratch_allocations,
		stats->decoded_buffer_reuses,
		stats->code_view_uses,
		stats->code_copy_uses,
		stats->block_local_scored_count,
		stats->block_local_survivor_count,
		stats->block_local_rejected_count
	);

	return written >= 0 && (size_t) written < buffer_len;
}

bool
tq_candidate_heap_init(TqCandidateHeap *heap, size_t capacity)
{
	if (heap == NULL || capacity == 0)
		return false;

	memset(heap, 0, sizeof(*heap));
	heap->entries = (TqCandidateEntry *) calloc(capacity, sizeof(TqCandidateEntry));
	if (heap->entries == NULL)
		return false;

	heap->capacity = capacity;
	return true;
}

void
tq_candidate_heap_reset(TqCandidateHeap *heap)
{
	if (heap == NULL)
		return;

	free(heap->entries);
	memset(heap, 0, sizeof(*heap));
}

bool
tq_candidate_heap_push(TqCandidateHeap *heap,
					   float score,
					   uint32_t block_number,
					   uint16_t offset_number)
{
	return tq_candidate_heap_push_internal(heap,
											 score,
											 block_number,
											 offset_number,
											 TQ_CANDIDATE_HEAP_METRICS_GLOBAL);
}

static bool
tq_candidate_heap_push_internal(TqCandidateHeap *heap,
								 float score,
								 uint32_t block_number,
								 uint16_t offset_number,
								 TqCandidateHeapMetricsMode metrics_mode)
{
	TqCandidateEntry entry;

	if (heap == NULL || heap->entries == NULL || heap->capacity == 0)
		return false;

	entry.score = score;
	entry.tid.block_number = block_number;
	entry.tid.offset_number = offset_number;

	if (heap->count < heap->capacity)
	{
		heap->entries[heap->count] = entry;
		tq_candidate_sift_up(heap, heap->count);
		heap->count++;
		heap->sorted = false;
		tq_scan_stats_record_heap_insert(metrics_mode);
		return true;
	}

	if (tq_candidate_compare_worst(&entry, &heap->entries[0]) >= 0)
	{
		tq_scan_stats_record_heap_reject(metrics_mode);
		return true;
	}

	heap->entries[0] = entry;
	tq_candidate_sift_down(heap, 0);
	heap->sorted = false;
	tq_scan_stats_record_heap_replace(metrics_mode);
	return true;
}

static bool
tq_candidate_heap_merge_into_global(TqCandidateHeap *global_heap,
									 const TqCandidateHeap *local_heap)
{
	size_t index = 0;

	if (global_heap == NULL || local_heap == NULL)
		return false;

	for (index = 0; index < local_heap->count; index++)
	{
		tq_scan_stats_record_local_candidate_merge();
		if (!tq_candidate_heap_push_internal(global_heap,
												 local_heap->entries[index].score,
												 local_heap->entries[index].tid.block_number,
												 local_heap->entries[index].tid.offset_number,
												 TQ_CANDIDATE_HEAP_METRICS_GLOBAL))
			return false;
	}

	return true;
}

bool
tq_candidate_heap_pop_best(TqCandidateHeap *heap, TqCandidateEntry *entry)
{
	if (heap == NULL || entry == NULL || heap->entries == NULL)
		return false;

	if (!heap->sorted)
	{
		qsort(heap->entries, heap->count, sizeof(TqCandidateEntry),
			  tq_candidate_compare_best_qsort);
		heap->sorted = true;
		heap->pop_index = 0;
	}

	if (heap->pop_index >= heap->count)
		return false;

	*entry = heap->entries[heap->pop_index++];
	return true;
}

bool
tq_metric_distance_from_ip_score(TqDistanceKind distance,
								 float ip_score,
								 float *distance_value,
								 char *errmsg,
								 size_t errmsg_len)
{
	if (distance_value == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: distance output must be non-null");
		return false;
	}

	switch (distance)
	{
		case TQ_DISTANCE_COSINE:
			*distance_value = 1.0f - ip_score;
			return true;
		case TQ_DISTANCE_IP:
			*distance_value = -ip_score;
			return true;
		case TQ_DISTANCE_L2:
			*distance_value = 2.0f - (2.0f * ip_score);
			if (*distance_value < 0.0f && *distance_value > -1e-6f)
				*distance_value = 0.0f;
			return true;
		default:
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant scan: unsupported distance kind");
			return false;
	}
}

bool
tq_scan_should_prune_page(const TqCandidateHeap *heap,
						  float optimistic_distance,
						  bool *should_prune,
						  char *errmsg,
						  size_t errmsg_len)
{
	if (heap == NULL || should_prune == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: heap and prune output must be non-null");
		return false;
	}

	if (heap->capacity == 0 || heap->count < heap->capacity || heap->entries == NULL)
	{
		*should_prune = false;
		return true;
	}

	*should_prune = optimistic_distance > (heap->entries[0].score + 1e-6f);
	return true;
}

bool
tq_scan_page_bounds_are_safe_for_pruning(bool normalized, TqDistanceKind distance)
{
	return tq_scan_can_use_prod_code_domain(normalized, distance);
}

bool
tq_scan_summary_optimistic_distance_bound(const TqProdCodecConfig *config,
										  const TqProdLut *lut,
										  const TqBatchPageSummary *summary,
										  const uint8_t *representative_code,
										  size_t representative_code_len,
										  bool normalized,
										  TqDistanceKind distance,
										  const float *query_values,
										  size_t query_len,
										  float *optimistic_distance,
										  char *errmsg,
										  size_t errmsg_len)
{
	float rep_ip = 0.0f;
	float query_weight_norm = 0.0f;
	float optimistic_ip = 0.0f;

	if (config == NULL || lut == NULL || summary == NULL
		|| representative_code == NULL || optimistic_distance == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: summary bound scorer requires codec, lut, summary, code, and output");
		return false;
	}

	if (!normalized || !tq_scan_can_use_prod_code_domain(normalized, distance))
	{
		*optimistic_distance = -INFINITY;
		return true;
	}

	if (query_values == NULL || query_len != config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: bound scorer requires query values matching codec dimension");
		return false;
	}

	if (summary->representative_lane == TQ_BATCH_PAGE_NO_REPRESENTATIVE)
	{
		*optimistic_distance = INFINITY;
		return true;
	}

	if (!tq_prod_score_code_from_lut(config, lut, representative_code,
									 representative_code_len, &rep_ip,
									 errmsg, errmsg_len))
		return false;

	if (!tq_prod_query_weight_l2_norm(config, lut, &query_weight_norm, errmsg, errmsg_len))
		return false;

	optimistic_ip = rep_ip + (query_weight_norm * summary->residual_radius);

	return tq_metric_distance_from_ip_score(distance, optimistic_ip,
											optimistic_distance, errmsg, errmsg_len);
}

bool
tq_scan_page_optimistic_distance_bound(const TqProdCodecConfig *config,
									   const TqProdLut *lut,
									   const void *page,
									   size_t page_size,
									   bool normalized,
									   TqDistanceKind distance,
									   const float *query_values,
									   size_t query_len,
									   float *optimistic_distance,
									   char *errmsg,
									   size_t errmsg_len)
{
	TqBatchPageHeaderView header;
	TqBatchPageSummary summary;
	const uint8_t *code = NULL;
	size_t code_len = 0;
	bool ok = false;

	if (config == NULL || lut == NULL || page == NULL || optimistic_distance == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: bound scorer requires codec, lut, page, and output");
		return false;
	}

	memset(&header, 0, sizeof(header));
	memset(&summary, 0, sizeof(summary));

	if (!tq_batch_page_read_header(page, page_size, &header, errmsg, errmsg_len)
		|| !tq_batch_page_get_summary(page, page_size, &summary, errmsg, errmsg_len))
		return false;

	if (!normalized || !tq_scan_can_use_prod_code_domain(normalized, distance))
	{
		*optimistic_distance = -INFINITY;
		return true;
	}

	if (query_values == NULL || query_len != config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: bound scorer requires query values matching codec dimension");
		return false;
	}

	if (header.live_count == 0 || summary.representative_lane == TQ_BATCH_PAGE_NO_REPRESENTATIVE)
	{
		*optimistic_distance = INFINITY;
		return true;
	}

	if (tq_batch_page_is_soa(page, page_size))
	{
		if (!tq_batch_page_get_representative_code_view(page, page_size,
														&code, &code_len,
														errmsg, errmsg_len))
			return false;
	}
	else
	{
		if (!tq_batch_page_code_view(page,
									 page_size,
									 summary.representative_lane,
									 &code,
									 &code_len,
									 errmsg,
									 errmsg_len))
			return false;
	}
	tq_scan_stats_record_code_view_uses(1);

	ok = tq_scan_summary_optimistic_distance_bound(config,
												   lut,
												   &summary,
												   code,
												   code_len,
												   normalized,
												   distance,
												   query_values,
												   query_len,
												   optimistic_distance,
												   errmsg,
												   errmsg_len);
	return ok;
}

bool
tq_batch_page_scan_prod(const void *page,
						size_t page_size,
						const TqProdCodecConfig *config,
						bool normalized,
						TqDistanceKind distance,
						const TqProdLut *lut,
						const float *query_values,
						size_t query_len,
						bool filter_enabled,
						int32_t filter_value,
						TqCandidateHeap *heap,
						TqCandidateHeap *shadow_decode_heap,
						char *errmsg,
						size_t errmsg_len)
{
	TqScanScratch scratch;
	bool ok = false;

	memset(&scratch, 0, sizeof(scratch));
	ok = tq_batch_page_scan_prod_with_scratch(page,
											  page_size,
											  config,
											  normalized,
											  distance,
											  lut,
											  query_values,
											  query_len,
											  filter_enabled,
											  filter_value,
											  heap,
											  shadow_decode_heap,
											  &scratch,
											  errmsg,
											  errmsg_len);
	tq_scan_scratch_reset(&scratch);
	return ok;
}

bool
tq_batch_page_scan_prod_with_scratch(const void *page,
									 size_t page_size,
									 const TqProdCodecConfig *config,
									 bool normalized,
									 TqDistanceKind distance,
									 const TqProdLut *lut,
									 const float *query_values,
									 size_t query_len,
									 bool filter_enabled,
									 int32_t filter_value,
									 TqCandidateHeap *heap,
									 TqCandidateHeap *shadow_decode_heap,
									 TqScanScratch *scratch,
									 char *errmsg,
									 size_t errmsg_len)
{
	TqBatchPageHeaderView header;
	TqCandidateHeap local_heap;
	const uint8_t *code = NULL;
	size_t code_len = 0;
	float *decoded = NULL;
	bool		use_code_domain = false;
	bool		use_shadow_decode = false;
	bool		use_block16 = false;
	float		query_norm_squared = 0.0f;
	uint16_t	lane = 0;
	bool		page_has_filter = false;
	bool		ok = false;

	if (page == NULL || config == NULL || lut == NULL || heap == NULL || scratch == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: page, codec, lut, heap, and scratch must be non-null");
		return false;
	}

	memset(&header, 0, sizeof(header));
	memset(&local_heap, 0, sizeof(local_heap));

	if (query_values == NULL || query_len != config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: query values must match codec dimension");
		return false;
	}

	if (heap->entries == NULL || heap->capacity == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: candidate heap must be initialized");
		return false;
	}

	if (!tq_batch_page_read_header(page, page_size, &header, errmsg, errmsg_len))
		return false;

	if (filter_enabled)
	{
		if (!tq_batch_page_has_filter_int4(page, page_size, &page_has_filter,
										   errmsg, errmsg_len))
			return false;
		if (!page_has_filter)
		{
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant scan: filtered ordered scans require int4 filter payloads on batch pages");
			return false;
		}
	}

	if (header.live_count > 0
		&& !tq_candidate_heap_init(&local_heap,
								   header.live_count < heap->capacity ? header.live_count : heap->capacity))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: failed to allocate page-local candidate heap");
		return false;
	}

	use_code_domain = tq_scan_can_use_prod_code_domain(normalized, distance)
		&& !tq_guc_force_decode_score_diagnostics;
	use_shadow_decode = use_code_domain
		&& shadow_decode_heap != NULL
		&& shadow_decode_heap->entries != NULL
		&& shadow_decode_heap->capacity > 0;
	use_block16 = use_code_domain
		&& !use_shadow_decode
		&& !filter_enabled
		&& scratch->lut16 != NULL
		&& scratch->lut16->quantized_ready
		&& tq_prod_lut16_is_supported(config, NULL, 0);
	tq_scan_stats_set_score_mode(use_code_domain ? TQ_SCAN_SCORE_MODE_CODE_DOMAIN
												 : TQ_SCAN_SCORE_MODE_DECODE);
	tq_scan_stats_set_score_kernel(use_code_domain
								   ? tq_prod_code_domain_preferred_kernel(config)
								   : TQ_PROD_SCORE_SCALAR);
	tq_scan_stats_set_path_flags(use_code_domain, !use_code_domain);
	tq_scan_stats_record_page_visit();

	if (!use_code_domain || use_shadow_decode)
	{
		if (!tq_scan_scratch_ensure_decoded_capacity(scratch,
													 query_len,
													 errmsg,
													 errmsg_len))
			return false;

		decoded = scratch->decoded_values;
		query_norm_squared = tq_norm_squared_scalar(query_values, query_len);
	}

	/*
	 * SoA zero-copy fast path: when the page stores 4-bit packed nibbles
	 * in dimension-major layout and all lanes are live with count == 16,
	 * score directly from the page buffer with no transpose or copy.
	 * The kernel unpacks 4-bit pairs inline via mask+shift.
	 */
	if (use_block16 && header.live_count > 0
		&& tq_batch_page_is_soa(page, page_size)
		&& header.live_count == header.occupied_count
		&& header.occupied_count == TQ_BLOCK16_MAX_CANDIDATES)
	{
		const uint8_t *page_nibbles = NULL;
		const float   *page_gammas = NULL;
		uint32_t	soa_dim = 0;
		uint16_t	soa_lc = 0;

		if (!tq_batch_page_get_nibble_ptr(page, page_size, &page_nibbles,
										  &soa_dim, &soa_lc, errmsg, errmsg_len)
			|| !tq_batch_page_get_gamma_ptr(page, page_size, &page_gammas,
											errmsg, errmsg_len))
			return false;

		if (soa_lc == TQ_BLOCK16_MAX_CANDIDATES && soa_dim == config->dimension)
		{
			const TqProdLut16 *lut16 = scratch->lut16;
			uint32_t	dim = soa_dim;
			size_t		pair_cols = (size_t) soa_lc / 2u;
			float		block_scores[TQ_BLOCK16_MAX_CANDIDATES];
			int32_t		base_sums[TQ_BLOCK16_MAX_CANDIDATES];
			int32_t		qjl_sums[TQ_BLOCK16_MAX_CANDIDATES];
			uint32_t	d;
			uint32_t	i;

			memset(base_sums, 0, sizeof(base_sums));
			memset(qjl_sums, 0, sizeof(qjl_sums));

			for (d = 0; d < dim; d++)
			{
				const uint8_t *packed_row = page_nibbles + (size_t) d * pair_cols;
				const int8_t  *base_row = lut16->base_quantized + (size_t) d * 16u;
				const int8_t  *qjl_row = lut16->qjl_quantized + (size_t) d * 16u;
				size_t		p;

				for (p = 0; p < pair_cols; p++)
				{
					uint8_t	lo_nib = packed_row[p] & 0x0Fu;
					uint8_t	hi_nib = (packed_row[p] >> 4u) & 0x0Fu;

					base_sums[p * 2]     += (int32_t) base_row[lo_nib];
					base_sums[p * 2 + 1] += (int32_t) base_row[hi_nib];
					qjl_sums[p * 2]      += (int32_t) qjl_row[lo_nib];
					qjl_sums[p * 2 + 1]  += (int32_t) qjl_row[hi_nib];
				}
			}

			for (i = 0; i < TQ_BLOCK16_MAX_CANDIDATES; i++)
				block_scores[i] = (float) base_sums[i] * lut16->base_global_scale
							   + page_gammas[i] * (float) qjl_sums[i] * lut16->qjl_global_scale;

			tq_scan_stats_set_score_kernel(TQ_PROD_SCORE_SCALAR);

			for (i = 0; i < TQ_BLOCK16_MAX_CANDIDATES; i++)
			{
				float	distance_value = 0.0f;
				TqTid	tid;

				tq_scan_stats_record_code_visit(false);

				if (!tq_metric_distance_from_ip_score(distance,
													  block_scores[i],
													  &distance_value,
													  errmsg, errmsg_len))
					return false;

				memset(&tid, 0, sizeof(tid));
				if (!tq_batch_page_get_tid(page, page_size, (uint16_t) i,
										   &tid, errmsg, errmsg_len))
					return false;

				if (!tq_candidate_heap_push_internal(&local_heap,
													 distance_value,
													 tid.block_number,
													 tid.offset_number,
													 TQ_CANDIDATE_HEAP_METRICS_LOCAL))
					return false;
			}

			tq_scan_stats_record_block_local_selection(TQ_BLOCK16_MAX_CANDIDATES,
													   TQ_BLOCK16_MAX_CANDIDATES);
			goto merge_and_return;
		}
	}

	if (use_block16 && header.live_count > 0)
	{
		/*
		 * Block-16 transpose path: for AoS pages or partial SoA pages,
		 * extract + transpose nibbles, then score via SIMD dispatch.
		 */
		float		block_scores[TQ_BLOCK16_MAX_CANDIDATES];
		uint32_t	b;

		if (!scratch->block16_set_initialized
			|| scratch->block16_set.dimension != config->dimension
			|| scratch->block16_set.block_capacity * TQ_BLOCK16_MAX_CANDIDATES < header.live_count)
		{
			if (scratch->block16_set_initialized)
				tq_scratch_block16_set_reset(&scratch->block16_set);

			if (!tq_scratch_block16_set_init(&scratch->block16_set,
											 config->dimension,
											 header.live_count,
											 errmsg, errmsg_len))
				return false;
			scratch->block16_set_initialized = true;
		}

		if (!tq_batch_page_transpose_block16(page, page_size, config,
											 &scratch->block16_set,
											 errmsg, errmsg_len))
			return false;

		for (b = 0; b < scratch->block16_set.block_count; b++)
		{
			TqScratchBlock16 *blk = &scratch->block16_set.blocks[b];
			TqProdScoreKernel used_kernel = TQ_PROD_SCORE_SCALAR;
			uint32_t	i;

			if (!tq_prod_score_block16_dispatch(scratch->lut16,
												blk->nibbles,
												blk->gammas,
												blk->count,
												TQ_PROD_SCORE_AUTO,
												block_scores,
												&used_kernel,
												errmsg, errmsg_len))
				return false;

			tq_scan_stats_set_score_kernel(used_kernel);

			for (i = 0; i < blk->count; i++)
			{
				float	distance_value = 0.0f;

				tq_scan_stats_record_code_visit(false);

				if (!tq_metric_distance_from_ip_score(distance,
													  block_scores[i],
													  &distance_value,
													  errmsg, errmsg_len))
					return false;

				if (!tq_candidate_heap_push_internal(&local_heap,
													 distance_value,
													 blk->tids[i].block_number,
													 blk->tids[i].offset_number,
													 TQ_CANDIDATE_HEAP_METRICS_LOCAL))
					return false;
			}

			tq_scan_stats_record_block_local_selection(blk->count, blk->count);
		}
	}
	else if (tq_batch_page_next_live_lane(page, page_size, -1, &lane, errmsg, errmsg_len))
	{
		/*
		 * For SoA pages in the per-lane fallback, we reconstruct packed
		 * codes from nibbles + gamma so existing scoring functions work.
		 */
		bool		page_is_soa = tq_batch_page_is_soa(page, page_size);
		const uint8_t *soa_nibbles_ptr = NULL;
		const float *soa_gammas_ptr = NULL;
		uint32_t	soa_dim = 0;
		uint16_t	soa_lane_count = 0;
		uint8_t	   *reconstructed_code = NULL;

		if (page_is_soa)
		{
			TqProdPackedLayout layout;

			memset(&layout, 0, sizeof(layout));
			if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len))
				return false;

			if (!tq_batch_page_get_nibble_ptr(page, page_size, &soa_nibbles_ptr,
											  &soa_dim, &soa_lane_count,
											  errmsg, errmsg_len))
				return false;
			if (!tq_batch_page_get_gamma_ptr(page, page_size, &soa_gammas_ptr,
											 errmsg, errmsg_len))
				return false;

			reconstructed_code = (uint8_t *) malloc(layout.total_bytes);
			if (reconstructed_code == NULL)
			{
				tq_set_error(errmsg, errmsg_len,
							 "invalid turboquant scan: out of memory for SoA code reconstruction");
				return false;
			}
		}

		do
		{
			TqTid		tid;
			float		distance_value = 0.0f;
			float		ip_score = 0.0f;
			int32_t		lane_filter_value = 0;

			memset(&tid, 0, sizeof(tid));
			if (!tq_batch_page_get_tid(page, page_size, lane, &tid, errmsg, errmsg_len))
			{
				free(reconstructed_code);
				return false;
			}

			if (filter_enabled)
			{
				if (!tq_batch_page_get_filter_int4(page, page_size, lane,
												   &lane_filter_value,
												   errmsg, errmsg_len))
				{
					free(reconstructed_code);
					return false;
				}
				if (lane_filter_value != filter_value)
					continue;
			}

			if (page_is_soa)
			{
				TqProdPackedLayout layout;
				uint8_t	   *lane_nibbles;
				uint32_t	d;

				memset(&layout, 0, sizeof(layout));
				if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len))
				{
					free(reconstructed_code);
					return false;
				}

				lane_nibbles = (uint8_t *) malloc(soa_dim);
				if (lane_nibbles == NULL)
				{
					free(reconstructed_code);
					tq_set_error(errmsg, errmsg_len,
								 "invalid turboquant scan: out of memory for SoA nibble extraction");
					return false;
				}
				for (d = 0; d < soa_dim; d++)
					lane_nibbles[d] = soa_nibbles_ptr[(size_t) d * soa_lane_count + lane];

				if (!tq_prod_nibbles_gamma_to_packed(config, lane_nibbles, soa_dim,
													 soa_gammas_ptr[lane],
													 reconstructed_code, layout.total_bytes,
													 errmsg, errmsg_len))
				{
					free(lane_nibbles);
					free(reconstructed_code);
					return false;
				}
				free(lane_nibbles);

				code = reconstructed_code;
				code_len = layout.total_bytes;
			}
			else
			{
				if (!tq_batch_page_code_view(page, page_size, lane, &code, &code_len, errmsg, errmsg_len))
					return false;
			}

			scratch->code_view_uses += 1;
			tq_scan_stats_record_code_view_uses(1);

			if (use_code_domain)
			{
				TqProdScoreKernel used_kernel = TQ_PROD_SCORE_SCALAR;
				float		shadow_distance_value = 0.0f;

				if (!tq_prod_score_code_from_lut_dispatch(config, lut, code, code_len,
														  tq_prod_code_domain_preferred_kernel(config),
														  &ip_score, &used_kernel, errmsg, errmsg_len))
				{
					free(reconstructed_code);
					return false;
				}

				tq_scan_stats_set_score_kernel(used_kernel);
				tq_scan_stats_record_code_visit(false);

				if (!tq_metric_distance_from_ip_score(distance,
													  ip_score,
													  &distance_value,
													  errmsg,
													  errmsg_len))
				{
					free(reconstructed_code);
					return false;
				}

				if (use_shadow_decode)
				{
					if (!tq_prod_decode(config, code, code_len, decoded, query_len,
										 errmsg, errmsg_len))
					{
						free(reconstructed_code);
						return false;
					}

					tq_scan_stats_record_shadow_decoded_vector();

					if (!tq_metric_distance_from_decoded_vector(distance,
																query_values,
																query_len,
																decoded,
																query_len,
																query_norm_squared,
																&shadow_distance_value,
																errmsg,
																errmsg_len)
						|| !tq_candidate_heap_push_internal(shadow_decode_heap,
															 shadow_distance_value,
															 tid.block_number,
															 tid.offset_number,
															 TQ_CANDIDATE_HEAP_METRICS_NONE))
					{
						free(reconstructed_code);
						return false;
					}
				}
			}
			else if (!tq_prod_decode(config, code, code_len, decoded, query_len,
									 errmsg, errmsg_len))
			{
				free(reconstructed_code);
				return false;
			}
			else
			{
				tq_scan_stats_record_code_visit(true);

				if (!tq_metric_distance_from_decoded_vector(distance,
															query_values,
															query_len,
															decoded,
															query_len,
															query_norm_squared,
															&distance_value,
															errmsg,
															errmsg_len))
				{
					free(reconstructed_code);
					return false;
				}
			}

			if (!tq_candidate_heap_push_internal(&local_heap,
												 distance_value,
												 tid.block_number,
												 tid.offset_number,
												 TQ_CANDIDATE_HEAP_METRICS_LOCAL))
			{
				free(reconstructed_code);
				return false;
			}
		} while (tq_batch_page_next_live_lane(page, page_size, (int) lane, &lane, errmsg, errmsg_len));

		free(reconstructed_code);
	}

merge_and_return:
	if (!tq_candidate_heap_merge_into_global(heap, &local_heap))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: failed to merge page-local candidates");
		goto cleanup;
	}

	tq_scan_stats_set_candidate_heap_metrics(heap->capacity, heap->count);
	ok = true;

cleanup:
	tq_candidate_heap_reset(&local_heap);
	return ok;
}

bool
tq_batch_page_rescore_prod_candidates(const void *page,
									  size_t page_size,
									  const TqProdCodecConfig *config,
									  bool normalized,
									  TqDistanceKind distance,
									  const float *query_values,
									  size_t query_len,
									  const TqTid *candidate_tids,
									  size_t candidate_tid_count,
									  TqCandidateHeap *heap,
									  char *errmsg,
									  size_t errmsg_len)
{
	TqScanScratch scratch;
	bool ok = false;

	memset(&scratch, 0, sizeof(scratch));
	ok = tq_batch_page_rescore_prod_candidates_with_scratch(page,
															page_size,
															config,
															normalized,
															distance,
															query_values,
															query_len,
															candidate_tids,
															candidate_tid_count,
															heap,
															&scratch,
															errmsg,
															errmsg_len);
	tq_scan_scratch_reset(&scratch);
	return ok;
}

bool
tq_batch_page_rescore_prod_candidates_with_scratch(const void *page,
												   size_t page_size,
												   const TqProdCodecConfig *config,
												   bool normalized,
												   TqDistanceKind distance,
												   const float *query_values,
												   size_t query_len,
												   const TqTid *candidate_tids,
												   size_t candidate_tid_count,
												   TqCandidateHeap *heap,
												   TqScanScratch *scratch,
												   char *errmsg,
												   size_t errmsg_len)
{
	TqBatchPageHeaderView header;
	const uint8_t *code = NULL;
	size_t code_len = 0;
	float *decoded = NULL;
	float query_norm_squared = 0.0f;
	uint16_t lane = 0;

	if (page == NULL || config == NULL || query_values == NULL || heap == NULL || scratch == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: decode rescoring requires page, codec, query, candidate tids, heap, and scratch");
		return false;
	}

	if (candidate_tid_count == 0)
		return true;

	if (candidate_tids == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: decode rescoring requires candidate tids when candidate count is non-zero");
		return false;
	}

	if (!normalized || !tq_scan_can_use_prod_code_domain(normalized, distance))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: decode rescoring requires normalized code-domain capable scans");
		return false;
	}

	if (query_len != config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: decode rescoring requires query values matching codec dimension");
		return false;
	}

	memset(&header, 0, sizeof(header));
	if (!tq_batch_page_read_header(page, page_size, &header, errmsg, errmsg_len))
		return false;

	tq_scan_stats_set_score_mode(TQ_SCAN_SCORE_MODE_DECODE_RESCORE);
	tq_scan_stats_set_score_kernel(TQ_PROD_SCORE_SCALAR);
	tq_scan_stats_set_path_flags(false, true);

	if (!tq_scan_scratch_ensure_decoded_capacity(scratch,
												 query_len,
												 errmsg,
												 errmsg_len))
		return false;

	decoded = scratch->decoded_values;
	query_norm_squared = tq_norm_squared_scalar(query_values, query_len);

	if (tq_batch_page_next_live_lane(page, page_size, -1, &lane, errmsg, errmsg_len))
	{
		bool		page_is_soa = tq_batch_page_is_soa(page, page_size);
		const uint8_t *soa_nibbles_ptr = NULL;
		const float *soa_gammas_ptr = NULL;
		uint32_t	soa_dim = 0;
		uint16_t	soa_lane_count = 0;
		uint8_t	   *reconstructed_code = NULL;

		if (page_is_soa)
		{
			TqProdPackedLayout layout;

			memset(&layout, 0, sizeof(layout));
			if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len))
				return false;

			if (!tq_batch_page_get_nibble_ptr(page, page_size, &soa_nibbles_ptr,
											  &soa_dim, &soa_lane_count,
											  errmsg, errmsg_len))
				return false;
			if (!tq_batch_page_get_gamma_ptr(page, page_size, &soa_gammas_ptr,
											 errmsg, errmsg_len))
				return false;

			reconstructed_code = (uint8_t *) malloc(layout.total_bytes);
			if (reconstructed_code == NULL)
			{
				tq_set_error(errmsg, errmsg_len,
							 "invalid turboquant scan: out of memory for SoA rescore code reconstruction");
				return false;
			}
		}

		do
		{
			TqTid tid;
			float distance_value = 0.0f;

			memset(&tid, 0, sizeof(tid));
			if (!tq_batch_page_get_tid(page, page_size, lane, &tid, errmsg, errmsg_len))
			{
				free(reconstructed_code);
				return false;
			}

			if (!tq_candidate_tid_is_selected(candidate_tids, candidate_tid_count, &tid))
				continue;

			if (page_is_soa)
			{
				TqProdPackedLayout layout;
				uint8_t	   *lane_nibbles;
				uint32_t	d;

				memset(&layout, 0, sizeof(layout));
				if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len))
				{
					free(reconstructed_code);
					return false;
				}

				lane_nibbles = (uint8_t *) malloc(soa_dim);
				if (lane_nibbles == NULL)
				{
					free(reconstructed_code);
					tq_set_error(errmsg, errmsg_len,
								 "invalid turboquant scan: out of memory for SoA rescore nibble extraction");
					return false;
				}
				for (d = 0; d < soa_dim; d++)
					lane_nibbles[d] = soa_nibbles_ptr[(size_t) d * soa_lane_count + lane];

				if (!tq_prod_nibbles_gamma_to_packed(config, lane_nibbles, soa_dim,
													 soa_gammas_ptr[lane],
													 reconstructed_code, layout.total_bytes,
													 errmsg, errmsg_len))
				{
					free(lane_nibbles);
					free(reconstructed_code);
					return false;
				}
				free(lane_nibbles);

				code = reconstructed_code;
				code_len = layout.total_bytes;
			}
			else
			{
				if (!tq_batch_page_code_view(page, page_size, lane, &code, &code_len, errmsg, errmsg_len))
					return false;
			}

			scratch->code_view_uses += 1;
			tq_scan_stats_record_code_view_uses(1);

			if (!tq_prod_decode(config, code, code_len, decoded, query_len, errmsg, errmsg_len))
			{
				free(reconstructed_code);
				return false;
			}

			tq_scan_stats_record_decoded_vector_only();

			if (!tq_metric_distance_from_decoded_vector(distance,
														query_values,
														query_len,
														decoded,
														query_len,
														query_norm_squared,
														&distance_value,
														errmsg,
														errmsg_len)
				|| !tq_candidate_heap_push(heap,
											 distance_value,
											 tid.block_number,
											 tid.offset_number))
			{
				free(reconstructed_code);
				return false;
			}
		} while (tq_batch_page_next_live_lane(page, page_size, (int) lane, &lane, errmsg, errmsg_len));

		free(reconstructed_code);
	}

	return true;
}

bool
tq_batch_page_scan_prod_cosine(const void *page,
							   size_t page_size,
							   const TqProdCodecConfig *config,
							   bool normalized,
							   const TqProdLut *lut,
							   const float *query_values,
							   size_t query_len,
							   bool filter_enabled,
							   int32_t filter_value,
							   TqCandidateHeap *heap,
							   TqCandidateHeap *shadow_decode_heap,
							   char *errmsg,
							   size_t errmsg_len)
{
	return tq_batch_page_scan_prod(page, page_size, config, normalized, TQ_DISTANCE_COSINE,
								   lut, query_values, query_len,
								   filter_enabled, filter_value,
								   heap, shadow_decode_heap,
								   errmsg, errmsg_len);
}
