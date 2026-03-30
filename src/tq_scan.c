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
static int tq_candidate_compare_best_qsort(const void *left, const void *right);
static int tq_candidate_tid_compare_qsort(const void *left, const void *right);
static int tq_tid_compare(const TqTid *left, const TqTid *right);
static bool tq_candidate_tid_is_selected(const TqTid *candidate_tids,
										 size_t candidate_tid_count,
										 const TqTid *tid);
static const char *tq_page_bound_mode_name(TqPageBoundMode mode);
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
	tq_last_scan_stats.faithful_fast_path = false;
	tq_last_scan_stats.compatibility_fallback = false;
	tq_last_scan_stats.safe_pruning_enabled = false;
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
tq_scan_stats_reset_candidate_heap_metrics(void)
{
	tq_last_scan_stats.retained_candidate_count = 0;
	tq_last_scan_stats.candidate_heap_capacity = 0;
	tq_last_scan_stats.candidate_heap_count = 0;
	tq_last_scan_stats.candidate_heap_insert_count = 0;
	tq_last_scan_stats.candidate_heap_replace_count = 0;
	tq_last_scan_stats.candidate_heap_reject_count = 0;
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
tq_scan_stats_record_shadow_decoded_vector(void)
{
	tq_last_scan_stats.shadow_decoded_vector_count += 1;
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
		"\"faithful_fast_path\":%s,\"compatibility_fallback\":%s,\"safe_pruning_enabled\":%s,"
		"\"configured_probe_count\":%zu,"
		"\"nominal_probe_count\":%zu,\"effective_probe_count\":%zu,"
		"\"max_visited_codes\":%zu,\"max_visited_pages\":%zu,"
		"\"selected_list_count\":%zu,\"selected_live_count\":%zu,\"selected_page_count\":%zu,"
		"\"visited_page_count\":%zu,\"visited_code_count\":%zu,"
		"\"retained_candidate_count\":%zu,\"candidate_heap_capacity\":%zu,"
		"\"candidate_heap_count\":%zu,\"candidate_heap_insert_count\":%zu,"
		"\"candidate_heap_replace_count\":%zu,\"candidate_heap_reject_count\":%zu,"
		"\"shadow_decoded_vector_count\":%zu,\"shadow_decode_candidate_count\":%zu,"
		"\"shadow_decode_overlap_count\":%zu,\"shadow_decode_primary_only_count\":%zu,"
		"\"shadow_decode_only_count\":%zu,"
		"\"decoded_vector_count\":%zu,"
		"\"bound_data_page_reads\":%zu,"
		"\"page_prune_count\":%zu,\"early_stop_count\":%zu}",
		tq_scan_mode_name(stats->mode),
		tq_scan_score_mode_name(stats->score_mode),
		tq_scan_score_kernel_name(stats),
		tq_page_bound_mode_name(stats->page_bound_mode),
		stats->faithful_fast_path ? "true" : "false",
		stats->compatibility_fallback ? "true" : "false",
		stats->safe_pruning_enabled ? "true" : "false",
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
		stats->shadow_decoded_vector_count,
		stats->shadow_decode_candidate_count,
		stats->shadow_decode_overlap_count,
		stats->shadow_decode_primary_only_count,
		stats->shadow_decode_only_count,
		stats->decoded_vector_count,
		stats->bound_data_page_reads,
		stats->page_prune_count,
		stats->early_stop_count
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
		tq_scan_stats_record_candidate_heap_insert();
		return true;
	}

	if (tq_candidate_compare_worst(&entry, &heap->entries[0]) >= 0)
	{
		tq_scan_stats_record_candidate_heap_reject();
		return true;
	}

	heap->entries[0] = entry;
	tq_candidate_sift_down(heap, 0);
	heap->sorted = false;
	tq_scan_stats_record_candidate_heap_replace();
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
	uint8_t *code = NULL;
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

	code = (uint8_t *) malloc(header.code_bytes);
	if (code == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: out of memory");
		return false;
	}

	ok = tq_batch_page_get_code(page, page_size, summary.representative_lane, code,
								header.code_bytes, errmsg, errmsg_len);
	if (!ok)
	{
		free(code);
		return false;
	}

	ok = tq_scan_summary_optimistic_distance_bound(config,
												   lut,
												   &summary,
												   code,
												   header.code_bytes,
												   normalized,
												   distance,
												   query_values,
												   query_len,
												   optimistic_distance,
												   errmsg,
												   errmsg_len);
	free(code);

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
						TqCandidateHeap *heap,
						TqCandidateHeap *shadow_decode_heap,
						char *errmsg,
						size_t errmsg_len)
{
	TqBatchPageHeaderView header;
	uint8_t    *code = NULL;
	float	   *decoded = NULL;
	bool		use_code_domain = false;
	bool		use_shadow_decode = false;
	float		query_norm_squared = 0.0f;
	uint16_t	lane = 0;

	if (page == NULL || config == NULL || lut == NULL || heap == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: page, codec, lut, and heap must be non-null");
		return false;
	}

	memset(&header, 0, sizeof(header));

	if (query_values == NULL || query_len != config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: query values must match codec dimension");
		return false;
	}

	if (!tq_batch_page_read_header(page, page_size, &header, errmsg, errmsg_len))
		return false;

	use_code_domain = tq_scan_can_use_prod_code_domain(normalized, distance)
		&& !tq_guc_force_decode_score_diagnostics;
	use_shadow_decode = use_code_domain
		&& shadow_decode_heap != NULL
		&& shadow_decode_heap->entries != NULL
		&& shadow_decode_heap->capacity > 0;
	tq_scan_stats_set_score_mode(use_code_domain ? TQ_SCAN_SCORE_MODE_CODE_DOMAIN
												 : TQ_SCAN_SCORE_MODE_DECODE);
	tq_scan_stats_set_score_kernel(use_code_domain
								   ? tq_prod_code_domain_preferred_kernel(config)
								   : TQ_PROD_SCORE_SCALAR);
	tq_scan_stats_set_path_flags(use_code_domain, !use_code_domain);
	tq_scan_stats_record_page_visit();

	code = (uint8_t *) malloc(header.code_bytes);
	if (code == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: out of memory");
		return false;
	}

	if (!use_code_domain || use_shadow_decode)
	{
		decoded = (float *) malloc(sizeof(float) * query_len);
		if (decoded == NULL)
		{
			free(code);
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant scan: out of memory");
			return false;
		}

		query_norm_squared = tq_norm_squared_scalar(query_values, query_len);
	}

	if (tq_batch_page_next_live_lane(page, page_size, -1, &lane, errmsg, errmsg_len))
	{
		do
		{
			TqTid		tid;
			float		distance_value = 0.0f;
			float		ip_score = 0.0f;

			memset(&tid, 0, sizeof(tid));
			if (!tq_batch_page_get_tid(page, page_size, lane, &tid, errmsg, errmsg_len)
				|| !tq_batch_page_get_code(page, page_size, lane, code, header.code_bytes, errmsg, errmsg_len))
			{
				free(decoded);
				free(code);
				return false;
			}

			if (use_code_domain)
			{
				TqProdScoreKernel used_kernel = TQ_PROD_SCORE_SCALAR;
				float		shadow_distance_value = 0.0f;

				if (!tq_prod_score_code_from_lut_dispatch(config, lut, code, header.code_bytes,
														  tq_prod_code_domain_preferred_kernel(config),
														  &ip_score, &used_kernel, errmsg, errmsg_len))
				{
					free(decoded);
					free(code);
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
					free(decoded);
					free(code);
					return false;
				}

				if (use_shadow_decode)
				{
					if (!tq_prod_decode(config, code, header.code_bytes, decoded, query_len,
										 errmsg, errmsg_len))
					{
						free(decoded);
						free(code);
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
						|| !tq_candidate_heap_push(shadow_decode_heap,
													 shadow_distance_value,
													 tid.block_number,
													 tid.offset_number))
					{
						free(decoded);
						free(code);
						return false;
					}
				}
			}
			else if (!tq_prod_decode(config, code, header.code_bytes, decoded, query_len,
									 errmsg, errmsg_len))
			{
				free(decoded);
				free(code);
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
					free(decoded);
					free(code);
					return false;
				}
			}

			if (!tq_candidate_heap_push(heap, distance_value, tid.block_number, tid.offset_number))
			{
				free(decoded);
				free(code);
				return false;
			}

			tq_scan_stats_set_candidate_heap_metrics(heap->capacity, heap->count);
		} while (tq_batch_page_next_live_lane(page, page_size, (int) lane, &lane, errmsg, errmsg_len));
	}

	free(decoded);
	free(code);
	return true;
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
	TqBatchPageHeaderView header;
	uint8_t *code = NULL;
	float *decoded = NULL;
	float query_norm_squared = 0.0f;
	uint16_t lane = 0;

	if (page == NULL || config == NULL || query_values == NULL || heap == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: decode rescoring requires page, codec, query, candidate tids, and heap");
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

	code = (uint8_t *) malloc(header.code_bytes);
	decoded = (float *) malloc(sizeof(float) * query_len);
	if (code == NULL || decoded == NULL)
	{
		free(decoded);
		free(code);
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant scan: out of memory");
		return false;
	}

	query_norm_squared = tq_norm_squared_scalar(query_values, query_len);

	if (tq_batch_page_next_live_lane(page, page_size, -1, &lane, errmsg, errmsg_len))
	{
		do
		{
			TqTid tid;
			float distance_value = 0.0f;

			memset(&tid, 0, sizeof(tid));
			if (!tq_batch_page_get_tid(page, page_size, lane, &tid, errmsg, errmsg_len))
			{
				free(decoded);
				free(code);
				return false;
			}

			if (!tq_candidate_tid_is_selected(candidate_tids, candidate_tid_count, &tid))
				continue;

			if (!tq_batch_page_get_code(page, page_size, lane, code, header.code_bytes, errmsg, errmsg_len)
				|| !tq_prod_decode(config, code, header.code_bytes, decoded, query_len, errmsg, errmsg_len))
			{
				free(decoded);
				free(code);
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
				free(decoded);
				free(code);
				return false;
			}
		} while (tq_batch_page_next_live_lane(page, page_size, (int) lane, &lane, errmsg, errmsg_len));
	}

	free(decoded);
	free(code);
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
							   TqCandidateHeap *heap,
							   TqCandidateHeap *shadow_decode_heap,
							   char *errmsg,
							   size_t errmsg_len)
{
	return tq_batch_page_scan_prod(page, page_size, config, normalized, TQ_DISTANCE_COSINE,
								   lut, query_values, query_len, heap, shadow_decode_heap,
								   errmsg, errmsg_len);
}
