#ifndef TQ_SCAN_H
#define TQ_SCAN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/tq_codec_prod.h"
#include "src/tq_page.h"
#include "src/tq_simd_avx2.h"

typedef struct TqCandidateEntry
{
	float		score;
	TqTid		tid;
} TqCandidateEntry;

typedef struct TqCandidateHeap
{
	size_t		capacity;
	size_t		count;
	size_t		pop_index;
	bool		sorted;
	TqCandidateEntry *entries;
} TqCandidateHeap;

typedef enum TqScanMode
{
	TQ_SCAN_MODE_NONE = 0,
	TQ_SCAN_MODE_FLAT,
	TQ_SCAN_MODE_IVF,
	TQ_SCAN_MODE_BITMAP
} TqScanMode;

typedef enum TqScanScoreMode
{
	TQ_SCAN_SCORE_MODE_NONE = 0,
	TQ_SCAN_SCORE_MODE_DECODE,
	TQ_SCAN_SCORE_MODE_CODE_DOMAIN,
	TQ_SCAN_SCORE_MODE_BITMAP_FILTER
} TqScanScoreMode;

typedef struct TqScanStats
{
	TqScanMode	mode;
	TqScanScoreMode score_mode;
	TqProdScoreKernel score_kernel;
	size_t		configured_probe_count;
	size_t		nominal_probe_count;
	size_t		effective_probe_count;
	size_t		max_visited_codes;
	size_t		max_visited_pages;
	size_t		selected_list_count;
	size_t		selected_live_count;
	size_t		visited_page_count;
	size_t		visited_code_count;
	size_t		retained_candidate_count;
	size_t		candidate_heap_capacity;
	size_t		candidate_heap_count;
	size_t		decoded_vector_count;
	size_t		page_prune_count;
	size_t		early_stop_count;
} TqScanStats;

extern bool tq_candidate_heap_init(TqCandidateHeap *heap, size_t capacity);
extern void tq_candidate_heap_reset(TqCandidateHeap *heap);
extern bool tq_candidate_heap_push(TqCandidateHeap *heap,
								   float score,
								   uint32_t block_number,
								   uint16_t offset_number);
extern bool tq_candidate_heap_pop_best(TqCandidateHeap *heap,
									   TqCandidateEntry *entry);
extern bool tq_metric_distance_from_ip_score(TqDistanceKind distance,
											 float ip_score,
											 float *distance_value,
											 char *errmsg,
											 size_t errmsg_len);
extern bool tq_scan_page_optimistic_distance_bound(const TqProdCodecConfig *config,
												   const TqProdLut *lut,
												   const void *page,
												   size_t page_size,
												   bool normalized,
												   TqDistanceKind distance,
												   const float *query_values,
												   size_t query_len,
												   float *optimistic_distance,
												   char *errmsg,
												   size_t errmsg_len);
extern bool tq_scan_should_prune_page(const TqCandidateHeap *heap,
									  float optimistic_distance,
									  bool *should_prune,
									  char *errmsg,
									  size_t errmsg_len);
extern void tq_scan_stats_reset_last(void);
extern void tq_scan_stats_begin(TqScanMode mode, size_t configured_probe_count);
extern void tq_scan_stats_set_score_mode(TqScanScoreMode score_mode);
extern void tq_scan_stats_set_score_kernel(TqProdScoreKernel score_kernel);
extern void tq_scan_stats_set_probe_budget(size_t nominal_probe_count,
										   size_t effective_probe_count,
										   size_t max_visited_codes,
										   size_t max_visited_pages);
extern void tq_scan_stats_record_selected_list(size_t live_count);
extern void tq_scan_stats_add_selected_live(size_t live_count);
extern void tq_scan_stats_record_page_visit(void);
extern void tq_scan_stats_record_code_visit(bool decoded_vector);
extern void tq_scan_stats_add_page_prunes(size_t count);
extern void tq_scan_stats_add_early_stops(size_t count);
extern void tq_scan_stats_set_candidate_heap_metrics(size_t capacity, size_t count);
extern void tq_scan_stats_snapshot(TqScanStats *stats);
extern bool tq_scan_stats_serialize_json(const TqScanStats *stats,
										 char *buffer,
										 size_t buffer_len);
extern bool tq_batch_page_scan_prod(const void *page,
									size_t page_size,
									const TqProdCodecConfig *config,
									bool normalized,
									TqDistanceKind distance,
									const TqProdLut *lut,
									const float *query_values,
									size_t query_len,
									TqCandidateHeap *heap,
									char *errmsg,
									size_t errmsg_len);
extern bool tq_batch_page_scan_prod_cosine(const void *page,
										   size_t page_size,
										   const TqProdCodecConfig *config,
										   bool normalized,
										   const TqProdLut *lut,
										   const float *query_values,
										   size_t query_len,
										   TqCandidateHeap *heap,
										   char *errmsg,
										   size_t errmsg_len);

#endif
