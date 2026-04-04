#ifndef TQ_SCAN_H
#define TQ_SCAN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/tq_codec_prod.h"
#include "src/tq_metadata.h"
#include "src/tq_page.h"
#include "src/tq_simd_avx2.h"

#define TQ_MAX_FILTER_VALUES_PER_CLAUSE 16

typedef struct TqMetadataFilterClause
{
	uint16_t	attribute_index;
	TqMetadataKind kind;
	bool		match_null;
	uint16_t	value_count;
	uint8_t		values[TQ_MAX_FILTER_VALUES_PER_CLAUSE][TQ_METADATA_SLOT_BYTES];
} TqMetadataFilterClause;

typedef struct TqCandidateEntry
{
	float		score;
	TqTid		tid;
	TqExactKeyRef exact_key_ref;
	uint16_t	metadata_attribute_count;
	uint16_t	metadata_nullmask;
	uint8_t		metadata_values[TQ_MAX_STORED_METADATA_ATTRIBUTES * TQ_METADATA_SLOT_BYTES];
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
	TQ_SCAN_SCORE_MODE_DECODE_RESCORE,
	TQ_SCAN_SCORE_MODE_CODE_DOMAIN,
	TQ_SCAN_SCORE_MODE_BITMAP_FILTER
} TqScanScoreMode;

typedef enum TqPageBoundMode
{
	TQ_PAGE_BOUND_MODE_NONE = 0,
	TQ_PAGE_BOUND_MODE_DISABLED,
	TQ_PAGE_BOUND_MODE_SAFE_SUMMARY_PRUNING,
	TQ_PAGE_BOUND_MODE_ORDERING_ONLY_SUMMARY,
	TQ_PAGE_BOUND_MODE_DATA_PAGE_FALLBACK,
	TQ_PAGE_BOUND_MODE_MIXED
} TqPageBoundMode;

typedef enum TqScanOrchestration
{
	TQ_SCAN_ORCHESTRATION_NONE = 0,
	TQ_SCAN_ORCHESTRATION_FLAT_STREAMING,
	TQ_SCAN_ORCHESTRATION_FLAT_BOUNDED_PAGES,
	TQ_SCAN_ORCHESTRATION_IVF_BOUNDED_PAGES,
	TQ_SCAN_ORCHESTRATION_IVF_NEAR_EXHAUSTIVE,
	TQ_SCAN_ORCHESTRATION_BITMAP_FILTER
} TqScanOrchestration;

typedef enum TqRouterSelectionMethod
{
	TQ_ROUTER_SELECTION_NONE = 0,
	TQ_ROUTER_SELECTION_PARTIAL,
	TQ_ROUTER_SELECTION_FULL_SORT
} TqRouterSelectionMethod;

typedef struct TqScanStats
{
	TqScanMode	mode;
	TqScanScoreMode score_mode;
	TqProdScoreKernel score_kernel;
	TqPageBoundMode page_bound_mode;
	TqScanOrchestration scan_orchestration;
	TqRouterSelectionMethod router_selection_method;
	bool		faithful_fast_path;
	bool		compatibility_fallback;
	bool		safe_pruning_enabled;
	bool		near_exhaustive_crossover;
	size_t		configured_probe_count;
	size_t		nominal_probe_count;
	size_t		effective_probe_count;
	size_t		max_visited_codes;
	size_t		max_visited_pages;
	size_t		selected_list_count;
	size_t		selected_live_count;
	size_t		selected_page_count;
	size_t		visited_page_count;
	size_t		visited_code_count;
	size_t		retained_candidate_count;
	size_t		candidate_heap_capacity;
	size_t		candidate_heap_count;
	size_t		candidate_heap_insert_count;
	size_t		candidate_heap_replace_count;
	size_t		candidate_heap_reject_count;
	size_t		local_candidate_heap_insert_count;
	size_t		local_candidate_heap_replace_count;
	size_t		local_candidate_heap_reject_count;
	size_t		local_candidate_merge_count;
	size_t		shadow_decoded_vector_count;
	size_t		shadow_decode_candidate_count;
	size_t		shadow_decode_overlap_count;
	size_t		shadow_decode_primary_only_count;
	size_t		shadow_decode_only_count;
	size_t		decoded_vector_count;
	size_t		bound_data_page_reads;
	size_t		page_prune_count;
	size_t		early_stop_count;
	size_t		scratch_allocations;
	size_t		decoded_buffer_reuses;
	size_t		code_view_uses;
	size_t		code_copy_uses;
	size_t		block_local_scored_count;
	size_t		block_local_survivor_count;
	size_t		block_local_rejected_count;
} TqScanStats;

/*
 * Scratch block-16 microblock for transposed scoring.
 * nibbles: dimension-major layout, nibbles[d * 16 + c]
 *          (16 candidates per dimension, contiguous for SIMD loads)
 * gammas:  one float per candidate
 * tids:    one TqTid per candidate
 * count:   number of valid candidates (1..16)
 */
#define TQ_BLOCK16_MAX_CANDIDATES 16

typedef struct TqScratchBlock16
{
	uint8_t    *nibbles;
	float	   *gammas;
	TqTid	   *tids;
	uint32_t	count;
	uint32_t	dimension;
} TqScratchBlock16;

typedef struct TqScratchBlock16Set
{
	TqScratchBlock16 *blocks;
	uint32_t	block_count;
	uint32_t	block_capacity;
	uint8_t    *nibble_storage;
	float	   *gamma_storage;
	TqTid	   *tid_storage;
	uint32_t	dimension;
	uint32_t	total_candidates;
} TqScratchBlock16Set;

typedef struct TqScanScratch
{
	float	   *decoded_values;
	size_t		decoded_capacity;
	size_t		scratch_allocations;
	size_t		decoded_buffer_reuses;
	size_t		code_view_uses;
	size_t		code_copy_uses;
	TqScratchBlock16Set block16_set;
	bool		block16_set_initialized;
	const TqProdLut16 *lut16;	/* optional: enables block-16 fast path */
} TqScanScratch;

extern bool tq_candidate_heap_init(TqCandidateHeap *heap, size_t capacity);
extern void tq_candidate_heap_reset(TqCandidateHeap *heap);
extern bool tq_candidate_heap_push(TqCandidateHeap *heap,
								   float score,
								   uint32_t block_number,
								   uint16_t offset_number,
								   const TqExactKeyRef *exact_key_ref,
								   const uint8_t *metadata_values,
								   uint16_t metadata_attribute_count);
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
extern bool tq_scan_summary_optimistic_distance_bound(const TqProdCodecConfig *config,
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
														  size_t errmsg_len);
extern bool tq_scan_page_bounds_are_safe_for_pruning(bool normalized,
													 TqDistanceKind distance);
extern bool tq_scan_should_prune_page(const TqCandidateHeap *heap,
									  float optimistic_distance,
									  bool *should_prune,
									  char *errmsg,
									  size_t errmsg_len);
extern void tq_scan_stats_reset_last(void);
extern void tq_scan_stats_begin(TqScanMode mode, size_t configured_probe_count);
extern void tq_scan_stats_set_score_mode(TqScanScoreMode score_mode);
extern void tq_scan_stats_set_score_kernel(TqProdScoreKernel score_kernel);
extern void tq_scan_stats_set_path_flags(bool faithful_fast_path,
										 bool compatibility_fallback);
extern void tq_scan_stats_set_scan_orchestration(TqScanOrchestration scan_orchestration,
												 bool near_exhaustive_crossover);
extern void tq_scan_stats_record_page_bound_mode(TqPageBoundMode page_bound_mode,
												 bool safe_pruning_enabled);
extern void tq_scan_stats_set_router_selection_method(TqRouterSelectionMethod method);
extern void tq_scan_stats_reset_candidate_heap_metrics(void);
extern void tq_scan_stats_set_probe_budget(size_t nominal_probe_count,
										   size_t effective_probe_count,
										   size_t max_visited_codes,
										   size_t max_visited_pages);
extern void tq_scan_stats_record_selected_list(size_t live_count,
											   size_t page_count);
extern void tq_scan_stats_add_selected_live(size_t live_count);
extern void tq_scan_stats_record_page_visit(void);
extern void tq_scan_stats_record_code_visit(bool decoded_vector);
extern void tq_scan_stats_record_bound_data_page_read(void);
extern void tq_scan_stats_add_page_prunes(size_t count);
extern void tq_scan_stats_add_early_stops(size_t count);
extern void tq_scan_stats_record_candidate_heap_insert(void);
extern void tq_scan_stats_record_candidate_heap_replace(void);
extern void tq_scan_stats_record_candidate_heap_reject(void);
extern void tq_scan_stats_record_local_candidate_heap_insert(void);
extern void tq_scan_stats_record_local_candidate_heap_replace(void);
extern void tq_scan_stats_record_local_candidate_heap_reject(void);
extern void tq_scan_stats_record_local_candidate_merge(void);
extern void tq_scan_stats_record_decoded_vector_only(void);
extern void tq_scan_stats_record_shadow_decoded_vector(void);
extern void tq_scan_stats_record_scratch_allocations(size_t count);
extern void tq_scan_stats_record_decoded_buffer_reuses(size_t count);
extern void tq_scan_stats_record_code_view_uses(size_t count);
extern void tq_scan_stats_record_code_copy_uses(size_t count);
extern void tq_scan_stats_record_block_local_selection(size_t scored, size_t survivors);
extern void tq_scan_stats_set_candidate_heap_metrics(size_t capacity, size_t count);
extern void tq_scan_stats_set_shadow_decode_metrics(const TqCandidateHeap *primary,
													const TqCandidateHeap *shadow);
extern bool tq_candidate_heap_copy_sorted_tids(const TqCandidateHeap *heap,
											   TqTid *dest,
											   size_t capacity,
											   size_t *count);
extern bool tq_scan_stats_copy_shadow_decode_tids(TqTid *dest,
												  size_t capacity,
												  size_t *count);
extern void tq_scan_stats_snapshot(TqScanStats *stats);
extern bool tq_scan_stats_serialize_json(const TqScanStats *stats,
										 char *buffer,
										 size_t buffer_len);
extern bool tq_scan_active_uses_prod_code_domain(bool normalized, TqDistanceKind distance);
extern void tq_scan_scratch_reset(TqScanScratch *scratch);

extern uint32_t tq_block16_select_top_m(const float *scores,
										uint32_t candidate_count,
										uint32_t top_m,
										uint32_t *selected_indices);
extern bool tq_scratch_block16_set_init(TqScratchBlock16Set *set,
										uint32_t dimension,
										uint32_t max_candidates,
										char *errmsg,
										size_t errmsg_len);
extern void tq_scratch_block16_set_reset(TqScratchBlock16Set *set);
extern bool tq_batch_page_transpose_block16(const void *page,
											size_t page_size,
											const TqProdCodecConfig *config,
											TqScratchBlock16Set *set,
											char *errmsg,
											size_t errmsg_len);
extern bool tq_batch_page_scan_prod(const void *page,
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
									size_t errmsg_len);
extern bool tq_batch_page_scan_prod_filtered(const void *page,
											 size_t page_size,
											 const TqProdCodecConfig *config,
											 bool normalized,
											 TqDistanceKind distance,
											 const TqProdLut *lut,
											 const float *query_values,
											 size_t query_len,
											 const TqMetadataFilterClause *filter_clauses,
											 uint16_t filter_clause_count,
											 uint16_t metadata_attribute_count,
											 TqCandidateHeap *heap,
											 TqCandidateHeap *shadow_decode_heap,
											 char *errmsg,
											 size_t errmsg_len);
extern bool tq_batch_page_scan_prod_with_scratch(const void *page,
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
												 size_t errmsg_len);
extern bool tq_batch_page_scan_prod_with_scratch_filtered(const void *page,
														  size_t page_size,
														  const TqProdCodecConfig *config,
														  bool normalized,
														  TqDistanceKind distance,
														  const TqProdLut *lut,
														  const float *query_values,
														  size_t query_len,
														  const TqMetadataFilterClause *filter_clauses,
														  uint16_t filter_clause_count,
														  uint16_t metadata_attribute_count,
														  TqCandidateHeap *heap,
														  TqCandidateHeap *shadow_decode_heap,
														  TqScanScratch *scratch,
														  char *errmsg,
														  size_t errmsg_len);
extern bool tq_batch_page_rescore_prod_candidates(const void *page,
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
												  size_t errmsg_len);
extern bool tq_batch_page_rescore_prod_candidates_with_scratch(const void *page,
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
															   size_t errmsg_len);
extern bool tq_batch_page_scan_prod_cosine(const void *page,
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
										   size_t errmsg_len);

#endif
