#ifndef TQ_SCAN_H
#define TQ_SCAN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/tq_codec_prod.h"
#include "src/tq_page.h"

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
extern bool tq_batch_page_scan_prod(const void *page,
									size_t page_size,
									const TqProdCodecConfig *config,
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
										   const TqProdLut *lut,
										   const float *query_values,
										   size_t query_len,
										   TqCandidateHeap *heap,
										   char *errmsg,
										   size_t errmsg_len);

#endif
