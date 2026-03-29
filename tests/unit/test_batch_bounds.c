#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/tq_codec_prod.h"
#include "src/tq_page.h"
#include "src/tq_scan.h"

static void
normalize(float *values, size_t len)
{
	float norm = 0.0f;
	size_t i = 0;

	for (i = 0; i < len; i++)
		norm += values[i] * values[i];

	norm = sqrtf(norm);
	assert(norm > 0.0f);

	for (i = 0; i < len; i++)
		values[i] /= norm;
}

static void
seeded_unit_vector(uint32_t seed, float *values, size_t len)
{
	size_t i = 0;

	for (i = 0; i < len; i++)
	{
		uint32_t mixed = seed * 1664525u + 1013904223u + (uint32_t) i * 2654435761u;
		values[i] = ((float) (mixed % 2001u) / 1000.0f) - 1.0f;
	}
	normalize(values, len);
}

static void
append_encoded_lane(uint8_t *page,
					size_t page_size,
					const TqProdCodecConfig *config,
					const float *values,
					uint16_t offset_number)
{
	TqProdPackedLayout layout;
	uint8_t packed[64];
	uint16_t lane = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(packed, 0, sizeof(packed));

	assert(tq_prod_packed_layout(config, &layout, errmsg, sizeof(errmsg)));
	assert(layout.total_bytes <= sizeof(packed));
	assert(tq_prod_encode(config, values, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, page_size,
									 &(TqTid){.block_number = 1, .offset_number = offset_number},
									 &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, page_size, lane, packed, layout.total_bytes,
								  errmsg, sizeof(errmsg)));
}

static void
test_page_summary_round_trip(void)
{
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout layout;
	TqBatchPageParams params = {
		.lane_count = 4,
		.code_bytes = 0,
		.list_id = 7,
		.next_block = TQ_INVALID_BLOCK_NUMBER
	};
	TqBatchPageSummary written = {
		.representative_lane = 1,
		.residual_radius = 0.375f
	};
	TqBatchPageSummary readback;
	float left[8];
	float right[8];
	char errmsg[256];

	memset(page, 0, sizeof(page));
	memset(&layout, 0, sizeof(layout));
	memset(&readback, 0, sizeof(readback));
	memset(left, 0, sizeof(left));
	memset(right, 0, sizeof(right));

	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	params.code_bytes = (uint32_t) layout.total_bytes;
	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	seeded_unit_vector(1u, left, 8);
	seeded_unit_vector(2u, right, 8);
	append_encoded_lane(page, sizeof(page), &config, left, 1);
	append_encoded_lane(page, sizeof(page), &config, right, 2);
	assert(tq_batch_page_set_summary(page, sizeof(page), &written, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_get_summary(page, sizeof(page), &readback, errmsg, sizeof(errmsg)));
	assert(readback.representative_lane == written.representative_lane);
	assert(fabsf(readback.residual_radius - written.residual_radius) <= 1e-6f);
}

static void
test_optimistic_bound_prefers_clearly_better_page(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdLut lut;
	TqProdPackedLayout layout;
	TqBatchPageParams params;
	uint8_t near_page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t far_page[TQ_DEFAULT_BLOCK_SIZE];
	float query[8];
	float near_bound = 0.0f;
	float far_bound = 0.0f;
	char errmsg[256];

	memset(&lut, 0, sizeof(lut));
	memset(&layout, 0, sizeof(layout));
	memset(&params, 0, sizeof(params));
	memset(near_page, 0, sizeof(near_page));
	memset(far_page, 0, sizeof(far_page));
	memset(query, 0, sizeof(query));

	seeded_unit_vector(10u, query, 8);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));

	params.lane_count = 4;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;

	assert(tq_batch_page_init(near_page, sizeof(near_page), &params, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_init(far_page, sizeof(far_page), &params, errmsg, sizeof(errmsg)));

	append_encoded_lane(near_page, sizeof(near_page), &config, query, 1);
	append_encoded_lane(far_page, sizeof(far_page), &config, (float[8]){-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 1);

	assert(tq_batch_page_set_summary(near_page, sizeof(near_page),
									 &(TqBatchPageSummary){.representative_lane = 0, .residual_radius = 0.05f},
									 errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_summary(far_page, sizeof(far_page),
									 &(TqBatchPageSummary){.representative_lane = 0, .residual_radius = 0.05f},
									 errmsg, sizeof(errmsg)));

	assert(tq_scan_page_optimistic_distance_bound(&config, &lut, near_page, sizeof(near_page),
												  true, TQ_DISTANCE_COSINE, query, 8,
												  &near_bound, errmsg, sizeof(errmsg)));
	assert(tq_scan_page_optimistic_distance_bound(&config, &lut, far_page, sizeof(far_page),
												  true, TQ_DISTANCE_COSINE, query, 8,
												  &far_bound, errmsg, sizeof(errmsg)));
	assert(near_bound < far_bound);

	tq_prod_lut_reset(&lut);
}

static void
test_empty_page_bound_is_safe(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdLut lut;
	TqProdPackedLayout layout;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	float query[8];
	float optimistic_distance = 0.0f;
	char errmsg[256];

	memset(&lut, 0, sizeof(lut));
	memset(&layout, 0, sizeof(layout));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(query, 0, sizeof(query));

	seeded_unit_vector(12u, query, 8);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));

	params.lane_count = 4;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_summary(page, sizeof(page),
									 &(TqBatchPageSummary){
										.representative_lane = TQ_BATCH_PAGE_NO_REPRESENTATIVE,
										.residual_radius = 0.0f
									 },
									 errmsg, sizeof(errmsg)));
	assert(tq_scan_page_optimistic_distance_bound(&config, &lut, page, sizeof(page),
												  true, TQ_DISTANCE_COSINE, query, 8,
												  &optimistic_distance, errmsg, sizeof(errmsg)));
	assert(isinf(optimistic_distance));

	tq_prod_lut_reset(&lut);
}

static void
test_early_stop_requires_full_heap(void)
{
	TqCandidateHeap heap;
	bool should_prune = false;
	char errmsg[256];

	memset(&heap, 0, sizeof(heap));
	assert(tq_candidate_heap_init(&heap, 2));
	assert(tq_candidate_heap_push(&heap, 0.10f, 1, 1));
	assert(tq_scan_should_prune_page(&heap, 0.50f, &should_prune, errmsg, sizeof(errmsg)));
	assert(!should_prune);
	tq_candidate_heap_reset(&heap);
}

static void
test_full_heap_prunes_worse_page_bounds(void)
{
	TqCandidateHeap heap;
	bool should_prune = false;
	char errmsg[256];

	memset(&heap, 0, sizeof(heap));
	assert(tq_candidate_heap_init(&heap, 2));
	assert(tq_candidate_heap_push(&heap, 0.10f, 1, 1));
	assert(tq_candidate_heap_push(&heap, 0.20f, 1, 2));
	assert(tq_scan_should_prune_page(&heap, 0.25f, &should_prune, errmsg, sizeof(errmsg)));
	assert(should_prune);
	tq_candidate_heap_reset(&heap);
}

int
main(void)
{
	test_page_summary_round_trip();
	test_optimistic_bound_prefers_clearly_better_page();
	test_empty_page_bound_is_safe();
	test_early_stop_requires_full_heap();
	test_full_heap_prunes_worse_page_bounds();
	return 0;
}
