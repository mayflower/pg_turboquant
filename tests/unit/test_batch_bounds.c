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

static float
vector_l2_distance(const float *left, const float *right, size_t len)
{
	float sum = 0.0f;
	size_t i = 0;

	for (i = 0; i < len; i++)
	{
		float delta = left[i] - right[i];

		sum += delta * delta;
	}

	return sqrtf(sum);
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

static float
best_actual_page_distance(const uint8_t *page,
						  size_t page_size,
						  const TqProdCodecConfig *config,
						  const TqProdLut *lut)
{
	TqBatchPageHeaderView header;
	uint8_t code[64];
	float best_distance = INFINITY;
	uint16_t lane = 0;
	char errmsg[256];

	memset(&header, 0, sizeof(header));
	memset(code, 0, sizeof(code));
	assert(tq_batch_page_read_header(page, page_size, &header, errmsg, sizeof(errmsg)));
	assert(header.code_bytes <= sizeof(code));

	if (!tq_batch_page_next_live_lane(page, page_size, -1, &lane, errmsg, sizeof(errmsg)))
		return INFINITY;

	do
	{
		float ip_score = 0.0f;
		float distance = 0.0f;

		assert(tq_batch_page_get_code(page, page_size, lane, code, header.code_bytes,
									  errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut(config, lut, code, header.code_bytes,
										   &ip_score, errmsg, sizeof(errmsg)));
		assert(tq_metric_distance_from_ip_score(TQ_DISTANCE_COSINE,
												ip_score,
												&distance,
												errmsg,
												sizeof(errmsg)));
		if (distance < best_distance)
			best_distance = distance;
	}
	while (tq_batch_page_next_live_lane(page, page_size, (int) lane, &lane,
										errmsg, sizeof(errmsg)));

	return best_distance;
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
test_summary_side_page_round_trip(void)
{
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	TqBatchPageSummary written = {
		.representative_lane = 2,
		.residual_radius = 0.1875f
	};
	TqBatchPageSummary readback;
	uint8_t written_code[8] = {3, 1, 4, 1, 5, 9, 2, 6};
	uint8_t readback_code[8];
	uint32_t block_number = 0;
	TqBatchSummaryPageHeaderView header;
	char errmsg[256];

	memset(page, 0, sizeof(page));
	memset(&readback, 0, sizeof(readback));
	memset(readback_code, 0, sizeof(readback_code));
	memset(&header, 0, sizeof(header));

	assert(tq_batch_summary_page_init(page, sizeof(page), sizeof(written_code), 4, 91,
									  errmsg, sizeof(errmsg)));
	assert(tq_batch_summary_page_set_entry(page, sizeof(page), 0, 77, &written,
										   written_code, sizeof(written_code),
										   errmsg, sizeof(errmsg)));
	assert(tq_batch_summary_page_read_header(page, sizeof(page), &header,
											errmsg, sizeof(errmsg)));
	assert(header.code_bytes == sizeof(written_code));
	assert(header.entry_count == 1);
	assert(header.next_block == 91);
	assert(tq_batch_summary_page_get_entry(page, sizeof(page), 0, &block_number, &readback,
										   readback_code, sizeof(readback_code),
										   errmsg, sizeof(errmsg)));
	assert(block_number == 77);
	assert(readback.representative_lane == written.representative_lane);
	assert(fabsf(readback.residual_radius - written.residual_radius) <= 1e-6f);
	assert(memcmp(readback_code, written_code, sizeof(written_code)) == 0);
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
test_summary_bound_matches_page_bound(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdLut lut;
	TqProdPackedLayout layout;
	TqBatchPageSummary summary;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t representative_code[64];
	float query[8];
	float page_bound = 0.0f;
	float summary_bound = 0.0f;
	char errmsg[256];

	memset(&lut, 0, sizeof(lut));
	memset(&layout, 0, sizeof(layout));
	memset(&summary, 0, sizeof(summary));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(representative_code, 0, sizeof(representative_code));
	memset(query, 0, sizeof(query));

	seeded_unit_vector(21u, query, 8);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(layout.total_bytes <= sizeof(representative_code));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));

	params.lane_count = 4;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	append_encoded_lane(page, sizeof(page), &config, query, 1);
	assert(tq_batch_page_set_summary(page, sizeof(page),
									 &(TqBatchPageSummary){.representative_lane = 0, .residual_radius = 0.05f},
									 errmsg, sizeof(errmsg)));
	assert(tq_batch_page_get_summary(page, sizeof(page), &summary, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_get_code(page, sizeof(page), summary.representative_lane,
								  representative_code, layout.total_bytes,
								  errmsg, sizeof(errmsg)));
	assert(tq_scan_page_optimistic_distance_bound(&config, &lut, page, sizeof(page),
												  true, TQ_DISTANCE_COSINE, query, 8,
												  &page_bound, errmsg, sizeof(errmsg)));
	assert(tq_scan_summary_optimistic_distance_bound(&config, &lut, &summary,
													 representative_code, layout.total_bytes,
													 true, TQ_DISTANCE_COSINE, query, 8,
													 &summary_bound, errmsg, sizeof(errmsg)));
	assert(isfinite(summary_bound));
	assert(fabsf(summary_bound - page_bound) <= 1e-6f);

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

static void
test_code_domain_page_bounds_are_not_pruning_safe(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdLut lut;
	TqProdPackedLayout layout;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	float query[8];
	float representative[8];
	float lane_values[3][8];
	uint8_t representative_code[64];
	TqBatchPageSummary summary;
	float optimistic_distance = 0.0f;
	float best_distance = 0.0f;
	float residual_radius = 0.0f;
	uint32_t query_seed = 0;
	bool saw_violation = false;
	char errmsg[256];

	memset(&lut, 0, sizeof(lut));
	memset(&layout, 0, sizeof(layout));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(query, 0, sizeof(query));
	memset(representative, 0, sizeof(representative));
	memset(lane_values, 0, sizeof(lane_values));
	memset(representative_code, 0, sizeof(representative_code));
	memset(&summary, 0, sizeof(summary));

	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(layout.total_bytes <= sizeof(representative_code));
	assert(!tq_scan_page_bounds_are_safe_for_pruning(true, TQ_DISTANCE_COSINE));

	for (query_seed = 1; query_seed <= 256; query_seed++)
	{
		uint32_t lane_seed = 0;

		memset(page, 0, sizeof(page));
		memset(&summary, 0, sizeof(summary));
		seeded_unit_vector(query_seed, query, 8);
		seeded_unit_vector(query_seed * 17u + 3u, representative, 8);
		assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));

		params.lane_count = 4;
		params.code_bytes = (uint32_t) layout.total_bytes;
		params.list_id = 0;
		params.next_block = TQ_INVALID_BLOCK_NUMBER;
		assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));

		append_encoded_lane(page, sizeof(page), &config, representative, 1);
		for (lane_seed = 0; lane_seed < 3; lane_seed++)
		{
			seeded_unit_vector(query_seed * 31u + (lane_seed + 1u) * 101u,
							   lane_values[lane_seed], 8);
			append_encoded_lane(page, sizeof(page), &config, lane_values[lane_seed],
								(uint16_t) (lane_seed + 2u));
			if (vector_l2_distance(representative, lane_values[lane_seed], 8) > residual_radius)
				residual_radius = vector_l2_distance(representative, lane_values[lane_seed], 8);
		}

		summary.representative_lane = 0;
		summary.residual_radius = residual_radius;
		assert(tq_batch_page_set_summary(page, sizeof(page), &summary, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_get_code(page, sizeof(page), summary.representative_lane,
									  representative_code, layout.total_bytes,
									  errmsg, sizeof(errmsg)));
		assert(tq_scan_summary_optimistic_distance_bound(&config,
														 &lut,
														 &summary,
														 representative_code,
														 layout.total_bytes,
														 true,
														 TQ_DISTANCE_COSINE,
														 query,
														 8,
														 &optimistic_distance,
														 errmsg,
														 sizeof(errmsg)));
		best_distance = best_actual_page_distance(page, sizeof(page), &config, &lut);
		if (optimistic_distance > best_distance + 1e-5f)
		{
			saw_violation = true;
			tq_prod_lut_reset(&lut);
			break;
		}
		tq_prod_lut_reset(&lut);
		residual_radius = 0.0f;
	}

	assert(saw_violation);
}

int
main(void)
{
	test_page_summary_round_trip();
	test_summary_side_page_round_trip();
	test_optimistic_bound_prefers_clearly_better_page();
	test_summary_bound_matches_page_bound();
	test_empty_page_bound_is_safe();
	test_early_stop_requires_full_heap();
	test_full_heap_prunes_worse_page_bounds();
	test_code_domain_page_bounds_are_not_pruning_safe();
	return 0;
}
