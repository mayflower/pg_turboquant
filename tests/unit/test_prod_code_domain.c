#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "src/tq_codec_prod.h"
#include "src/tq_guc.h"
#include "src/tq_page.h"
#include "src/tq_scan.h"

typedef struct RankedDistance
{
	uint16_t	offset;
	float		distance;
} RankedDistance;

static float
dot_product(const float *left, const float *right, size_t len)
{
	float sum = 0.0f;
	size_t i = 0;

	for (i = 0; i < len; i++)
		sum += left[i] * right[i];

	return sum;
}

static int
compare_ranked_distance(const void *left, const void *right)
{
	const RankedDistance *lhs = (const RankedDistance *) left;
	const RankedDistance *rhs = (const RankedDistance *) right;

	if (lhs->distance < rhs->distance)
		return -1;
	if (lhs->distance > rhs->distance)
		return 1;
	if (lhs->offset < rhs->offset)
		return -1;
	if (lhs->offset > rhs->offset)
		return 1;
	return 0;
}

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

static float
blend_with_query_component(float query_component,
						   float noise_component,
						   float query_weight)
{
	return (query_weight * query_component) + ((1.0f - query_weight) * noise_component);
}

static void
seeded_overlap_vector(const float *query,
					  uint32_t seed,
					  float query_weight,
					  float *values,
					  size_t len)
{
	size_t i = 0;

	seeded_unit_vector(seed, values, len);
	for (i = 0; i < len; i++)
		values[i] = blend_with_query_component(query[i], values[i], query_weight);
	normalize(values, len);
}

static float
qjl_scale(uint32_t sketch_dimension)
{
	return 1.253314137315500251f / (float) sketch_dimension;
}

static uint32_t
test_unpack_bits(const uint8_t *packed, uint32_t bit_offset, uint32_t bit_count)
{
	uint32_t value = 0;
	uint32_t bit = 0;

	for (bit = 0; bit < bit_count; bit++)
	{
		uint32_t target_bit = bit_offset + bit;
		uint32_t byte_index = target_bit / 8u;
		uint32_t byte_shift = target_bit % 8u;

		if ((packed[byte_index] >> byte_shift) & 1u)
			value |= UINT32_C(1) << bit;
	}

	return value;
}

static float
quantized_qjl_score(const TqProdCodecConfig *config,
					  const TqProdLut *lut,
					  const uint8_t *packed,
					  size_t packed_len)
{
	TqProdPackedLayout layout;
	float base_sum = 0.0f;
	float residual_sum = 0.0f;
	float gamma = 0.0f;
	uint32_t dim = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(errmsg, 0, sizeof(errmsg));

	assert(lut->qjl_quantized_enabled);
	assert(lut->qjl_quantized_values != NULL);
	assert(tq_prod_packed_layout(config, &layout, errmsg, sizeof(errmsg)));
	assert(packed_len >= layout.total_bytes);
	assert(tq_prod_read_gamma(config, packed, packed_len, &gamma, errmsg, sizeof(errmsg)));

	for (dim = 0; dim < config->dimension; dim++)
	{
		uint32_t idx_code = 0;
		size_t index = 0;

		idx_code = test_unpack_bits(packed, dim * 3u, 3u);
		index = ((size_t) dim * (size_t) lut->level_count) + (size_t) idx_code;
		base_sum += lut->values[index];
	}

	for (dim = 0; dim < lut->qjl_dimension; dim++)
	{
		float sign = test_unpack_bits(packed + layout.idx_bytes, dim, 1u) ? 1.0f : -1.0f;
		float reconstructed = (float) lut->qjl_quantized_values[dim] * lut->qjl_quantization_scale;

		residual_sum += sign * reconstructed;
	}

	return base_sum + (gamma * residual_sum);
}

static void
test_qjl_structured_projection_is_deterministic_and_seeded(void)
{
	TqProdCodecConfig first = {.dimension = 6, .bits = 4, .qjl_seed = 17u, .qjl_dimension = 4};
	TqProdCodecConfig second = {.dimension = 6, .bits = 4, .qjl_seed = 23u, .qjl_dimension = 4};
	float input[6];
	float left[4];
	float right[4];
	float alternate[4];
	bool any_difference = false;
	size_t i = 0;
	char errmsg[256];

	memset(input, 0, sizeof(input));
	memset(left, 0, sizeof(left));
	memset(right, 0, sizeof(right));
	memset(alternate, 0, sizeof(alternate));

	seeded_unit_vector(41u, input, 6);
	assert(tq_prod_qjl_project(&first, input, left, 4, errmsg, sizeof(errmsg)));
	assert(tq_prod_qjl_project(&first, input, right, 4, errmsg, sizeof(errmsg)));
	assert(tq_prod_qjl_project(&second, input, alternate, 4, errmsg, sizeof(errmsg)));

	for (i = 0; i < 4; i++)
		assert(fabsf(left[i] - right[i]) <= 1e-6f);

	for (i = 0; i < 4; i++)
	{
		if (fabsf(left[i] - alternate[i]) > 1e-4f)
		{
			any_difference = true;
			break;
		}
	}

	assert(any_difference);
}

static void
test_qjl_lut_matches_structured_projection(void)
{
	TqProdCodecConfig config = {.dimension = 6, .bits = 4, .qjl_seed = 99u, .qjl_dimension = 4};
	TqProdLut lut;
	float query[6];
	float projection[4];
	uint32_t dim = 0;
	char errmsg[256];

	memset(&lut, 0, sizeof(lut));
	memset(query, 0, sizeof(query));
	memset(projection, 0, sizeof(projection));

	seeded_unit_vector(57u, query, 6);
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_qjl_project(&config, query, projection, 4, errmsg, sizeof(errmsg)));

	for (dim = 0; dim < config.qjl_dimension; dim++)
		assert(fabsf(lut.qjl_values[dim] - (qjl_scale(config.qjl_dimension) * projection[dim])) <= 1e-6f);

	tq_prod_lut_reset(&lut);
}

static void
test_quantized_qjl_lut_reconstructs_with_small_relative_error(void)
{
	TqProdCodecConfig config = {.dimension = 32, .bits = 4, .qjl_seed = 199u, .qjl_dimension = 32};
	TqProdLut lut;
	float query[32];
	float max_abs = 0.0f;
	uint32_t dim = 0;
	char errmsg[256];

	memset(&lut, 0, sizeof(lut));
	memset(query, 0, sizeof(query));

	seeded_unit_vector(157u, query, 32);
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(lut.qjl_quantized_enabled);
	assert(lut.qjl_quantized_values != NULL);
	assert(lut.qjl_quantization_scale > 0.0f);

	for (dim = 0; dim < lut.qjl_dimension; dim++)
	{
		float reconstructed = (float) lut.qjl_quantized_values[dim] * lut.qjl_quantization_scale;
		float error = fabsf(reconstructed - lut.qjl_values[dim]);

		if (fabsf(lut.qjl_values[dim]) > max_abs)
			max_abs = fabsf(lut.qjl_values[dim]);
		assert(error <= 1e-4f);
	}

	assert(max_abs > 0.0f);
	assert((lut.qjl_quantization_max_error / max_abs) <= (1.0f / 16384.0f));

	tq_prod_lut_reset(&lut);
}

static void
test_qjl_backprojection_matches_decode_residual_component(void)
{
	TqProdCodecConfig config = {.dimension = 6, .bits = 4, .qjl_seed = 7u, .qjl_dimension = 4};
	TqProdPackedLayout layout;
	uint8_t packed[64];
	uint8_t stage1_only[64];
	float input[6];
	float decoded[6];
	float stage1_decoded[6];
	float residual_component[6];
	float reconstructed[6];
	float gamma = 0.0f;
	uint32_t dim = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(packed, 0, sizeof(packed));
	memset(stage1_only, 0, sizeof(stage1_only));
	memset(input, 0, sizeof(input));
	memset(decoded, 0, sizeof(decoded));
	memset(stage1_decoded, 0, sizeof(stage1_decoded));
	memset(residual_component, 0, sizeof(residual_component));
	memset(reconstructed, 0, sizeof(reconstructed));

	seeded_unit_vector(73u, input, 6);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(layout.total_bytes <= sizeof(packed));
	assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
	assert(tq_prod_read_gamma(&config, packed, layout.total_bytes, &gamma, errmsg, sizeof(errmsg)));
	assert(tq_prod_decode(&config, packed, layout.total_bytes, decoded, 6, errmsg, sizeof(errmsg)));

	memcpy(stage1_only, packed, layout.total_bytes);
	memset(stage1_only + layout.idx_bytes, 0, layout.qjl_bytes + layout.gamma_bytes);
	assert(tq_prod_decode(&config, stage1_only, layout.total_bytes, stage1_decoded, 6, errmsg, sizeof(errmsg)));
	assert(tq_prod_qjl_backproject_signs(&config,
										 packed + layout.idx_bytes,
										 layout.qjl_bytes,
										 reconstructed,
										 6,
										 errmsg,
										 sizeof(errmsg)));

	for (dim = 0; dim < config.dimension; dim++)
	{
		residual_component[dim] = decoded[dim] - stage1_decoded[dim];
		assert(fabsf(residual_component[dim]
					 - (gamma * qjl_scale(config.qjl_dimension) * reconstructed[dim])) <= 1e-5f);
	}
}

static void
test_quantized_qjl_score_stays_close_to_float_reference(void)
{
	TqProdCodecConfig config = {.dimension = 32, .bits = 4, .qjl_seed = 211u, .qjl_dimension = 32};
	TqProdPackedLayout layout;
	TqProdLut lut;
	uint8_t packed[256];
	float query[32];
	uint32_t seed = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));

	seeded_unit_vector(701u, query, 32);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(layout.total_bytes <= sizeof(packed));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(lut.qjl_quantized_enabled);

	for (seed = 0; seed < 64u; seed++)
	{
		float input[32];
		float float_score = 0.0f;
		float quantized_score = 0.0f;

		memset(input, 0, sizeof(input));
		seeded_unit_vector(1701u + seed, input, 32);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut(&config, &lut, packed, layout.total_bytes,
										   &float_score, errmsg, sizeof(errmsg)));
		quantized_score = quantized_qjl_score(&config, &lut, packed, layout.total_bytes);
		assert(fabsf(quantized_score - float_score) <= 3e-4f);
	}

	tq_prod_lut_reset(&lut);
}

static void
test_qjl_sketch_dimension_controls_packed_layout(void)
{
	TqProdCodecConfig config = {.dimension = 17, .bits = 4, .qjl_seed = 101u, .qjl_dimension = 9};
	TqProdPackedLayout layout;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(layout.idx_bytes == 7);
	assert(layout.qjl_bytes == 2);
	assert(layout.gamma_bytes == 4);
	assert(layout.total_bytes == 13);
}

static void
test_quantized_qjl_retains_seeded_ordering(void)
{
	TqProdCodecConfig config = {.dimension = 32, .bits = 4, .qjl_seed = 307u, .qjl_dimension = 32};
	TqProdPackedLayout layout;
	TqProdLut lut;
	uint8_t packed[16][256];
	float query[32];
	RankedDistance float_ranking[16];
	RankedDistance quantized_ranking[16];
	size_t i = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));
	memset(float_ranking, 0, sizeof(float_ranking));
	memset(quantized_ranking, 0, sizeof(quantized_ranking));

	seeded_unit_vector(901u, query, 32);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(layout.total_bytes <= sizeof(packed[0]));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(lut.qjl_quantized_enabled);

	for (i = 0; i < 16u; i++)
	{
		float input[32];
		float float_score = 0.0f;
		float quantized_score = 0.0f;

		memset(input, 0, sizeof(input));
		seeded_unit_vector((uint32_t) (1901u + i), input, 32);
		assert(tq_prod_encode(&config, input, packed[i], layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut(&config, &lut, packed[i], layout.total_bytes,
										   &float_score, errmsg, sizeof(errmsg)));
		quantized_score = quantized_qjl_score(&config, &lut, packed[i], layout.total_bytes);
		float_ranking[i].offset = (uint16_t) (i + 1u);
		float_ranking[i].distance = -float_score;
		quantized_ranking[i].offset = (uint16_t) (i + 1u);
		quantized_ranking[i].distance = -quantized_score;
	}

	qsort(float_ranking, 16u, sizeof(float_ranking[0]), compare_ranked_distance);
	qsort(quantized_ranking, 16u, sizeof(quantized_ranking[0]), compare_ranked_distance);

	for (i = 0; i < 10u; i++)
		assert(float_ranking[i].offset == quantized_ranking[i].offset);

	tq_prod_lut_reset(&lut);
}

static void
test_code_domain_score_matches_decode_baseline(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	uint8_t packed[64];
	float query[8];
	float decoded[8];
	float code_score = 0.0f;
	float decode_score = 0.0f;
	char errmsg[256];
	uint32_t vector_seed = 0;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));
	memset(decoded, 0, sizeof(decoded));

	seeded_unit_vector(7u, query, 8);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(layout.total_bytes <= sizeof(packed));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));

	for (vector_seed = 11u; vector_seed < 27u; vector_seed++)
	{
		float input[8];

		memset(input, 0, sizeof(input));
		seeded_unit_vector(vector_seed, input, 8);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut(&config, &lut, packed, layout.total_bytes,
										   &code_score, errmsg, sizeof(errmsg)));
		assert(tq_prod_decode(&config, packed, layout.total_bytes, decoded, 8, errmsg, sizeof(errmsg)));
		decode_score = dot_product(query, decoded, 8);
		assert(fabsf(code_score - decode_score) <= 0.03f);
	}

	tq_prod_lut_reset(&lut);
}

static void
test_ranking_matches_decode_baseline_for_normalized_cosine_and_ip(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t packed[64];
	float query[8];
	float decoded[8];
	RankedDistance expected_cosine[4];
	RankedDistance expected_ip[4];
	TqCandidateHeap cosine_heap;
	TqCandidateHeap ip_heap;
	TqCandidateEntry entry;
	TqScanStats stats;
	char errmsg[256];
	uint16_t lane = 0;
	size_t i = 0;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));
	memset(decoded, 0, sizeof(decoded));
	memset(expected_cosine, 0, sizeof(expected_cosine));
	memset(expected_ip, 0, sizeof(expected_ip));
	memset(&cosine_heap, 0, sizeof(cosine_heap));
	memset(&ip_heap, 0, sizeof(ip_heap));
	memset(&entry, 0, sizeof(entry));
	memset(&stats, 0, sizeof(stats));

	seeded_unit_vector(3u, query, 8);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	params.lane_count = 4;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_candidate_heap_init(&cosine_heap, 4));
	assert(tq_candidate_heap_init(&ip_heap, 4));

	for (i = 0; i < 4; i++)
	{
		float input[8];
		float ip_score = 0.0f;

		seeded_unit_vector((uint32_t) (100 + i), input, 8);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_decode(&config, packed, layout.total_bytes, decoded, 8, errmsg, sizeof(errmsg)));
		ip_score = dot_product(query, decoded, 8);
		expected_cosine[i].offset = (uint16_t) (i + 1);
		expected_cosine[i].distance = 1.0f - ip_score;
		expected_ip[i].offset = (uint16_t) (i + 1);
		expected_ip[i].distance = -ip_score;

		assert(tq_batch_page_append_lane(page, sizeof(page),
										 &(TqTid){.block_number = 1, .offset_number = (uint16_t) (i + 1)},
										 &lane, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, layout.total_bytes,
									  errmsg, sizeof(errmsg)));
	}

	qsort(expected_cosine, 4, sizeof(expected_cosine[0]), compare_ranked_distance);
	qsort(expected_ip, 4, sizeof(expected_ip[0]), compare_ranked_distance);

	tq_prod_decode_counter_reset();
	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1);
	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, true, TQ_DISTANCE_COSINE, &lut,
								   query, 8, &cosine_heap, NULL, errmsg, sizeof(errmsg)));
	assert(tq_prod_decode_counter_get() == 0);
	tq_scan_stats_snapshot(&stats);
	assert(stats.score_mode == TQ_SCAN_SCORE_MODE_CODE_DOMAIN);
	assert(stats.decoded_vector_count == 0);

	tq_prod_decode_counter_reset();
	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1);
	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, true, TQ_DISTANCE_IP, &lut,
								   query, 8, &ip_heap, NULL, errmsg, sizeof(errmsg)));
	assert(tq_prod_decode_counter_get() == 0);
	tq_scan_stats_snapshot(&stats);
	assert(stats.score_mode == TQ_SCAN_SCORE_MODE_CODE_DOMAIN);
	assert(stats.decoded_vector_count == 0);

	for (i = 0; i < 4; i++)
	{
		assert(tq_candidate_heap_pop_best(&cosine_heap, &entry));
		assert(entry.tid.offset_number == expected_cosine[i].offset);
	}

	for (i = 0; i < 4; i++)
	{
		assert(tq_candidate_heap_pop_best(&ip_heap, &entry));
		assert(entry.tid.offset_number == expected_ip[i].offset);
	}

	tq_candidate_heap_reset(&cosine_heap);
	tq_candidate_heap_reset(&ip_heap);
	tq_prod_lut_reset(&lut);
}

static void
test_distance_conversion_contract(void)
{
	float distance = 0.0f;
	char errmsg[256];

	assert(tq_metric_distance_from_ip_score(TQ_DISTANCE_COSINE, 0.75f, &distance, errmsg, sizeof(errmsg)));
	assert(fabsf(distance - 0.25f) <= 1e-6f);

	assert(tq_metric_distance_from_ip_score(TQ_DISTANCE_IP, 0.75f, &distance, errmsg, sizeof(errmsg)));
	assert(fabsf(distance + 0.75f) <= 1e-6f);

	assert(tq_metric_distance_from_ip_score(TQ_DISTANCE_L2, 0.75f, &distance, errmsg, sizeof(errmsg)));
	assert(fabsf(distance - 0.5f) <= 1e-6f);
}

static void
test_force_decode_diagnostics_switches_active_scoring_path(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t packed[64];
	float query[8];
	float vector[8];
	TqCandidateHeap heap;
	TqScanStats stats;
	uint16_t lane = 0;
	char errmsg[256];
	bool previous_force_decode = tq_guc_force_decode_score_diagnostics;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(&stats, 0, sizeof(stats));
	memset(query, 0, sizeof(query));
	memset(vector, 0, sizeof(vector));

	assert(tq_candidate_heap_init(&heap, 4));
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	params.lane_count = 4;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;
	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));

	seeded_unit_vector(501, query, 8);
	seeded_unit_vector(502, vector, 8);
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_encode(&config, vector, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page),
									 &(TqTid){.block_number = 1, .offset_number = 1},
									 &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, layout.total_bytes,
								  errmsg, sizeof(errmsg)));

	tq_guc_force_decode_score_diagnostics = true;
	tq_prod_decode_counter_reset();
	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1);
	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, true, TQ_DISTANCE_COSINE, &lut,
								   query, 8, &heap, NULL, errmsg, sizeof(errmsg)));
	tq_scan_stats_snapshot(&stats);

	assert(stats.score_mode == TQ_SCAN_SCORE_MODE_DECODE);
	assert(stats.decoded_vector_count == 1);
	assert(tq_prod_decode_counter_get() == 1);

	tq_guc_force_decode_score_diagnostics = previous_force_decode;
	tq_candidate_heap_reset(&heap);
	tq_prod_lut_reset(&lut);
}

static void
test_batch_page_code_view_matches_copied_code(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout layout;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t packed[64];
	uint8_t copied[64];
	float vector[8];
	const uint8_t *view = NULL;
	size_t view_len = 0;
	uint16_t lane = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(packed, 0, sizeof(packed));
	memset(copied, 0, sizeof(copied));
	memset(vector, 0, sizeof(vector));

	seeded_unit_vector(601u, vector, 8);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(layout.total_bytes <= sizeof(packed));
	assert(tq_prod_encode(&config, vector, packed, layout.total_bytes, errmsg, sizeof(errmsg)));

	params.lane_count = 1;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;
	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page,
									 sizeof(page),
									 &(TqTid){.block_number = 1, .offset_number = 1},
									 &lane,
									 errmsg,
									 sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, layout.total_bytes,
								  errmsg, sizeof(errmsg)));

	assert(tq_batch_page_code_view(page,
								   sizeof(page),
								   lane,
								   &view,
								   &view_len,
								   errmsg,
								   sizeof(errmsg)));
	assert(tq_batch_page_get_code(page,
								  sizeof(page),
								  lane,
								  copied,
								  sizeof(copied),
								  errmsg,
								  sizeof(errmsg)));
	assert(view != NULL);
	assert(view_len == layout.total_bytes);
	assert(memcmp(view, copied, view_len) == 0);
}

static void
test_scan_scratch_reuse_across_repeated_page_scans(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t packed[64];
	float query[8];
	TqCandidateHeap primary_heap;
	TqCandidateHeap shadow_heap;
	TqCandidateEntry entry;
	TqScanScratch scratch;
	uint16_t lane = 0;
	size_t iteration = 0;
	size_t lane_index = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));
	memset(&primary_heap, 0, sizeof(primary_heap));
	memset(&shadow_heap, 0, sizeof(shadow_heap));
	memset(&entry, 0, sizeof(entry));
	memset(&scratch, 0, sizeof(scratch));

	seeded_unit_vector(701u, query, 8);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_candidate_heap_init(&primary_heap, 4));
	assert(tq_candidate_heap_init(&shadow_heap, 4));

	params.lane_count = 4;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;
	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));

	for (lane_index = 0; lane_index < 4; lane_index++)
	{
		float vector[8];

		memset(vector, 0, sizeof(vector));
		seeded_unit_vector((uint32_t) (800u + lane_index), vector, 8);
		assert(tq_prod_encode(&config, vector, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_append_lane(page,
										 sizeof(page),
										 &(TqTid){.block_number = 1, .offset_number = (uint16_t) (lane_index + 1u)},
										 &lane,
										 errmsg,
										 sizeof(errmsg)));
		assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, layout.total_bytes,
									  errmsg, sizeof(errmsg)));
	}

	for (iteration = 0; iteration < 32; iteration++)
	{
		tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1);
		assert(tq_batch_page_scan_prod_with_scratch(page,
													sizeof(page),
													&config,
													true,
													TQ_DISTANCE_IP,
													&lut,
													query,
													8,
													&primary_heap,
													&shadow_heap,
													&scratch,
													errmsg,
													sizeof(errmsg)));
		assert(tq_candidate_heap_pop_best(&primary_heap, &entry));
		assert(entry.tid.offset_number >= 1u && entry.tid.offset_number <= 4u);
		assert(tq_candidate_heap_pop_best(&shadow_heap, &entry));
		assert(entry.tid.offset_number >= 1u && entry.tid.offset_number <= 4u);
		primary_heap.count = 0;
		primary_heap.pop_index = 0;
		primary_heap.sorted = false;
		shadow_heap.count = 0;
		shadow_heap.pop_index = 0;
		shadow_heap.sorted = false;
	}

	assert(scratch.scratch_allocations == 1u);
	assert(scratch.decoded_buffer_reuses >= 31u);
	assert(scratch.code_view_uses == (size_t) (32u * 4u));
	assert(scratch.code_copy_uses == 0u);

	tq_scan_scratch_reset(&scratch);
	tq_candidate_heap_reset(&primary_heap);
	tq_candidate_heap_reset(&shadow_heap);
	tq_prod_lut_reset(&lut);
}

static void
test_page_local_selection_preserves_single_page_top_k(void)
{
	TqProdCodecConfig config = {.dimension = 32, .bits = 4, .qjl_seed = 409u, .qjl_dimension = 32};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t packed[256];
	float query[32];
	const float query_weights[8] = {0.99f, 0.97f, 0.95f, 0.93f, 0.45f, 0.35f, 0.25f, 0.15f};
	RankedDistance expected[8];
	TqCandidateHeap heap;
	TqCandidateEntry entry;
	TqScanScratch scratch;
	TqScanStats stats;
	uint16_t lane = 0;
	size_t i = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));
	memset(expected, 0, sizeof(expected));
	memset(&heap, 0, sizeof(heap));
	memset(&entry, 0, sizeof(entry));
	memset(&scratch, 0, sizeof(scratch));
	memset(&stats, 0, sizeof(stats));

	seeded_unit_vector(901u, query, 32);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(layout.total_bytes <= sizeof(packed));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_candidate_heap_init(&heap, 4));

	params.lane_count = 8;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;
	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));

	for (i = 0; i < 8u; i++)
	{
		float vector[32];
		float decoded[32];
		float ip_score = 0.0f;

		memset(vector, 0, sizeof(vector));
		memset(decoded, 0, sizeof(decoded));
		seeded_overlap_vector(query, (uint32_t) (1100u + i), query_weights[i], vector, 32);
		assert(tq_prod_encode(&config, vector, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_decode(&config, packed, layout.total_bytes, decoded, 32, errmsg, sizeof(errmsg)));
		ip_score = dot_product(query, decoded, 32);
		expected[i].offset = (uint16_t) (i + 1u);
		expected[i].distance = 1.0f - ip_score;
		assert(tq_batch_page_append_lane(page,
										 sizeof(page),
										 &(TqTid){.block_number = 1u, .offset_number = (uint16_t) (i + 1u)},
										 &lane,
										 errmsg,
										 sizeof(errmsg)));
		assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, layout.total_bytes,
									  errmsg, sizeof(errmsg)));
	}

	qsort(expected, 8u, sizeof(expected[0]), compare_ranked_distance);

	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1u);
	assert(tq_batch_page_scan_prod_with_scratch(page,
												sizeof(page),
												&config,
												true,
												TQ_DISTANCE_COSINE,
												&lut,
												query,
												32u,
												&heap,
												NULL,
												&scratch,
												errmsg,
												sizeof(errmsg)));
	tq_scan_stats_snapshot(&stats);

	for (i = 0; i < 4u; i++)
	{
		assert(tq_candidate_heap_pop_best(&heap, &entry));
		assert(entry.tid.offset_number == expected[i].offset);
	}

	assert(stats.visited_page_count == 1u);
	assert(stats.visited_code_count == 8u);
	assert(stats.local_candidate_heap_insert_count == 4u);
	assert(stats.local_candidate_heap_insert_count
		   + stats.local_candidate_heap_replace_count
		   + stats.local_candidate_heap_reject_count
		   == stats.visited_code_count);
	assert(stats.local_candidate_merge_count == 4u);
	assert(stats.candidate_heap_insert_count == 4u);
	assert(stats.candidate_heap_replace_count == 0u);
	assert(stats.candidate_heap_reject_count == 0u);
	assert(stats.candidate_heap_insert_count < stats.visited_code_count);

	tq_scan_scratch_reset(&scratch);
	tq_candidate_heap_reset(&heap);
	tq_prod_lut_reset(&lut);
}

static void
test_page_local_selection_retains_overlap_fixture_across_pages(void)
{
	TqProdCodecConfig config = {.dimension = 32, .bits = 4, .qjl_seed = 557u, .qjl_dimension = 32};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqBatchPageParams params;
	uint8_t pages[3][TQ_DEFAULT_BLOCK_SIZE];
	uint8_t packed[256];
	float query[32];
	const float query_weights[3][8] = {
		{0.998f, 0.994f, 0.990f, 0.986f, 0.42f, 0.36f, 0.30f, 0.24f},
		{0.997f, 0.993f, 0.989f, 0.985f, 0.48f, 0.40f, 0.20f, 0.12f},
		{0.996f, 0.992f, 0.60f, 0.54f, 0.18f, 0.10f, 0.05f, 0.01f},
	};
	RankedDistance expected[24];
	TqCandidateHeap heap;
	TqCandidateEntry entry;
	TqScanScratch scratch;
	TqScanStats stats;
	size_t expected_count = 0;
	size_t page_index = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&params, 0, sizeof(params));
	memset(pages, 0, sizeof(pages));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));
	memset(expected, 0, sizeof(expected));
	memset(&heap, 0, sizeof(heap));
	memset(&entry, 0, sizeof(entry));
	memset(&scratch, 0, sizeof(scratch));
	memset(&stats, 0, sizeof(stats));

	seeded_unit_vector(1301u, query, 32);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(layout.total_bytes <= sizeof(packed));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_candidate_heap_init(&heap, 4));

	params.lane_count = 8;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;

	for (page_index = 0; page_index < 3u; page_index++)
	{
		uint16_t lane = 0;
		size_t lane_index = 0;

		assert(tq_batch_page_init(pages[page_index], sizeof(pages[page_index]), &params, errmsg, sizeof(errmsg)));
		for (lane_index = 0; lane_index < 8u; lane_index++)
		{
			float vector[32];
			float decoded[32];
			float ip_score = 0.0f;

			memset(vector, 0, sizeof(vector));
			memset(decoded, 0, sizeof(decoded));
			seeded_overlap_vector(query,
								  (uint32_t) (2000u + (page_index * 16u) + lane_index),
								  query_weights[page_index][lane_index],
								  vector,
								  32);
			assert(tq_prod_encode(&config, vector, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
			assert(tq_prod_decode(&config, packed, layout.total_bytes, decoded, 32, errmsg, sizeof(errmsg)));
			ip_score = dot_product(query, decoded, 32);
			expected[expected_count].offset = (uint16_t) ((page_index * 8u) + lane_index + 1u);
			expected[expected_count].distance = 1.0f - ip_score;
			expected_count += 1u;
			assert(tq_batch_page_append_lane(
				pages[page_index],
				sizeof(pages[page_index]),
				&(TqTid){.block_number = (uint32_t) (page_index + 1u), .offset_number = (uint16_t) (lane_index + 1u)},
				&lane,
				errmsg,
				sizeof(errmsg)));
			assert(tq_batch_page_set_code(pages[page_index],
									  sizeof(pages[page_index]),
									  lane,
									  packed,
									  layout.total_bytes,
									  errmsg,
									  sizeof(errmsg)));
		}
	}

	qsort(expected, expected_count, sizeof(expected[0]), compare_ranked_distance);

	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1u);
	for (page_index = 0; page_index < 3u; page_index++)
	{
		assert(tq_batch_page_scan_prod_with_scratch(pages[page_index],
												sizeof(pages[page_index]),
												&config,
												true,
												TQ_DISTANCE_COSINE,
												&lut,
												query,
												32u,
												&heap,
												NULL,
												&scratch,
												errmsg,
												sizeof(errmsg)));
	}
	tq_scan_stats_snapshot(&stats);

	for (page_index = 0; page_index < 4u; page_index++)
	{
		uint32_t expected_block = ((uint32_t) expected[page_index].offset - 1u) / 8u + 1u;
		uint16_t expected_offset = (uint16_t) ((((uint32_t) expected[page_index].offset - 1u) % 8u) + 1u);

		assert(tq_candidate_heap_pop_best(&heap, &entry));
		assert(entry.tid.block_number == expected_block);
		assert(entry.tid.offset_number == expected_offset);
	}

	assert(stats.visited_page_count == 3u);
	assert(stats.visited_code_count == 24u);
	assert(stats.local_candidate_heap_insert_count == 12u);
	assert(stats.local_candidate_heap_insert_count
		   + stats.local_candidate_heap_replace_count
		   + stats.local_candidate_heap_reject_count
		   == stats.visited_code_count);
	assert(stats.local_candidate_merge_count == 12u);
	assert(stats.local_candidate_merge_count < stats.visited_code_count);
	assert(stats.candidate_heap_insert_count
		   + stats.candidate_heap_replace_count
		   + stats.candidate_heap_reject_count
		   == stats.local_candidate_merge_count);

	tq_scan_scratch_reset(&scratch);
	tq_candidate_heap_reset(&heap);
	tq_prod_lut_reset(&lut);
}

int
main(void)
{
	test_qjl_structured_projection_is_deterministic_and_seeded();
	test_qjl_lut_matches_structured_projection();
	test_quantized_qjl_lut_reconstructs_with_small_relative_error();
	test_qjl_backprojection_matches_decode_residual_component();
	test_quantized_qjl_score_stays_close_to_float_reference();
	test_qjl_sketch_dimension_controls_packed_layout();
	test_quantized_qjl_retains_seeded_ordering();
	test_code_domain_score_matches_decode_baseline();
	test_ranking_matches_decode_baseline_for_normalized_cosine_and_ip();
	test_distance_conversion_contract();
	test_force_decode_diagnostics_switches_active_scoring_path();
	test_batch_page_code_view_matches_copied_code();
	test_scan_scratch_reuse_across_repeated_page_scans();
	test_page_local_selection_preserves_single_page_top_k();
	test_page_local_selection_retains_overlap_fixture_across_pages();
	return 0;
}
