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

/*
 * Transpose nibbles from candidate-major (nibbles[c * dim + d]) to
 * dimension-major (out[d * 16 + c]) layout for block-16 scorers.
 */
static void
transpose_nibbles_to_dim_major(const uint8_t *candidate_major,
							   uint8_t *dim_major,
							   uint32_t candidate_count,
							   uint32_t dimension)
{
	uint32_t c, d;

	memset(dim_major, 0, (size_t) dimension * 16u);
	for (c = 0; c < candidate_count; c++)
		for (d = 0; d < dimension; d++)
			dim_major[(size_t) d * 16u + c] = candidate_major[(size_t) c * dimension + d];
}

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
								   query, 8, false, 0, &cosine_heap, NULL, errmsg, sizeof(errmsg)));
	assert(tq_prod_decode_counter_get() == 0);
	tq_scan_stats_snapshot(&stats);
	assert(stats.score_mode == TQ_SCAN_SCORE_MODE_CODE_DOMAIN);
	assert(stats.decoded_vector_count == 0);

	tq_prod_decode_counter_reset();
	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1);
	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, true, TQ_DISTANCE_IP, &lut,
								   query, 8, false, 0, &ip_heap, NULL, errmsg, sizeof(errmsg)));
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
								   query, 8, false, 0, &heap, NULL, errmsg, sizeof(errmsg)));
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
													false,
													0,
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
												false,
												0,
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
												false,
												0,
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

static void
test_lut16_build_rejects_unsupported_shapes(void)
{
	TqProdCodecConfig config_bad_bits = {.dimension = 32, .bits = 3, .qjl_dimension = 32};
	TqProdCodecConfig config_bad_dim = {.dimension = 31, .bits = 4, .qjl_dimension = 31};
	TqProdCodecConfig config_bad_qjl = {.dimension = 32, .bits = 4, .qjl_dimension = 16};
	TqProdLut16 lut16;
	char errmsg[256];

	memset(&lut16, 0, sizeof(lut16));
	memset(errmsg, 0, sizeof(errmsg));

	assert(!tq_prod_lut16_is_supported(&config_bad_bits, errmsg, sizeof(errmsg)));
	assert(!tq_prod_lut16_is_supported(&config_bad_dim, errmsg, sizeof(errmsg)));
	assert(!tq_prod_lut16_is_supported(&config_bad_qjl, errmsg, sizeof(errmsg)));

	/* lut16_build should also reject, since it checks support internally */
	assert(!tq_prod_lut16_build(&config_bad_bits, NULL, &lut16, errmsg, sizeof(errmsg)));
}

static void
test_lut16_scalar_matches_code_domain_scorer(void)
{
	TqProdCodecConfig config = {.dimension = 32, .bits = 4, .qjl_seed = 997u, .qjl_dimension = 32};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	uint8_t packed[256];
	uint8_t nibbles[32];
	uint8_t dim_major_nibbles[32 * 16];
	float query[32];
	float input[32];
	float gamma = 0.0f;
	float reference_score = 0.0f;
	float lut16_score = 0.0f;
	uint32_t vec_idx = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));
	memset(packed, 0, sizeof(packed));
	memset(nibbles, 0, sizeof(nibbles));

	seeded_unit_vector(101u, query, 32);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_build(&config, &lut, &lut16, errmsg, sizeof(errmsg)));

	for (vec_idx = 0; vec_idx < 8; vec_idx++)
	{
		memset(packed, 0, sizeof(packed));
		seeded_unit_vector(200u + vec_idx, input, 32);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_read_gamma(&config, packed, layout.total_bytes, &gamma, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut(&config, &lut, packed, layout.total_bytes,
										   &reference_score, errmsg, sizeof(errmsg)));
		assert(tq_prod_extract_nibbles(&config, packed, layout.total_bytes,
									   nibbles, sizeof(nibbles), errmsg, sizeof(errmsg)));
		transpose_nibbles_to_dim_major(nibbles, dim_major_nibbles, 1, 32);
		assert(tq_prod_score_block16_scalar(&lut16, dim_major_nibbles, &gamma, 1, &lut16_score,
											errmsg, sizeof(errmsg)));

		assert(fabsf(lut16_score - reference_score) < 1e-5f);
	}

	tq_prod_lut16_reset(&lut16);
	tq_prod_lut_reset(&lut);
}

static void
test_lut16_block16_preserves_score_ordering(void)
{
	const uint32_t N = 16;
	TqProdCodecConfig config = {.dimension = 32, .bits = 4, .qjl_seed = 443u, .qjl_dimension = 32};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	uint8_t packed[256];
	uint8_t all_nibbles[16 * 32];
	uint8_t dim_major_nibbles[32 * 16];
	float gammas[16];
	float reference_scores[16];
	float block_scores[16];
	float query[32];
	float input[32];
	uint32_t i = 0;
	uint32_t j = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));
	memset(all_nibbles, 0, sizeof(all_nibbles));

	seeded_unit_vector(501u, query, 32);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_build(&config, &lut, &lut16, errmsg, sizeof(errmsg)));

	/* Encode N vectors, extract nibbles and gammas, score with reference */
	for (i = 0; i < N; i++)
	{
		memset(packed, 0, sizeof(packed));
		seeded_unit_vector(600u + i, input, 32);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_read_gamma(&config, packed, layout.total_bytes, &gammas[i], errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut(&config, &lut, packed, layout.total_bytes,
										   &reference_scores[i], errmsg, sizeof(errmsg)));
		assert(tq_prod_extract_nibbles(&config, packed, layout.total_bytes,
									   all_nibbles + (size_t) i * 32u, 32, errmsg, sizeof(errmsg)));
	}

	transpose_nibbles_to_dim_major(all_nibbles, dim_major_nibbles, N, 32);

	/* Score all 16 at once via block scorer */
	assert(tq_prod_score_block16_scalar(&lut16, dim_major_nibbles, gammas, N, block_scores,
										errmsg, sizeof(errmsg)));

	/* Verify exact score match */
	for (i = 0; i < N; i++)
		assert(fabsf(block_scores[i] - reference_scores[i]) < 1e-5f);

	/* Verify ordering is preserved: same relative order */
	for (i = 0; i < N; i++)
		for (j = i + 1; j < N; j++)
		{
			if (reference_scores[i] > reference_scores[j])
				assert(block_scores[i] > block_scores[j] - 1e-6f);
			else if (reference_scores[i] < reference_scores[j])
				assert(block_scores[i] < block_scores[j] + 1e-6f);
		}

	tq_prod_lut16_reset(&lut16);
	tq_prod_lut_reset(&lut);
}

static void
test_scratch_transpose_preserves_lane_order_and_tid(void)
{
	TqProdCodecConfig config = {.dimension = 32, .bits = 4, .qjl_seed = 811u, .qjl_dimension = 32};
	TqProdPackedLayout layout;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t packed[256];
	TqScratchBlock16Set set;
	TqTid expected_tids[32];
	uint32_t lane_count = 0;
	uint32_t i = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(packed, 0, sizeof(packed));
	memset(&set, 0, sizeof(set));
	memset(expected_tids, 0, sizeof(expected_tids));

	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	params.lane_count = 20;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;
	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));

	for (i = 0; i < params.lane_count; i++)
	{
		float input[32];
		uint16_t lane_idx = 0;

		seeded_unit_vector(900u + i, input, 32);
		memset(packed, 0, sizeof(packed));
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_append_lane(page, sizeof(page),
			&(TqTid){.block_number = 10u, .offset_number = (uint16_t)(i + 1u)},
			&lane_idx, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_set_code(page, sizeof(page), lane_idx, packed,
									  layout.total_bytes, errmsg, sizeof(errmsg)));
		expected_tids[i].block_number = 10u;
		expected_tids[i].offset_number = (uint16_t)(i + 1u);
		lane_count++;
	}

	assert(tq_scratch_block16_set_init(&set, config.dimension, lane_count, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_transpose_block16(page, sizeof(page), &config, &set, errmsg, sizeof(errmsg)));

	/* Verify total candidates match live lanes */
	assert(set.total_candidates == lane_count);
	assert(set.block_count == 2);  /* 20 candidates → 2 blocks of 16+4 */
	assert(set.blocks[0].count == 16);
	assert(set.blocks[1].count == 4);

	/* Verify TID order matches insertion order */
	for (i = 0; i < set.total_candidates; i++)
	{
		assert(set.tid_storage[i].block_number == expected_tids[i].block_number);
		assert(set.tid_storage[i].offset_number == expected_tids[i].offset_number);
	}

	tq_scratch_block16_set_reset(&set);
}

static void
test_scratch_transpose_block16_scores_match_per_lane(void)
{
	TqProdCodecConfig config = {.dimension = 32, .bits = 4, .qjl_seed = 557u, .qjl_dimension = 32};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t packed[256];
	float query[32];
	float reference_scores[32];
	TqScratchBlock16Set set;
	uint32_t i = 0;
	uint32_t b = 0;
	uint32_t c = 0;
	uint32_t score_idx = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(&set, 0, sizeof(set));

	seeded_unit_vector(701u, query, 32);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_build(&config, &lut, &lut16, errmsg, sizeof(errmsg)));

	params.lane_count = 24;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;
	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));

	for (i = 0; i < params.lane_count; i++)
	{
		float input[32];
		uint16_t lane_idx = 0;

		seeded_unit_vector(800u + i, input, 32);
		memset(packed, 0, sizeof(packed));
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_append_lane(page, sizeof(page),
			&(TqTid){.block_number = 5u, .offset_number = (uint16_t)(i + 1u)},
			&lane_idx, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_set_code(page, sizeof(page), lane_idx, packed,
									  layout.total_bytes, errmsg, sizeof(errmsg)));

		/* Score each lane individually for reference */
		assert(tq_prod_score_code_from_lut(&config, &lut, packed, layout.total_bytes,
										   &reference_scores[i], errmsg, sizeof(errmsg)));
	}

	/* Transpose and score via block16 */
	assert(tq_scratch_block16_set_init(&set, config.dimension, params.lane_count, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_transpose_block16(page, sizeof(page), &config, &set, errmsg, sizeof(errmsg)));

	score_idx = 0;
	for (b = 0; b < set.block_count; b++)
	{
		float block_scores[TQ_BLOCK16_MAX_CANDIDATES];

		memset(block_scores, 0, sizeof(block_scores));
		assert(tq_prod_score_block16_scalar(&lut16,
											set.blocks[b].nibbles,
											set.blocks[b].gammas,
											set.blocks[b].count,
											block_scores,
											errmsg, sizeof(errmsg)));

		for (c = 0; c < set.blocks[b].count; c++)
		{
			assert(fabsf(block_scores[c] - reference_scores[score_idx]) < 1e-4f);
			score_idx++;
		}
	}

	assert(score_idx == params.lane_count);

	tq_scratch_block16_set_reset(&set);
	tq_prod_lut16_reset(&lut16);
	tq_prod_lut_reset(&lut);
}

static void
test_scratch_transpose_partial_block_pads_safely(void)
{
	TqProdCodecConfig config = {.dimension = 16, .bits = 4, .qjl_seed = 331u, .qjl_dimension = 16};
	TqProdPackedLayout layout;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t packed[128];
	TqScratchBlock16Set set;
	uint32_t i = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(&set, 0, sizeof(set));

	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	params.lane_count = 5;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;
	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));

	for (i = 0; i < params.lane_count; i++)
	{
		float input[16];
		uint16_t lane_idx = 0;

		seeded_unit_vector(400u + i, input, 16);
		memset(packed, 0, sizeof(packed));
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_append_lane(page, sizeof(page),
			&(TqTid){.block_number = 2u, .offset_number = (uint16_t)(i + 1u)},
			&lane_idx, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_set_code(page, sizeof(page), lane_idx, packed,
									  layout.total_bytes, errmsg, sizeof(errmsg)));
	}

	assert(tq_scratch_block16_set_init(&set, config.dimension, params.lane_count, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_transpose_block16(page, sizeof(page), &config, &set, errmsg, sizeof(errmsg)));

	/* 5 candidates → 1 block with count=5 */
	assert(set.block_count == 1);
	assert(set.blocks[0].count == 5);
	assert(set.total_candidates == 5);

	/* Verify each nibble is valid (0..15) */
	for (i = 0; i < set.blocks[0].count * config.dimension; i++)
		assert(set.blocks[0].nibbles[i] < 16u);

	tq_scratch_block16_set_reset(&set);
}

static void
test_quantized_lut16_reconstruction_error_is_small(void)
{
	TqProdCodecConfig config = {.dimension = 64, .bits = 4, .qjl_seed = 131u, .qjl_dimension = 64};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	float query[64];
	uint32_t dim = 0;
	uint32_t nib = 0;
	float max_rel_error = 0.0f;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));

	seeded_unit_vector(421u, query, 64);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_build(&config, &lut, &lut16, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_quantize(&lut16, errmsg, sizeof(errmsg)));

	{
		float global_max = lut16.base_global_scale * 127.0f;

		for (dim = 0; dim < config.dimension; dim++)
		{
			for (nib = 0; nib < 16u; nib++)
			{
				size_t idx = (size_t) dim * 16u + nib;
				float original = lut16.base_values[idx];
				float reconstructed = (float) lut16.base_quantized[idx] * lut16.base_global_scale;
				float abs_error = fabsf(original - reconstructed);
				/* Normalize error against global range (avoids division by near-zero) */
				float rel_error = (global_max > 1e-8f) ? abs_error / global_max : abs_error;

				if (rel_error > max_rel_error)
					max_rel_error = rel_error;
			}
		}
	}

	/* int8 quantization with 127 levels: max error per entry is ~1/127 ≈ 0.8% of range */
	assert(max_rel_error < 0.02f);

	tq_prod_lut16_reset(&lut16);
	tq_prod_lut_reset(&lut);
}

static void
test_quantized_block16_fused_score_stays_close_to_float(void)
{
	const uint32_t dim = 32;
	const uint32_t N = 16;
	TqProdCodecConfig config = {.dimension = dim, .bits = 4, .qjl_seed = 557u, .qjl_dimension = dim};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	uint8_t packed[256];
	uint8_t all_nibbles[16 * 32];
	uint8_t dim_major_nibbles[32 * 16];
	float gammas[16];
	float float_scores[16];
	float quantized_scores[16];
	float query[32];
	float input[32];
	float max_abs_diff = 0.0f;
	uint32_t i = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));
	memset(all_nibbles, 0, sizeof(all_nibbles));

	seeded_unit_vector(771u, query, dim);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_build(&config, &lut, &lut16, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_quantize(&lut16, errmsg, sizeof(errmsg)));

	for (i = 0; i < N; i++)
	{
		memset(packed, 0, sizeof(packed));
		seeded_unit_vector(800u + i, input, dim);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_read_gamma(&config, packed, layout.total_bytes, &gammas[i], errmsg, sizeof(errmsg)));
		assert(tq_prod_extract_nibbles(&config, packed, layout.total_bytes,
									   all_nibbles + (size_t) i * dim, dim, errmsg, sizeof(errmsg)));
	}

	transpose_nibbles_to_dim_major(all_nibbles, dim_major_nibbles, N, dim);

	/* Float reference */
	assert(tq_prod_score_block16_scalar(&lut16, dim_major_nibbles, gammas, N, float_scores,
										errmsg, sizeof(errmsg)));

	/* Quantized fused scorer */
	assert(tq_prod_score_block16_quantized_scalar(&lut16, dim_major_nibbles, gammas, N, quantized_scores,
												  errmsg, sizeof(errmsg)));

	for (i = 0; i < N; i++)
	{
		float diff = fabsf(quantized_scores[i] - float_scores[i]);

		if (diff > max_abs_diff)
			max_abs_diff = diff;
	}

	/* Quantized fused score should be close to float (< 5% of max score magnitude) */
	{
		float max_mag = 0.0f;

		for (i = 0; i < N; i++)
		{
			float mag = fabsf(float_scores[i]);
			if (mag > max_mag)
				max_mag = mag;
		}
		assert(max_abs_diff < 0.05f * max_mag);
	}

	tq_prod_lut16_reset(&lut16);
	tq_prod_lut_reset(&lut);
}

static void
test_quantized_block16_preserves_ranking_at_top_k(void)
{
	const uint32_t dim = 64;
	const uint32_t N = 16;
	const uint32_t top_k = 5;
	TqProdCodecConfig config = {.dimension = dim, .bits = 4, .qjl_seed = 887u, .qjl_dimension = dim};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	uint8_t packed[512];
	uint8_t *all_nibbles = NULL;
	uint8_t *dim_major_nibbles = NULL;
	float gammas[16];
	float float_scores[16];
	float quantized_scores[16];
	RankedDistance float_ranked[16];
	RankedDistance quant_ranked[16];
	float *query = NULL;
	float *input = NULL;
	uint32_t i = 0;
	uint32_t overlap = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));

	query = (float *) calloc(dim, sizeof(float));
	input = (float *) calloc(dim, sizeof(float));
	all_nibbles = (uint8_t *) calloc((size_t) N * dim, sizeof(uint8_t));
	dim_major_nibbles = (uint8_t *) calloc((size_t) dim * 16u, sizeof(uint8_t));
	assert(query != NULL && input != NULL && all_nibbles != NULL && dim_major_nibbles != NULL);

	seeded_unit_vector(991u, query, dim);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_build(&config, &lut, &lut16, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_quantize(&lut16, errmsg, sizeof(errmsg)));

	for (i = 0; i < N; i++)
	{
		memset(packed, 0, sizeof(packed));
		seeded_unit_vector(1000u + i, input, dim);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_read_gamma(&config, packed, layout.total_bytes, &gammas[i], errmsg, sizeof(errmsg)));
		assert(tq_prod_extract_nibbles(&config, packed, layout.total_bytes,
									   all_nibbles + (size_t) i * dim, dim, errmsg, sizeof(errmsg)));
	}

	transpose_nibbles_to_dim_major(all_nibbles, dim_major_nibbles, N, dim);

	assert(tq_prod_score_block16_scalar(&lut16, dim_major_nibbles, gammas, N, float_scores,
										errmsg, sizeof(errmsg)));
	assert(tq_prod_score_block16_quantized_scalar(&lut16, dim_major_nibbles, gammas, N, quantized_scores,
												  errmsg, sizeof(errmsg)));

	/* Sort both by distance (1 - score for cosine-like) and compare top-k overlap */
	for (i = 0; i < N; i++)
	{
		float_ranked[i].offset = (uint16_t) i;
		float_ranked[i].distance = 1.0f - float_scores[i];
		quant_ranked[i].offset = (uint16_t) i;
		quant_ranked[i].distance = 1.0f - quantized_scores[i];
	}

	qsort(float_ranked, N, sizeof(RankedDistance), compare_ranked_distance);
	qsort(quant_ranked, N, sizeof(RankedDistance), compare_ranked_distance);

	for (i = 0; i < top_k; i++)
	{
		uint32_t j = 0;

		for (j = 0; j < top_k; j++)
		{
			if (float_ranked[i].offset == quant_ranked[j].offset)
			{
				overlap++;
				break;
			}
		}
	}

	/* At least 80% of top-k should overlap between float and quantized ranking */
	assert(overlap >= (top_k * 4u) / 5u);

	tq_prod_lut16_reset(&lut16);
	tq_prod_lut_reset(&lut);
	free(dim_major_nibbles);
	free(all_nibbles);
	free(input);
	free(query);
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
	test_lut16_build_rejects_unsupported_shapes();
	test_lut16_scalar_matches_code_domain_scorer();
	test_lut16_block16_preserves_score_ordering();
	test_scratch_transpose_preserves_lane_order_and_tid();
	test_scratch_transpose_block16_scores_match_per_lane();
	test_scratch_transpose_partial_block_pads_safely();
	test_quantized_lut16_reconstruction_error_is_small();
	test_quantized_block16_fused_score_stays_close_to_float();
	test_quantized_block16_preserves_ranking_at_top_k();
	return 0;
}
