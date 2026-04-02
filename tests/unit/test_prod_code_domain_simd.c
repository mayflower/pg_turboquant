#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "src/tq_codec_prod.h"
#include "src/tq_page.h"
#include "src/tq_scan.h"
#include "src/tq_simd_avx2.h"

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

static TqProdScoreKernel
expected_code_domain_kernel(const TqProdCodecConfig *config)
{
	if (config == NULL || config->bits != 4 || config->dimension == 0
		|| (config->dimension % 8u) != 0u)
		return TQ_PROD_SCORE_SCALAR;

	if (tq_simd_avx2_runtime_available())
		return TQ_PROD_SCORE_AVX2;
	if (tq_simd_neon_runtime_available())
		return TQ_PROD_SCORE_NEON;
	return TQ_PROD_SCORE_SCALAR;
}

static void
test_supported_shape_matches_scalar_scores(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	uint8_t packed[64];
	float query[8];
	float scalar_score = 0.0f;
	float dispatch_score = 0.0f;
	TqProdScoreKernel used_kernel = TQ_PROD_SCORE_AUTO;
	char errmsg[256];
	uint32_t seed = 0;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));

	seeded_unit_vector(7u, query, 8);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));

	for (seed = 11u; seed < 27u; seed++)
	{
		float input[8];

		memset(input, 0, sizeof(input));
		seeded_unit_vector(seed, input, 8);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut(&config, &lut, packed, layout.total_bytes,
										   &scalar_score, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut_dispatch(&config, &lut, packed, layout.total_bytes,
													TQ_PROD_SCORE_AUTO, &dispatch_score, &used_kernel,
													errmsg, sizeof(errmsg)));
		assert(fabsf(dispatch_score - scalar_score) <= 1e-5f);
		assert(used_kernel == tq_prod_code_domain_preferred_kernel(&config));
	}

	tq_prod_lut_reset(&lut);
}

static void
test_supported_shape_avx2_matches_scalar_scores_for_multiple_dimensions_when_available(void)
{
	const uint32_t dimensions[] = {8u, 32u, 128u};
	size_t config_index = 0;

	if (!tq_simd_avx2_runtime_available())
		return;

	for (config_index = 0; config_index < (sizeof(dimensions) / sizeof(dimensions[0])); config_index++)
	{
		TqProdCodecConfig config = {.dimension = dimensions[config_index], .bits = 4};
		TqProdPackedLayout layout;
		TqProdLut lut;
		uint8_t packed[512];
		float query[128];
		float input[128];
		float scalar_score = 0.0f;
		float avx2_score = 0.0f;
		TqProdScoreKernel used_kernel = TQ_PROD_SCORE_SCALAR;
		char errmsg[256];
		uint32_t seed = 0;

		memset(&layout, 0, sizeof(layout));
		memset(&lut, 0, sizeof(lut));
		memset(packed, 0, sizeof(packed));
		memset(query, 0, sizeof(query));
		memset(input, 0, sizeof(input));

		seeded_unit_vector(1000u + dimensions[config_index], query, config.dimension);
		assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
		assert(layout.total_bytes <= sizeof(packed));
		assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));

		for (seed = 0; seed < 32u; seed++)
		{
			seeded_unit_vector(2000u + dimensions[config_index] + seed, input, config.dimension);
			assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
			assert(tq_prod_score_code_from_lut(&config, &lut, packed, layout.total_bytes,
											   &scalar_score, errmsg, sizeof(errmsg)));
			assert(tq_prod_score_code_from_lut_dispatch(&config, &lut, packed, layout.total_bytes,
														TQ_PROD_SCORE_AVX2, &avx2_score, &used_kernel,
														errmsg, sizeof(errmsg)));
			assert(used_kernel == TQ_PROD_SCORE_AVX2);
			assert(fabsf(avx2_score - scalar_score) <= 1e-5f);
		}

		tq_prod_lut_reset(&lut);
	}
}

static void
test_supported_shape_neon_matches_scalar_scores_when_available(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	uint8_t packed[64];
	float query[8];
	float scalar_score = 0.0f;
	float neon_score = 0.0f;
	TqProdScoreKernel used_kernel = TQ_PROD_SCORE_SCALAR;
	char errmsg[256];
	uint32_t seed = 0;

	if (!tq_simd_neon_runtime_available())
		return;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));

	seeded_unit_vector(9u, query, 8);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));

	for (seed = 101u; seed < 117u; seed++)
	{
		float input[8];

		memset(input, 0, sizeof(input));
		seeded_unit_vector(seed, input, 8);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut(&config, &lut, packed, layout.total_bytes,
										   &scalar_score, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut_dispatch(&config, &lut, packed, layout.total_bytes,
													TQ_PROD_SCORE_NEON, &neon_score, &used_kernel,
													errmsg, sizeof(errmsg)));
		assert(used_kernel == TQ_PROD_SCORE_NEON);
		assert(fabsf(neon_score - scalar_score) <= 1e-5f);
	}

	tq_prod_lut_reset(&lut);
}

static void
test_randomized_avx2_property_matches_scalar_when_available(void)
{
	const uint32_t dimensions[] = {8u, 32u, 128u};
	size_t config_index = 0;

	if (!tq_simd_avx2_runtime_available())
		return;

	for (config_index = 0; config_index < (sizeof(dimensions) / sizeof(dimensions[0])); config_index++)
	{
		TqProdCodecConfig config = {
			.dimension = dimensions[config_index],
			.bits = 4,
			.qjl_dimension = dimensions[config_index],
			.qjl_seed = 12345u + dimensions[config_index]
		};
		TqProdPackedLayout layout;
		TqProdLut lut;
		uint8_t packed[512];
		float query[128];
		float input[128];
		uint32_t seed = 0;
		char errmsg[256];

		memset(&layout, 0, sizeof(layout));
		memset(&lut, 0, sizeof(lut));
		memset(packed, 0, sizeof(packed));
		memset(query, 0, sizeof(query));
		memset(input, 0, sizeof(input));

		seeded_unit_vector(5000u + dimensions[config_index], query, config.dimension);
		assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
		assert(layout.total_bytes <= sizeof(packed));
		assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));

		for (seed = 0; seed < 96u; seed++)
		{
			float scalar_score = 0.0f;
			float avx2_score = 0.0f;
			TqProdScoreKernel used_kernel = TQ_PROD_SCORE_SCALAR;

			seeded_unit_vector(7000u + dimensions[config_index] + seed, input, config.dimension);
			assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
			assert(tq_prod_score_code_from_lut(&config, &lut, packed, layout.total_bytes,
											   &scalar_score, errmsg, sizeof(errmsg)));
			assert(tq_prod_score_code_from_lut_dispatch(&config, &lut, packed, layout.total_bytes,
														TQ_PROD_SCORE_AVX2, &avx2_score, &used_kernel,
														errmsg, sizeof(errmsg)));
			assert(used_kernel == TQ_PROD_SCORE_AVX2);
			assert(fabsf(avx2_score - scalar_score) <= 1e-5f);
		}

		tq_prod_lut_reset(&lut);
	}
}

static void
test_randomized_neon_property_matches_scalar_when_available(void)
{
	const uint32_t dimensions[] = {8u, 32u, 128u};
	size_t config_index = 0;

	if (!tq_simd_neon_runtime_available())
		return;

	for (config_index = 0; config_index < (sizeof(dimensions) / sizeof(dimensions[0])); config_index++)
	{
		TqProdCodecConfig config = {
			.dimension = dimensions[config_index],
			.bits = 4,
			.qjl_dimension = dimensions[config_index],
			.qjl_seed = 22345u + dimensions[config_index]
		};
		TqProdPackedLayout layout;
		TqProdLut lut;
		uint8_t packed[512];
		float query[128];
		float input[128];
		uint32_t seed = 0;
		char errmsg[256];

		memset(&layout, 0, sizeof(layout));
		memset(&lut, 0, sizeof(lut));
		memset(packed, 0, sizeof(packed));
		memset(query, 0, sizeof(query));
		memset(input, 0, sizeof(input));

		seeded_unit_vector(8000u + dimensions[config_index], query, config.dimension);
		assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
		assert(layout.total_bytes <= sizeof(packed));
		assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));

		for (seed = 0; seed < 96u; seed++)
		{
			float scalar_score = 0.0f;
			float neon_score = 0.0f;
			TqProdScoreKernel used_kernel = TQ_PROD_SCORE_SCALAR;

			seeded_unit_vector(9000u + dimensions[config_index] + seed, input, config.dimension);
			assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
			assert(tq_prod_score_code_from_lut(&config, &lut, packed, layout.total_bytes,
											   &scalar_score, errmsg, sizeof(errmsg)));
			assert(tq_prod_score_code_from_lut_dispatch(&config, &lut, packed, layout.total_bytes,
														TQ_PROD_SCORE_NEON, &neon_score, &used_kernel,
														errmsg, sizeof(errmsg)));
			assert(used_kernel == TQ_PROD_SCORE_NEON);
			assert(fabsf(neon_score - scalar_score) <= 1e-5f);
		}

		tq_prod_lut_reset(&lut);
	}
}

static void
test_preferred_kernel_matches_supported_runtime(void)
{
	TqProdCodecConfig supported_config = {.dimension = 8, .bits = 4};
	TqProdCodecConfig unsupported_config = {.dimension = 6, .bits = 4};

	assert(tq_prod_code_domain_preferred_kernel(&supported_config)
		   == expected_code_domain_kernel(&supported_config));
	assert(tq_prod_code_domain_preferred_kernel(&unsupported_config) == TQ_PROD_SCORE_SCALAR);
}

static void
test_explicit_kernel_selection_reports_requested_or_scalar_fallback(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	uint8_t packed[64];
	float query[8];
	float input[8];
	float scalar_score = 0.0f;
	float explicit_score = 0.0f;
	TqProdScoreKernel used_kernel = TQ_PROD_SCORE_AUTO;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));
	memset(input, 0, sizeof(input));

	seeded_unit_vector(901u, query, 8);
	seeded_unit_vector(902u, input, 8);

	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
	assert(tq_prod_score_code_from_lut_dispatch(&config,
											   &lut,
											   packed,
											   layout.total_bytes,
											   TQ_PROD_SCORE_SCALAR,
											   &scalar_score,
											   &used_kernel,
											   errmsg,
											   sizeof(errmsg)));
	assert(used_kernel == TQ_PROD_SCORE_SCALAR);

	assert(tq_prod_score_code_from_lut_dispatch(&config,
											   &lut,
											   packed,
											   layout.total_bytes,
											   TQ_PROD_SCORE_AVX2,
											   &explicit_score,
											   &used_kernel,
											   errmsg,
											   sizeof(errmsg)));
	assert(fabsf(explicit_score - scalar_score) <= 1e-5f);
	if (tq_simd_avx2_runtime_available())
		assert(used_kernel == TQ_PROD_SCORE_AVX2);
	else
		assert(used_kernel == TQ_PROD_SCORE_SCALAR);

	assert(tq_prod_score_code_from_lut_dispatch(&config,
											   &lut,
											   packed,
											   layout.total_bytes,
											   TQ_PROD_SCORE_NEON,
											   &explicit_score,
											   &used_kernel,
											   errmsg,
											   sizeof(errmsg)));
	assert(fabsf(explicit_score - scalar_score) <= 1e-5f);
	if (tq_simd_neon_runtime_available())
		assert(used_kernel == TQ_PROD_SCORE_NEON);
	else
		assert(used_kernel == TQ_PROD_SCORE_SCALAR);

	tq_prod_lut_reset(&lut);
}

static void
test_score_decompose_stays_consistent_with_packed_ip(void)
{
	TqProdCodecConfig config = {.dimension = 32, .qjl_dimension = 16, .bits = 4, .qjl_seed = 91u};
	TqProdPackedLayout layout;
	TqProdLut lut;
	uint8_t packed[256];
	float query[32];
	char errmsg[256];
	uint32_t seed = 0;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));

	seeded_unit_vector(701u, query, 32);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(layout.total_bytes <= sizeof(packed));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));

	for (seed = 0; seed < 24u; seed++)
	{
		float input[32];
		float mse_contribution = 0.0f;
		float qjl_contribution = 0.0f;
		float combined_score = 0.0f;
		float packed_score = 0.0f;

		memset(input, 0, sizeof(input));
		memset(packed, 0, sizeof(packed));

		seeded_unit_vector(1700u + seed, input, 32);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_decompose_ip(&config,
										  &lut,
										  packed,
										  layout.total_bytes,
										  &mse_contribution,
										  &qjl_contribution,
										  &combined_score,
										  errmsg,
										  sizeof(errmsg)));
		assert(tq_prod_score_packed_ip(&config,
									   &lut,
									   packed,
									   layout.total_bytes,
									   &packed_score,
									   errmsg,
									   sizeof(errmsg)));
		assert(fabsf(combined_score - (mse_contribution + qjl_contribution)) <= 1e-5f);
		assert(fabsf(combined_score - packed_score) <= 1e-5f);
	}

	tq_prod_lut_reset(&lut);
}

static void
test_code_domain_and_decode_fallback_match_ordering_on_seeded_fixture(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t packed[64];
	float query[8];
	TqCandidateHeap primary_heap;
	TqCandidateHeap shadow_decode_heap;
	TqCandidateEntry primary_entry;
	TqCandidateEntry shadow_entry;
	TqScanStats stats;
	TqProdScoreKernel preferred_kernel = TQ_PROD_SCORE_SCALAR;
	char errmsg[256];
	uint16_t lane = 0;
	size_t i = 0;
	const float ordering_tolerance = 1e-5f;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));
	memset(&primary_heap, 0, sizeof(primary_heap));
	memset(&shadow_decode_heap, 0, sizeof(shadow_decode_heap));
	memset(&primary_entry, 0, sizeof(primary_entry));
	memset(&shadow_entry, 0, sizeof(shadow_entry));
	memset(&stats, 0, sizeof(stats));

	seeded_unit_vector(3u, query, 8);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_candidate_heap_init(&primary_heap, 8));
	assert(tq_candidate_heap_init(&shadow_decode_heap, 8));

	params.lane_count = 8;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;
	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));

	for (i = 0; i < 8; i++)
	{
		float input[8];

		seeded_unit_vector((uint32_t) (100 + i), input, 8);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_append_lane(page, sizeof(page),
										 &(TqTid){.block_number = 1, .offset_number = (uint16_t) (i + 1)},
										 &lane, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, layout.total_bytes,
									  errmsg, sizeof(errmsg)));
	}

	preferred_kernel = tq_prod_code_domain_preferred_kernel(&config);
	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1);
	assert(tq_batch_page_scan_prod(page,
								   sizeof(page),
								   &config,
								   true,
								   TQ_DISTANCE_IP,
								   &lut,
								   query,
								   8,
								   &primary_heap,
								   &shadow_decode_heap,
								   errmsg,
								   sizeof(errmsg)));
	tq_scan_stats_set_shadow_decode_metrics(&primary_heap, &shadow_decode_heap);
	tq_scan_stats_snapshot(&stats);
	assert(stats.score_mode == TQ_SCAN_SCORE_MODE_CODE_DOMAIN);
	assert(stats.score_kernel == preferred_kernel);
	assert(stats.faithful_fast_path);
	assert(!stats.compatibility_fallback);
	assert(stats.shadow_decode_candidate_count == 8);
	assert(stats.shadow_decoded_vector_count == 8);
	assert(stats.decoded_vector_count == 0);

	for (i = 0; i < 8; i++)
	{
		assert(tq_candidate_heap_pop_best(&primary_heap, &primary_entry));
		assert(tq_candidate_heap_pop_best(&shadow_decode_heap, &shadow_entry));
		assert(primary_entry.tid.block_number == shadow_entry.tid.block_number);
		assert(primary_entry.tid.offset_number == shadow_entry.tid.offset_number);
		assert(fabsf(primary_entry.score - shadow_entry.score) <= ordering_tolerance);
	}

	tq_candidate_heap_reset(&primary_heap);
	tq_candidate_heap_reset(&shadow_decode_heap);
	tq_prod_lut_reset(&lut);
}

static void
test_dispatch_falls_back_to_scalar_for_unsupported_shape_or_disabled_runtime(void)
{
	TqProdCodecConfig unsupported_config = {.dimension = 6, .bits = 4};
	TqProdCodecConfig supported_config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout unsupported_layout;
	TqProdPackedLayout supported_layout;
	TqProdLut unsupported_lut;
	TqProdLut supported_lut;
	uint8_t unsupported_packed[64];
	uint8_t supported_packed[64];
	float unsupported_query[6];
	float supported_query[8];
	float scalar_score = 0.0f;
	float dispatch_score = 0.0f;
	TqProdScoreKernel used_kernel = TQ_PROD_SCORE_AVX2;
	char errmsg[256];

	memset(&unsupported_layout, 0, sizeof(unsupported_layout));
	memset(&supported_layout, 0, sizeof(supported_layout));
	memset(&unsupported_lut, 0, sizeof(unsupported_lut));
	memset(&supported_lut, 0, sizeof(supported_lut));
	memset(unsupported_packed, 0, sizeof(unsupported_packed));
	memset(supported_packed, 0, sizeof(supported_packed));
	memset(unsupported_query, 0, sizeof(unsupported_query));
	memset(supported_query, 0, sizeof(supported_query));

	seeded_unit_vector(21u, unsupported_query, 6);
	seeded_unit_vector(31u, supported_query, 8);

	assert(tq_prod_packed_layout(&unsupported_config, &unsupported_layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&unsupported_config, unsupported_query, &unsupported_lut, errmsg, sizeof(errmsg)));
	{
		float input[6];

		seeded_unit_vector(41u, input, 6);
		assert(tq_prod_encode(&unsupported_config, input, unsupported_packed, unsupported_layout.total_bytes,
							  errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut(&unsupported_config, &unsupported_lut,
										   unsupported_packed, unsupported_layout.total_bytes,
										   &scalar_score, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut_dispatch(&unsupported_config, &unsupported_lut,
													unsupported_packed, unsupported_layout.total_bytes,
													TQ_PROD_SCORE_AVX2, &dispatch_score, &used_kernel,
													errmsg, sizeof(errmsg)));
		assert(used_kernel == TQ_PROD_SCORE_SCALAR);
		assert(fabsf(dispatch_score - scalar_score) <= 1e-5f);
		assert(tq_prod_score_code_from_lut_dispatch(&unsupported_config, &unsupported_lut,
													unsupported_packed, unsupported_layout.total_bytes,
													TQ_PROD_SCORE_NEON, &dispatch_score, &used_kernel,
													errmsg, sizeof(errmsg)));
		assert(used_kernel == TQ_PROD_SCORE_SCALAR);
		assert(fabsf(dispatch_score - scalar_score) <= 1e-5f);
	}

	assert(tq_prod_packed_layout(&supported_config, &supported_layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&supported_config, supported_query, &supported_lut, errmsg, sizeof(errmsg)));
	{
		float input[8];

		seeded_unit_vector(51u, input, 8);
		assert(tq_prod_encode(&supported_config, input, supported_packed, supported_layout.total_bytes,
							  errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut(&supported_config, &supported_lut,
										   supported_packed, supported_layout.total_bytes,
										   &scalar_score, errmsg, sizeof(errmsg)));
		tq_simd_force_disable(true);
		assert(tq_prod_score_code_from_lut_dispatch(&supported_config, &supported_lut,
													supported_packed, supported_layout.total_bytes,
													TQ_PROD_SCORE_AUTO, &dispatch_score, &used_kernel,
													errmsg, sizeof(errmsg)));
		tq_simd_force_disable(false);
		assert(used_kernel == TQ_PROD_SCORE_SCALAR);
		assert(fabsf(dispatch_score - scalar_score) <= 1e-5f);
	}

	tq_prod_lut_reset(&unsupported_lut);
	tq_prod_lut_reset(&supported_lut);
}

static void
test_malformed_packed_input_rejected_cleanly(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	uint8_t packed[64];
	float query[8];
	float score = 0.0f;
	TqProdScoreKernel used_kernel = TQ_PROD_SCORE_AUTO;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));
	memset(errmsg, 0, sizeof(errmsg));

	seeded_unit_vector(77u, query, 8);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_encode(&config, query, packed, layout.total_bytes, errmsg, sizeof(errmsg)));

	assert(!tq_prod_score_code_from_lut_dispatch(&config, &lut, packed, layout.total_bytes - 1,
												 TQ_PROD_SCORE_AUTO, &score, &used_kernel,
												 errmsg, sizeof(errmsg)));
	assert(strstr(errmsg, "too small") != NULL);
	assert(used_kernel == expected_code_domain_kernel(&config));

	tq_prod_lut_reset(&lut);
}

static void
test_lut16_reference_matches_legacy_scalar_and_simd_code_domain(void)
{
	const uint32_t dimensions[] = {16, 32, 64, 128};
	const size_t dim_count = sizeof(dimensions) / sizeof(dimensions[0]);
	const uint32_t vec_count = 8;
	size_t d = 0;

	for (d = 0; d < dim_count; d++)
	{
		uint32_t dim = dimensions[d];
		TqProdCodecConfig config = {.dimension = dim, .bits = 4, .qjl_seed = 113u, .qjl_dimension = dim};
		TqProdPackedLayout layout;
		TqProdLut lut;
		TqProdLut16 lut16;
		float *query = NULL;
		float *input = NULL;
		uint8_t *packed = NULL;
		uint8_t *nibbles = NULL;
		uint8_t *dim_major_nibbles = NULL;
		uint32_t v = 0;
		char errmsg[256];

		memset(&layout, 0, sizeof(layout));
		memset(&lut, 0, sizeof(lut));
		memset(&lut16, 0, sizeof(lut16));

		query = (float *) calloc(dim, sizeof(float));
		input = (float *) calloc(dim, sizeof(float));
		nibbles = (uint8_t *) calloc(dim, sizeof(uint8_t));
		dim_major_nibbles = (uint8_t *) calloc((size_t) dim * 16u, sizeof(uint8_t));
		assert(query != NULL && input != NULL && nibbles != NULL && dim_major_nibbles != NULL);

		seeded_unit_vector(77u + dim, query, dim);
		assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
		packed = (uint8_t *) calloc(layout.total_bytes, sizeof(uint8_t));
		assert(packed != NULL);
		assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
		assert(tq_prod_lut16_build(&config, &lut, &lut16, errmsg, sizeof(errmsg)));

		for (v = 0; v < vec_count; v++)
		{
			float scalar_score = 0.0f;
			float simd_score = 0.0f;
			float lut16_score = 0.0f;
			float gamma = 0.0f;
			TqProdScoreKernel used = TQ_PROD_SCORE_SCALAR;

			memset(packed, 0, layout.total_bytes);
			seeded_unit_vector(300u + v + dim, input, dim);
			assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
			assert(tq_prod_read_gamma(&config, packed, layout.total_bytes, &gamma, errmsg, sizeof(errmsg)));

			/* Legacy scalar path */
			assert(tq_prod_score_code_from_lut(&config, &lut, packed, layout.total_bytes,
											   &scalar_score, errmsg, sizeof(errmsg)));

			/* SIMD dispatch (auto) */
			assert(tq_prod_score_code_from_lut_dispatch(&config, &lut, packed, layout.total_bytes,
														TQ_PROD_SCORE_AUTO, &simd_score, &used,
														errmsg, sizeof(errmsg)));

			/* LUT16 reference */
			assert(tq_prod_extract_nibbles(&config, packed, layout.total_bytes,
										   nibbles, dim, errmsg, sizeof(errmsg)));
			transpose_nibbles_to_dim_major(nibbles, dim_major_nibbles, 1, dim);
			assert(tq_prod_score_block16_scalar(&lut16, dim_major_nibbles, &gamma, 1, &lut16_score,
												errmsg, sizeof(errmsg)));

			assert(fabsf(lut16_score - scalar_score) < 1e-4f);
			assert(fabsf(lut16_score - simd_score) < 1e-4f);
		}

		tq_prod_lut16_reset(&lut16);
		tq_prod_lut_reset(&lut);
		free(packed);
		free(dim_major_nibbles);
		free(nibbles);
		free(input);
		free(query);
	}
}

static void
test_block16_dispatch_selects_kernel_and_falls_back_to_scalar(void)
{
	const uint32_t dim = 32;
	const uint32_t N = 8;
	TqProdCodecConfig config = {.dimension = dim, .bits = 4, .qjl_seed = 199u, .qjl_dimension = dim};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	uint8_t packed[256];
	uint8_t all_nibbles[8 * 32];
	uint8_t dim_major_nibbles[32 * 16];
	float gammas[8];
	float scalar_scores[8];
	float quantized_scalar_scores[8];
	float dispatch_scores[8];
	float query[32];
	float input[32];
	TqProdScoreKernel used_kernel = TQ_PROD_SCORE_AUTO;
	uint32_t i = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));
	memset(all_nibbles, 0, sizeof(all_nibbles));

	seeded_unit_vector(501u, query, dim);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_build(&config, &lut, &lut16, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_quantize(&lut16, errmsg, sizeof(errmsg)));

	for (i = 0; i < N; i++)
	{
		memset(packed, 0, sizeof(packed));
		seeded_unit_vector(600u + i, input, dim);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_read_gamma(&config, packed, layout.total_bytes, &gammas[i], errmsg, sizeof(errmsg)));
		assert(tq_prod_extract_nibbles(&config, packed, layout.total_bytes,
									   all_nibbles + (size_t) i * dim, dim, errmsg, sizeof(errmsg)));
	}

	transpose_nibbles_to_dim_major(all_nibbles, dim_major_nibbles, N, dim);

	/* Float scalar reference */
	assert(tq_prod_score_block16_scalar(&lut16, dim_major_nibbles, gammas, N, scalar_scores,
										errmsg, sizeof(errmsg)));

	/* Quantized scalar reference (for comparing against SIMD dispatch) */
	assert(tq_prod_score_block16_quantized_scalar(&lut16, dim_major_nibbles, gammas, N,
												  quantized_scalar_scores, errmsg, sizeof(errmsg)));

	/* Dispatch with auto: uses quantized SIMD if available, else float scalar */
	assert(tq_prod_score_block16_dispatch(&lut16, dim_major_nibbles, gammas, N,
										  TQ_PROD_SCORE_AUTO, dispatch_scores, &used_kernel,
										  errmsg, sizeof(errmsg)));

	if (used_kernel == TQ_PROD_SCORE_SCALAR)
	{
		for (i = 0; i < N; i++)
			assert(fabsf(dispatch_scores[i] - scalar_scores[i]) < 1e-4f);
	}
	else
	{
		for (i = 0; i < N; i++)
			assert(fabsf(dispatch_scores[i] - quantized_scalar_scores[i]) < 1e-4f);
	}

	/* Dispatch with explicit AVX2 request */
	assert(tq_prod_score_block16_dispatch(&lut16, dim_major_nibbles, gammas, N,
										  TQ_PROD_SCORE_AVX2, dispatch_scores, &used_kernel,
										  errmsg, sizeof(errmsg)));
	if (!tq_simd_avx2_runtime_available())
		assert(used_kernel == TQ_PROD_SCORE_SCALAR);

	if (used_kernel == TQ_PROD_SCORE_SCALAR)
	{
		for (i = 0; i < N; i++)
			assert(fabsf(dispatch_scores[i] - scalar_scores[i]) < 1e-4f);
	}
	else
	{
		for (i = 0; i < N; i++)
			assert(fabsf(dispatch_scores[i] - quantized_scalar_scores[i]) < 1e-4f);
	}

	/* Dispatch with explicit scalar */
	assert(tq_prod_score_block16_dispatch(&lut16, dim_major_nibbles, gammas, N,
										  TQ_PROD_SCORE_SCALAR, dispatch_scores, &used_kernel,
										  errmsg, sizeof(errmsg)));
	assert(used_kernel == TQ_PROD_SCORE_SCALAR);

	for (i = 0; i < N; i++)
		assert(fabsf(dispatch_scores[i] - scalar_scores[i]) < 1e-4f);

	tq_prod_lut16_reset(&lut16);
	tq_prod_lut_reset(&lut);
}

static void
test_block16_dispatch_neon_matches_scalar_on_arm64(void)
{
	const uint32_t dim = 64;
	const uint32_t N = 16;
	TqProdCodecConfig config = {.dimension = dim, .bits = 4, .qjl_seed = 317u, .qjl_dimension = dim};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	uint8_t packed[512];
	uint8_t *all_nibbles = NULL;
	uint8_t *dim_major_nibbles = NULL;
	float gammas[16];
	float scalar_scores[16];
	float quantized_scalar_scores[16];
	float neon_scores[16];
	float auto_scores[16];
	float *query = NULL;
	float *input = NULL;
	TqProdScoreKernel used_kernel = TQ_PROD_SCORE_AUTO;
	uint32_t i = 0;
	char errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));

	query = (float *) calloc(dim, sizeof(float));
	input = (float *) calloc(dim, sizeof(float));
	all_nibbles = (uint8_t *) calloc((size_t) N * dim, sizeof(uint8_t));
	dim_major_nibbles = (uint8_t *) calloc((size_t) dim * 16u, sizeof(uint8_t));
	assert(query != NULL && input != NULL && all_nibbles != NULL && dim_major_nibbles != NULL);

	seeded_unit_vector(777u, query, dim);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_build(&config, &lut, &lut16, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut16_quantize(&lut16, errmsg, sizeof(errmsg)));

	for (i = 0; i < N; i++)
	{
		memset(packed, 0, sizeof(packed));
		seeded_unit_vector(900u + i, input, dim);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_read_gamma(&config, packed, layout.total_bytes, &gammas[i], errmsg, sizeof(errmsg)));
		assert(tq_prod_extract_nibbles(&config, packed, layout.total_bytes,
									   all_nibbles + (size_t) i * dim, dim, errmsg, sizeof(errmsg)));
	}

	transpose_nibbles_to_dim_major(all_nibbles, dim_major_nibbles, N, dim);

	/* Float scalar reference */
	assert(tq_prod_score_block16_dispatch(&lut16, dim_major_nibbles, gammas, N,
										  TQ_PROD_SCORE_SCALAR, scalar_scores, &used_kernel,
										  errmsg, sizeof(errmsg)));
	assert(used_kernel == TQ_PROD_SCORE_SCALAR);

	/* Quantized scalar reference (for comparing against SIMD dispatch) */
	assert(tq_prod_score_block16_quantized_scalar(&lut16, dim_major_nibbles, gammas, N,
												  quantized_scalar_scores, errmsg, sizeof(errmsg)));

	/* Explicit NEON request */
	assert(tq_prod_score_block16_dispatch(&lut16, dim_major_nibbles, gammas, N,
										  TQ_PROD_SCORE_NEON, neon_scores, &used_kernel,
										  errmsg, sizeof(errmsg)));
	if (tq_simd_neon_runtime_available())
		assert(used_kernel == TQ_PROD_SCORE_NEON);
	else
		assert(used_kernel == TQ_PROD_SCORE_SCALAR);

	/* Auto dispatch */
	assert(tq_prod_score_block16_dispatch(&lut16, dim_major_nibbles, gammas, N,
										  TQ_PROD_SCORE_AUTO, auto_scores, &used_kernel,
										  errmsg, sizeof(errmsg)));

	/*
	 * NEON/auto dispatch uses quantized kernel when SIMD is available,
	 * otherwise falls back to float scalar.
	 */
	for (i = 0; i < N; i++)
	{
		if (tq_simd_neon_runtime_available() || tq_simd_avx2_runtime_available())
		{
			assert(fabsf(neon_scores[i] - quantized_scalar_scores[i]) < 1e-4f);
			assert(fabsf(auto_scores[i] - quantized_scalar_scores[i]) < 1e-4f);
		}
		else
		{
			assert(fabsf(neon_scores[i] - scalar_scores[i]) < 1e-4f);
			assert(fabsf(auto_scores[i] - scalar_scores[i]) < 1e-4f);
		}
	}

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
	test_supported_shape_matches_scalar_scores();
	test_supported_shape_avx2_matches_scalar_scores_for_multiple_dimensions_when_available();
	test_supported_shape_neon_matches_scalar_scores_when_available();
	test_randomized_avx2_property_matches_scalar_when_available();
	test_randomized_neon_property_matches_scalar_when_available();
	test_preferred_kernel_matches_supported_runtime();
	test_explicit_kernel_selection_reports_requested_or_scalar_fallback();
	test_score_decompose_stays_consistent_with_packed_ip();
	test_code_domain_and_decode_fallback_match_ordering_on_seeded_fixture();
	test_dispatch_falls_back_to_scalar_for_unsupported_shape_or_disabled_runtime();
	test_malformed_packed_input_rejected_cleanly();
	test_lut16_reference_matches_legacy_scalar_and_simd_code_domain();
	test_block16_dispatch_selects_kernel_and_falls_back_to_scalar();
	test_block16_dispatch_neon_matches_scalar_on_arm64();
	return 0;
}
