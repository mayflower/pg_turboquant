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

typedef struct RankedDistance
{
	uint16_t	offset;
	float		distance;
} RankedDistance;

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
test_preferred_kernel_matches_supported_runtime(void)
{
	TqProdCodecConfig supported_config = {.dimension = 8, .bits = 4};
	TqProdCodecConfig unsupported_config = {.dimension = 6, .bits = 4};

	assert(tq_prod_code_domain_preferred_kernel(&supported_config)
		   == expected_code_domain_kernel(&supported_config));
	assert(tq_prod_code_domain_preferred_kernel(&unsupported_config) == TQ_PROD_SCORE_SCALAR);
}

static void
test_ranking_equivalence_on_seeded_fixture(void)
{
	TqProdCodecConfig config = {.dimension = 8, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t packed[64];
	float query[8];
	RankedDistance expected[8];
	TqCandidateHeap heap;
	TqCandidateEntry entry;
	TqScanStats stats;
	TqProdScoreKernel preferred_kernel = TQ_PROD_SCORE_SCALAR;
	char errmsg[256];
	uint16_t lane = 0;
	size_t i = 0;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));
	memset(expected, 0, sizeof(expected));
	memset(&heap, 0, sizeof(heap));
	memset(&entry, 0, sizeof(entry));
	memset(&stats, 0, sizeof(stats));

	seeded_unit_vector(3u, query, 8);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_candidate_heap_init(&heap, 8));

	params.lane_count = 8;
	params.code_bytes = (uint32_t) layout.total_bytes;
	params.list_id = 0;
	params.next_block = TQ_INVALID_BLOCK_NUMBER;
	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));

	for (i = 0; i < 8; i++)
	{
		float input[8];
		float score = 0.0f;

		seeded_unit_vector((uint32_t) (100 + i), input, 8);
		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_code_from_lut(&config, &lut, packed, layout.total_bytes,
										   &score, errmsg, sizeof(errmsg)));
		expected[i].offset = (uint16_t) (i + 1);
		expected[i].distance = 1.0f - score;

		assert(tq_batch_page_append_lane(page, sizeof(page),
										 &(TqTid){.block_number = 1, .offset_number = (uint16_t) (i + 1)},
										 &lane, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, layout.total_bytes,
									  errmsg, sizeof(errmsg)));
	}

	qsort(expected, 8, sizeof(expected[0]), compare_ranked_distance);

	preferred_kernel = tq_prod_code_domain_preferred_kernel(&config);
	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1);
	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, true, TQ_DISTANCE_COSINE, &lut,
								   query, 8, &heap, NULL, errmsg, sizeof(errmsg)));
	tq_scan_stats_snapshot(&stats);
	assert(stats.score_mode == TQ_SCAN_SCORE_MODE_CODE_DOMAIN);
	assert(stats.score_kernel == preferred_kernel);

	for (i = 0; i < 8; i++)
	{
		assert(tq_candidate_heap_pop_best(&heap, &entry));
		assert(entry.tid.offset_number == expected[i].offset);
	}

	tq_candidate_heap_reset(&heap);
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

int
main(void)
{
	test_supported_shape_matches_scalar_scores();
	test_supported_shape_avx2_matches_scalar_scores_for_multiple_dimensions_when_available();
	test_supported_shape_neon_matches_scalar_scores_when_available();
	test_preferred_kernel_matches_supported_runtime();
	test_ranking_equivalence_on_seeded_fixture();
	test_dispatch_falls_back_to_scalar_for_unsupported_shape_or_disabled_runtime();
	test_malformed_packed_input_rejected_cleanly();
	return 0;
}
