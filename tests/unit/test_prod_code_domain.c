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
qjl_scale(uint32_t sketch_dimension)
{
	return 1.253314137315500251f / (float) sketch_dimension;
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

int
main(void)
{
	test_qjl_structured_projection_is_deterministic_and_seeded();
	test_qjl_lut_matches_structured_projection();
	test_qjl_backprojection_matches_decode_residual_component();
	test_qjl_sketch_dimension_controls_packed_layout();
	test_code_domain_score_matches_decode_baseline();
	test_ranking_matches_decode_baseline_for_normalized_cosine_and_ip();
	test_distance_conversion_contract();
	test_force_decode_diagnostics_switches_active_scoring_path();
	return 0;
}
