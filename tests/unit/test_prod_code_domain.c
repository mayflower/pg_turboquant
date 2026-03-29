#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "src/tq_codec_prod.h"
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
								   query, 8, &cosine_heap, errmsg, sizeof(errmsg)));
	assert(tq_prod_decode_counter_get() == 0);
	tq_scan_stats_snapshot(&stats);
	assert(stats.score_mode == TQ_SCAN_SCORE_MODE_CODE_DOMAIN);
	assert(stats.decoded_vector_count == 0);

	tq_prod_decode_counter_reset();
	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1);
	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, true, TQ_DISTANCE_IP, &lut,
								   query, 8, &ip_heap, errmsg, sizeof(errmsg)));
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

int
main(void)
{
	test_code_domain_score_matches_decode_baseline();
	test_ranking_matches_decode_baseline_for_normalized_cosine_and_ip();
	test_distance_conversion_contract();
	return 0;
}
