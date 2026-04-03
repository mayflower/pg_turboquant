#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "src/tq_codec_prod.h"
#include "src/tq_page.h"
#include "src/tq_router.h"
#include "src/tq_scan.h"
#include "src/tq_simd_avx2.h"

typedef struct MicrobenchResult
{
	const char *benchmark;
	const char *requested_kernel;
	const char *kernel;
	const char *qjl_lut_mode;
	const char *scan_layout;
	const char *lookup_style;
	const char *qjl_path;
	const char *gamma_path;
	uint32_t dimension;
	uint8_t bits;
	uint32_t iterations;
	uint32_t lane_count;
	uint32_t block_width;
	uint64_t visited_code_count;
	uint64_t visited_page_count;
	uint64_t candidate_heap_insert_count;
	uint64_t candidate_heap_replace_count;
	uint64_t candidate_heap_reject_count;
	uint64_t local_candidate_heap_insert_count;
	uint64_t local_candidate_heap_replace_count;
	uint64_t local_candidate_heap_reject_count;
	uint64_t local_candidate_merge_count;
	uint64_t scratch_allocations;
	uint64_t decoded_buffer_reuses;
	uint64_t code_view_uses;
	uint64_t code_copy_uses;
	uint32_t list_count;
	uint32_t probe_count;
	uint64_t total_ns;
	double ns_per_op;
	double codes_per_second;
	double pages_per_second;
	bool requested_kernel_honored;
	bool runtime_available;
} MicrobenchResult;

static void
normalize(float *values, size_t len)
{
	float norm = 0.0f;
	size_t i = 0;

	for (i = 0; i < len; i++)
		norm += values[i] * values[i];

	norm = sqrtf(norm);
	if (norm <= 0.0f)
		norm = 1.0f;

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

static uint64_t
monotonic_now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((uint64_t) ts.tv_sec * UINT64_C(1000000000)) + (uint64_t) ts.tv_nsec;
}

static double
throughput_per_second(uint64_t count, uint64_t total_ns)
{
	if (total_ns == 0)
		return 0.0;

	return ((double) count * 1000000000.0) / (double) total_ns;
}

static uint32_t
iterations_for_dimension(uint32_t dimension)
{
	if (dimension <= 32u)
		return 25000u;
	if (dimension <= 128u)
		return 10000u;
	return 4000u;
}

static bool
router_top_probe_prefix_matches(const TqRouterProbeScore *full_scores,
								   const TqRouterProbeScore *partial_scores,
								   uint32_t probe_count)
{
	uint32_t index = 0;

	for (index = 0; index < probe_count; index++)
	{
		if (full_scores[index].list_id != partial_scores[index].list_id)
			return false;
		if (fabsf(full_scores[index].score - partial_scores[index].score) > 1e-6f)
			return false;
	}

	return true;
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

static void
expand_single_candidate_nibbles(const uint8_t *candidate_nibbles,
								  uint32_t dimension,
								  uint8_t *block16_nibbles)
{
	uint32_t dim = 0;

	memset(block16_nibbles, 0, (size_t) dimension * 16u);
	for (dim = 0; dim < dimension; dim++)
		block16_nibbles[(size_t) dim * 16u] = candidate_nibbles[dim];
}

static float
quantized_qjl_score(const TqProdCodecConfig *config,
					  const TqProdLut *lut,
					  const uint8_t *packed,
					  size_t packed_len,
					  char *errmsg,
					  size_t errmsg_len)
{
	TqProdPackedLayout layout;
	float base_sum = 0.0f;
	float residual_sum = 0.0f;
	float gamma = 0.0f;
	uint32_t dim = 0;

	memset(&layout, 0, sizeof(layout));

	if (!lut->qjl_quantized_enabled || lut->qjl_quantized_values == NULL)
	{
		snprintf(errmsg, errmsg_len, "quantized qjl score requires quantized lut sidecar");
		return 0.0f;
	}

	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len)
		|| packed_len < layout.total_bytes
		|| !tq_prod_read_gamma(config, packed, packed_len, &gamma, errmsg, errmsg_len))
		return 0.0f;

	for (dim = 0; dim < config->dimension; dim++)
	{
		uint32_t idx_code = test_unpack_bits(packed, dim * 3u, 3u);
		size_t index = ((size_t) dim * (size_t) lut->level_count) + (size_t) idx_code;

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

static bool
build_page_scan_fixture(uint32_t dimension,
						  TqProdCodecConfig *config,
						  TqProdPackedLayout *layout,
						  TqProdLut *lut,
						  TqBatchPageParams *params,
						  uint8_t *page,
						  size_t page_size,
						  float *query,
						  size_t query_capacity,
						  char *errmsg,
						  size_t errmsg_len)
{
	uint8_t packed[512];
	uint16_t lane = 0;
	uint32_t i = 0;

	if (dimension > query_capacity)
		return false;

	memset(layout, 0, sizeof(*layout));
	memset(lut, 0, sizeof(*lut));
	memset(params, 0, sizeof(*params));
	memset(page, 0, page_size);
	memset(query, 0, sizeof(float) * query_capacity);
	memset(packed, 0, sizeof(packed));

	config->dimension = dimension;
	config->bits = 4;

	seeded_unit_vector(7u + dimension, query, dimension);
	if (!tq_prod_packed_layout(config, layout, errmsg, errmsg_len)
		|| !tq_prod_lut_build(config, query, lut, errmsg, errmsg_len))
		return false;

	params->lane_count = 32;
	params->code_bytes = (uint32_t) layout->total_bytes;
	params->list_id = 0;
	params->next_block = TQ_INVALID_BLOCK_NUMBER;
	if (!tq_batch_page_init(page, page_size, params, errmsg, errmsg_len))
		return false;

	for (i = 0; i < params->lane_count; i++)
	{
		float input[128];

		memset(input, 0, sizeof(input));
		seeded_unit_vector(100u + i + dimension, input, dimension);
		if (!tq_prod_encode(config, input, packed, layout->total_bytes, errmsg, errmsg_len)
			|| !tq_batch_page_append_lane(page,
										  page_size,
										  &(TqTid){.block_number = 1u, .offset_number = (uint16_t) (i + 1u)},
										  &lane,
										  errmsg,
										  errmsg_len)
			|| !tq_batch_page_set_code(page,
									  page_size,
									  lane,
									  packed,
									  layout->total_bytes,
									  errmsg,
									  errmsg_len))
			return false;
	}

	return true;
}

static bool
run_code_domain_bench(TqProdScoreKernel requested_kernel,
					  uint32_t dimension,
					  MicrobenchResult *result,
					  char *errmsg,
					  size_t errmsg_len)
{
	TqProdCodecConfig config = {.dimension = dimension, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	uint8_t *packed = NULL;
	float *query = NULL;
	float *input = NULL;
	float score = 0.0f;
	volatile float sink = 0.0f;
	uint64_t started_ns = 0;
	uint64_t ended_ns = 0;
	uint32_t iterations = iterations_for_dimension(dimension);
	uint32_t i = 0;
	TqProdScoreKernel used_kernel = TQ_PROD_SCORE_AUTO;
	bool ok = false;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));

	query = (float *) calloc(dimension, sizeof(float));
	input = (float *) calloc(dimension, sizeof(float));
	if (query == NULL || input == NULL)
		goto cleanup;

	seeded_unit_vector(17u + dimension, query, dimension);
	seeded_unit_vector(31u + dimension, input, dimension);

	if (!tq_prod_packed_layout(&config, &layout, errmsg, errmsg_len))
		goto cleanup;

	packed = (uint8_t *) calloc(layout.total_bytes, sizeof(uint8_t));
	if (packed == NULL)
		goto cleanup;

	if (!tq_prod_lut_build(&config, query, &lut, errmsg, errmsg_len))
		goto cleanup;

	if (!tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, errmsg_len))
		goto cleanup;

	started_ns = monotonic_now_ns();
	for (i = 0; i < iterations; i++)
	{
		if (!tq_prod_score_code_from_lut_dispatch(&config,
												  &lut,
												  packed,
												  layout.total_bytes,
												  requested_kernel,
												  &score,
												  &used_kernel,
												  errmsg,
												  errmsg_len))
			goto cleanup;
		sink += score;
	}
	ended_ns = monotonic_now_ns();
	if (sink == 0.1234567f)
		fprintf(stderr, "ignore %f\n", sink);

	result->benchmark = "score_code_from_lut";
	result->requested_kernel = tq_prod_score_kernel_name(requested_kernel);
	result->kernel = tq_prod_score_kernel_name(used_kernel);
	result->qjl_lut_mode =
		(lut.qjl_quantized_enabled && used_kernel != TQ_PROD_SCORE_SCALAR) ? "quantized" : "float";
	result->scan_layout = "row_major";
	result->lookup_style = tq_lookup_style_name(tq_lookup_style_for_kernel(used_kernel));
	result->block_width = 1u;
	result->qjl_path = tq_qjl_path_name(tq_qjl_path_for_kernel(used_kernel, lut.qjl_quantized_enabled));
	result->gamma_path = tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_SCALAR);
	result->dimension = dimension;
	result->bits = config.bits;
	result->iterations = iterations;
	result->lane_count = 1u;
	result->visited_code_count = iterations;
	result->visited_page_count = 0u;
	result->candidate_heap_insert_count = 0u;
	result->candidate_heap_replace_count = 0u;
	result->candidate_heap_reject_count = 0u;
	result->local_candidate_heap_insert_count = 0u;
	result->local_candidate_heap_replace_count = 0u;
	result->local_candidate_heap_reject_count = 0u;
	result->local_candidate_merge_count = 0u;
	result->scratch_allocations = 0u;
	result->decoded_buffer_reuses = 0u;
	result->code_view_uses = 0u;
	result->code_copy_uses = 0u;
	result->list_count = 0u;
	result->probe_count = 0u;
	result->total_ns = ended_ns - started_ns;
	result->ns_per_op = (double) result->total_ns / (double) iterations;
	result->codes_per_second = throughput_per_second(result->visited_code_count,
													 result->total_ns);
	result->pages_per_second = 0.0;
	result->requested_kernel_honored =
		requested_kernel == TQ_PROD_SCORE_AUTO || used_kernel == requested_kernel;
	switch (requested_kernel == TQ_PROD_SCORE_AUTO ? used_kernel : requested_kernel)
	{
		case TQ_PROD_SCORE_AVX2:
			result->runtime_available = tq_simd_avx2_runtime_available();
			break;
		case TQ_PROD_SCORE_AVX512:
			result->runtime_available = tq_simd_avx512_runtime_available();
			break;
		case TQ_PROD_SCORE_NEON:
			result->runtime_available = tq_simd_neon_runtime_available();
			break;
		case TQ_PROD_SCORE_SCALAR:
		default:
			result->runtime_available = true;
			break;
	}
	ok = true;

cleanup:
	tq_prod_lut_reset(&lut);
	free(packed);
	free(query);
	free(input);
	return ok;
}

static bool
run_quantized_reference_bench(uint32_t dimension,
								MicrobenchResult *result,
								char *errmsg,
								size_t errmsg_len)
{
	TqProdCodecConfig config = {.dimension = dimension, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	uint8_t *packed = NULL;
	float *query = NULL;
	float *input = NULL;
	float score = 0.0f;
	volatile float sink = 0.0f;
	uint64_t started_ns = 0;
	uint64_t ended_ns = 0;
	uint32_t iterations = iterations_for_dimension(dimension);
	uint32_t i = 0;
	bool ok = false;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));

	query = (float *) calloc(dimension, sizeof(float));
	input = (float *) calloc(dimension, sizeof(float));
	if (query == NULL || input == NULL)
		goto cleanup;

	seeded_unit_vector(17u + dimension, query, dimension);
	seeded_unit_vector(31u + dimension, input, dimension);
	if (!tq_prod_packed_layout(&config, &layout, errmsg, errmsg_len))
		goto cleanup;

	packed = (uint8_t *) calloc(layout.total_bytes, sizeof(uint8_t));
	if (packed == NULL)
		goto cleanup;

	if (!tq_prod_lut_build(&config, query, &lut, errmsg, errmsg_len)
		|| !lut.qjl_quantized_enabled
		|| !tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, errmsg_len))
		goto cleanup;

	errmsg[0] = '\0';
	started_ns = monotonic_now_ns();
	for (i = 0; i < iterations; i++)
	{
		score = quantized_qjl_score(&config, &lut, packed, layout.total_bytes, errmsg, errmsg_len);
		if (errmsg[0] != '\0')
			goto cleanup;
		sink += score;
	}
	ended_ns = monotonic_now_ns();
	if (sink == 0.1234567f)
		fprintf(stderr, "ignore %f\n", sink);

	result->benchmark = "score_code_from_lut_quantized_reference";
	result->requested_kernel = "scalar";
	result->kernel = "scalar";
	result->qjl_lut_mode = "quantized";
	result->scan_layout = "row_major";
	result->lookup_style = tq_lookup_style_name(TQ_LOOKUP_STYLE_SCALAR_LOOP);
	result->block_width = 1u;
	result->qjl_path = tq_qjl_path_name(TQ_QJL_PATH_INT16_QUANTIZED);
	result->gamma_path = tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_SCALAR);
	result->dimension = dimension;
	result->bits = config.bits;
	result->iterations = iterations;
	result->lane_count = 1u;
	result->visited_code_count = iterations;
	result->visited_page_count = 0u;
	result->candidate_heap_insert_count = 0u;
	result->candidate_heap_replace_count = 0u;
	result->candidate_heap_reject_count = 0u;
	result->local_candidate_heap_insert_count = 0u;
	result->local_candidate_heap_replace_count = 0u;
	result->local_candidate_heap_reject_count = 0u;
	result->local_candidate_merge_count = 0u;
	result->scratch_allocations = 0u;
	result->decoded_buffer_reuses = 0u;
	result->code_view_uses = 0u;
	result->code_copy_uses = 0u;
	result->list_count = 0u;
	result->probe_count = 0u;
	result->total_ns = ended_ns - started_ns;
	result->ns_per_op = (double) result->total_ns / (double) iterations;
	result->codes_per_second = throughput_per_second(result->visited_code_count,
													 result->total_ns);
	result->pages_per_second = throughput_per_second(result->visited_page_count,
													 result->total_ns);
	result->requested_kernel_honored = true;
	result->runtime_available = true;
	ok = true;

cleanup:
	tq_prod_lut_reset(&lut);
	free(packed);
	free(query);
	free(input);
	return ok;
}

static bool
run_page_scan_bench(uint32_t dimension,
					MicrobenchResult *result,
					char *errmsg,
					size_t errmsg_len)
{
	TqProdCodecConfig config;
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	float query[128];
	TqCandidateHeap heap;
	TqScanScratch scratch;
	volatile float sink = 0.0f;
	uint32_t iterations = 0;
	uint64_t started_ns = 0;
	uint64_t ended_ns = 0;
	TqScanStats stats;
	bool ok = false;

	memset(&config, 0, sizeof(config));
	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(query, 0, sizeof(query));
	memset(&heap, 0, sizeof(heap));
	memset(&scratch, 0, sizeof(scratch));
	memset(&stats, 0, sizeof(stats));

	if (!build_page_scan_fixture(dimension,
								  &config,
								  &layout,
								  &lut,
								  &params,
								  page,
								  sizeof(page),
								  query,
								  sizeof(query) / sizeof(query[0]),
								  errmsg,
								  errmsg_len)
		|| !tq_candidate_heap_init(&heap, 8))
		goto cleanup;

	iterations = iterations_for_dimension(dimension) / params.lane_count;
	if (iterations == 0)
		iterations = 1;

	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1u);
	started_ns = monotonic_now_ns();
	for (uint32_t i = 0; i < iterations; i++)
	{
		if (!tq_batch_page_scan_prod_with_scratch(page,
												  sizeof(page),
												  &config,
												  true,
												  TQ_DISTANCE_COSINE,
												  &lut,
												  query,
												  dimension,
												  false,
												  0,
												  &heap,
												  NULL,
												  &scratch,
												  errmsg,
												  errmsg_len))
			goto cleanup;
		sink += (float) heap.count;
		heap.count = 0;
	}
	ended_ns = monotonic_now_ns();
	if (sink == 0.1234567f)
		fprintf(stderr, "ignore %f\n", sink);
	tq_scan_stats_snapshot(&stats);

	result->benchmark = "page_scan";
	result->requested_kernel = "auto";
	result->kernel = tq_prod_score_kernel_name(stats.score_kernel);
	result->qjl_lut_mode =
		(lut.qjl_quantized_enabled && stats.score_kernel != TQ_PROD_SCORE_SCALAR) ? "quantized" : "float";
	result->scan_layout = "row_major";
	result->lookup_style = tq_lookup_style_name(tq_lookup_style_for_kernel(stats.score_kernel));
	result->block_width = 1u;
	result->qjl_path = tq_qjl_path_name(tq_qjl_path_for_kernel(stats.score_kernel, lut.qjl_quantized_enabled));
	result->gamma_path = tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_SCALAR);
	result->dimension = dimension;
	result->bits = config.bits;
	result->iterations = iterations;
	result->lane_count = params.lane_count;
	result->visited_code_count = stats.visited_code_count;
	result->visited_page_count = stats.visited_page_count;
	result->candidate_heap_insert_count = stats.candidate_heap_insert_count;
	result->candidate_heap_replace_count = stats.candidate_heap_replace_count;
	result->candidate_heap_reject_count = stats.candidate_heap_reject_count;
	result->local_candidate_heap_insert_count = stats.local_candidate_heap_insert_count;
	result->local_candidate_heap_replace_count = stats.local_candidate_heap_replace_count;
	result->local_candidate_heap_reject_count = stats.local_candidate_heap_reject_count;
	result->local_candidate_merge_count = stats.local_candidate_merge_count;
	result->scratch_allocations = scratch.scratch_allocations;
	result->decoded_buffer_reuses = scratch.decoded_buffer_reuses;
	result->code_view_uses = scratch.code_view_uses;
	result->code_copy_uses = scratch.code_copy_uses;
	result->list_count = 0u;
	result->probe_count = 0u;
	result->total_ns = ended_ns - started_ns;
	result->ns_per_op = (double) result->total_ns / (double) iterations;
	result->codes_per_second = throughput_per_second(result->visited_code_count, result->total_ns);
	result->pages_per_second = throughput_per_second(result->visited_page_count, result->total_ns);
	result->requested_kernel_honored = true;
	switch (stats.score_kernel)
	{
		case TQ_PROD_SCORE_AVX2:
			result->runtime_available = tq_simd_avx2_runtime_available();
			break;
		case TQ_PROD_SCORE_AVX512:
			result->runtime_available = tq_simd_avx512_runtime_available();
			break;
		case TQ_PROD_SCORE_NEON:
			result->runtime_available = tq_simd_neon_runtime_available();
			break;
		case TQ_PROD_SCORE_SCALAR:
		default:
			result->runtime_available = true;
			break;
	}
	ok = true;

cleanup:
	tq_scan_scratch_reset(&scratch);
	tq_candidate_heap_reset(&heap);
	tq_prod_lut_reset(&lut);
	return ok;
}

static bool
run_page_scan_global_heap_only_bench(uint32_t dimension,
									   MicrobenchResult *result,
									   char *errmsg,
									   size_t errmsg_len)
{
	TqProdCodecConfig config;
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	float query[128];
	TqCandidateHeap heap;
	volatile float sink = 0.0f;
	uint32_t iterations = 0;
	uint64_t started_ns = 0;
	uint64_t ended_ns = 0;
	TqScanStats stats;
	TqProdScoreKernel used_kernel = TQ_PROD_SCORE_SCALAR;
	bool ok = false;

	memset(&config, 0, sizeof(config));
	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(query, 0, sizeof(query));
	memset(&heap, 0, sizeof(heap));
	memset(&stats, 0, sizeof(stats));

	if (!build_page_scan_fixture(dimension,
								  &config,
								  &layout,
								  &lut,
								  &params,
								  page,
								  sizeof(page),
								  query,
								  sizeof(query) / sizeof(query[0]),
								  errmsg,
								  errmsg_len)
		|| !tq_candidate_heap_init(&heap, 8))
		goto cleanup;

	iterations = iterations_for_dimension(dimension) / params.lane_count;
	if (iterations == 0)
		iterations = 1;

	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1u);
	tq_scan_stats_set_score_mode(TQ_SCAN_SCORE_MODE_CODE_DOMAIN);
	started_ns = monotonic_now_ns();
	for (uint32_t i = 0; i < iterations; i++)
	{
		uint16_t lane = 0;

		tq_scan_stats_record_page_visit();
		if (tq_batch_page_next_live_lane(page, sizeof(page), -1, &lane, errmsg, errmsg_len))
		{
			do
			{
				TqTid tid;
				const uint8_t *code = NULL;
				size_t code_len = 0;
				float ip_score = 0.0f;
				float distance_value = 0.0f;

				memset(&tid, 0, sizeof(tid));
				if (!tq_batch_page_get_tid(page, sizeof(page), lane, &tid, errmsg, errmsg_len)
					|| !tq_batch_page_code_view(page, sizeof(page), lane, &code, &code_len, errmsg, errmsg_len)
					|| !tq_prod_score_code_from_lut_dispatch(&config,
															 &lut,
															 code,
															 code_len,
															 tq_prod_code_domain_preferred_kernel(&config),
															 &ip_score,
															 &used_kernel,
															 errmsg,
															 errmsg_len)
					|| !tq_metric_distance_from_ip_score(TQ_DISTANCE_COSINE,
														  ip_score,
														  &distance_value,
														  errmsg,
														  errmsg_len)
					|| !tq_candidate_heap_push(&heap,
											   distance_value,
											   tid.block_number,
											   tid.offset_number))
					goto cleanup;

				tq_scan_stats_set_score_kernel(used_kernel);
				tq_scan_stats_record_code_view_uses(1);
				tq_scan_stats_record_code_visit(false);
			} while (tq_batch_page_next_live_lane(page, sizeof(page), (int) lane, &lane, errmsg, errmsg_len));
		}

		sink += (float) heap.count;
		heap.count = 0;
		heap.pop_index = 0;
		heap.sorted = false;
	}
	ended_ns = monotonic_now_ns();
	if (sink == 0.1234567f)
		fprintf(stderr, "ignore %f\n", sink);
	tq_scan_stats_snapshot(&stats);

	result->benchmark = "page_scan_global_heap_only";
	result->requested_kernel = "auto";
	result->kernel = tq_prod_score_kernel_name(used_kernel);
	result->qjl_lut_mode =
		(lut.qjl_quantized_enabled && used_kernel != TQ_PROD_SCORE_SCALAR) ? "quantized" : "float";
	result->scan_layout = "row_major";
	result->lookup_style = tq_lookup_style_name(tq_lookup_style_for_kernel(used_kernel));
	result->block_width = 1u;
	result->qjl_path = tq_qjl_path_name(tq_qjl_path_for_kernel(used_kernel, lut.qjl_quantized_enabled));
	result->gamma_path = tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_SCALAR);
	result->dimension = dimension;
	result->bits = config.bits;
	result->iterations = iterations;
	result->lane_count = params.lane_count;
	result->visited_code_count = stats.visited_code_count;
	result->visited_page_count = stats.visited_page_count;
	result->candidate_heap_insert_count = stats.candidate_heap_insert_count;
	result->candidate_heap_replace_count = stats.candidate_heap_replace_count;
	result->candidate_heap_reject_count = stats.candidate_heap_reject_count;
	result->local_candidate_heap_insert_count = stats.local_candidate_heap_insert_count;
	result->local_candidate_heap_replace_count = stats.local_candidate_heap_replace_count;
	result->local_candidate_heap_reject_count = stats.local_candidate_heap_reject_count;
	result->local_candidate_merge_count = stats.local_candidate_merge_count;
	result->scratch_allocations = 0u;
	result->decoded_buffer_reuses = 0u;
	result->code_view_uses = stats.code_view_uses;
	result->code_copy_uses = 0u;
	result->list_count = 0u;
	result->probe_count = 0u;
	result->total_ns = ended_ns - started_ns;
	result->ns_per_op = (double) result->total_ns / (double) iterations;
	result->codes_per_second = throughput_per_second(result->visited_code_count, result->total_ns);
	result->pages_per_second = throughput_per_second(result->visited_page_count, result->total_ns);
	result->requested_kernel_honored = true;
	result->runtime_available = true;
	ok = true;

cleanup:
	tq_candidate_heap_reset(&heap);
	tq_prod_lut_reset(&lut);
	return ok;
}

static bool
run_page_transpose_block16_topm_bench(uint32_t dimension,
									  MicrobenchResult *result,
									  char *errmsg,
									  size_t errmsg_len)
{
	TqProdCodecConfig config;
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	float query[128];
	TqScratchBlock16Set set;
	volatile float sink = 0.0f;
	uint32_t iterations = 0;
	uint32_t top_m = 8;
	uint64_t total_scored = 0;
	uint64_t total_survivors = 0;
	uint64_t started_ns = 0;
	uint64_t ended_ns = 0;
	bool ok = false;

	memset(&config, 0, sizeof(config));
	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(query, 0, sizeof(query));
	memset(&set, 0, sizeof(set));

	if (!build_page_scan_fixture(dimension, &config, &layout, &lut, &params,
								 page, sizeof(page), query,
								 sizeof(query) / sizeof(query[0]),
								 errmsg, errmsg_len))
		goto cleanup;

	if (!tq_prod_lut16_build(&config, &lut, &lut16, errmsg, errmsg_len))
		goto cleanup;

	if (!tq_scratch_block16_set_init(&set, config.dimension, params.lane_count, errmsg, errmsg_len))
		goto cleanup;

	iterations = iterations_for_dimension(dimension) / params.lane_count;
	if (iterations == 0)
		iterations = 1;

	started_ns = monotonic_now_ns();
	for (uint32_t i = 0; i < iterations; i++)
	{
		if (!tq_batch_page_transpose_block16(page, sizeof(page), &config, &set,
											 errmsg, errmsg_len))
			goto cleanup;

		for (uint32_t b = 0; b < set.block_count; b++)
		{
			float scores[TQ_BLOCK16_MAX_CANDIDATES];
			uint32_t selected[TQ_BLOCK16_MAX_CANDIDATES];
			uint32_t survivors = 0;

			memset(scores, 0, sizeof(scores));
			if (!tq_prod_score_block16_scalar(&lut16,
											  set.blocks[b].nibbles,
											  set.blocks[b].gammas,
											  set.blocks[b].count,
											  scores,
											  errmsg, errmsg_len))
				goto cleanup;

			survivors = tq_block16_select_top_m(scores, set.blocks[b].count, top_m, selected);
			total_scored += set.blocks[b].count;
			total_survivors += survivors;

			for (uint32_t s = 0; s < survivors; s++)
				sink += scores[selected[s]];
		}
	}
	ended_ns = monotonic_now_ns();
	if (sink == 0.1234567f)
		fprintf(stderr, "ignore %f\n", sink);

	result->benchmark = "page_transpose_block16_topm";
	result->requested_kernel = "scalar";
	result->kernel = "scalar";
	result->qjl_lut_mode = "float";
	result->scan_layout = "scratch_transposed_block16";
	result->lookup_style = tq_lookup_style_name(TQ_LOOKUP_STYLE_LUT16_SCALAR);
	result->block_width = TQ_BLOCK16_MAX_CANDIDATES;
	result->qjl_path = tq_qjl_path_name(TQ_QJL_PATH_FLOAT);
	result->gamma_path = tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_SCALAR);
	result->dimension = dimension;
	result->bits = config.bits;
	result->iterations = iterations;
	result->lane_count = params.lane_count;
	result->visited_code_count = (uint64_t) iterations * (uint64_t) params.lane_count;
	result->visited_page_count = iterations;
	result->candidate_heap_insert_count = total_survivors;
	result->candidate_heap_replace_count = 0u;
	result->candidate_heap_reject_count = total_scored - total_survivors;
	result->local_candidate_heap_insert_count = 0u;
	result->local_candidate_heap_replace_count = 0u;
	result->local_candidate_heap_reject_count = 0u;
	result->local_candidate_merge_count = 0u;
	result->scratch_allocations = 0u;
	result->decoded_buffer_reuses = 0u;
	result->code_view_uses = 0u;
	result->code_copy_uses = 0u;
	result->list_count = 0u;
	result->probe_count = 0u;
	result->total_ns = ended_ns - started_ns;
	result->ns_per_op = (double) result->total_ns / (double) iterations;
	result->codes_per_second = throughput_per_second(result->visited_code_count, result->total_ns);
	result->pages_per_second = throughput_per_second(result->visited_page_count, result->total_ns);
	result->requested_kernel_honored = true;
	result->runtime_available = true;
	ok = true;

cleanup:
	tq_scratch_block16_set_reset(&set);
	tq_prod_lut16_reset(&lut16);
	tq_prod_lut_reset(&lut);
	return ok;
}

static bool
run_router_top_probes_bench(bool partial_selection,
							   MicrobenchResult *result,
							   char *errmsg,
							   size_t errmsg_len)
{
	const uint32_t dimension = 32u;
	const uint32_t list_count = 256u;
	const uint32_t probe_count = 8u;
	const uint32_t iterations = 10000u;
	float *centroids = NULL;
	float *query = NULL;
	TqRouterProbeScore *scores = NULL;
	TqRouterProbeScore *full_scores = NULL;
	TqRouterProbeScore *partial_scores = NULL;
	TqRouterModel model;
	volatile uint32_t sink = 0;
	uint64_t started_ns = 0;
	uint64_t ended_ns = 0;
	uint32_t i = 0;
	bool ok = false;

	memset(&model, 0, sizeof(model));

	centroids = (float *) calloc((size_t) list_count * (size_t) dimension, sizeof(float));
	query = (float *) calloc(dimension, sizeof(float));
	scores = (TqRouterProbeScore *) calloc(partial_selection ? probe_count : list_count,
										   sizeof(TqRouterProbeScore));
	full_scores = (TqRouterProbeScore *) calloc(list_count, sizeof(TqRouterProbeScore));
	partial_scores = (TqRouterProbeScore *) calloc(probe_count, sizeof(TqRouterProbeScore));
	if (centroids == NULL || query == NULL || scores == NULL || full_scores == NULL || partial_scores == NULL)
		goto cleanup;

	for (i = 0; i < list_count; i++)
		seeded_unit_vector(100u + i, centroids + ((size_t) i * (size_t) dimension), dimension);
	seeded_unit_vector(17u, query, dimension);

	model.dimension = dimension;
	model.list_count = list_count;
	model.centroids = centroids;

	if (!tq_router_rank_probes(&model, query, full_scores, list_count, errmsg, errmsg_len))
		goto cleanup;
	if (!tq_router_rank_probes(&model, query, partial_scores, probe_count, errmsg, errmsg_len))
		goto cleanup;
	if (!router_top_probe_prefix_matches(full_scores, partial_scores, probe_count))
	{
		snprintf(errmsg, errmsg_len,
				 "router microbench expected partial top-probe prefix to match full sort");
		goto cleanup;
	}

	started_ns = monotonic_now_ns();
	for (i = 0; i < iterations; i++)
	{
		size_t capacity = partial_selection ? (size_t) probe_count : (size_t) list_count;

		if (!tq_router_rank_probes(&model, query, scores, capacity, errmsg, errmsg_len))
			goto cleanup;
		sink += scores[0].list_id;
	}
	ended_ns = monotonic_now_ns();
	if (sink == 0u)
		fprintf(stderr, "ignore %u\n", sink);

	result->benchmark = partial_selection
		? "router_top_probes_partial"
		: "router_top_probes_full_sort";
	result->requested_kernel = "scalar";
	result->kernel = "scalar";
	result->qjl_lut_mode = "float";
	result->scan_layout = "row_major";
	result->lookup_style = tq_lookup_style_name(TQ_LOOKUP_STYLE_SCALAR_LOOP);
	result->block_width = 0u;
	result->qjl_path = tq_qjl_path_name(TQ_QJL_PATH_FLOAT);
	result->gamma_path = tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_SCALAR);
	result->dimension = dimension;
	result->bits = 0u;
	result->iterations = iterations;
	result->lane_count = 0u;
	result->visited_code_count = (uint64_t) iterations * (uint64_t) list_count;
	result->visited_page_count = 0u;
	result->candidate_heap_insert_count = 0u;
	result->candidate_heap_replace_count = 0u;
	result->candidate_heap_reject_count = 0u;
	result->local_candidate_heap_insert_count = 0u;
	result->local_candidate_heap_replace_count = 0u;
	result->local_candidate_heap_reject_count = 0u;
	result->local_candidate_merge_count = 0u;
	result->scratch_allocations = 0u;
	result->decoded_buffer_reuses = 0u;
	result->code_view_uses = 0u;
	result->code_copy_uses = 0u;
	result->list_count = list_count;
	result->probe_count = probe_count;
	result->total_ns = ended_ns - started_ns;
	result->ns_per_op = (double) result->total_ns / (double) iterations;
	result->codes_per_second = throughput_per_second(result->visited_code_count,
													 result->total_ns);
	result->pages_per_second = 0.0;
	result->requested_kernel_honored = true;
	result->runtime_available = true;
	ok = true;

cleanup:
	free(centroids);
	free(query);
	free(scores);
	free(full_scores);
	free(partial_scores);
	return ok;
}

static bool
run_lut16_reference_bench(uint32_t dimension,
						  MicrobenchResult *result,
						  char *errmsg,
						  size_t errmsg_len)
{
	TqProdCodecConfig config = {.dimension = dimension, .bits = 4, .qjl_dimension = dimension};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	uint8_t *packed = NULL;
	uint8_t *candidate_nibbles = NULL;
	uint8_t *block16_nibbles = NULL;
	float *query = NULL;
	float *input = NULL;
	float gamma = 0.0f;
	float score = 0.0f;
	volatile float sink = 0.0f;
	uint64_t started_ns = 0;
	uint64_t ended_ns = 0;
	uint32_t iterations = iterations_for_dimension(dimension);
	uint32_t i = 0;
	bool ok = false;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));

	query = (float *) calloc(dimension, sizeof(float));
	input = (float *) calloc(dimension, sizeof(float));
	candidate_nibbles = (uint8_t *) calloc(dimension, sizeof(uint8_t));
	block16_nibbles = (uint8_t *) calloc((size_t) dimension * 16u, sizeof(uint8_t));
	if (query == NULL || input == NULL || candidate_nibbles == NULL
		|| block16_nibbles == NULL)
		goto cleanup;

	seeded_unit_vector(17u + dimension, query, dimension);
	seeded_unit_vector(31u + dimension, input, dimension);

	if (!tq_prod_packed_layout(&config, &layout, errmsg, errmsg_len))
		goto cleanup;

	packed = (uint8_t *) calloc(layout.total_bytes, sizeof(uint8_t));
	if (packed == NULL)
		goto cleanup;

	if (!tq_prod_lut_build(&config, query, &lut, errmsg, errmsg_len)
		|| !tq_prod_lut16_build(&config, &lut, &lut16, errmsg, errmsg_len)
		|| !tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, errmsg_len)
		|| !tq_prod_read_gamma(&config, packed, layout.total_bytes, &gamma, errmsg, errmsg_len)
		|| !tq_prod_extract_nibbles(&config, packed, layout.total_bytes,
									candidate_nibbles, dimension, errmsg, errmsg_len))
		goto cleanup;

	/* The block16 scorer consumes a dimension*16 lane-major nibble matrix. */
	expand_single_candidate_nibbles(candidate_nibbles, dimension, block16_nibbles);

	started_ns = monotonic_now_ns();
	for (i = 0; i < iterations; i++)
	{
		if (!tq_prod_score_block16_scalar(&lut16, block16_nibbles, &gamma, 1, &score,
										  errmsg, errmsg_len))
			goto cleanup;
		sink += score;
	}
	ended_ns = monotonic_now_ns();
	if (sink == 0.1234567f)
		fprintf(stderr, "ignore %f\n", sink);

	result->benchmark = "score_lut16_reference";
	result->requested_kernel = "scalar";
	result->kernel = "scalar";
	result->qjl_lut_mode = "float";
	result->scan_layout = "row_major";
	result->lookup_style = tq_lookup_style_name(TQ_LOOKUP_STYLE_LUT16_SCALAR);
	result->block_width = 1u;
	result->qjl_path = tq_qjl_path_name(TQ_QJL_PATH_FLOAT);
	result->gamma_path = tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_SCALAR);
	result->dimension = dimension;
	result->bits = config.bits;
	result->iterations = iterations;
	result->lane_count = 1u;
	result->visited_code_count = iterations;
	result->visited_page_count = 0u;
	result->candidate_heap_insert_count = 0u;
	result->candidate_heap_replace_count = 0u;
	result->candidate_heap_reject_count = 0u;
	result->local_candidate_heap_insert_count = 0u;
	result->local_candidate_heap_replace_count = 0u;
	result->local_candidate_heap_reject_count = 0u;
	result->local_candidate_merge_count = 0u;
	result->scratch_allocations = 0u;
	result->decoded_buffer_reuses = 0u;
	result->code_view_uses = 0u;
	result->code_copy_uses = 0u;
	result->list_count = 0u;
	result->probe_count = 0u;
	result->total_ns = ended_ns - started_ns;
	result->ns_per_op = (double) result->total_ns / (double) iterations;
	result->codes_per_second = throughput_per_second(result->visited_code_count,
													 result->total_ns);
	result->pages_per_second = 0.0;
	result->requested_kernel_honored = true;
	result->runtime_available = true;
	ok = true;

cleanup:
	tq_prod_lut16_reset(&lut16);
	tq_prod_lut_reset(&lut);
	free(packed);
	free(candidate_nibbles);
	free(block16_nibbles);
	free(query);
	free(input);
	return ok;
}

static bool
run_lut16_dispatch_bench(uint32_t dimension,
						 MicrobenchResult *result,
						 char *errmsg,
						 size_t errmsg_len)
{
	TqProdCodecConfig config = {.dimension = dimension, .bits = 4, .qjl_dimension = dimension};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	uint8_t *packed = NULL;
	uint8_t *candidate_nibbles = NULL;
	uint8_t *block16_nibbles = NULL;
	float *query = NULL;
	float *input = NULL;
	float gamma = 0.0f;
	float score = 0.0f;
	volatile float sink = 0.0f;
	uint64_t started_ns = 0;
	uint64_t ended_ns = 0;
	uint32_t iterations = iterations_for_dimension(dimension);
	uint32_t i = 0;
	TqProdScoreKernel used_kernel = TQ_PROD_SCORE_AUTO;
	bool ok = false;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));

	query = (float *) calloc(dimension, sizeof(float));
	input = (float *) calloc(dimension, sizeof(float));
	candidate_nibbles = (uint8_t *) calloc(dimension, sizeof(uint8_t));
	block16_nibbles = (uint8_t *) calloc((size_t) dimension * 16u, sizeof(uint8_t));
	if (query == NULL || input == NULL || candidate_nibbles == NULL
		|| block16_nibbles == NULL)
		goto cleanup;

	seeded_unit_vector(17u + dimension, query, dimension);
	seeded_unit_vector(31u + dimension, input, dimension);

	if (!tq_prod_packed_layout(&config, &layout, errmsg, errmsg_len))
		goto cleanup;

	packed = (uint8_t *) calloc(layout.total_bytes, sizeof(uint8_t));
	if (packed == NULL)
		goto cleanup;

	if (!tq_prod_lut_build(&config, query, &lut, errmsg, errmsg_len)
		|| !tq_prod_lut16_build(&config, &lut, &lut16, errmsg, errmsg_len)
		|| !tq_prod_lut16_quantize(&lut16, errmsg, errmsg_len)
		|| !tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, errmsg_len)
		|| !tq_prod_read_gamma(&config, packed, layout.total_bytes, &gamma, errmsg, errmsg_len)
		|| !tq_prod_extract_nibbles(&config, packed, layout.total_bytes,
									candidate_nibbles, dimension, errmsg, errmsg_len))
		goto cleanup;

	expand_single_candidate_nibbles(candidate_nibbles, dimension, block16_nibbles);

	started_ns = monotonic_now_ns();
	for (i = 0; i < iterations; i++)
	{
		if (!tq_prod_score_block16_dispatch(&lut16, block16_nibbles, &gamma, 1,
											TQ_PROD_SCORE_AUTO, &score, &used_kernel,
											errmsg, errmsg_len))
			goto cleanup;
		sink += score;
	}
	ended_ns = monotonic_now_ns();
	if (sink == 0.1234567f)
		fprintf(stderr, "ignore %f\n", sink);

	result->benchmark = "score_lut16_dispatch";
	result->requested_kernel = "auto";
	result->kernel = tq_prod_score_kernel_name(used_kernel);
	result->qjl_lut_mode = "float";
	result->scan_layout = "row_major";
	result->lookup_style = tq_lookup_style_name(
		used_kernel == TQ_PROD_SCORE_AVX2 ? TQ_LOOKUP_STYLE_LUT16_AVX2 :
		used_kernel == TQ_PROD_SCORE_NEON ? TQ_LOOKUP_STYLE_LUT16_NEON :
		TQ_LOOKUP_STYLE_LUT16_SCALAR);
	result->block_width = 1u;
	result->qjl_path = tq_qjl_path_name(TQ_QJL_PATH_FLOAT);
	result->gamma_path = tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_SCALAR);
	result->dimension = dimension;
	result->bits = config.bits;
	result->iterations = iterations;
	result->lane_count = 1u;
	result->visited_code_count = iterations;
	result->visited_page_count = 0u;
	result->candidate_heap_insert_count = 0u;
	result->candidate_heap_replace_count = 0u;
	result->candidate_heap_reject_count = 0u;
	result->local_candidate_heap_insert_count = 0u;
	result->local_candidate_heap_replace_count = 0u;
	result->local_candidate_heap_reject_count = 0u;
	result->local_candidate_merge_count = 0u;
	result->scratch_allocations = 0u;
	result->decoded_buffer_reuses = 0u;
	result->code_view_uses = 0u;
	result->code_copy_uses = 0u;
	result->list_count = 0u;
	result->probe_count = 0u;
	result->total_ns = ended_ns - started_ns;
	result->ns_per_op = (double) result->total_ns / (double) iterations;
	result->codes_per_second = throughput_per_second(result->visited_code_count,
													 result->total_ns);
	result->pages_per_second = 0.0;
	result->requested_kernel_honored = true;
	result->runtime_available = true;
	ok = true;

cleanup:
	tq_prod_lut16_reset(&lut16);
	tq_prod_lut_reset(&lut);
	free(packed);
	free(candidate_nibbles);
	free(block16_nibbles);
	free(query);
	free(input);
	return ok;
}

static bool
run_lut16_quantized_fused_bench(uint32_t dimension,
								MicrobenchResult *result,
								char *errmsg,
								size_t errmsg_len)
{
	TqProdCodecConfig config = {.dimension = dimension, .bits = 4, .qjl_dimension = dimension};
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	uint8_t *packed = NULL;
	uint8_t *candidate_nibbles = NULL;
	uint8_t *block16_nibbles = NULL;
	float *query = NULL;
	float *input = NULL;
	float gamma = 0.0f;
	float score = 0.0f;
	volatile float sink = 0.0f;
	uint64_t started_ns = 0;
	uint64_t ended_ns = 0;
	uint32_t iterations = iterations_for_dimension(dimension);
	uint32_t i = 0;
	bool ok = false;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));

	query = (float *) calloc(dimension, sizeof(float));
	input = (float *) calloc(dimension, sizeof(float));
	candidate_nibbles = (uint8_t *) calloc(dimension, sizeof(uint8_t));
	block16_nibbles = (uint8_t *) calloc((size_t) dimension * 16u, sizeof(uint8_t));
	if (query == NULL || input == NULL || candidate_nibbles == NULL
		|| block16_nibbles == NULL)
		goto cleanup;

	seeded_unit_vector(17u + dimension, query, dimension);
	seeded_unit_vector(31u + dimension, input, dimension);

	if (!tq_prod_packed_layout(&config, &layout, errmsg, errmsg_len))
		goto cleanup;

	packed = (uint8_t *) calloc(layout.total_bytes, sizeof(uint8_t));
	if (packed == NULL)
		goto cleanup;

	if (!tq_prod_lut_build(&config, query, &lut, errmsg, errmsg_len)
		|| !tq_prod_lut16_build(&config, &lut, &lut16, errmsg, errmsg_len)
		|| !tq_prod_lut16_quantize(&lut16, errmsg, errmsg_len)
		|| !tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, errmsg_len)
		|| !tq_prod_read_gamma(&config, packed, layout.total_bytes, &gamma, errmsg, errmsg_len)
		|| !tq_prod_extract_nibbles(&config, packed, layout.total_bytes,
									candidate_nibbles, dimension, errmsg, errmsg_len))
		goto cleanup;

	expand_single_candidate_nibbles(candidate_nibbles, dimension, block16_nibbles);

	started_ns = monotonic_now_ns();
	for (i = 0; i < iterations; i++)
	{
		if (!tq_prod_score_block16_quantized_scalar(&lut16, block16_nibbles, &gamma, 1, &score,
													errmsg, errmsg_len))
			goto cleanup;
		sink += score;
	}
	ended_ns = monotonic_now_ns();
	if (sink == 0.1234567f)
		fprintf(stderr, "ignore %f\n", sink);

	result->benchmark = "score_lut16_quantized_fused";
	result->requested_kernel = "scalar";
	result->kernel = "scalar";
	result->qjl_lut_mode = "quantized";
	result->scan_layout = "row_major";
	result->lookup_style = tq_lookup_style_name(TQ_LOOKUP_STYLE_LUT16_SCALAR);
	result->block_width = 1u;
	result->qjl_path = tq_qjl_path_name(TQ_QJL_PATH_LUT16_QUANTIZED);
	result->gamma_path = tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_SCALAR);
	result->dimension = dimension;
	result->bits = config.bits;
	result->iterations = iterations;
	result->lane_count = 1u;
	result->visited_code_count = iterations;
	result->visited_page_count = 0u;
	result->candidate_heap_insert_count = 0u;
	result->candidate_heap_replace_count = 0u;
	result->candidate_heap_reject_count = 0u;
	result->local_candidate_heap_insert_count = 0u;
	result->local_candidate_heap_replace_count = 0u;
	result->local_candidate_heap_reject_count = 0u;
	result->local_candidate_merge_count = 0u;
	result->scratch_allocations = 0u;
	result->decoded_buffer_reuses = 0u;
	result->code_view_uses = 0u;
	result->code_copy_uses = 0u;
	result->list_count = 0u;
	result->probe_count = 0u;
	result->total_ns = ended_ns - started_ns;
	result->ns_per_op = (double) result->total_ns / (double) iterations;
	result->codes_per_second = throughput_per_second(result->visited_code_count,
													 result->total_ns);
	result->pages_per_second = 0.0;
	result->requested_kernel_honored = true;
	result->runtime_available = true;
	ok = true;

cleanup:
	tq_prod_lut16_reset(&lut16);
	tq_prod_lut_reset(&lut);
	free(packed);
	free(candidate_nibbles);
	free(block16_nibbles);
	free(query);
	free(input);
	return ok;
}

static bool
run_page_transpose_only_bench(uint32_t dimension,
							  MicrobenchResult *result,
							  char *errmsg,
							  size_t errmsg_len)
{
	TqProdCodecConfig config;
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	float query[128];
	TqScratchBlock16Set set;
	uint32_t iterations = 0;
	uint64_t started_ns = 0;
	uint64_t ended_ns = 0;
	bool ok = false;

	memset(&config, 0, sizeof(config));
	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(query, 0, sizeof(query));
	memset(&set, 0, sizeof(set));

	if (!build_page_scan_fixture(dimension, &config, &layout, &lut, &params,
								 page, sizeof(page), query,
								 sizeof(query) / sizeof(query[0]),
								 errmsg, errmsg_len))
		goto cleanup;

	if (!tq_scratch_block16_set_init(&set, config.dimension, params.lane_count, errmsg, errmsg_len))
		goto cleanup;

	iterations = iterations_for_dimension(dimension) / params.lane_count;
	if (iterations == 0)
		iterations = 1;

	started_ns = monotonic_now_ns();
	for (uint32_t i = 0; i < iterations; i++)
	{
		if (!tq_batch_page_transpose_block16(page, sizeof(page), &config, &set,
											 errmsg, errmsg_len))
			goto cleanup;
	}
	ended_ns = monotonic_now_ns();

	result->benchmark = "page_transpose_only";
	result->requested_kernel = "scalar";
	result->kernel = "scalar";
	result->qjl_lut_mode = "float";
	result->scan_layout = "scratch_transposed_block16";
	result->lookup_style = tq_lookup_style_name(TQ_LOOKUP_STYLE_SCALAR_LOOP);
	result->block_width = TQ_BLOCK16_MAX_CANDIDATES;
	result->qjl_path = tq_qjl_path_name(TQ_QJL_PATH_FLOAT);
	result->gamma_path = tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_SCALAR);
	result->dimension = dimension;
	result->bits = config.bits;
	result->iterations = iterations;
	result->lane_count = params.lane_count;
	result->visited_code_count = (uint64_t) iterations * (uint64_t) params.lane_count;
	result->visited_page_count = iterations;
	result->candidate_heap_insert_count = 0u;
	result->candidate_heap_replace_count = 0u;
	result->candidate_heap_reject_count = 0u;
	result->local_candidate_heap_insert_count = 0u;
	result->local_candidate_heap_replace_count = 0u;
	result->local_candidate_heap_reject_count = 0u;
	result->local_candidate_merge_count = 0u;
	result->scratch_allocations = 0u;
	result->decoded_buffer_reuses = 0u;
	result->code_view_uses = 0u;
	result->code_copy_uses = 0u;
	result->list_count = 0u;
	result->probe_count = 0u;
	result->total_ns = ended_ns - started_ns;
	result->ns_per_op = (double) result->total_ns / (double) iterations;
	result->codes_per_second = throughput_per_second(result->visited_code_count, result->total_ns);
	result->pages_per_second = throughput_per_second(result->visited_page_count, result->total_ns);
	result->requested_kernel_honored = true;
	result->runtime_available = true;
	ok = true;

cleanup:
	tq_scratch_block16_set_reset(&set);
	tq_prod_lut_reset(&lut);
	return ok;
}

static bool
run_page_transpose_block16_bench(uint32_t dimension,
								 MicrobenchResult *result,
								 char *errmsg,
								 size_t errmsg_len)
{
	TqProdCodecConfig config;
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqProdLut16 lut16;
	TqBatchPageParams params;
	uint8_t page[TQ_DEFAULT_BLOCK_SIZE];
	float query[128];
	TqScratchBlock16Set set;
	volatile float sink = 0.0f;
	uint32_t iterations = 0;
	uint64_t started_ns = 0;
	uint64_t ended_ns = 0;
	bool ok = false;

	memset(&config, 0, sizeof(config));
	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(&lut16, 0, sizeof(lut16));
	memset(&params, 0, sizeof(params));
	memset(page, 0, sizeof(page));
	memset(query, 0, sizeof(query));
	memset(&set, 0, sizeof(set));

	if (!build_page_scan_fixture(dimension, &config, &layout, &lut, &params,
								 page, sizeof(page), query,
								 sizeof(query) / sizeof(query[0]),
								 errmsg, errmsg_len))
		goto cleanup;

	if (!tq_prod_lut16_build(&config, &lut, &lut16, errmsg, errmsg_len))
		goto cleanup;

	if (!tq_scratch_block16_set_init(&set, config.dimension, params.lane_count, errmsg, errmsg_len))
		goto cleanup;

	iterations = iterations_for_dimension(dimension) / params.lane_count;
	if (iterations == 0)
		iterations = 1;

	started_ns = monotonic_now_ns();
	for (uint32_t i = 0; i < iterations; i++)
	{
		if (!tq_batch_page_transpose_block16(page, sizeof(page), &config, &set,
											 errmsg, errmsg_len))
			goto cleanup;

		for (uint32_t b = 0; b < set.block_count; b++)
		{
			float scores[TQ_BLOCK16_MAX_CANDIDATES];

			memset(scores, 0, sizeof(scores));
			if (!tq_prod_score_block16_scalar(&lut16,
											  set.blocks[b].nibbles,
											  set.blocks[b].gammas,
											  set.blocks[b].count,
											  scores,
											  errmsg, errmsg_len))
				goto cleanup;

			for (uint32_t c = 0; c < set.blocks[b].count; c++)
				sink += scores[c];
		}
	}
	ended_ns = monotonic_now_ns();
	if (sink == 0.1234567f)
		fprintf(stderr, "ignore %f\n", sink);

	result->benchmark = "page_transpose_block16_scalar";
	result->requested_kernel = "scalar";
	result->kernel = "scalar";
	result->qjl_lut_mode = "float";
	result->scan_layout = "scratch_transposed_block16";
	result->lookup_style = tq_lookup_style_name(TQ_LOOKUP_STYLE_LUT16_SCALAR);
	result->block_width = TQ_BLOCK16_MAX_CANDIDATES;
	result->qjl_path = tq_qjl_path_name(TQ_QJL_PATH_FLOAT);
	result->gamma_path = tq_gamma_path_name(TQ_GAMMA_PATH_FLOAT32_SCALAR);
	result->dimension = dimension;
	result->bits = config.bits;
	result->iterations = iterations;
	result->lane_count = params.lane_count;
	result->visited_code_count = (uint64_t) iterations * (uint64_t) params.lane_count;
	result->visited_page_count = iterations;
	result->candidate_heap_insert_count = 0u;
	result->candidate_heap_replace_count = 0u;
	result->candidate_heap_reject_count = 0u;
	result->local_candidate_heap_insert_count = 0u;
	result->local_candidate_heap_replace_count = 0u;
	result->local_candidate_heap_reject_count = 0u;
	result->local_candidate_merge_count = 0u;
	result->scratch_allocations = 0u;
	result->decoded_buffer_reuses = 0u;
	result->code_view_uses = 0u;
	result->code_copy_uses = 0u;
	result->list_count = 0u;
	result->probe_count = 0u;
	result->total_ns = ended_ns - started_ns;
	result->ns_per_op = (double) result->total_ns / (double) iterations;
	result->codes_per_second = throughput_per_second(result->visited_code_count, result->total_ns);
	result->pages_per_second = throughput_per_second(result->visited_page_count, result->total_ns);
	result->requested_kernel_honored = true;
	result->runtime_available = true;
	ok = true;

cleanup:
	tq_scratch_block16_set_reset(&set);
	tq_prod_lut16_reset(&lut16);
	tq_prod_lut_reset(&lut);
	return ok;
}

static void
emit_results_json(const MicrobenchResult *results, size_t count)
{
	const char *arch = "unknown";
	bool first = true;
	size_t i = 0;

#if defined(__aarch64__) || defined(_M_ARM64)
	arch = "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
	arch = "x86_64";
#endif

	printf("{\"architecture\":\"%s\",\"simd\":{", arch);
	printf("\"scalar\":{\"compiled\":true,\"runtime_available\":true},");
	printf("\"avx2\":{\"compiled\":%s,\"runtime_available\":%s},",
		   tq_simd_avx2_compile_available() ? "true" : "false",
		   tq_simd_avx2_runtime_available() ? "true" : "false");
	printf("\"avx512\":{\"compiled\":%s,\"runtime_available\":%s},",
		   tq_simd_avx512_compile_available() ? "true" : "false",
		   tq_simd_avx512_runtime_available() ? "true" : "false");
	printf("\"neon\":{\"compiled\":%s,\"runtime_available\":%s}",
		   tq_simd_neon_compile_available() ? "true" : "false",
		   tq_simd_neon_runtime_available() ? "true" : "false");
	printf("},\"results\":[");

	for (i = 0; i < count; i++)
	{
		if (!first)
			printf(",");
		first = false;
		printf(
			"{\"benchmark\":\"%s\",\"requested_kernel\":\"%s\",\"kernel\":\"%s\","
			"\"qjl_lut_mode\":\"%s\",\"scan_layout\":\"%s\","
			"\"lookup_style\":\"%s\",\"block_width\":%u,"
			"\"qjl_path\":\"%s\",\"gamma_path\":\"%s\","
			"\"requested_kernel_honored\":%s,\"dimension\":%u,\"bits\":%u,"
			"\"iterations\":%u,\"lane_count\":%u,\"visited_code_count\":%llu,"
			"\"visited_page_count\":%llu,\"candidate_heap_insert_count\":%llu,"
			"\"candidate_heap_replace_count\":%llu,\"candidate_heap_reject_count\":%llu,"
			"\"local_candidate_heap_insert_count\":%llu,"
			"\"local_candidate_heap_replace_count\":%llu,"
			"\"local_candidate_heap_reject_count\":%llu,"
			"\"local_candidate_merge_count\":%llu,"
			"\"scratch_allocations\":%llu,\"decoded_buffer_reuses\":%llu,"
			"\"code_view_uses\":%llu,\"code_copy_uses\":%llu,"
			"\"list_count\":%u,\"probe_count\":%u,"
			"\"total_ns\":%llu,\"ns_per_op\":%.3f,\"codes_per_second\":%.3f,"
			"\"pages_per_second\":%.3f,"
			"\"runtime_available\":%s}",
			results[i].benchmark,
			results[i].requested_kernel,
			results[i].kernel,
			results[i].qjl_lut_mode,
			results[i].scan_layout,
			results[i].lookup_style,
			results[i].block_width,
			results[i].qjl_path,
			results[i].gamma_path,
			results[i].requested_kernel_honored ? "true" : "false",
			results[i].dimension,
			(unsigned int) results[i].bits,
			results[i].iterations,
			results[i].lane_count,
			(unsigned long long) results[i].visited_code_count,
			(unsigned long long) results[i].visited_page_count,
			(unsigned long long) results[i].candidate_heap_insert_count,
			(unsigned long long) results[i].candidate_heap_replace_count,
			(unsigned long long) results[i].candidate_heap_reject_count,
			(unsigned long long) results[i].local_candidate_heap_insert_count,
			(unsigned long long) results[i].local_candidate_heap_replace_count,
			(unsigned long long) results[i].local_candidate_heap_reject_count,
			(unsigned long long) results[i].local_candidate_merge_count,
			(unsigned long long) results[i].scratch_allocations,
			(unsigned long long) results[i].decoded_buffer_reuses,
			(unsigned long long) results[i].code_view_uses,
			(unsigned long long) results[i].code_copy_uses,
			results[i].list_count,
			results[i].probe_count,
			(unsigned long long) results[i].total_ns,
			results[i].ns_per_op,
			results[i].codes_per_second,
			results[i].pages_per_second,
			results[i].runtime_available ? "true" : "false");
	}

	printf("],\"gates\":[");

	/* Gate: LUT16 dispatch kernel must match the best available SIMD */
	{
		const MicrobenchResult *lut16_dispatch = NULL;
		const MicrobenchResult *scalar_baseline = NULL;
		const MicrobenchResult *lut16_ref = NULL;
		bool simd_available = tq_simd_avx2_runtime_available() || tq_simd_neon_runtime_available();
		bool first_gate = true;

		for (i = 0; i < count; i++)
		{
			if (strcmp(results[i].benchmark, "score_lut16_dispatch") == 0)
				lut16_dispatch = &results[i];
			else if (strcmp(results[i].benchmark, "score_code_from_lut") == 0
					 && strcmp(results[i].requested_kernel, "scalar") == 0)
				scalar_baseline = &results[i];
			else if (strcmp(results[i].benchmark, "score_lut16_reference") == 0)
				lut16_ref = &results[i];
		}

		if (lut16_dispatch != NULL)
		{
			bool kernel_ok = true;

			if (simd_available)
				kernel_ok = strcmp(lut16_dispatch->kernel, "scalar") != 0;

			if (!first_gate)
				printf(",");
			first_gate = false;
			printf("{\"gate\":\"lut16_dispatch_kernel_selection\","
				   "\"passed\":%s,"
				   "\"simd_available\":%s,"
				   "\"selected_kernel\":\"%s\","
				   "\"reason\":\"%s\"}",
				   kernel_ok ? "true" : "false",
				   simd_available ? "true" : "false",
				   lut16_dispatch->kernel,
				   kernel_ok ? "kernel matches best available SIMD"
							: "supported host fell back to scalar unexpectedly");
		}

		if (lut16_ref != NULL && scalar_baseline != NULL
			&& scalar_baseline->ns_per_op > 0.0)
		{
			double ratio = scalar_baseline->ns_per_op / lut16_ref->ns_per_op;

			if (!first_gate)
				printf(",");
			first_gate = false;
			printf("{\"gate\":\"lut16_reference_not_slower_than_scalar\","
				   "\"passed\":%s,"
				   "\"scalar_ns_per_op\":%.3f,"
				   "\"lut16_ns_per_op\":%.3f,"
				   "\"ratio\":%.3f,"
				   "\"reason\":\"%s\"}",
				   ratio >= 0.5 ? "true" : "false",
				   scalar_baseline->ns_per_op,
				   lut16_ref->ns_per_op,
				   ratio,
				   ratio >= 0.5 ? "lut16 reference is within acceptable range"
							   : "lut16 reference is more than 2x slower than scalar");
		}

		/* Gate: page_scan local heap reduces global inserts */
		{
			const MicrobenchResult *page_scan = NULL;

			for (i = 0; i < count; i++)
			{
				if (strcmp(results[i].benchmark, "page_scan") == 0)
					page_scan = &results[i];
			}

			if (page_scan != NULL && page_scan->visited_code_count > 0)
			{
				bool reduces = page_scan->candidate_heap_insert_count < page_scan->visited_code_count;

				if (!first_gate)
					printf(",");
				first_gate = false;
				printf("{\"gate\":\"page_scan_local_heap_reduces_global_inserts\","
					   "\"passed\":%s,"
					   "\"heap_inserts\":%llu,"
					   "\"visited_codes\":%llu,"
					   "\"reason\":\"%s\"}",
					   reduces ? "true" : "false",
					   (unsigned long long) page_scan->candidate_heap_insert_count,
					   (unsigned long long) page_scan->visited_code_count,
					   reduces ? "local heap selection reduces global heap pressure"
							  : "global heap saw every visited code without reduction");
			}
		}
	}

	printf("]}\n");
}

int
main(void)
{
	MicrobenchResult results[20];
	size_t result_count = 0;
	char errmsg[256];

	memset(results, 0, sizeof(results));
	memset(errmsg, 0, sizeof(errmsg));

	if (!run_code_domain_bench(TQ_PROD_SCORE_SCALAR, 32u, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "scalar microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_code_domain_bench(TQ_PROD_SCORE_AUTO, 32u, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "auto microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_code_domain_bench(TQ_PROD_SCORE_AVX2, 32u, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "explicit avx2 microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_code_domain_bench(TQ_PROD_SCORE_NEON, 32u, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "explicit neon microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_quantized_reference_bench(32u, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "quantized reference microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_lut16_reference_bench(32u, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "lut16 reference microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_page_scan_bench(32u,
							 &results[result_count],
							 errmsg,
							 sizeof(errmsg)))
	{
		fprintf(stderr, "page scan microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_page_scan_global_heap_only_bench(32u, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "global-heap page scan microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_lut16_dispatch_bench(32u, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "lut16 dispatch microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_lut16_quantized_fused_bench(32u, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "lut16 quantized fused microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_page_transpose_only_bench(32u, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "page transpose only microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_page_transpose_block16_bench(32u, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "page transpose+block16 microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_page_transpose_block16_topm_bench(32u, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "page transpose+block16+topM microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_router_top_probes_bench(false, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "router full-sort microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	if (!run_router_top_probes_bench(true, &results[result_count], errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "router partial microbench failed: %s\n", errmsg);
		return 1;
	}
	result_count++;

	emit_results_json(results, result_count);
	return 0;
}
