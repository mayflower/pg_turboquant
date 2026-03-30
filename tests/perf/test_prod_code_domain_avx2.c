#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "src/tq_codec_prod.h"
#include "src/tq_simd_avx2.h"

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

static double
monotonic_now_seconds(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double) ts.tv_sec + ((double) ts.tv_nsec / 1000000000.0);
}

static uint32_t
iterations_for_dimension(uint32_t dimension)
{
	if (dimension <= 32u)
		return 250000u;
	if (dimension <= 128u)
		return 150000u;
	if (dimension <= 768u)
		return 30000u;
	return 15000u;
}

static int
run_shape(uint32_t dimension)
{
	TqProdCodecConfig config = {.dimension = dimension, .bits = 4};
	TqProdPackedLayout layout;
	TqProdLut lut;
	uint8_t *packed = NULL;
	float *query = NULL;
	float *input = NULL;
	float scalar_score = 0.0f;
	float avx2_score = 0.0f;
	volatile float sink = 0.0f;
	uint32_t iterations = iterations_for_dimension(dimension);
	uint32_t i = 0;
	char errmsg[256];
	double scalar_start = 0.0;
	double scalar_end = 0.0;
	double avx2_start = 0.0;
	double avx2_end = 0.0;
	int exit_code = 1;

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));

	query = (float *) calloc(dimension, sizeof(float));
	input = (float *) calloc(dimension, sizeof(float));
	if (query == NULL || input == NULL)
		goto cleanup;

	seeded_unit_vector(17u + dimension, query, dimension);
	seeded_unit_vector(31u + dimension, input, dimension);

	if (!tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "layout failed for d=%u: %s\n", dimension, errmsg);
		goto cleanup;
	}

	packed = (uint8_t *) calloc(layout.total_bytes, sizeof(uint8_t));
	if (packed == NULL)
		goto cleanup;

	if (!tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "lut failed for d=%u: %s\n", dimension, errmsg);
		goto cleanup;
	}

	if (!tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)))
	{
		fprintf(stderr, "encode failed for d=%u: %s\n", dimension, errmsg);
		goto cleanup;
	}

	scalar_start = monotonic_now_seconds();
	for (i = 0; i < iterations; i++)
	{
		if (!tq_prod_score_code_from_lut_dispatch(&config, &lut, packed, layout.total_bytes,
												  TQ_PROD_SCORE_SCALAR, &scalar_score, NULL,
												  errmsg, sizeof(errmsg)))
		{
			fprintf(stderr, "scalar dispatch failed for d=%u: %s\n", dimension, errmsg);
			goto cleanup;
		}
		sink += scalar_score;
	}
	scalar_end = monotonic_now_seconds();

	avx2_start = monotonic_now_seconds();
	for (i = 0; i < iterations; i++)
	{
		if (!tq_prod_score_code_from_lut_dispatch(&config, &lut, packed, layout.total_bytes,
												  TQ_PROD_SCORE_AVX2, &avx2_score, NULL,
												  errmsg, sizeof(errmsg)))
		{
			fprintf(stderr, "avx2 dispatch failed for d=%u: %s\n", dimension, errmsg);
			goto cleanup;
		}
		sink += avx2_score;
	}
	avx2_end = monotonic_now_seconds();

	printf(
		"d=%u iterations=%u scalar_ns=%.1f avx2_ns=%.1f speedup=%.3fx\n",
		dimension,
		iterations,
		((scalar_end - scalar_start) * 1000000000.0) / (double) iterations,
		((avx2_end - avx2_start) * 1000000000.0) / (double) iterations,
		(scalar_end - scalar_start) / (avx2_end - avx2_start));
	if (sink == 0.1234567f)
		fprintf(stderr, "ignore %f\n", sink);

	exit_code = 0;

cleanup:
	tq_prod_lut_reset(&lut);
	free(packed);
	free(query);
	free(input);
	return exit_code;
}

int
main(void)
{
	const uint32_t dimensions[] = {32u, 128u, 768u, 1536u};
	size_t index = 0;

	if (!tq_simd_avx2_runtime_available())
	{
		printf("AVX2 code-domain microbenchmark skipped: runtime unavailable on this machine.\n");
		return 0;
	}

	for (index = 0; index < (sizeof(dimensions) / sizeof(dimensions[0])); index++)
	{
		if (run_shape(dimensions[index]) != 0)
			return 1;
	}

	return 0;
}
