#include "src/tq_simd_avx2.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define TQ_CAN_COMPILE_AVX2 1
#define TQ_CAN_COMPILE_AVX512 1
#else
#define TQ_CAN_COMPILE_AVX2 0
#define TQ_CAN_COMPILE_AVX512 0
#endif

#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || defined(_M_ARM)
#include <arm_neon.h>
#define TQ_CAN_COMPILE_NEON 1
#else
#define TQ_CAN_COMPILE_NEON 0
#endif

static bool tq_simd_force_disabled = false;

static void
tq_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg == NULL || errmsg_len == 0)
		return;

	snprintf(errmsg, errmsg_len, "%s", message);
}

static bool
tq_validate_score_inputs(const TqProdCodecConfig *config,
						 const float *query,
						 size_t query_len,
						 const uint8_t *packed,
						 float *score,
						 char *errmsg,
						 size_t errmsg_len)
{
	if (config == NULL || query == NULL || packed == NULL || score == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant score dispatch: config, query, packed input, and score output must be non-null");
		return false;
	}

	if (query_len != config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant score dispatch: query length must match codec dimension");
		return false;
	}

	return true;
}

static bool
tq_validate_code_domain_inputs(const TqProdCodecConfig *config,
							   const TqProdLut *lut,
							   const uint8_t *packed,
							   float *score,
							   char *errmsg,
							   size_t errmsg_len)
{
	if (config == NULL || lut == NULL || packed == NULL || score == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant code-domain dispatch: config, lut, packed input, and score output must be non-null");
		return false;
	}

	if (lut->dimension != config->dimension
		|| lut->level_count != ((uint32_t) 1u << ((uint32_t) config->bits - 1u))
		|| lut->values == NULL
		|| lut->query_signs == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant code-domain dispatch: lut shape must match codec config");
		return false;
	}

	return true;
}

static bool
tq_prod_code_domain_shape_supports_avx2(const TqProdCodecConfig *config)
{
	if (config == NULL)
		return false;

	return config->bits == 4
		&& config->dimension > 0
		&& (config->dimension % 8u) == 0u;
}

static float
tq_dot_product_scalar(const float *left, const float *right, size_t len)
{
	float		sum = 0.0f;
	size_t		i = 0;

	for (i = 0; i < len; i++)
		sum += left[i] * right[i];

	return sum;
}

#if TQ_CAN_COMPILE_AVX2
static uint32_t
tq_unpack_bits_local(const uint8_t *packed, uint32_t bit_offset, uint32_t bit_count)
{
	uint32_t value = 0;
	uint32_t bit = 0;

	for (bit = 0; bit < bit_count; bit++)
	{
		uint32_t target_bit = bit_offset + bit;
		uint32_t byte_index = target_bit / 8;
		uint32_t byte_shift = target_bit % 8;

		if ((packed[byte_index] >> byte_shift) & 1u)
			value |= UINT32_C(1) << bit;
	}

	return value;
}

__attribute__((target("avx2")))
static float
tq_dot_product_avx2_impl(const float *left, const float *right, size_t len)
{
	__m256		sum = _mm256_setzero_ps();
	float		lanes[8];
	float		total = 0.0f;
	size_t		i = 0;

	for (; i + 8 <= len; i += 8)
	{
		__m256		lhs = _mm256_loadu_ps(left + i);
		__m256		rhs = _mm256_loadu_ps(right + i);

		sum = _mm256_add_ps(sum, _mm256_mul_ps(lhs, rhs));
	}

	_mm256_storeu_ps(lanes, sum);
	for (i = 0; i < 8; i++)
		total += lanes[i];

	for (i = len & ~(size_t) 7; i < len; i++)
		total += left[i] * right[i];

	return total;
}

static bool
tq_cpu_supports_avx2(void)
{
#if defined(__clang__) || defined(__GNUC__)
	return __builtin_cpu_supports("avx2");
#else
	return false;
#endif
}
#else
static bool
tq_cpu_supports_avx2(void)
{
	return false;
}
#endif

#if TQ_CAN_COMPILE_AVX512
__attribute__((target("avx512f")))
static float
tq_dot_product_avx512_impl(const float *left, const float *right, size_t len)
{
	__m512		sum = _mm512_setzero_ps();
	float		lanes[16];
	float		total = 0.0f;
	size_t		i = 0;

	for (; i + 16 <= len; i += 16)
	{
		__m512		lhs = _mm512_loadu_ps(left + i);
		__m512		rhs = _mm512_loadu_ps(right + i);

		sum = _mm512_add_ps(sum, _mm512_mul_ps(lhs, rhs));
	}

	_mm512_storeu_ps(lanes, sum);
	for (i = 0; i < 16; i++)
		total += lanes[i];

	for (i = len & ~(size_t) 15; i < len; i++)
		total += left[i] * right[i];

	return total;
}

static bool
tq_cpu_supports_avx512(void)
{
#if defined(__clang__) || defined(__GNUC__)
	return __builtin_cpu_supports("avx512f");
#else
	return false;
#endif
}
#else
static bool
tq_cpu_supports_avx512(void)
{
	return false;
}
#endif

#if TQ_CAN_COMPILE_NEON
static float
tq_dot_product_neon_impl(const float *left, const float *right, size_t len)
{
	float32x4_t	sum = vdupq_n_f32(0.0f);
	float		lanes[4];
	float		total = 0.0f;
	size_t		i = 0;

	for (; i + 4 <= len; i += 4)
	{
		float32x4_t	lhs = vld1q_f32(left + i);
		float32x4_t	rhs = vld1q_f32(right + i);

		sum = vmlaq_f32(sum, lhs, rhs);
	}

	vst1q_f32(lanes, sum);
	for (i = 0; i < 4; i++)
		total += lanes[i];

	for (i = len & ~(size_t) 3; i < len; i++)
		total += left[i] * right[i];

	return total;
}
#endif

#if TQ_CAN_COMPILE_AVX2
__attribute__((target("avx2")))
static float
tq_hsum_avx2(__m256 vector)
{
	float lanes[8];
	float total = 0.0f;
	size_t i = 0;

	_mm256_storeu_ps(lanes, vector);
	for (i = 0; i < 8; i++)
		total += lanes[i];

	return total;
}

__attribute__((target("avx2")))
static bool
tq_prod_score_code_from_lut_avx2_impl(const TqProdCodecConfig *config,
									  const TqProdLut *lut,
									  const uint8_t *packed,
									  size_t packed_len,
									  float *score,
									  char *errmsg,
									  size_t errmsg_len)
{
	TqProdPackedLayout layout;
	float gamma = 0.0f;
	__m256 base_sum = _mm256_setzero_ps();
	__m256 qjl_sum = _mm256_setzero_ps();
	uint32_t dim = 0;

	memset(&layout, 0, sizeof(layout));

	if (!tq_validate_code_domain_inputs(config, lut, packed, score, errmsg, errmsg_len))
		return false;

	if (!tq_prod_code_domain_shape_supports_avx2(config))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant code-domain dispatch: AVX2 shape requires bits=4 and dimension divisible by 8");
		return false;
	}

	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len)
		|| !tq_prod_read_gamma(config, packed, packed_len, &gamma, errmsg, errmsg_len))
		return false;

	for (dim = 0; dim < config->dimension; dim += 8)
	{
		int32_t gather_indices[8];
		float qjl_scale[8];
		uint32_t lane = 0;

		for (lane = 0; lane < 8; lane++)
		{
			uint32_t current_dim = dim + lane;
			uint32_t idx_code = tq_unpack_bits_local(packed, current_dim * 3u, 3u);
			uint32_t sign_bit = tq_unpack_bits_local(packed + layout.idx_bytes, current_dim, 1u);

			gather_indices[lane] = (int32_t) (current_dim * lut->level_count + idx_code);
			qjl_scale[lane] = (sign_bit == lut->query_signs[current_dim]) ? 2.0f : 0.0f;
		}

		{
			__m256i gather_index_vec = _mm256_loadu_si256((const __m256i *) gather_indices);
			__m256 weights = _mm256_i32gather_ps(lut->values, gather_index_vec, 4);
			__m256 qjl_scale_vec = _mm256_loadu_ps(qjl_scale);

			base_sum = _mm256_add_ps(base_sum, weights);
			qjl_sum = _mm256_add_ps(qjl_sum, _mm256_mul_ps(weights, qjl_scale_vec));
		}
	}

	*score = gamma * (tq_hsum_avx2(qjl_sum) - tq_hsum_avx2(base_sum));
	return true;
}
#endif

static bool
tq_prod_score_query_scalar(const TqProdCodecConfig *config,
						   const float *query,
						   size_t query_len,
						   const uint8_t *packed,
						   size_t packed_len,
						   float *score,
						   char *errmsg,
						   size_t errmsg_len)
{
	float	   *decoded = NULL;
	bool		ok = false;

	if (!tq_validate_score_inputs(config, query, query_len, packed, score,
								  errmsg, errmsg_len))
		return false;

	decoded = (float *) malloc(sizeof(float) * query_len);
	if (decoded == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant score dispatch: out of memory");
		return false;
	}

	ok = tq_prod_decode(config, packed, packed_len, decoded, query_len,
						errmsg, errmsg_len);
	if (ok)
		*score = tq_dot_product_scalar(query, decoded, query_len);

	free(decoded);
	return ok;
}

static bool
tq_prod_score_query_avx2(const TqProdCodecConfig *config,
						 const float *query,
						 size_t query_len,
						 const uint8_t *packed,
						 size_t packed_len,
						 float *score,
						 char *errmsg,
						 size_t errmsg_len)
{
	float	   *decoded = NULL;
	bool		ok = false;

	if (!tq_validate_score_inputs(config, query, query_len, packed, score,
								  errmsg, errmsg_len))
		return false;

	if (!tq_simd_avx2_runtime_available())
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant score dispatch: AVX2 is not available on this machine");
		return false;
	}

	decoded = (float *) malloc(sizeof(float) * query_len);
	if (decoded == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant score dispatch: out of memory");
		return false;
	}

	ok = tq_prod_decode(config, packed, packed_len, decoded, query_len,
						errmsg, errmsg_len);
	if (ok)
	{
#if TQ_CAN_COMPILE_AVX2
		*score = tq_dot_product_avx2_impl(query, decoded, query_len);
#else
		*score = tq_dot_product_scalar(query, decoded, query_len);
#endif
	}

	free(decoded);
	return ok;
}

static bool
tq_prod_score_query_avx512(const TqProdCodecConfig *config,
						   const float *query,
						   size_t query_len,
						   const uint8_t *packed,
						   size_t packed_len,
						   float *score,
						   char *errmsg,
						   size_t errmsg_len)
{
	float	   *decoded = NULL;
	bool		ok = false;

	if (!tq_validate_score_inputs(config, query, query_len, packed, score,
								  errmsg, errmsg_len))
		return false;

	if (!tq_simd_avx512_runtime_available())
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant score dispatch: AVX-512 is not available on this machine");
		return false;
	}

	decoded = (float *) malloc(sizeof(float) * query_len);
	if (decoded == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant score dispatch: out of memory");
		return false;
	}

	ok = tq_prod_decode(config, packed, packed_len, decoded, query_len,
						errmsg, errmsg_len);
	if (ok)
	{
#if TQ_CAN_COMPILE_AVX512
		*score = tq_dot_product_avx512_impl(query, decoded, query_len);
#else
		*score = tq_dot_product_scalar(query, decoded, query_len);
#endif
	}

	free(decoded);
	return ok;
}

static bool
tq_prod_score_query_neon(const TqProdCodecConfig *config,
						 const float *query,
						 size_t query_len,
						 const uint8_t *packed,
						 size_t packed_len,
						 float *score,
						 char *errmsg,
						 size_t errmsg_len)
{
	float	   *decoded = NULL;
	bool		ok = false;

	if (!tq_validate_score_inputs(config, query, query_len, packed, score,
								  errmsg, errmsg_len))
		return false;

	if (!tq_simd_neon_runtime_available())
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant score dispatch: NEON is not available on this machine");
		return false;
	}

	decoded = (float *) malloc(sizeof(float) * query_len);
	if (decoded == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant score dispatch: out of memory");
		return false;
	}

	ok = tq_prod_decode(config, packed, packed_len, decoded, query_len,
						errmsg, errmsg_len);
	if (ok)
	{
#if TQ_CAN_COMPILE_NEON
		*score = tq_dot_product_neon_impl(query, decoded, query_len);
#else
		*score = tq_dot_product_scalar(query, decoded, query_len);
#endif
	}

	free(decoded);
	return ok;
}

bool
tq_simd_scalar_runtime_available(void)
{
	return true;
}

bool
tq_simd_avx2_compile_available(void)
{
	return TQ_CAN_COMPILE_AVX2 != 0;
}

bool
tq_simd_avx2_runtime_available(void)
{
	if (tq_simd_force_disabled)
		return false;

	return tq_cpu_supports_avx2();
}

bool
tq_simd_avx512_compile_available(void)
{
	return TQ_CAN_COMPILE_AVX512 != 0;
}

bool
tq_simd_avx512_runtime_available(void)
{
	if (tq_simd_force_disabled)
		return false;

	return tq_cpu_supports_avx512();
}

bool
tq_simd_neon_compile_available(void)
{
	return TQ_CAN_COMPILE_NEON != 0;
}

bool
tq_simd_neon_runtime_available(void)
{
	if (tq_simd_force_disabled)
		return false;

#if TQ_CAN_COMPILE_NEON
	return true;
#else
	return false;
#endif
}

void
tq_simd_force_disable(bool disabled)
{
	tq_simd_force_disabled = disabled;
}

void
tq_simd_avx2_force_disable(bool disabled)
{
	tq_simd_force_disabled = disabled;
}

TqProdScoreKernel
tq_prod_score_preferred_kernel(void)
{
	if (tq_simd_avx512_runtime_available())
		return TQ_PROD_SCORE_AVX512;
	if (tq_simd_avx2_runtime_available())
		return TQ_PROD_SCORE_AVX2;
	if (tq_simd_neon_runtime_available())
		return TQ_PROD_SCORE_NEON;
	return TQ_PROD_SCORE_SCALAR;
}

TqProdScoreKernel
tq_prod_code_domain_preferred_kernel(const TqProdCodecConfig *config)
{
	if (tq_simd_avx2_runtime_available()
		&& tq_prod_code_domain_shape_supports_avx2(config))
		return TQ_PROD_SCORE_AVX2;

	return TQ_PROD_SCORE_SCALAR;
}

const char *
tq_prod_score_kernel_name(TqProdScoreKernel kernel)
{
	switch (kernel)
	{
		case TQ_PROD_SCORE_SCALAR:
			return "scalar";
		case TQ_PROD_SCORE_AVX2:
			return "avx2";
		case TQ_PROD_SCORE_AVX512:
			return "avx512";
		case TQ_PROD_SCORE_NEON:
			return "neon";
		case TQ_PROD_SCORE_AUTO:
			return "auto";
	}

	return "unknown";
}

bool
tq_prod_score_query_dispatch(const TqProdCodecConfig *config,
							 const float *query,
							 size_t query_len,
							 const uint8_t *packed,
							 size_t packed_len,
							 TqProdScoreKernel kernel,
							 float *score,
							 TqProdScoreKernel *used_kernel,
							 char *errmsg,
							 size_t errmsg_len)
{
	if (used_kernel != NULL)
		*used_kernel = TQ_PROD_SCORE_SCALAR;

	switch (kernel)
	{
		case TQ_PROD_SCORE_SCALAR:
			if (used_kernel != NULL)
				*used_kernel = TQ_PROD_SCORE_SCALAR;
			return tq_prod_score_query_scalar(config, query, query_len, packed,
											  packed_len, score, errmsg, errmsg_len);
		case TQ_PROD_SCORE_AVX2:
			if (used_kernel != NULL)
				*used_kernel = TQ_PROD_SCORE_AVX2;
			return tq_prod_score_query_avx2(config, query, query_len, packed,
											packed_len, score, errmsg, errmsg_len);
		case TQ_PROD_SCORE_AVX512:
			if (used_kernel != NULL)
				*used_kernel = TQ_PROD_SCORE_AVX512;
			return tq_prod_score_query_avx512(config, query, query_len, packed,
											  packed_len, score, errmsg, errmsg_len);
		case TQ_PROD_SCORE_NEON:
			if (used_kernel != NULL)
				*used_kernel = TQ_PROD_SCORE_NEON;
			return tq_prod_score_query_neon(config, query, query_len, packed,
											packed_len, score, errmsg, errmsg_len);
		case TQ_PROD_SCORE_AUTO:
			{
				TqProdScoreKernel preferred = tq_prod_score_preferred_kernel();

				if (used_kernel != NULL)
					*used_kernel = preferred;

				switch (preferred)
				{
					case TQ_PROD_SCORE_AVX512:
						return tq_prod_score_query_avx512(config, query, query_len, packed,
														  packed_len, score, errmsg, errmsg_len);
					case TQ_PROD_SCORE_AVX2:
						return tq_prod_score_query_avx2(config, query, query_len, packed,
														packed_len, score, errmsg, errmsg_len);
					case TQ_PROD_SCORE_NEON:
						return tq_prod_score_query_neon(config, query, query_len, packed,
														packed_len, score, errmsg, errmsg_len);
					case TQ_PROD_SCORE_SCALAR:
					default:
						return tq_prod_score_query_scalar(config, query, query_len, packed,
														  packed_len, score, errmsg, errmsg_len);
				}
			}
		default:
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant score dispatch: unsupported kernel");
			return false;
	}
}

bool
tq_prod_score_code_from_lut_dispatch(const TqProdCodecConfig *config,
									 const TqProdLut *lut,
									 const uint8_t *packed,
									 size_t packed_len,
									 TqProdScoreKernel kernel,
									 float *score,
									 TqProdScoreKernel *used_kernel,
									 char *errmsg,
									 size_t errmsg_len)
{
	TqProdScoreKernel resolved_kernel = TQ_PROD_SCORE_SCALAR;

	if (used_kernel != NULL)
		*used_kernel = TQ_PROD_SCORE_SCALAR;

	if (!tq_validate_code_domain_inputs(config, lut, packed, score, errmsg, errmsg_len))
		return false;

	switch (kernel)
	{
		case TQ_PROD_SCORE_AUTO:
			resolved_kernel = tq_prod_code_domain_preferred_kernel(config);
			break;
		case TQ_PROD_SCORE_AVX2:
			resolved_kernel = tq_prod_code_domain_preferred_kernel(config) == TQ_PROD_SCORE_AVX2
				? TQ_PROD_SCORE_AVX2
				: TQ_PROD_SCORE_SCALAR;
			break;
		case TQ_PROD_SCORE_SCALAR:
			resolved_kernel = TQ_PROD_SCORE_SCALAR;
			break;
		default:
			tq_set_error(errmsg, errmsg_len,
						 "invalid turboquant code-domain dispatch: unsupported kernel");
			return false;
	}

	if (used_kernel != NULL)
		*used_kernel = resolved_kernel;

	if (resolved_kernel == TQ_PROD_SCORE_AVX2)
	{
#if TQ_CAN_COMPILE_AVX2
		return tq_prod_score_code_from_lut_avx2_impl(config, lut, packed, packed_len,
													 score, errmsg, errmsg_len);
#else
		resolved_kernel = TQ_PROD_SCORE_SCALAR;
		if (used_kernel != NULL)
			*used_kernel = resolved_kernel;
#endif
	}

	return tq_prod_score_code_from_lut(config, lut, packed, packed_len,
									   score, errmsg, errmsg_len);
}
