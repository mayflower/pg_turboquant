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
		|| lut->qjl_dimension != (config->qjl_dimension == 0 ? config->dimension : config->qjl_dimension)
		|| lut->level_count != ((uint32_t) 1u << ((uint32_t) config->bits - 1u))
		|| lut->values == NULL
		|| lut->qjl_values == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant code-domain dispatch: lut shape must match codec config");
		return false;
	}

	return true;
}

static bool
tq_prod_code_domain_shape_supports_vectorized(const TqProdCodecConfig *config)
{
	if (config == NULL)
		return false;

	return config->bits == 4
		&& (config->qjl_dimension == 0 || config->qjl_dimension == config->dimension)
		&& config->dimension > 0
		&& (config->dimension % 8u) == 0u;
}

static bool
tq_prod_lut_uses_quantized_qjl(const TqProdLut *lut)
{
	return lut != NULL
		&& lut->qjl_quantized_enabled
		&& lut->qjl_quantized_values != NULL
		&& lut->qjl_quantization_scale > 0.0f;
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

static uint32_t
tq_load_u24_le(const uint8_t *packed)
{
	return ((uint32_t) packed[0])
		| (((uint32_t) packed[1]) << 8)
		| (((uint32_t) packed[2]) << 16);
}

#if TQ_CAN_COMPILE_AVX2

__attribute__((target("avx2")))
static float
tq_hsum_avx2(__m256 vector)
{
	__m128 low = _mm256_castps256_ps128(vector);
	__m128 high = _mm256_extractf128_ps(vector, 1);
	__m128 sum = _mm_add_ps(low, high);

	sum = _mm_hadd_ps(sum, sum);
	sum = _mm_hadd_ps(sum, sum);
	return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2")))
static float
tq_dot_product_avx2_impl(const float *left, const float *right, size_t len)
{
	__m256		sum = _mm256_setzero_ps();
	float		total = 0.0f;
	size_t		i = 0;

	for (; i + 8 <= len; i += 8)
	{
		__m256		lhs = _mm256_loadu_ps(left + i);
		__m256		rhs = _mm256_loadu_ps(right + i);

		sum = _mm256_add_ps(sum, _mm256_mul_ps(lhs, rhs));
	}

	total = tq_hsum_avx2(sum);
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
static __m256i
tq_prod_avx2_idx_code_vector(uint32_t idx_chunk)
{
	return _mm256_set_epi32(
		(int) ((idx_chunk >> 21u) & 0x7u),
		(int) ((idx_chunk >> 18u) & 0x7u),
		(int) ((idx_chunk >> 15u) & 0x7u),
		(int) ((idx_chunk >> 12u) & 0x7u),
		(int) ((idx_chunk >> 9u) & 0x7u),
		(int) ((idx_chunk >> 6u) & 0x7u),
		(int) ((idx_chunk >> 3u) & 0x7u),
		(int) (idx_chunk & 0x7u));
}

__attribute__((target("avx2")))
static __m256
tq_prod_avx2_sign_vector(uint8_t sign_chunk)
{
	__m256i sign_bits = _mm256_set_epi32(
		(int) ((sign_chunk >> 7u) & 0x1u),
		(int) ((sign_chunk >> 6u) & 0x1u),
		(int) ((sign_chunk >> 5u) & 0x1u),
		(int) ((sign_chunk >> 4u) & 0x1u),
		(int) ((sign_chunk >> 3u) & 0x1u),
		(int) ((sign_chunk >> 2u) & 0x1u),
		(int) ((sign_chunk >> 1u) & 0x1u),
		(int) (sign_chunk & 0x1u));
	__m256i signed_int = _mm256_sub_epi32(_mm256_slli_epi32(sign_bits, 1),
										  _mm256_set1_epi32(1));

	return _mm256_cvtepi32_ps(signed_int);
}

__attribute__((target("avx2")))
static __m256i
tq_prod_avx2_sign_vector_epi32(uint8_t sign_chunk)
{
	__m256i sign_bits = _mm256_set_epi32(
		(int) ((sign_chunk >> 7u) & 0x1u),
		(int) ((sign_chunk >> 6u) & 0x1u),
		(int) ((sign_chunk >> 5u) & 0x1u),
		(int) ((sign_chunk >> 4u) & 0x1u),
		(int) ((sign_chunk >> 3u) & 0x1u),
		(int) ((sign_chunk >> 2u) & 0x1u),
		(int) ((sign_chunk >> 1u) & 0x1u),
		(int) (sign_chunk & 0x1u));

	return _mm256_sub_epi32(_mm256_slli_epi32(sign_bits, 1), _mm256_set1_epi32(1));
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
	__m256 base_sum = _mm256_setzero_ps();
	__m256 residual_sum = _mm256_setzero_ps();
	__m256i residual_sum_i32 = _mm256_setzero_si256();
	__m256i lane_value_offsets;
	bool use_quantized_qjl = tq_prod_lut_uses_quantized_qjl(lut);
	uint32_t dim = 0;

	memset(&layout, 0, sizeof(layout));

	if (!tq_validate_code_domain_inputs(config, lut, packed, score, errmsg, errmsg_len))
		return false;

	if (!tq_prod_code_domain_shape_supports_vectorized(config))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant code-domain dispatch: AVX2 shape requires bits=4 and dimension divisible by 8");
		return false;
	}

	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len))
		return false;
	if (packed_len < layout.total_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: packed input buffer is too small");
		return false;
	}

	lane_value_offsets = _mm256_set_epi32(
		(int) (7u * lut->level_count),
		(int) (6u * lut->level_count),
		(int) (5u * lut->level_count),
		(int) (4u * lut->level_count),
		(int) (3u * lut->level_count),
		(int) (2u * lut->level_count),
		(int) (1u * lut->level_count),
		0);

	for (dim = 0; dim < config->dimension; dim += 8)
	{
		const uint8_t *idx_bytes = packed + ((size_t) dim * 3u / 8u);
		const uint8_t *sign_bytes = packed + layout.idx_bytes + ((size_t) dim / 8u);
		uint32_t idx_chunk = tq_load_u24_le(idx_bytes);
		__m256i gather_indices = _mm256_add_epi32(
			_mm256_add_epi32(_mm256_set1_epi32((int) (dim * lut->level_count)),
							 lane_value_offsets),
			tq_prod_avx2_idx_code_vector(idx_chunk));
		__m256 base = _mm256_i32gather_ps(lut->values, gather_indices, 4);

		base_sum = _mm256_add_ps(base_sum, base);
		if (use_quantized_qjl)
		{
			__m128i residual_qjl_i16 = _mm_loadu_si128((const __m128i *) (lut->qjl_quantized_values + dim));
			__m256i residual_qjl_i32 = _mm256_cvtepi16_epi32(residual_qjl_i16);
			__m256i residual_sign_vec_i32 = tq_prod_avx2_sign_vector_epi32(sign_bytes[0]);

			residual_sum_i32 = _mm256_add_epi32(residual_sum_i32,
												 _mm256_mullo_epi32(residual_qjl_i32, residual_sign_vec_i32));
		}
		else
		{
			__m256 residual = _mm256_loadu_ps(lut->qjl_values + dim);
			__m256 residual_sign_vec = tq_prod_avx2_sign_vector(sign_bytes[0]);

			residual_sum = _mm256_add_ps(residual_sum, _mm256_mul_ps(residual, residual_sign_vec));
		}
	}

	{
		float gamma = 0.0f;
		float qjl_contribution = 0.0f;

		if (!tq_prod_read_gamma(config, packed, packed_len, &gamma, errmsg, errmsg_len))
			return false;
		if (use_quantized_qjl)
		{
			qjl_contribution = gamma
				* lut->qjl_quantization_scale
				* tq_hsum_avx2(_mm256_cvtepi32_ps(residual_sum_i32));
		}
		else
		{
			qjl_contribution = gamma * tq_hsum_avx2(residual_sum);
		}
		*score = tq_hsum_avx2(base_sum) + qjl_contribution;
	}
	return true;
}
#endif

#if TQ_CAN_COMPILE_NEON
static float
tq_hsum_neon(float32x4_t vector)
{
#if defined(__aarch64__) || defined(_M_ARM64)
	return vaddvq_f32(vector);
#else
	float32x2_t low = vget_low_f32(vector);
	float32x2_t high = vget_high_f32(vector);
	float32x2_t pair = vpadd_f32(low, high);
	pair = vpadd_f32(pair, pair);
	return vget_lane_f32(pair, 0);
#endif
}

static int32_t
tq_hsum_neon_s32(int32x4_t vector)
{
#if defined(__aarch64__) || defined(_M_ARM64)
	return vaddvq_s32(vector);
#else
	int32x2_t low = vget_low_s32(vector);
	int32x2_t high = vget_high_s32(vector);
	int32x2_t pair = vpadd_s32(low, high);
	pair = vpadd_s32(pair, pair);
	return vget_lane_s32(pair, 0);
#endif
}

static uint32_t
tq_prod_idx_code_from_chunk(uint32_t idx_chunk, uint32_t lane)
{
	return (idx_chunk >> (lane * 3u)) & 0x7u;
}

static float32x4_t
tq_prod_neon_base_vector(const TqProdLut *lut, uint32_t dim, uint32_t idx_chunk)
{
	float32x4_t base = vdupq_n_f32(0.0f);
	const size_t level_count = (size_t) lut->level_count;
	const size_t dim_base = (size_t) dim * level_count;

	base = vsetq_lane_f32(lut->values[dim_base + tq_prod_idx_code_from_chunk(idx_chunk, 0u)], base, 0);
	base = vsetq_lane_f32(lut->values[dim_base + level_count + tq_prod_idx_code_from_chunk(idx_chunk, 1u)],
						  base,
						  1);
	base = vsetq_lane_f32(lut->values[dim_base + (2u * level_count) + tq_prod_idx_code_from_chunk(idx_chunk, 2u)],
						  base,
						  2);
	base = vsetq_lane_f32(lut->values[dim_base + (3u * level_count) + tq_prod_idx_code_from_chunk(idx_chunk, 3u)],
						  base,
						  3);
	return base;
}

static float32x4_t
tq_prod_neon_sign_vector(uint8_t sign_chunk, uint32_t lane_base)
{
	int32x4_t sign = vdupq_n_s32(-1);

	sign = vsetq_lane_s32((sign_chunk & (uint8_t) (1u << lane_base)) ? 1 : -1, sign, 0);
	sign = vsetq_lane_s32((sign_chunk & (uint8_t) (1u << (lane_base + 1u))) ? 1 : -1, sign, 1);
	sign = vsetq_lane_s32((sign_chunk & (uint8_t) (1u << (lane_base + 2u))) ? 1 : -1, sign, 2);
	sign = vsetq_lane_s32((sign_chunk & (uint8_t) (1u << (lane_base + 3u))) ? 1 : -1, sign, 3);
	return vcvtq_f32_s32(sign);
}

static int16x8_t
tq_prod_neon_sign_vector_s16(uint8_t sign_chunk)
{
	int16x8_t sign = vdupq_n_s16(-1);

	sign = vsetq_lane_s16((sign_chunk & 0x01u) ? 1 : -1, sign, 0);
	sign = vsetq_lane_s16((sign_chunk & 0x02u) ? 1 : -1, sign, 1);
	sign = vsetq_lane_s16((sign_chunk & 0x04u) ? 1 : -1, sign, 2);
	sign = vsetq_lane_s16((sign_chunk & 0x08u) ? 1 : -1, sign, 3);
	sign = vsetq_lane_s16((sign_chunk & 0x10u) ? 1 : -1, sign, 4);
	sign = vsetq_lane_s16((sign_chunk & 0x20u) ? 1 : -1, sign, 5);
	sign = vsetq_lane_s16((sign_chunk & 0x40u) ? 1 : -1, sign, 6);
	sign = vsetq_lane_s16((sign_chunk & 0x80u) ? 1 : -1, sign, 7);
	return sign;
}

static bool
tq_prod_score_code_from_lut_neon_impl(const TqProdCodecConfig *config,
									  const TqProdLut *lut,
									  const uint8_t *packed,
									  size_t packed_len,
									  float *score,
									  char *errmsg,
									  size_t errmsg_len)
{
	TqProdPackedLayout layout;
	float32x4_t base_sum_lo = vdupq_n_f32(0.0f);
	float32x4_t base_sum_hi = vdupq_n_f32(0.0f);
	float32x4_t residual_sum_lo = vdupq_n_f32(0.0f);
	float32x4_t residual_sum_hi = vdupq_n_f32(0.0f);
	int32x4_t residual_sum_lo_i32 = vdupq_n_s32(0);
	int32x4_t residual_sum_hi_i32 = vdupq_n_s32(0);
	bool use_quantized_qjl = tq_prod_lut_uses_quantized_qjl(lut);
	uint32_t dim = 0;

	memset(&layout, 0, sizeof(layout));

	if (!tq_validate_code_domain_inputs(config, lut, packed, score, errmsg, errmsg_len))
		return false;

	if (!tq_prod_code_domain_shape_supports_vectorized(config))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant code-domain dispatch: NEON shape requires bits=4 and dimension divisible by 8");
		return false;
	}

	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len))
		return false;
	if (packed_len < layout.total_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: packed input buffer is too small");
		return false;
	}

	for (dim = 0; dim < config->dimension; dim += 8)
	{
		const uint8_t *idx_bytes = packed + ((size_t) dim * 3u / 8u);
		const uint8_t *sign_bytes = packed + layout.idx_bytes + ((size_t) dim / 8u);
		uint32_t idx_chunk = tq_load_u24_le(idx_bytes);
		uint8_t sign_chunk = sign_bytes[0];
		float32x4_t base_lo = tq_prod_neon_base_vector(lut, dim, idx_chunk);
		float32x4_t base_hi = tq_prod_neon_base_vector(lut, dim + 4u, idx_chunk >> 12u);

		base_sum_lo = vaddq_f32(base_sum_lo, base_lo);
		base_sum_hi = vaddq_f32(base_sum_hi, base_hi);
		if (use_quantized_qjl)
		{
			int16x8_t residual_qjl = vld1q_s16(lut->qjl_quantized_values + dim);
			int16x8_t residual_signed = vmulq_s16(residual_qjl, tq_prod_neon_sign_vector_s16(sign_chunk));

			residual_sum_lo_i32 = vaddw_s16(residual_sum_lo_i32, vget_low_s16(residual_signed));
			residual_sum_hi_i32 = vaddw_s16(residual_sum_hi_i32, vget_high_s16(residual_signed));
		}
		else
		{
			float32x4_t residual_lo = vmulq_f32(vld1q_f32(lut->qjl_values + dim),
												 tq_prod_neon_sign_vector(sign_chunk, 0u));
			float32x4_t residual_hi = vmulq_f32(vld1q_f32(lut->qjl_values + dim + 4u),
												 tq_prod_neon_sign_vector(sign_chunk, 4u));

			residual_sum_lo = vaddq_f32(residual_sum_lo, residual_lo);
			residual_sum_hi = vaddq_f32(residual_sum_hi, residual_hi);
		}
	}

	{
		float gamma = 0.0f;
		float qjl_contribution = 0.0f;

		if (!tq_prod_read_gamma(config, packed, packed_len, &gamma, errmsg, errmsg_len))
			return false;
		if (use_quantized_qjl)
		{
			qjl_contribution = gamma
				* lut->qjl_quantization_scale
				* (float) (tq_hsum_neon_s32(residual_sum_lo_i32) + tq_hsum_neon_s32(residual_sum_hi_i32));
		}
		else
		{
			qjl_contribution = gamma * (tq_hsum_neon(residual_sum_lo) + tq_hsum_neon(residual_sum_hi));
		}
		*score = (tq_hsum_neon(base_sum_lo) + tq_hsum_neon(base_sum_hi)) + qjl_contribution;
	}
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
		&& tq_prod_code_domain_shape_supports_vectorized(config))
		return TQ_PROD_SCORE_AVX2;
	if (tq_simd_neon_runtime_available()
		&& tq_prod_code_domain_shape_supports_vectorized(config))
		return TQ_PROD_SCORE_NEON;

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
		case TQ_PROD_SCORE_NEON:
			resolved_kernel = tq_prod_code_domain_preferred_kernel(config) == TQ_PROD_SCORE_NEON
				? TQ_PROD_SCORE_NEON
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
	else if (resolved_kernel == TQ_PROD_SCORE_NEON)
	{
#if TQ_CAN_COMPILE_NEON
		return tq_prod_score_code_from_lut_neon_impl(config, lut, packed, packed_len,
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
