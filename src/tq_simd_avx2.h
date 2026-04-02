#ifndef TQ_SIMD_AVX2_H
#define TQ_SIMD_AVX2_H

#include <stdbool.h>
#include <stddef.h>

#include "src/tq_codec_prod.h"

typedef enum TqProdScoreKernel
{
	TQ_PROD_SCORE_SCALAR = 0,
	TQ_PROD_SCORE_AVX2 = 1,
	TQ_PROD_SCORE_AVX512 = 2,
	TQ_PROD_SCORE_NEON = 3,
	TQ_PROD_SCORE_AUTO = 4
} TqProdScoreKernel;

typedef enum TqLookupStyle
{
	TQ_LOOKUP_STYLE_SCALAR_LOOP = 0,
	TQ_LOOKUP_STYLE_FLOAT_GATHER = 1,
	TQ_LOOKUP_STYLE_LUT16_SCALAR = 2,
	TQ_LOOKUP_STYLE_LUT16_AVX2 = 3,
	TQ_LOOKUP_STYLE_LUT16_NEON = 4,
	TQ_LOOKUP_STYLE_LUT16_AVX512 = 5
} TqLookupStyle;

typedef enum TqGammaPath
{
	TQ_GAMMA_PATH_FLOAT32_SCALAR = 0,
	TQ_GAMMA_PATH_FLOAT32_VECTOR = 1,
	TQ_GAMMA_PATH_FP16_VECTOR = 2
} TqGammaPath;

typedef enum TqQjlPath
{
	TQ_QJL_PATH_FLOAT = 0,
	TQ_QJL_PATH_INT16_QUANTIZED = 1,
	TQ_QJL_PATH_LUT16_QUANTIZED = 2
} TqQjlPath;

extern bool tq_simd_scalar_runtime_available(void);
extern bool tq_simd_avx2_compile_available(void);
extern bool tq_simd_avx2_runtime_available(void);
extern bool tq_simd_avx512_compile_available(void);
extern bool tq_simd_avx512_runtime_available(void);
extern bool tq_simd_neon_compile_available(void);
extern bool tq_simd_neon_runtime_available(void);
extern void tq_simd_force_disable(bool disabled);
extern void tq_simd_avx2_force_disable(bool disabled);
extern TqProdScoreKernel tq_prod_score_preferred_kernel(void);
extern TqProdScoreKernel tq_prod_code_domain_preferred_kernel(const TqProdCodecConfig *config);
extern const char *tq_prod_score_kernel_name(TqProdScoreKernel kernel);
extern const char *tq_lookup_style_name(TqLookupStyle style);
extern const char *tq_gamma_path_name(TqGammaPath path);
extern const char *tq_qjl_path_name(TqQjlPath path);
extern TqLookupStyle tq_lookup_style_for_kernel(TqProdScoreKernel kernel);
extern TqQjlPath tq_qjl_path_for_kernel(TqProdScoreKernel kernel, bool qjl_quantized);
extern bool tq_prod_score_query_dispatch(const TqProdCodecConfig *config,
										 const float *query,
										 size_t query_len,
										 const uint8_t *packed,
										 size_t packed_len,
										 TqProdScoreKernel kernel,
										 float *score,
										 TqProdScoreKernel *used_kernel,
										 char *errmsg,
										 size_t errmsg_len);
extern bool tq_prod_score_code_from_lut_dispatch(const TqProdCodecConfig *config,
												 const TqProdLut *lut,
												 const uint8_t *packed,
												 size_t packed_len,
												 TqProdScoreKernel kernel,
												 float *score,
												 TqProdScoreKernel *used_kernel,
												 char *errmsg,
												 size_t errmsg_len);
extern bool tq_prod_score_block16_dispatch(const TqProdLut16 *lut16,
										   const uint8_t *nibbles,
										   const float *gammas,
										   uint32_t candidate_count,
										   TqProdScoreKernel kernel,
										   float *scores,
										   TqProdScoreKernel *used_kernel,
										   char *errmsg,
										   size_t errmsg_len);

#endif
