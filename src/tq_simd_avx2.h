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
extern const char *tq_prod_score_kernel_name(TqProdScoreKernel kernel);
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

#endif
