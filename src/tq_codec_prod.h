#ifndef TQ_CODEC_PROD_H
#define TQ_CODEC_PROD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct TqProdCodecConfig
{
	uint32_t	dimension;
	uint8_t		bits;
} TqProdCodecConfig;

typedef struct TqProdPackedLayout
{
	size_t		idx_bytes;
	size_t		qjl_bytes;
	size_t		gamma_bytes;
	size_t		total_bytes;
} TqProdPackedLayout;

typedef struct TqProdLut
{
	uint32_t	dimension;
	uint32_t	level_count;
	float	   *values;
	uint8_t	   *query_signs;
} TqProdLut;

extern bool tq_prod_packed_layout(const TqProdCodecConfig *config,
								  TqProdPackedLayout *layout,
								  char *errmsg,
								  size_t errmsg_len);
extern bool tq_prod_encode(const TqProdCodecConfig *config,
						   const float *input,
						   uint8_t *packed,
						   size_t packed_len,
						   char *errmsg,
						   size_t errmsg_len);
extern bool tq_prod_read_gamma(const TqProdCodecConfig *config,
							   const uint8_t *packed,
							   size_t packed_len,
							   float *gamma,
							   char *errmsg,
							   size_t errmsg_len);
extern bool tq_prod_decode(const TqProdCodecConfig *config,
						   const uint8_t *packed,
						   size_t packed_len,
						   float *output,
						   size_t output_len,
						   char *errmsg,
						   size_t errmsg_len);
extern bool tq_prod_lut_build(const TqProdCodecConfig *config,
							  const float *query,
							  TqProdLut *lut,
							  char *errmsg,
							  size_t errmsg_len);
extern void tq_prod_lut_reset(TqProdLut *lut);
extern bool tq_prod_score_code_from_lut(const TqProdCodecConfig *config,
										const TqProdLut *lut,
										const uint8_t *packed,
										size_t packed_len,
										float *score,
										char *errmsg,
										size_t errmsg_len);
extern bool tq_prod_score_decompose_ip(const TqProdCodecConfig *config,
									   const TqProdLut *lut,
									   const uint8_t *packed,
									   size_t packed_len,
									   float *mse_contribution,
									   float *qjl_contribution,
									   float *combined_score,
									   char *errmsg,
									   size_t errmsg_len);
extern bool tq_prod_score_packed_ip(const TqProdCodecConfig *config,
									const TqProdLut *lut,
									const uint8_t *packed,
									size_t packed_len,
									float *score,
									char *errmsg,
									size_t errmsg_len);
extern void tq_prod_decode_counter_reset(void);
extern size_t tq_prod_decode_counter_get(void);

#endif
