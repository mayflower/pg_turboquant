#ifndef TQ_CODEC_MSE_H
#define TQ_CODEC_MSE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct TqMseCodecConfig
{
	uint32_t	dimension;
	uint8_t		bits;
	float		min_value;
	float		max_value;
} TqMseCodecConfig;

typedef struct TqMseLut
{
	uint32_t	dimension;
	uint32_t	level_count;
	float	   *values;
} TqMseLut;

extern bool tq_mse_packed_bytes(const TqMseCodecConfig *config,
								size_t *packed_bytes,
								char *errmsg,
								size_t errmsg_len);
extern bool tq_mse_encode(const TqMseCodecConfig *config,
						  const float *input,
						  uint8_t *packed,
						  size_t packed_len,
						  char *errmsg,
						  size_t errmsg_len);
extern bool tq_mse_decode(const TqMseCodecConfig *config,
						  const uint8_t *packed,
						  size_t packed_len,
						  float *output,
						  size_t output_len,
						  char *errmsg,
						  size_t errmsg_len);
extern bool tq_mse_lut_build(const TqMseCodecConfig *config,
							 const float *query,
							 TqMseLut *lut,
							 char *errmsg,
							 size_t errmsg_len);
extern void tq_mse_lut_reset(TqMseLut *lut);
extern bool tq_mse_score_packed_l2(const TqMseCodecConfig *config,
								   const TqMseLut *lut,
								   const uint8_t *packed,
								   size_t packed_len,
								   float *score,
								   char *errmsg,
								   size_t errmsg_len);

#endif
