#ifndef TQ_CODEC_PROD_H
#define TQ_CODEC_PROD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct TqProdCodecConfig
{
	uint32_t	dimension;
	uint32_t	qjl_dimension;
	uint8_t		bits;
	uint64_t	qjl_seed;
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
	uint32_t	qjl_dimension;
	uint32_t	level_count;
	float	   *values;
	float	   *qjl_values;
	int16_t	   *qjl_quantized_values;
	float		qjl_quantization_scale;
	float		qjl_quantization_max_error;
	bool		qjl_quantized_enabled;
	float		feature_weight_norm;
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
extern bool tq_prod_qjl_project(const TqProdCodecConfig *config,
								const float *input,
								float *output,
								size_t output_len,
								char *errmsg,
								size_t errmsg_len);
extern bool tq_prod_qjl_backproject_signs(const TqProdCodecConfig *config,
										  const uint8_t *packed_signs,
										  size_t packed_signs_len,
										  float *output,
										  size_t output_len,
										  char *errmsg,
										  size_t errmsg_len);
extern bool tq_prod_feature_distance(const TqProdCodecConfig *config,
									 const uint8_t *left_packed,
									 size_t left_packed_len,
									 const uint8_t *right_packed,
									 size_t right_packed_len,
									 float *distance,
									 char *errmsg,
									 size_t errmsg_len);
extern bool tq_prod_query_weight_l2_norm(const TqProdCodecConfig *config,
										 const TqProdLut *lut,
										 float *norm,
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

/*
 * LUT16 block-scoring API for the fast-path contract:
 *   bits=4, normalized, qjl_dimension == dimension, dimension % 8 == 0.
 *
 * Packs the 3-bit idx code and 1-bit QJL sign into a 4-bit nibble.
 * base_values:  16 entries per dimension  (idx contribution only)
 * qjl_values:   16 entries per dimension  (signed QJL contribution)
 * Gamma is applied after accumulation since it varies per vector.
 */
typedef struct TqProdLut16
{
	uint32_t	dimension;
	float	   *base_values;		/* dimension * 16 floats: base_values[d*16 + nibble] */
	float	   *qjl_values;			/* dimension * 16 floats: qjl_values[d*16 + nibble] */
	int8_t	   *base_quantized;		/* dimension * 16 int8s for PSHUFB (NULL if not built) */
	int8_t	   *qjl_quantized;		/* dimension * 16 int8s for PSHUFB (NULL if not built) */
	float		base_global_scale;	/* single dequant scale for all base entries */
	float		qjl_global_scale;	/* single dequant scale for all QJL entries */
	bool		quantized_ready;
} TqProdLut16;

extern bool tq_prod_lut16_build(const TqProdCodecConfig *config,
								const TqProdLut *lut,
								TqProdLut16 *lut16,
								char *errmsg,
								size_t errmsg_len);
extern void tq_prod_lut16_reset(TqProdLut16 *lut16);
extern bool tq_prod_lut16_is_supported(const TqProdCodecConfig *config,
										char *errmsg,
										size_t errmsg_len);
extern bool tq_prod_score_block16_scalar(const TqProdLut16 *lut16,
										 const uint8_t *nibbles,
										 const float *gammas,
										 uint32_t candidate_count,
										 float *scores,
										 char *errmsg,
										 size_t errmsg_len);
extern bool tq_prod_score_block16_quantized_scalar(const TqProdLut16 *lut16,
												   const uint8_t *nibbles,
												   const float *gammas,
												   uint32_t candidate_count,
												   float *scores,
												   char *errmsg,
												   size_t errmsg_len);
extern bool tq_prod_lut16_quantize(TqProdLut16 *lut16,
								   char *errmsg,
								   size_t errmsg_len);
extern bool tq_prod_extract_nibbles(const TqProdCodecConfig *config,
									const uint8_t *packed,
									size_t packed_len,
									uint8_t *nibbles,
									size_t nibbles_len,
									char *errmsg,
									size_t errmsg_len);
extern bool tq_prod_nibbles_gamma_to_packed(const TqProdCodecConfig *config,
											const uint8_t *nibbles,
											uint32_t dimension,
											float gamma,
											uint8_t *packed,
											size_t packed_len,
											char *errmsg,
											size_t errmsg_len);

#endif
