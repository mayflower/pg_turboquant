#include "src/tq_codec_prod.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TQ_PROD_GAMMA_BYTES 4

static void
tq_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg_len == 0)
		return;

	snprintf(errmsg, errmsg_len, "%s", message);
}

static bool
tq_prod_validate_config(const TqProdCodecConfig *config,
						char *errmsg,
						size_t errmsg_len)
{
	if (config == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod codec config: config must be non-null");
		return false;
	}

	if (config->dimension == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod codec config: dimension must be positive");
		return false;
	}

	if (config->bits < 2 || config->bits > 8)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod codec config: bits must be between 2 and 8");
		return false;
	}

	return true;
}

static uint32_t
tq_prod_idx_bits(const TqProdCodecConfig *config)
{
	return (uint32_t) config->bits - 1;
}

static uint32_t
tq_prod_idx_levels(const TqProdCodecConfig *config)
{
	return UINT32_C(1) << tq_prod_idx_bits(config);
}

static float
tq_prod_gamma_scale_factor(const TqProdCodecConfig *config)
{
	static const float gamma_scale_by_bits[] = {
		1.0f,
		1.0f,
		1.00f,
		1.00f,
		1.00f,
		0.99f,
		0.98f,
		0.97f,
		0.96f
	};

	return gamma_scale_by_bits[config->bits];
}

static float
tq_prod_companding_mu(const TqProdCodecConfig *config)
{
	static const float mu_by_bits[] = {
		0.0f,
		0.0f,
		0.25f,
		0.50f,
		0.50f,
		1.00f,
		1.25f,
		1.50f,
		2.00f
	};

	return mu_by_bits[config->bits];
}

static void
tq_pack_bits(uint8_t *packed, uint32_t bit_offset, uint32_t bit_count, uint32_t value)
{
	uint32_t	bit = 0;

	for (bit = 0; bit < bit_count; bit++)
	{
		uint32_t	target_bit = bit_offset + bit;
		uint32_t	byte_index = target_bit / 8;
		uint32_t	byte_shift = target_bit % 8;
		uint8_t		mask = (uint8_t) (1u << byte_shift);

		if ((value >> bit) & 1u)
			packed[byte_index] |= mask;
		else
			packed[byte_index] &= (uint8_t) ~mask;
	}
}

static uint32_t
tq_unpack_bits(const uint8_t *packed, uint32_t bit_offset, uint32_t bit_count)
{
	uint32_t	value = 0;
	uint32_t	bit = 0;

	for (bit = 0; bit < bit_count; bit++)
	{
		uint32_t	target_bit = bit_offset + bit;
		uint32_t	byte_index = target_bit / 8;
		uint32_t	byte_shift = target_bit % 8;

		if ((packed[byte_index] >> byte_shift) & 1u)
			value |= UINT32_C(1) << bit;
	}

	return value;
}

static float
tq_prod_max_abs(const float *input, uint32_t dimension)
{
	float		gamma = 0.0f;
	uint32_t	i = 0;

	for (i = 0; i < dimension; i++)
	{
		float		abs_value = fabsf(input[i]);

		if (abs_value > gamma)
			gamma = abs_value;
	}

	return gamma;
}

static float
tq_prod_derive_gamma(const TqProdCodecConfig *config, const float *input)
{
	float		max_abs = tq_prod_max_abs(input, config->dimension);

	if (max_abs <= 0.0f)
		return 0.0f;

	return max_abs * tq_prod_gamma_scale_factor(config);
}

static uint32_t
tq_prod_quantize_magnitude(const TqProdCodecConfig *config, float magnitude, float gamma)
{
	uint32_t	max_code = tq_prod_idx_levels(config) - 1;
	float		normalized = 0.0f;
	float		companded = 0.0f;
	float		mu = tq_prod_companding_mu(config);
	long		rounded = 0;

	if (gamma <= 0.0f)
		return 0;

	if (magnitude <= 0.0f)
		return 0;

	if (magnitude >= gamma)
		return max_code;

	normalized = magnitude / gamma;
	if (mu > 0.0f)
		companded = log1pf(mu * normalized) / log1pf(mu);
	else
		companded = normalized;

	rounded = lroundf(companded * (float) max_code);

	if (rounded < 0)
		return 0;

	if ((uint32_t) rounded > max_code)
		return max_code;

	return (uint32_t) rounded;
}

static float
tq_prod_decode_magnitude(const TqProdCodecConfig *config, uint32_t code, float gamma)
{
	uint32_t	max_code = tq_prod_idx_levels(config) - 1;
	float		mu = tq_prod_companding_mu(config);
	float		companded = 0.0f;
	float		normalized = 0.0f;

	if (max_code == 0 || gamma <= 0.0f)
		return 0.0f;

	companded = (float) code / (float) max_code;
	if (mu > 0.0f)
		normalized = expm1f(companded * log1pf(mu)) / mu;
	else
		normalized = companded;

	return gamma * normalized;
}

static bool
tq_prod_unpack_parts(const uint8_t *packed,
					 size_t packed_len,
					 const TqProdPackedLayout *layout,
					 float *gamma,
					 char *errmsg,
					 size_t errmsg_len)
{
	if (packed == NULL || gamma == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: packed input and gamma output must be non-null");
		return false;
	}

	if (packed_len < layout->total_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: packed input buffer is too small");
		return false;
	}

	memcpy(gamma, packed + layout->idx_bytes + layout->qjl_bytes, sizeof(float));
	return true;
}

bool
tq_prod_packed_layout(const TqProdCodecConfig *config,
					  TqProdPackedLayout *layout,
					  char *errmsg,
					  size_t errmsg_len)
{
	uint64_t	idx_bits = 0;

	if (layout == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: layout output must be non-null");
		return false;
	}

	if (!tq_prod_validate_config(config, errmsg, errmsg_len))
		return false;

	idx_bits = (uint64_t) config->dimension * (uint64_t) tq_prod_idx_bits(config);
	layout->idx_bytes = (size_t) ((idx_bits + 7) / 8);
	layout->qjl_bytes = (size_t) (((uint64_t) config->dimension + 7) / 8);
	layout->gamma_bytes = TQ_PROD_GAMMA_BYTES;
	layout->total_bytes = layout->idx_bytes + layout->qjl_bytes + layout->gamma_bytes;
	return true;
}

bool
tq_prod_encode(const TqProdCodecConfig *config,
			   const float *input,
			   uint8_t *packed,
			   size_t packed_len,
			   char *errmsg,
			   size_t errmsg_len)
{
	TqProdPackedLayout layout;
	float		gamma = 0.0f;
	uint32_t	i = 0;

	if (input == NULL || packed == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: input and packed output must be non-null");
		return false;
	}

	memset(&layout, 0, sizeof(layout));

	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len))
		return false;

	if (packed_len < layout.total_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: packed output buffer is too small");
		return false;
	}

	memset(packed, 0, packed_len);
	gamma = tq_prod_derive_gamma(config, input);

	for (i = 0; i < config->dimension; i++)
	{
		uint32_t	idx_code = tq_prod_quantize_magnitude(config, fabsf(input[i]), gamma);
		uint32_t	sign_bit = input[i] >= 0.0f ? 1u : 0u;

		tq_pack_bits(packed, i * tq_prod_idx_bits(config), tq_prod_idx_bits(config), idx_code);
		tq_pack_bits(packed + layout.idx_bytes, i, 1, sign_bit);
	}

	memcpy(packed + layout.idx_bytes + layout.qjl_bytes, &gamma, sizeof(float));
	return true;
}

bool
tq_prod_read_gamma(const TqProdCodecConfig *config,
				   const uint8_t *packed,
				   size_t packed_len,
				   float *gamma,
				   char *errmsg,
				   size_t errmsg_len)
{
	TqProdPackedLayout layout;

	memset(&layout, 0, sizeof(layout));

	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len))
		return false;

	return tq_prod_unpack_parts(packed, packed_len, &layout, gamma, errmsg, errmsg_len);
}

bool
tq_prod_decode(const TqProdCodecConfig *config,
			   const uint8_t *packed,
			   size_t packed_len,
			   float *output,
			   size_t output_len,
			   char *errmsg,
			   size_t errmsg_len)
{
	TqProdPackedLayout layout;
	float		gamma = 0.0f;
	uint32_t	i = 0;

	if (packed == NULL || output == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: packed input and decoded output must be non-null");
		return false;
	}

	memset(&layout, 0, sizeof(layout));

	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len))
		return false;

	if (output_len < config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: decoded output buffer is too small");
		return false;
	}

	if (!tq_prod_unpack_parts(packed, packed_len, &layout, &gamma, errmsg, errmsg_len))
		return false;

	for (i = 0; i < config->dimension; i++)
	{
		uint32_t	idx_code = tq_unpack_bits(packed, i * tq_prod_idx_bits(config), tq_prod_idx_bits(config));
		uint32_t	sign_bit = tq_unpack_bits(packed + layout.idx_bytes, i, 1);
		float		value = tq_prod_decode_magnitude(config, idx_code, gamma);

		output[i] = sign_bit ? value : -value;
	}

	return true;
}

void
tq_prod_lut_reset(TqProdLut *lut)
{
	if (lut == NULL)
		return;

	free(lut->values);
	free(lut->query_signs);
	memset(lut, 0, sizeof(*lut));
}

bool
tq_prod_lut_build(const TqProdCodecConfig *config,
				  const float *query,
				  TqProdLut *lut,
				  char *errmsg,
				  size_t errmsg_len)
{
	uint32_t	level_count = 0;
	uint32_t	dim = 0;
	uint32_t	code = 0;
	uint32_t	max_code = 0;

	if (query == NULL || lut == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod lut: query and lut must be non-null");
		return false;
	}

	if (!tq_prod_validate_config(config, errmsg, errmsg_len))
		return false;

	tq_prod_lut_reset(lut);

	level_count = tq_prod_idx_levels(config);
	max_code = level_count - 1;
	lut->values = (float *) malloc(sizeof(float) * (size_t) config->dimension * (size_t) level_count);
	lut->query_signs = (uint8_t *) malloc(sizeof(uint8_t) * (size_t) config->dimension);

	if (lut->values == NULL || lut->query_signs == NULL)
	{
		tq_prod_lut_reset(lut);
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod lut: out of memory");
		return false;
	}

	lut->dimension = config->dimension;
	lut->level_count = level_count;

	for (dim = 0; dim < config->dimension; dim++)
	{
		float		abs_query = fabsf(query[dim]);

		lut->query_signs[dim] = query[dim] >= 0.0f ? 1u : 0u;
		for (code = 0; code < level_count; code++)
		{
			float		normalized_level = 0.0f;

			if (max_code != 0)
			{
				float		companded = (float) code / (float) max_code;
				float		mu = tq_prod_companding_mu(config);

				if (mu > 0.0f)
					normalized_level = expm1f(companded * log1pf(mu)) / mu;
				else
					normalized_level = companded;
			}

			lut->values[(size_t) dim * (size_t) level_count + (size_t) code] = abs_query * normalized_level;
		}
	}

	return true;
}

bool
tq_prod_score_decompose_ip(const TqProdCodecConfig *config,
						   const TqProdLut *lut,
						   const uint8_t *packed,
						   size_t packed_len,
						   float *mse_contribution,
						   float *qjl_contribution,
						   float *combined_score,
						   char *errmsg,
						   size_t errmsg_len)
{
	TqProdPackedLayout layout;
	float		gamma = 0.0f;
	float		base_sum = 0.0f;
	float		qjl_sum = 0.0f;
	uint32_t	i = 0;

	if (lut == NULL || packed == NULL || mse_contribution == NULL
		|| qjl_contribution == NULL || combined_score == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod scorer: lut, packed input, and score outputs must be non-null");
		return false;
	}

	memset(&layout, 0, sizeof(layout));

	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len))
		return false;

	if (lut->dimension != config->dimension || lut->level_count != tq_prod_idx_levels(config)
		|| lut->values == NULL || lut->query_signs == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod scorer: lut shape does not match codec config");
		return false;
	}

	if (!tq_prod_unpack_parts(packed, packed_len, &layout, &gamma, errmsg, errmsg_len))
		return false;

	for (i = 0; i < config->dimension; i++)
	{
		uint32_t	idx_code = tq_unpack_bits(packed, i * tq_prod_idx_bits(config), tq_prod_idx_bits(config));
		uint32_t	sign_bit = tq_unpack_bits(packed + layout.idx_bytes, i, 1);
		float		weight = lut->values[(size_t) i * (size_t) lut->level_count + (size_t) idx_code];

		base_sum += weight;
		if (sign_bit == lut->query_signs[i])
			qjl_sum += (2.0f * weight);
	}

	*mse_contribution = -gamma * base_sum;
	*qjl_contribution = gamma * qjl_sum;
	*combined_score = *mse_contribution + *qjl_contribution;
	return true;
}

bool
tq_prod_score_packed_ip(const TqProdCodecConfig *config,
						const TqProdLut *lut,
						const uint8_t *packed,
						size_t packed_len,
						float *score,
						char *errmsg,
						size_t errmsg_len)
{
	float		mse_contribution = 0.0f;
	float		qjl_contribution = 0.0f;

	if (score == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod scorer: score output must be non-null");
		return false;
	}

	return tq_prod_score_decompose_ip(config, lut, packed, packed_len,
									  &mse_contribution, &qjl_contribution, score,
									  errmsg, errmsg_len);
}
