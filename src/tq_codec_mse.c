#include "src/tq_codec_mse.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
tq_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg_len == 0)
		return;

	snprintf(errmsg, errmsg_len, "%s", message);
}

static bool
tq_mse_validate_config(const TqMseCodecConfig *config,
					   char *errmsg,
					   size_t errmsg_len)
{
	if (config == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse codec config: config must be non-null");
		return false;
	}

	if (config->dimension == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse codec config: dimension must be positive");
		return false;
	}

	if (config->bits < 2 || config->bits > 8)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse codec config: bits must be between 2 and 8");
		return false;
	}

	if (!(config->max_value > config->min_value))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse codec config: max_value must be greater than min_value");
		return false;
	}

	return true;
}

static uint32_t
tq_mse_level_count(const TqMseCodecConfig *config)
{
	return UINT32_C(1) << config->bits;
}

static float
tq_mse_companding_mu(const TqMseCodecConfig *config)
{
	/* Embedded per-bit companding strengths for the deterministic v1.5 codebook. */
	static const float mu_by_bits[] = {
		0.0f,
		0.0f,
		0.5f,
		1.5f,
		2.5f,
		3.5f,
		4.5f,
		5.5f,
		6.5f
	};

	return mu_by_bits[config->bits];
}

static float
tq_mse_normalize_value(const TqMseCodecConfig *config, float value)
{
	float		position = 0.0f;

	if (value <= config->min_value)
		return -1.0f;

	if (value >= config->max_value)
		return 1.0f;

	position = (value - config->min_value) / (config->max_value - config->min_value);
	return (position * 2.0f) - 1.0f;
}

static float
tq_mse_denormalize_value(const TqMseCodecConfig *config, float value)
{
	float		clamped = fmaxf(-1.0f, fminf(1.0f, value));

	return config->min_value
		+ ((clamped + 1.0f) * 0.5f * (config->max_value - config->min_value));
}

static float
tq_mse_compand_value(const TqMseCodecConfig *config, float value)
{
	float		mu = tq_mse_companding_mu(config);
	float		normalized = tq_mse_normalize_value(config, value);

	if (mu <= 0.0f)
		return normalized;

	return copysignf(log1pf(mu * fabsf(normalized)) / log1pf(mu), normalized);
}

static float
tq_mse_inverse_compand(const TqMseCodecConfig *config, float value)
{
	float		mu = tq_mse_companding_mu(config);
	float		normalized = 0.0f;

	if (mu <= 0.0f)
		normalized = value;
	else
		normalized = copysignf(expm1f(fabsf(value) * log1pf(mu)) / mu, value);

	return tq_mse_denormalize_value(config, normalized);
}

static uint32_t
tq_mse_quantize_companded(const TqMseCodecConfig *config, float value)
{
	float		position = 0.0f;
	long		rounded = 0;
	uint32_t	max_code = tq_mse_level_count(config) - 1;
	float		companded = tq_mse_compand_value(config, value);

	if (companded <= -1.0f)
		return 0;

	if (companded >= 1.0f)
		return max_code;

	position = ((companded + 1.0f) * 0.5f) * (float) max_code;
	rounded = lroundf(position);

	if (rounded < 0)
		return 0;

	if ((uint32_t) rounded > max_code)
		return max_code;

	return (uint32_t) rounded;
}

static float
tq_mse_decode_code_value(const TqMseCodecConfig *config, uint32_t code)
{
	float		max_code = (float) (tq_mse_level_count(config) - 1);
	float		companded = -1.0f + (2.0f * ((float) code / max_code));

	return tq_mse_inverse_compand(config, companded);
}

static void
tq_mse_pack_code(uint8_t *packed, uint32_t index, uint8_t bits, uint32_t code)
{
	uint32_t	bit_offset = index * (uint32_t) bits;
	uint8_t		bit = 0;

	for (bit = 0; bit < bits; bit++)
	{
		uint32_t	target_bit = bit_offset + (uint32_t) bit;
		uint32_t	byte_index = target_bit / 8;
		uint32_t	byte_shift = target_bit % 8;
		uint8_t		mask = (uint8_t) (1u << byte_shift);

		if ((code >> bit) & 1u)
			packed[byte_index] |= mask;
		else
			packed[byte_index] &= (uint8_t) ~mask;
	}
}

static uint32_t
tq_mse_unpack_code(const uint8_t *packed, uint32_t index, uint8_t bits)
{
	uint32_t	bit_offset = index * (uint32_t) bits;
	uint32_t	byte_index = bit_offset / 8;
	uint32_t	bit_shift = bit_offset % 8;
	uint32_t	value = packed[byte_index];
	uint32_t	mask = (UINT32_C(1) << bits) - 1;

	if (bit_shift + bits > 8)
		value |= ((uint32_t) packed[byte_index + 1]) << 8;

	return (value >> bit_shift) & mask;
}

bool
tq_mse_packed_bytes(const TqMseCodecConfig *config,
					size_t *packed_bytes,
					char *errmsg,
					size_t errmsg_len)
{
	uint64_t	total_bits = 0;

	if (packed_bytes == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse buffer: packed byte output must be non-null");
		return false;
	}

	if (!tq_mse_validate_config(config, errmsg, errmsg_len))
		return false;

	total_bits = (uint64_t) config->dimension * (uint64_t) config->bits;
	*packed_bytes = (size_t) ((total_bits + 7) / 8);
	return true;
}

bool
tq_mse_encode(const TqMseCodecConfig *config,
			  const float *input,
			  uint8_t *packed,
			  size_t packed_len,
			  char *errmsg,
			  size_t errmsg_len)
{
	size_t		required_len = 0;
	uint32_t	i = 0;

	if (input == NULL || packed == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse buffer: input and packed output must be non-null");
		return false;
	}

	if (!tq_mse_packed_bytes(config, &required_len, errmsg, errmsg_len))
		return false;

	if (packed_len < required_len)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse buffer: packed output buffer is too small");
		return false;
	}

	memset(packed, 0, packed_len);
	for (i = 0; i < config->dimension; i++)
	{
		uint32_t	code = tq_mse_quantize_companded(config, input[i]);

		tq_mse_pack_code(packed, i, config->bits, code);
	}

	return true;
}

bool
tq_mse_decode(const TqMseCodecConfig *config,
			  const uint8_t *packed,
			  size_t packed_len,
			  float *output,
			  size_t output_len,
			  char *errmsg,
			  size_t errmsg_len)
{
	size_t		required_len = 0;
	uint32_t	i = 0;

	if (packed == NULL || output == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse buffer: packed input and decoded output must be non-null");
		return false;
	}

	if (!tq_mse_packed_bytes(config, &required_len, errmsg, errmsg_len))
		return false;

	if (packed_len < required_len)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse buffer: packed input buffer is too small");
		return false;
	}

	if (output_len < config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse buffer: decoded output buffer is too small");
		return false;
	}

	for (i = 0; i < config->dimension; i++)
	{
		uint32_t	code = tq_mse_unpack_code(packed, i, config->bits);

		output[i] = tq_mse_decode_code_value(config, code);
	}

	return true;
}

void
tq_mse_lut_reset(TqMseLut *lut)
{
	if (lut == NULL)
		return;

	free(lut->values);
	memset(lut, 0, sizeof(*lut));
}

bool
tq_mse_lut_build(const TqMseCodecConfig *config,
				 const float *query,
				 TqMseLut *lut,
				 char *errmsg,
				 size_t errmsg_len)
{
	uint32_t	level_count = 0;
	uint32_t	dim = 0;
	uint32_t	code = 0;

	if (query == NULL || lut == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse lut: query and lut must be non-null");
		return false;
	}

	if (!tq_mse_validate_config(config, errmsg, errmsg_len))
		return false;

	tq_mse_lut_reset(lut);

	level_count = tq_mse_level_count(config);
	lut->values = (float *) malloc(sizeof(float) * (size_t) config->dimension * (size_t) level_count);
	if (lut->values == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse lut: out of memory");
		return false;
	}

	lut->dimension = config->dimension;
	lut->level_count = level_count;

	for (dim = 0; dim < config->dimension; dim++)
	{
		for (code = 0; code < level_count; code++)
		{
			float		reconstructed = tq_mse_decode_code_value(config, code);
			float		diff = query[dim] - reconstructed;

			lut->values[(size_t) dim * (size_t) level_count + (size_t) code] = diff * diff;
		}
	}

	return true;
}

bool
tq_mse_score_packed_l2(const TqMseCodecConfig *config,
					   const TqMseLut *lut,
					   const uint8_t *packed,
					   size_t packed_len,
					   float *score,
					   char *errmsg,
					   size_t errmsg_len)
{
	size_t		required_len = 0;
	uint32_t	level_count = 0;
	uint32_t	dim = 0;
	float		total = 0.0f;

	if (lut == NULL || packed == NULL || score == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse scorer: lut, packed input, and score output must be non-null");
		return false;
	}

	if (!tq_mse_packed_bytes(config, &required_len, errmsg, errmsg_len))
		return false;

	if (packed_len < required_len)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse buffer: packed input buffer is too small");
		return false;
	}

	level_count = tq_mse_level_count(config);
	if (lut->dimension != config->dimension || lut->level_count != level_count || lut->values == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_mse scorer: lut shape does not match codec config");
		return false;
	}

	for (dim = 0; dim < config->dimension; dim++)
	{
		uint32_t	code = tq_mse_unpack_code(packed, dim, config->bits);

		total += lut->values[(size_t) dim * (size_t) level_count + (size_t) code];
	}

	*score = total;
	return true;
}
