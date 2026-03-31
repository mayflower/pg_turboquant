#include "src/tq_codec_prod.h"
#include "src/tq_transform.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TQ_PROD_CODEBOOK_GRID 8192u
#define TQ_PROD_LLOYD_MAX_ITERATIONS 64u
#define TQ_PROD_QJL_SCALE ((float) 1.253314137315500251f)
#define TQ_PROD_QJL_QUANT_RELATIVE_TOLERANCE (1.0f / 16384.0f)

typedef struct TqProdCodebook
{
	uint32_t	dimension;
	uint8_t		bits;
	uint32_t	level_count;
	float	   *centroids;
	float	   *boundaries;
} TqProdCodebook;

typedef struct TqProdSketch
{
	uint32_t	dimension;
	uint64_t	seed;
	TqTransformState transform;
} TqProdSketch;

static size_t tq_prod_decode_counter = 0;
static TqProdCodebook tq_prod_cached_codebook = {0};
static TqProdSketch tq_prod_cached_sketch = {0};

static uint32_t tq_prod_qjl_dimension(const TqProdCodecConfig *config);

static void
tq_prod_disable_quantized_qjl(TqProdLut *lut)
{
	if (lut == NULL)
		return;

	free(lut->qjl_quantized_values);
	lut->qjl_quantized_values = NULL;
	lut->qjl_quantization_scale = 0.0f;
	lut->qjl_quantization_max_error = 0.0f;
	lut->qjl_quantized_enabled = false;
}

static void
tq_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg_len == 0)
		return;

	snprintf(errmsg, errmsg_len, "%s", message);
}

static void
tq_pack_bits(uint8_t *packed, uint32_t bit_offset, uint32_t bit_count, uint32_t value)
{
	uint32_t	bit = 0;

	for (bit = 0; bit < bit_count; bit++)
	{
		uint32_t	target_bit = bit_offset + bit;
		uint32_t	byte_index = target_bit / 8u;
		uint32_t	byte_shift = target_bit % 8u;
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
		uint32_t	byte_index = target_bit / 8u;
		uint32_t	byte_shift = target_bit % 8u;

		if ((packed[byte_index] >> byte_shift) & 1u)
			value |= UINT32_C(1) << bit;
	}

	return value;
}

static void
tq_prod_codebook_reset(TqProdCodebook *codebook)
{
	if (codebook == NULL)
		return;

	free(codebook->centroids);
	free(codebook->boundaries);
	memset(codebook, 0, sizeof(*codebook));
}

static void
tq_prod_sketch_reset(TqProdSketch *sketch)
{
	if (sketch == NULL)
		return;

	tq_transform_reset(&sketch->transform);
	memset(sketch, 0, sizeof(*sketch));
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

	if (tq_prod_qjl_dimension(config) == 0 || tq_prod_qjl_dimension(config) > config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod codec config: qjl sketch dimension must be between 1 and the transformed dimension");
		return false;
	}

	return true;
}

static uint32_t
tq_prod_idx_bits(const TqProdCodecConfig *config)
{
	return (uint32_t) config->bits - 1u;
}

static uint32_t
tq_prod_idx_levels(const TqProdCodecConfig *config)
{
	return UINT32_C(1) << tq_prod_idx_bits(config);
}

static uint32_t
tq_prod_qjl_dimension(const TqProdCodecConfig *config)
{
	return config->qjl_dimension == 0 ? config->dimension : config->qjl_dimension;
}

static float
tq_prod_clamp_unit(float value)
{
	if (value > 1.0f)
		return 1.0f;
	if (value < -1.0f)
		return -1.0f;
	return value;
}

static float
tq_prod_qjl_scale(uint32_t qjl_dimension)
{
	return TQ_PROD_QJL_SCALE / (float) qjl_dimension;
}

static bool
tq_prod_get_sketch(const TqProdCodecConfig *config,
				   const TqProdSketch **sketch,
				   char *errmsg,
				   size_t errmsg_len)
{
	TqTransformConfig transform_config;

	if (sketch == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod codec config: sketch output must be non-null");
		return false;
	}

	if (!tq_prod_validate_config(config, errmsg, errmsg_len))
		return false;

	if (tq_prod_cached_sketch.dimension != config->dimension
		|| tq_prod_cached_sketch.seed != config->qjl_seed
		|| tq_prod_cached_sketch.transform.dimension != config->dimension
		|| tq_prod_cached_sketch.transform.padded_dimension == 0
		|| tq_prod_cached_sketch.transform.permutation == NULL
		|| tq_prod_cached_sketch.transform.signs == NULL)
	{
		memset(&transform_config, 0, sizeof(transform_config));
		tq_prod_sketch_reset(&tq_prod_cached_sketch);
		transform_config.kind = TQ_TRANSFORM_HADAMARD;
		transform_config.dimension = config->dimension;
		transform_config.seed = config->qjl_seed;

		if (!tq_transform_prepare(&transform_config,
								  &tq_prod_cached_sketch.transform,
								  errmsg,
								  errmsg_len))
			return false;

		tq_prod_cached_sketch.dimension = config->dimension;
		tq_prod_cached_sketch.seed = config->qjl_seed;
	}

	*sketch = &tq_prod_cached_sketch;
	return true;
}

static void
tq_prod_write_float32(uint8_t *dest, float value)
{
	memcpy(dest, &value, sizeof(float));
}

static float
tq_prod_read_float32(const uint8_t *src)
{
	float		value = 0.0f;

	memcpy(&value, src, sizeof(float));
	return value;
}

static double
tq_prod_coordinate_density(uint32_t dimension, double value)
{
	double		clamped = value;
	double		one_minus = 0.0;
	double		exponent = 0.0;

	if (clamped > 1.0)
		clamped = 1.0;
	if (clamped < -1.0)
		clamped = -1.0;

	one_minus = 1.0 - (clamped * clamped);
	if (one_minus <= 0.0)
		return 0.0;

	if (dimension > 3u)
		exponent = ((double) dimension - 3.0) * 0.5;

	if (exponent <= 0.0)
		return 1.0;

	return exp(exponent * log(one_minus));
}

static float
tq_prod_quantile_edge(const double *cdf,
					  uint32_t grid_count,
					  double total_mass,
					  double target_mass)
{
	uint32_t	low = 0;
	uint32_t	high = grid_count;
	double		dx = 2.0 / (double) grid_count;
	uint32_t	index = 0;

	if (target_mass <= 0.0)
		return -1.0f;
	if (target_mass >= total_mass)
		return 1.0f;

	while (low < high)
	{
		uint32_t	mid = low + ((high - low) / 2u);

		if (cdf[mid] < target_mass)
			low = mid + 1u;
		else
			high = mid;
	}

	index = low;
	if (index == 0)
		return -1.0f;
	if (index >= grid_count)
		return 1.0f;

	{
		double left_cdf = cdf[index - 1u];
		double right_cdf = cdf[index];
		double fraction = 0.0;
		double left_edge = -1.0 + ((double) (index - 1u) * dx);

		if (right_cdf > left_cdf)
			fraction = (target_mass - left_cdf) / (right_cdf - left_cdf);

		if (fraction < 0.0)
			fraction = 0.0;
		if (fraction > 1.0)
			fraction = 1.0;

		return (float) (left_edge + (fraction * dx));
	}
}

static float
tq_prod_interval_weighted_mean(const double *weights,
							   uint32_t grid_count,
							   float lower,
							   float upper,
							   float fallback)
{
	double		dx = 2.0 / (double) grid_count;
	double		mass = 0.0;
	double		moment = 0.0;
	uint32_t	cell = 0;

	if (upper <= lower)
		return fallback;

	for (cell = 0; cell < grid_count; cell++)
	{
		double cell_lower = -1.0 + ((double) cell * dx);
		double cell_upper = cell_lower + dx;
		double overlap_lower = cell_lower > lower ? cell_lower : lower;
		double overlap_upper = cell_upper < upper ? cell_upper : upper;

		if (overlap_upper <= overlap_lower)
			continue;

		{
			double overlap_fraction = (overlap_upper - overlap_lower) / dx;
			double midpoint = 0.5 * (overlap_lower + overlap_upper);
			double weight = weights[cell] * overlap_fraction;

			mass += weight;
			moment += midpoint * weight;
		}
	}

	if (mass <= 0.0)
		return fallback;

	return (float) (moment / mass);
}

static bool
tq_prod_build_codebook(const TqProdCodecConfig *config,
					   TqProdCodebook *codebook,
					   char *errmsg,
					   size_t errmsg_len)
{
	uint32_t	level_count = tq_prod_idx_levels(config);
	double	   *weights = NULL;
	double	   *cdf = NULL;
	double		total_mass = 0.0;
	double		dx = 2.0 / (double) TQ_PROD_CODEBOOK_GRID;
	float	   *next_centroids = NULL;
	uint32_t	cell = 0;
	uint32_t	level = 0;
	uint32_t	iteration = 0;

	tq_prod_codebook_reset(codebook);

	codebook->centroids = (float *) calloc((size_t) level_count, sizeof(float));
	codebook->boundaries = (float *) calloc((size_t) level_count + 1u, sizeof(float));
	weights = (double *) calloc((size_t) TQ_PROD_CODEBOOK_GRID, sizeof(double));
	cdf = (double *) calloc((size_t) TQ_PROD_CODEBOOK_GRID, sizeof(double));
	next_centroids = (float *) calloc((size_t) level_count, sizeof(float));

	if (codebook->centroids == NULL || codebook->boundaries == NULL
		|| weights == NULL || cdf == NULL || next_centroids == NULL)
	{
		tq_prod_codebook_reset(codebook);
		free(weights);
		free(cdf);
		free(next_centroids);
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod codec config: out of memory");
		return false;
	}

	for (cell = 0; cell < TQ_PROD_CODEBOOK_GRID; cell++)
	{
		double mid = -1.0 + (((double) cell + 0.5) * dx);
		double weight = tq_prod_coordinate_density(config->dimension, mid) * dx;

		weights[cell] = weight;
		total_mass += weight;
		cdf[cell] = total_mass;
	}

	for (level = 0; level < level_count; level++)
	{
		float lower_edge = -1.0f;
		float upper_edge = 1.0f;
		double lower_target = total_mass * ((double) level / (double) level_count);
		double upper_target = total_mass * ((double) (level + 1u) / (double) level_count);

		if (level > 0)
			lower_edge = tq_prod_quantile_edge(cdf, TQ_PROD_CODEBOOK_GRID, total_mass, lower_target);
		if (level + 1u < level_count)
			upper_edge = tq_prod_quantile_edge(cdf, TQ_PROD_CODEBOOK_GRID, total_mass, upper_target);

		codebook->centroids[level] = tq_prod_interval_weighted_mean(weights,
																	 TQ_PROD_CODEBOOK_GRID,
																	 lower_edge,
																	 upper_edge,
																	 0.5f * (lower_edge + upper_edge));
	}

	for (iteration = 0; iteration < TQ_PROD_LLOYD_MAX_ITERATIONS; iteration++)
	{
		float	max_shift = 0.0f;

		codebook->boundaries[0] = -1.0f;
		codebook->boundaries[level_count] = 1.0f;
		for (level = 1; level < level_count; level++)
			codebook->boundaries[level] = 0.5f * (codebook->centroids[level - 1u] + codebook->centroids[level]);

		for (level = 0; level < level_count; level++)
		{
			float lower = codebook->boundaries[level];
			float upper = codebook->boundaries[level + 1u];
			float fallback = 0.5f * (lower + upper);

			next_centroids[level] = tq_prod_interval_weighted_mean(weights,
																	 TQ_PROD_CODEBOOK_GRID,
																	 lower,
																	 upper,
																	 fallback);
			if (fabsf(next_centroids[level] - codebook->centroids[level]) > max_shift)
				max_shift = fabsf(next_centroids[level] - codebook->centroids[level]);
		}

		memcpy(codebook->centroids, next_centroids, sizeof(float) * (size_t) level_count);
		if (max_shift < 1e-6f)
			break;
	}

	codebook->boundaries[0] = -1.0f;
	codebook->boundaries[level_count] = 1.0f;
	for (level = 1; level < level_count; level++)
		codebook->boundaries[level] = 0.5f * (codebook->centroids[level - 1u] + codebook->centroids[level]);

	codebook->dimension = config->dimension;
	codebook->bits = config->bits;
	codebook->level_count = level_count;

	free(weights);
	free(cdf);
	free(next_centroids);
	return true;
}

static bool
tq_prod_get_codebook(const TqProdCodecConfig *config,
					 const TqProdCodebook **codebook,
					 char *errmsg,
					 size_t errmsg_len)
{
	if (codebook == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod codec config: codebook output must be non-null");
		return false;
	}

	if (!tq_prod_validate_config(config, errmsg, errmsg_len))
		return false;

	if (tq_prod_cached_codebook.dimension != config->dimension
		|| tq_prod_cached_codebook.bits != config->bits
		|| tq_prod_cached_codebook.level_count != tq_prod_idx_levels(config)
		|| tq_prod_cached_codebook.centroids == NULL
		|| tq_prod_cached_codebook.boundaries == NULL)
	{
		if (!tq_prod_build_codebook(config, &tq_prod_cached_codebook, errmsg, errmsg_len))
			return false;
	}

	*codebook = &tq_prod_cached_codebook;
	return true;
}

static uint32_t
tq_prod_find_code(const TqProdCodebook *codebook, float value)
{
	uint32_t	low = 0;
	uint32_t	high = codebook->level_count;

	while (low + 1u < high)
	{
		uint32_t	mid = low + ((high - low) / 2u);

		if (value < codebook->boundaries[mid])
			high = mid;
		else
			low = mid;
	}

	if (low >= codebook->level_count)
		return codebook->level_count - 1u;
	return low;
}

static bool
tq_prod_validate_packed_inputs(const TqProdPackedLayout *layout,
							   const uint8_t *packed,
							   size_t packed_len,
							   char *errmsg,
							   size_t errmsg_len)
{
	if (packed == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: packed input must be non-null");
		return false;
	}

	if (packed_len < layout->total_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: packed input buffer is too small");
		return false;
	}

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
	layout->idx_bytes = (size_t) ((idx_bits + 7u) / 8u);
	layout->qjl_bytes = (size_t) (((uint64_t) tq_prod_qjl_dimension(config) + 7u) / 8u);
	layout->gamma_bytes = sizeof(float);
	layout->total_bytes = layout->idx_bytes + layout->qjl_bytes + layout->gamma_bytes;
	return true;
}

bool
tq_prod_qjl_project(const TqProdCodecConfig *config,
					const float *input,
					float *output,
					size_t output_len,
					char *errmsg,
					size_t errmsg_len)
{
	const TqProdSketch *sketch = NULL;
	float	   *full_projection = NULL;
	uint32_t	qjl_dimension = 0;
	uint32_t	i = 0;
	bool		ok = false;

	if (input == NULL || output == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod qjl: input and output must be non-null");
		return false;
	}

	if (!tq_prod_get_sketch(config, &sketch, errmsg, errmsg_len))
		return false;

	qjl_dimension = tq_prod_qjl_dimension(config);
	if (output_len < qjl_dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod qjl: output buffer is too small");
		return false;
	}

	full_projection = (float *) calloc((size_t) sketch->transform.padded_dimension, sizeof(float));
	if (full_projection == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod qjl: out of memory");
		return false;
	}

	if (!tq_transform_apply(&sketch->transform,
							 input,
							 full_projection,
							 sketch->transform.padded_dimension,
							 errmsg,
							 errmsg_len))
		goto cleanup;

	for (i = 0; i < qjl_dimension; i++)
		output[i] = full_projection[i];

	ok = true;

cleanup:
	free(full_projection);
	return ok;
}

bool
tq_prod_qjl_backproject_signs(const TqProdCodecConfig *config,
							  const uint8_t *packed_signs,
							  size_t packed_signs_len,
							  float *output,
							  size_t output_len,
							  char *errmsg,
							  size_t errmsg_len)
{
	const TqProdSketch *sketch = NULL;
	float	   *full_signs = NULL;
	uint32_t	qjl_dimension = 0;
	uint32_t	i = 0;
	size_t		required_bytes = 0;
	bool		ok = false;

	if (packed_signs == NULL || output == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod qjl: packed signs and output must be non-null");
		return false;
	}

	if (!tq_prod_get_sketch(config, &sketch, errmsg, errmsg_len))
		return false;

	qjl_dimension = tq_prod_qjl_dimension(config);
	if (output_len < config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod qjl: output buffer is too small");
		return false;
	}

	required_bytes = (size_t) (((uint64_t) qjl_dimension + 7u) / 8u);
	if (packed_signs_len < required_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod qjl: packed sign buffer is too small");
		return false;
	}

	full_signs = (float *) calloc((size_t) sketch->transform.padded_dimension, sizeof(float));
	if (full_signs == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod qjl: out of memory");
		return false;
	}

	for (i = 0; i < qjl_dimension; i++)
		full_signs[i] = tq_unpack_bits(packed_signs, i, 1u) ? 1.0f : -1.0f;

	if (!tq_transform_inverse(&sketch->transform,
							   full_signs,
							   sketch->transform.padded_dimension,
							   output,
							   output_len,
							   errmsg,
							   errmsg_len))
		goto cleanup;

	ok = true;

cleanup:
	free(full_signs);
	return ok;
}

bool
tq_prod_feature_distance(const TqProdCodecConfig *config,
						 const uint8_t *left_packed,
						 size_t left_packed_len,
						 const uint8_t *right_packed,
						 size_t right_packed_len,
						 float *distance,
	char *errmsg,
	size_t errmsg_len)
{
	TqProdPackedLayout layout;
	float	   *left_decoded = NULL;
	float	   *right_decoded = NULL;
	double		distance_sq = 0.0;
	uint32_t	i = 0;
	bool		ok = false;

	if (distance == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod scorer: feature distance output must be non-null");
		return false;
	}

	memset(&layout, 0, sizeof(layout));
	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len)
		|| !tq_prod_validate_packed_inputs(&layout, left_packed, left_packed_len, errmsg, errmsg_len)
		|| !tq_prod_validate_packed_inputs(&layout, right_packed, right_packed_len, errmsg, errmsg_len))
		return false;

	left_decoded = (float *) calloc((size_t) config->dimension, sizeof(float));
	right_decoded = (float *) calloc((size_t) config->dimension, sizeof(float));
	if (left_decoded == NULL || right_decoded == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod scorer: out of memory");
		goto cleanup;
	}

	if (!tq_prod_decode(config,
						left_packed,
						left_packed_len,
						left_decoded,
						config->dimension,
						errmsg,
						errmsg_len)
		|| !tq_prod_decode(config,
						   right_packed,
						   right_packed_len,
						   right_decoded,
						   config->dimension,
						   errmsg,
						   errmsg_len))
		goto cleanup;

	for (i = 0; i < config->dimension; i++)
	{
		double delta = (double) left_decoded[i] - (double) right_decoded[i];
		distance_sq += delta * delta;
	}

	*distance = (float) sqrt(distance_sq);
	ok = true;

cleanup:
	free(left_decoded);
	free(right_decoded);
	return ok;
}

bool
tq_prod_query_weight_l2_norm(const TqProdCodecConfig *config,
							 const TqProdLut *lut,
							 float *norm,
							 char *errmsg,
							 size_t errmsg_len)
{
	if (norm == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod scorer: query weight norm output must be non-null");
		return false;
	}

	if (lut == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod scorer: lut must be non-null");
		return false;
	}

	if (!tq_prod_validate_config(config, errmsg, errmsg_len))
		return false;

	if (lut->dimension != config->dimension
		|| lut->qjl_dimension != tq_prod_qjl_dimension(config)
		|| lut->level_count != tq_prod_idx_levels(config)
		|| lut->values == NULL
		|| lut->qjl_values == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod scorer: lut shape does not match codec config");
		return false;
	}

	*norm = lut->feature_weight_norm;
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
	const TqProdCodebook *codebook = NULL;
	float	   *residual = NULL;
	float	   *projection = NULL;
	double		residual_norm_sq = 0.0;
	float		gamma = 0.0f;
	uint32_t	qjl_dimension = 0;
	uint32_t	i = 0;
	bool		ok = false;

	if (input == NULL || packed == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: input and packed output must be non-null");
		return false;
	}

	memset(&layout, 0, sizeof(layout));

	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len)
		|| !tq_prod_get_codebook(config, &codebook, errmsg, errmsg_len)
		)
		return false;

	if (packed_len < layout.total_bytes)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: packed output buffer is too small");
		return false;
	}

	qjl_dimension = tq_prod_qjl_dimension(config);
	residual = (float *) malloc(sizeof(float) * (size_t) config->dimension);
	projection = (float *) malloc(sizeof(float) * (size_t) qjl_dimension);
	if (residual == NULL || projection == NULL)
	{
		free(residual);
		free(projection);
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: out of memory");
		return false;
	}

	memset(packed, 0, packed_len);

	for (i = 0; i < config->dimension; i++)
	{
		float		clamped = tq_prod_clamp_unit(input[i]);
		uint32_t	idx_code = tq_prod_find_code(codebook, clamped);
		float		stage1 = codebook->centroids[idx_code];
		float		delta = clamped - stage1;

		tq_pack_bits(packed, i * tq_prod_idx_bits(config), tq_prod_idx_bits(config), idx_code);
		residual[i] = delta;
		residual_norm_sq += (double) delta * (double) delta;
	}

	gamma = (float) sqrt(residual_norm_sq);
	tq_prod_write_float32(packed + layout.idx_bytes + layout.qjl_bytes, gamma);

	if (gamma > 0.0f)
	{
		if (!tq_prod_qjl_project(config, residual, projection, qjl_dimension, errmsg, errmsg_len))
			goto cleanup;

		for (i = 0; i < qjl_dimension; i++)
		{
			uint32_t	bit = 0;

			bit = projection[i] >= 0.0f ? 1u : 0u;
			tq_pack_bits(packed + layout.idx_bytes, i, 1u, bit);
		}
	}

	ok = true;
cleanup:
	free(residual);
	free(projection);
	return ok;
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

	if (gamma == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: gamma output must be non-null");
		return false;
	}

	memset(&layout, 0, sizeof(layout));

	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len)
		|| !tq_prod_validate_packed_inputs(&layout, packed, packed_len, errmsg, errmsg_len))
		return false;

	*gamma = tq_prod_read_float32(packed + layout.idx_bytes + layout.qjl_bytes);
	return true;
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
	const TqProdCodebook *codebook = NULL;
	float	   *residual = NULL;
	float		gamma = 0.0f;
	float		scale = 0.0f;
	uint32_t	qjl_dimension = 0;
	uint32_t	i = 0;
	bool		ok = false;

	if (packed == NULL || output == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: packed input and decoded output must be non-null");
		return false;
	}

	memset(&layout, 0, sizeof(layout));

	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len)
		|| !tq_prod_get_codebook(config, &codebook, errmsg, errmsg_len)
		|| !tq_prod_validate_packed_inputs(&layout, packed, packed_len, errmsg, errmsg_len))
		return false;

	if (output_len < config->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: decoded output buffer is too small");
		return false;
	}

	tq_prod_decode_counter += 1;
	for (i = 0; i < config->dimension; i++)
	{
		uint32_t	idx_code = tq_unpack_bits(packed, i * tq_prod_idx_bits(config), tq_prod_idx_bits(config));

		output[i] = codebook->centroids[idx_code];
	}

	gamma = tq_prod_read_float32(packed + layout.idx_bytes + layout.qjl_bytes);
	if (gamma <= 0.0f)
		return true;

	qjl_dimension = tq_prod_qjl_dimension(config);
	residual = (float *) calloc((size_t) config->dimension, sizeof(float));
	if (residual == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod buffer: out of memory");
		return false;
	}

	if (!tq_prod_qjl_backproject_signs(config,
									   packed + layout.idx_bytes,
									   layout.qjl_bytes,
									   residual,
									   config->dimension,
									   errmsg,
									   errmsg_len))
		goto cleanup;

	scale = tq_prod_qjl_scale(qjl_dimension) * gamma;
	for (i = 0; i < config->dimension; i++)
		output[i] += scale * residual[i];

	ok = true;

cleanup:
	free(residual);
	return ok;
}

void
tq_prod_lut_reset(TqProdLut *lut)
{
	if (lut == NULL)
		return;

	free(lut->values);
	free(lut->qjl_values);
	free(lut->qjl_quantized_values);
	memset(lut, 0, sizeof(*lut));
}

static bool
tq_prod_build_quantized_qjl_lut(TqProdLut *lut,
								  char *errmsg,
								  size_t errmsg_len)
{
	int16_t *quantized = NULL;
	float max_abs = 0.0f;
	float max_error = 0.0f;
	float scale = 1.0f;
	uint32_t dim = 0;

	if (lut == NULL || lut->qjl_values == NULL || lut->qjl_dimension == 0)
		return true;

	quantized = (int16_t *) malloc(sizeof(int16_t) * (size_t) lut->qjl_dimension);
	if (quantized == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod lut: out of memory");
		return false;
	}

	for (dim = 0; dim < lut->qjl_dimension; dim++)
	{
		float magnitude = fabsf(lut->qjl_values[dim]);

		if (magnitude > max_abs)
			max_abs = magnitude;
	}

	if (max_abs > 0.0f)
		scale = max_abs / (float) INT16_MAX;

	for (dim = 0; dim < lut->qjl_dimension; dim++)
	{
		float value = lut->qjl_values[dim];
		float reconstructed = 0.0f;
		float error = 0.0f;
		long quantized_value = 0;

		quantized_value = lroundf(value / scale);
		if (quantized_value > INT16_MAX)
			quantized_value = INT16_MAX;
		else if (quantized_value < INT16_MIN)
			quantized_value = INT16_MIN;
		quantized[dim] = (int16_t) quantized_value;
		reconstructed = (float) quantized[dim] * scale;
		error = fabsf(reconstructed - value);
		if (error > max_error)
			max_error = error;
	}

	if (max_abs > 0.0f
		&& (max_error / max_abs) > TQ_PROD_QJL_QUANT_RELATIVE_TOLERANCE)
	{
		free(quantized);
		tq_prod_disable_quantized_qjl(lut);
		return true;
	}

	tq_prod_disable_quantized_qjl(lut);
	lut->qjl_quantized_values = quantized;
	lut->qjl_quantization_scale = scale;
	lut->qjl_quantization_max_error = max_error;
	lut->qjl_quantized_enabled = true;
	return true;
}

bool
tq_prod_lut_build(const TqProdCodecConfig *config,
				  const float *query,
				  TqProdLut *lut,
				  char *errmsg,
				  size_t errmsg_len)
{
	const TqProdCodebook *codebook = NULL;
	float	   *projection = NULL;
	uint32_t	qjl_dimension = 0;
	uint32_t	dim = 0;
	uint32_t	code = 0;
	float		scale = 0.0f;
	double		weight_norm_sq = 0.0;
	bool		ok = false;

	if (query == NULL || lut == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod lut: query and lut must be non-null");
		return false;
	}

	if (!tq_prod_get_codebook(config, &codebook, errmsg, errmsg_len))
		return false;

	tq_prod_lut_reset(lut);
	qjl_dimension = tq_prod_qjl_dimension(config);

	lut->values = (float *) malloc(sizeof(float) * (size_t) config->dimension * (size_t) codebook->level_count);
	lut->qjl_values = (float *) malloc(sizeof(float) * (size_t) qjl_dimension);
	projection = (float *) malloc(sizeof(float) * (size_t) qjl_dimension);
	if (lut->values == NULL || lut->qjl_values == NULL)
	{
		tq_prod_lut_reset(lut);
		free(projection);
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod lut: out of memory");
		return false;
	}
	if (projection == NULL)
	{
		tq_prod_lut_reset(lut);
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod lut: out of memory");
		return false;
	}

	lut->dimension = config->dimension;
	lut->qjl_dimension = qjl_dimension;
	lut->level_count = codebook->level_count;

	for (dim = 0; dim < config->dimension; dim++)
	{
		for (code = 0; code < codebook->level_count; code++)
		{
			size_t index = ((size_t) dim * (size_t) codebook->level_count) + (size_t) code;

			lut->values[index] = query[dim] * codebook->centroids[code];
		}
		weight_norm_sq += (double) query[dim] * (double) query[dim];
	}

	if (!tq_prod_qjl_project(config, query, projection, qjl_dimension, errmsg, errmsg_len))
		goto cleanup;

	scale = tq_prod_qjl_scale(qjl_dimension);
	for (dim = 0; dim < qjl_dimension; dim++)
	{
		lut->qjl_values[dim] = scale * projection[dim];
		weight_norm_sq += (double) lut->qjl_values[dim] * (double) lut->qjl_values[dim];
	}

	lut->feature_weight_norm = (float) sqrt(weight_norm_sq);
	if (!tq_prod_build_quantized_qjl_lut(lut, errmsg, errmsg_len))
		goto cleanup;

	ok = true;

cleanup:
	free(projection);
	if (!ok)
		tq_prod_lut_reset(lut);
	return ok;
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
	float		base_sum = 0.0f;
	float		residual_sum = 0.0f;
	float		gamma = 0.0f;
	uint32_t	qjl_dimension = 0;
	uint32_t	i = 0;

	if (lut == NULL || packed == NULL || mse_contribution == NULL
		|| qjl_contribution == NULL || combined_score == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod scorer: lut, packed input, and score outputs must be non-null");
		return false;
	}

	memset(&layout, 0, sizeof(layout));
	qjl_dimension = tq_prod_qjl_dimension(config);

	if (!tq_prod_packed_layout(config, &layout, errmsg, errmsg_len))
		return false;

	if (lut->dimension != config->dimension
		|| lut->qjl_dimension != qjl_dimension
		|| lut->level_count != tq_prod_idx_levels(config)
		|| lut->values == NULL
		|| lut->qjl_values == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod scorer: lut shape does not match codec config");
		return false;
	}

	if (!tq_prod_validate_packed_inputs(&layout, packed, packed_len, errmsg, errmsg_len))
		return false;

	for (i = 0; i < config->dimension; i++)
	{
		uint32_t	idx_code = tq_unpack_bits(packed, i * tq_prod_idx_bits(config), tq_prod_idx_bits(config));
		size_t		index = ((size_t) i * (size_t) lut->level_count) + (size_t) idx_code;
			base_sum += lut->values[index];
		}

	for (i = 0; i < qjl_dimension; i++)
	{
		float sign = tq_unpack_bits(packed + layout.idx_bytes, i, 1u) ? 1.0f : -1.0f;

		residual_sum += sign * lut->qjl_values[i];
	}

	gamma = tq_prod_read_float32(packed + layout.idx_bytes + layout.qjl_bytes);
	*mse_contribution = base_sum;
	*qjl_contribution = gamma * residual_sum;
	*combined_score = base_sum + (*qjl_contribution);
	return true;
}

bool
tq_prod_score_code_from_lut(const TqProdCodecConfig *config,
							const TqProdLut *lut,
							const uint8_t *packed,
							size_t packed_len,
							float *score,
							char *errmsg,
							size_t errmsg_len)
{
	return tq_prod_score_packed_ip(config, lut, packed, packed_len,
								   score, errmsg, errmsg_len);
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
	float		stage1_contribution = 0.0f;
	float		residual_contribution = 0.0f;

	if (score == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid tq_prod scorer: score output must be non-null");
		return false;
	}

	return tq_prod_score_decompose_ip(config,
									  lut,
									  packed,
									  packed_len,
									  &stage1_contribution,
									  &residual_contribution,
									  score,
									  errmsg,
									  errmsg_len);
}

void
tq_prod_decode_counter_reset(void)
{
	tq_prod_decode_counter = 0;
}

size_t
tq_prod_decode_counter_get(void)
{
	return tq_prod_decode_counter;
}
