#include "src/tq_transform.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TQ_UNIT_TEST
#undef snprintf
#endif

static void
tq_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg_len == 0)
		return;

	snprintf(errmsg, errmsg_len, "%s", message);
}

static uint64_t
tq_splitmix64(uint64_t *state)
{
	uint64_t	z = 0;

	*state += UINT64_C(0x9E3779B97F4A7C15);
	z = *state;
	z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
	return z ^ (z >> 31);
}

static bool
tq_validate_state(const TqTransformState *state, char *errmsg, size_t errmsg_len)
{
	if (state == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: state must be non-null");
		return false;
	}

	if (state->kind != TQ_TRANSFORM_HADAMARD)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: unsupported transform kind");
		return false;
	}

	if (state->dimension == 0 || state->padded_dimension == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: dimensions must be positive");
		return false;
	}

	if (state->permutation == NULL || state->signs == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: prepared state is incomplete");
		return false;
	}

	if (state->permutation_count != state->padded_dimension
		|| state->sign_count != state->padded_dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: prepared state has inconsistent counts");
		return false;
	}

	return true;
}

static void
tq_fwht_normalized(float *values, uint32_t len)
{
	uint32_t	step = 0;
	float		scale = 0.0f;
	uint32_t	base = 0;
	uint32_t	offset = 0;

	for (step = 1; step < len; step *= 2)
	{
		for (base = 0; base < len; base += step * 2)
		{
			for (offset = 0; offset < step; offset++)
			{
				float	a = values[base + offset];
				float	b = values[base + offset + step];

				values[base + offset] = a + b;
				values[base + offset + step] = a - b;
			}
		}
	}

	scale = 1.0f / sqrtf((float) len);
	for (base = 0; base < len; base++)
		values[base] *= scale;
}

static bool
tq_alloc_work_buffer(uint32_t padded_dimension,
					 float **buffer,
					 char *errmsg,
					 size_t errmsg_len)
{
	*buffer = (float *) calloc((size_t) padded_dimension, sizeof(float));
	if (*buffer == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: out of memory");
		return false;
	}

	return true;
}

static void
tq_apply_signed_permutation(const TqTransformState *state,
							const float *input,
							float *work)
{
	uint32_t	i = 0;

	for (i = 0; i < state->padded_dimension; i++)
	{
		float		value = 0.0f;

		if (i < state->dimension)
			value = input[i];

		work[state->permutation[i]] = value * (float) state->signs[i];
	}
}

static void
tq_invert_signed_permutation(const TqTransformState *state,
							 const float *work,
							 float *output,
							 size_t output_len)
{
	uint32_t	i = 0;

	for (i = 0; i < state->dimension && i < output_len; i++)
		output[i] = work[state->permutation[i]] * (float) state->signs[i];
}

uint32_t
tq_transform_padded_dimension(uint32_t dimension)
{
	uint32_t	padded = 1;

	if (dimension == 0)
		return 0;

	while (padded < dimension)
		padded <<= 1;

	return padded;
}

bool
tq_transform_metadata_init(const TqTransformConfig *config,
						   TqTransformMetadata *metadata,
						   char *errmsg,
						   size_t errmsg_len)
{
	if (config == NULL || metadata == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: config and metadata must be non-null");
		return false;
	}

	if (config->kind != TQ_TRANSFORM_HADAMARD)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: only hadamard is supported in v1");
		return false;
	}

	if (config->dimension == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: dimension must be positive");
		return false;
	}

	memset(metadata, 0, sizeof(*metadata));
	metadata->contract_version = TQ_TRANSFORM_CONTRACT_VERSION;
	metadata->kind = config->kind;
	metadata->input_dimension = config->dimension;
	metadata->output_dimension = tq_transform_padded_dimension(config->dimension);
	metadata->seed = config->seed;
	return true;
}

bool
tq_transform_prepare_metadata(const TqTransformMetadata *metadata,
							  TqTransformState *state,
							  char *errmsg,
							  size_t errmsg_len)
{
	TqTransformConfig config;

	if (metadata == NULL || state == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: metadata and state must be non-null");
		return false;
	}

	if (metadata->contract_version != TQ_TRANSFORM_CONTRACT_VERSION)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: unsupported contract version");
		return false;
	}

	if (metadata->kind != TQ_TRANSFORM_HADAMARD)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: unsupported transform kind");
		return false;
	}

	if (metadata->input_dimension == 0 || metadata->output_dimension == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: metadata dimensions must be positive");
		return false;
	}

	if (metadata->output_dimension != tq_transform_padded_dimension(metadata->input_dimension))
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: output dimension does not match padded hadamard contract");
		return false;
	}

	memset(&config, 0, sizeof(config));
	config.kind = metadata->kind;
	config.dimension = metadata->input_dimension;
	config.seed = metadata->seed;
	return tq_transform_prepare(&config, state, errmsg, errmsg_len);
}

void
tq_transform_reset(TqTransformState *state)
{
	if (state == NULL)
		return;

	free(state->permutation);
	free(state->signs);
	memset(state, 0, sizeof(*state));
}

bool
tq_transform_prepare(const TqTransformConfig *config,
					 TqTransformState *state,
					 char *errmsg,
					 size_t errmsg_len)
{
	TqTransformMetadata metadata;
	uint64_t	rng_state = 0;
	uint32_t	i = 0;

	if (config == NULL || state == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: config and state must be non-null");
		return false;
	}

	if (!tq_transform_metadata_init(config, &metadata, errmsg, errmsg_len))
		return false;

	tq_transform_reset(state);

	state->permutation = (uint32_t *) malloc(sizeof(uint32_t) * (size_t) metadata.output_dimension);
	state->signs = (int8_t *) malloc(sizeof(int8_t) * (size_t) metadata.output_dimension);

	if (state->permutation == NULL || state->signs == NULL)
	{
		tq_transform_reset(state);
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: out of memory");
		return false;
	}

	state->kind = config->kind;
	state->dimension = metadata.input_dimension;
	state->padded_dimension = metadata.output_dimension;
	state->seed = metadata.seed;
	state->permutation_count = metadata.output_dimension;
	state->sign_count = metadata.output_dimension;

	rng_state = metadata.seed;
	for (i = 0; i < metadata.output_dimension; i++)
	{
		state->permutation[i] = i;
		state->signs[i] = (tq_splitmix64(&rng_state) & UINT64_C(1)) ? 1 : -1;
	}

	for (i = metadata.output_dimension; i > 1; i--)
	{
		uint32_t	j = (uint32_t) (tq_splitmix64(&rng_state) % (uint64_t) i);
		uint32_t	tmp = state->permutation[i - 1];

		state->permutation[i - 1] = state->permutation[j];
		state->permutation[j] = tmp;
	}

	return true;
}

bool
tq_transform_apply_reference(const TqTransformState *state,
							 const float *input,
							 float *output,
							 size_t output_len,
							 char *errmsg,
							 size_t errmsg_len)
{
	float	   *work = NULL;

	if (input == NULL || output == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: input and output must be non-null");
		return false;
	}

	if (!tq_validate_state(state, errmsg, errmsg_len))
		return false;

	if (output_len < state->padded_dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: reference output buffer is too small");
		return false;
	}

	if (!tq_alloc_work_buffer(state->padded_dimension, &work, errmsg, errmsg_len))
		return false;

	tq_apply_signed_permutation(state, input, work);
	tq_fwht_normalized(work, state->padded_dimension);
	memcpy(output, work, sizeof(float) * (size_t) state->padded_dimension);
	free(work);
	return true;
}

bool
tq_transform_inverse_reference(const TqTransformState *state,
							   const float *input,
							   float *output,
							   size_t output_len,
							   char *errmsg,
							   size_t errmsg_len)
{
	float	   *work = NULL;

	if (input == NULL || output == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: input and output must be non-null");
		return false;
	}

	if (!tq_validate_state(state, errmsg, errmsg_len))
		return false;

	if (output_len < state->dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: inverse output buffer is too small");
		return false;
	}

	if (!tq_alloc_work_buffer(state->padded_dimension, &work, errmsg, errmsg_len))
		return false;

	memcpy(work, input, sizeof(float) * (size_t) state->padded_dimension);
	tq_fwht_normalized(work, state->padded_dimension);
	memset(output, 0, sizeof(float) * output_len);
	tq_invert_signed_permutation(state, work, output, output_len);
	free(work);
	return true;
}

bool
tq_transform_inverse(const TqTransformState *state,
					 const float *input,
					 size_t input_len,
					 float *output,
					 size_t output_len,
					 char *errmsg,
					 size_t errmsg_len)
{
	if (!tq_validate_state(state, errmsg, errmsg_len))
		return false;

	if (input_len < state->padded_dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: inverse input buffer is too small");
		return false;
	}

	return tq_transform_inverse_reference(state, input, output, output_len,
										  errmsg, errmsg_len);
}

bool
tq_transform_apply(const TqTransformState *state,
				   const float *input,
				   float *output,
				   size_t output_len,
				   char *errmsg,
				   size_t errmsg_len)
{
	if (input == NULL || output == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: input and output must be non-null");
		return false;
	}

	if (!tq_validate_state(state, errmsg, errmsg_len))
		return false;

	if (output_len < state->padded_dimension)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant transform: output buffer is too small");
		return false;
	}

	return tq_transform_apply_reference(state, input, output, output_len,
										errmsg, errmsg_len);
}
