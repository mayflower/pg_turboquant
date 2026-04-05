#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "src/tq_am_routine.h"
#include "src/tq_codec_mse.h"
#include "src/tq_codec_prod.h"
#include "src/tq_options.h"
#include "src/tq_page.h"
#include "src/tq_pgvector_compat.h"
#include "src/tq_query_tuning.h"
#include "src/tq_router.h"
#include "src/tq_scan.h"
#include "src/tq_simd_avx2.h"
#include "src/tq_transform.h"
#include "third_party/pgvector/src/halfutils.h"
#include "third_party/pgvector/src/halfvec.h"
#include "third_party/pgvector/src/vector.h"
#include "utils/uuid.h"

static void
normalize(float *values, size_t len)
{
	float norm = 0.0f;
	size_t i = 0;

	for (i = 0; i < len; i++)
		norm += values[i] * values[i];

	norm = sqrtf(norm);
	assert(norm > 0.0f);

	for (i = 0; i < len; i++)
		values[i] /= norm;
}

static void
seeded_unit_vector(uint32_t seed, float *values, size_t len)
{
	size_t i = 0;

	for (i = 0; i < len; i++)
	{
		uint32_t mixed = seed * 1664525u + 1013904223u + (uint32_t) i * 2654435761u;
		values[i] = ((float) (mixed % 2001u) / 1000.0f) - 1.0f;
	}
	normalize(values, len);
}

static void
test_tq_init_amroutine_flags(void)
{
	IndexAmRoutine amroutine;

	memset(&amroutine, 0xFF, sizeof(amroutine));
	tq_init_amroutine(&amroutine);

	assert(amroutine.amcanorder == false);
	assert(amroutine.amcanorderbyop == true);
	assert(amroutine.amcanbackward == false);
	assert(amroutine.amcanunique == false);
	assert(amroutine.amcanmulticol == true);
	assert(amroutine.amoptionalkey == true);
	assert(amroutine.amsearcharray == true);
	assert(amroutine.amsearchnulls == true);
	assert(amroutine.amclusterable == false);
	assert(amroutine.amcaninclude == true);
	assert(amroutine.amsummarizing == false);
	assert(amroutine.amcanreturn == NULL);
	assert(amroutine.amstrategies == 1);
	assert(amroutine.amgetbitmap == NULL);
}

static void
test_tq_validate_option_config_valid(void)
{
	TqOptionConfig config = {
		.bits = 4,
		.lists = 0,
		.normalized = true,
		.transform_name = "hadamard",
		.lanes_name = "auto"
	};
	char		errmsg[256];

	assert(tq_validate_option_config(&config, errmsg, sizeof(errmsg)));
}

static void
test_tq_validate_option_config_invalid_bits(void)
{
	TqOptionConfig config = {
		.bits = 1,
		.lists = 0,
		.normalized = true,
		.transform_name = "hadamard",
		.lanes_name = "auto"
	};
	char		errmsg[256];

	assert(!tq_validate_option_config(&config, errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid value for parameter \"bits\": turboquant bits must be between 2 and 8") == 0);
}

static void
test_tq_validate_option_config_invalid_lists(void)
{
	TqOptionConfig config = {
		.bits = 4,
		.lists = -1,
		.normalized = false,
		.transform_name = "hadamard",
		.lanes_name = "auto"
	};
	char		errmsg[256];

	assert(!tq_validate_option_config(&config, errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid value for parameter \"lists\": turboquant lists must be greater than or equal to 0") == 0);
}

static void
test_tq_validate_option_config_invalid_transform(void)
{
	TqOptionConfig config = {
		.bits = 4,
		.lists = 32,
		.normalized = false,
		.transform_name = "dense",
		.lanes_name = "auto"
	};
	char		errmsg[256];

	assert(!tq_validate_option_config(&config, errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid value for parameter \"transform\": turboquant transform must be \"hadamard\" in v1") == 0);
}

static void
test_tq_validate_option_config_invalid_lanes(void)
{
	TqOptionConfig config = {
		.bits = 4,
		.lists = 0,
		.normalized = true,
		.transform_name = "hadamard",
		.lanes_name = "8"
	};
	char		errmsg[256];

	assert(!tq_validate_option_config(&config, errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid value for parameter \"lanes\": turboquant lanes must be set to auto in v1") == 0);
}

static TqLaneConfig
default_lane_config(int dimension)
{
	TqLaneConfig config;

	memset(&config, 0, sizeof(config));
	config.block_size = TQ_DEFAULT_BLOCK_SIZE;
	config.dimension = dimension;
	config.bits = 4;
	config.codec = TQ_CODEC_PROD;
	config.normalized = true;
	config.page_header_bytes = TQ_PAGE_HEADER_BYTES;
	config.special_space_bytes = TQ_PAGE_SPECIAL_BYTES;
	config.reserve_bytes = TQ_PAGE_RESERVED_BYTES;
	config.tid_bytes = TQ_TID_BYTES;

	return config;
}

static void
test_tq_compute_code_bytes_prod_default(void)
{
	TqLaneConfig config = default_lane_config(1536);
	size_t		code_bytes = 0;
	char		errmsg[256];

	assert(tq_compute_code_bytes(&config, &code_bytes, errmsg, sizeof(errmsg)));
	assert(code_bytes == 778);
}

static void
test_tq_resolve_lane_count_default_prod(void)
{
	TqLaneConfig config = default_lane_config(1536);
	int			lane_count = 0;
	char		errmsg[256];

	assert(tq_resolve_lane_count(&config, &lane_count, errmsg, sizeof(errmsg)));
	assert(lane_count == 8);
}

static void
test_tq_resolve_lane_count_small_dimensions(void)
{
	TqLaneConfig config_256 = default_lane_config(256);
	TqLaneConfig config_768 = default_lane_config(768);
	TqLaneConfig config_1024 = default_lane_config(1024);
	int			lane_count = 0;
	char		errmsg[256];

	assert(tq_resolve_lane_count(&config_256, &lane_count, errmsg, sizeof(errmsg)));
	assert(lane_count == 16);

	assert(tq_resolve_lane_count(&config_768, &lane_count, errmsg, sizeof(errmsg)));
	assert(lane_count == 16);

	assert(tq_resolve_lane_count(&config_1024, &lane_count, errmsg, sizeof(errmsg)));
	assert(lane_count == 8);
}

static void
test_tq_resolve_lane_count_impossible(void)
{
	TqLaneConfig config = default_lane_config(20000);
	int			lane_count = 0;
	char		errmsg[256];

	assert(!tq_resolve_lane_count(&config, &lane_count, errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid page budget: one turboquant code does not fit on a page with the current settings") == 0);
}

static float
sum_squared(const float *values, size_t len)
{
	float		sum = 0.0f;
	size_t		i = 0;

	for (i = 0; i < len; i++)
		sum += values[i] * values[i];

	return sum;
}

static float
sum_abs_diff(const float *left, const float *right, size_t len)
{
	float		sum = 0.0f;
	size_t		i = 0;

	for (i = 0; i < len; i++)
		sum += fabsf(left[i] - right[i]);

	return sum;
}

static void
assert_float_close(float actual, float expected, float tolerance)
{
	assert(fabsf(actual - expected) <= tolerance);
}

static void
assert_float_array_close(const float *actual,
						   const float *expected,
						   size_t len,
						   float tolerance)
{
	size_t		i = 0;

	for (i = 0; i < len; i++)
		assert_float_close(actual[i], expected[i], tolerance);
}

static TqTransformConfig
default_transform_config(uint32_t dimension, uint64_t seed)
{
	TqTransformConfig config;

	memset(&config, 0, sizeof(config));
	config.kind = TQ_TRANSFORM_HADAMARD;
	config.dimension = dimension;
	config.seed = seed;

	return config;
}

static TqTransformMetadata
default_transform_metadata(uint32_t dimension, uint64_t seed)
{
	TqTransformMetadata metadata;
	TqTransformConfig config = default_transform_config(dimension, seed);
	char		errmsg[256];

	memset(&metadata, 0, sizeof(metadata));
	assert(tq_transform_metadata_init(&config, &metadata, errmsg, sizeof(errmsg)));
	return metadata;
}

static TqMseCodecConfig
default_mse_config(uint32_t dimension, uint8_t bits)
{
	TqMseCodecConfig config;

	memset(&config, 0, sizeof(config));
	config.dimension = dimension;
	config.bits = bits;
	config.min_value = -1.0f;
	config.max_value = 1.0f;

	return config;
}

static TqProdCodecConfig
default_prod_config(uint32_t dimension, uint8_t bits)
{
	TqProdCodecConfig config;

	memset(&config, 0, sizeof(config));
	config.dimension = dimension;
	config.bits = bits;
	config.qjl_seed = UINT64_C(0x13579BDF2468ACE0);

	return config;
}

static float
dot_product(const float *left, const float *right, size_t len)
{
	float		sum = 0.0f;
	size_t		i = 0;

	for (i = 0; i < len; i++)
		sum += left[i] * right[i];

	return sum;
}

static Vector *
alloc_test_vector(int dim, const float *values)
{
	Vector	   *vector = (Vector *) malloc(VECTOR_SIZE(dim));
	int			i = 0;

	assert(vector != NULL);
	memset(vector, 0, VECTOR_SIZE(dim));
	vector->vl_len_ = VECTOR_SIZE(dim);
	vector->dim = dim;
	vector->unused = 0;

	for (i = 0; i < dim; i++)
		vector->x[i] = values[i];

	return vector;
}

static HalfVector *
alloc_test_halfvec(int dim, const float *values)
{
	HalfVector *vector = (HalfVector *) malloc(HALFVEC_SIZE(dim));
	int			i = 0;

	assert(vector != NULL);
	memset(vector, 0, HALFVEC_SIZE(dim));
	vector->vl_len_ = HALFVEC_SIZE(dim);
	vector->dim = dim;
	vector->unused = 0;

	for (i = 0; i < dim; i++)
		vector->x[i] = Float4ToHalfUnchecked(values[i]);

	return vector;
}

static uint32_t
lcg_next(uint32_t *state)
{
	*state = (*state * UINT32_C(1664525)) + UINT32_C(1013904223);
	return *state;
}

static float
lcg_float_signed(uint32_t *state)
{
	return ((float) (lcg_next(state) & UINT32_C(0xFFFF)) / 32767.5f) - 1.0f;
}

static void
test_tq_transform_prepare_and_dimension_padding(void)
{
	TqTransformConfig config = default_transform_config(5, UINT64_C(7));
	TqTransformState state;
	char		errmsg[256];

	memset(&state, 0, sizeof(state));

	assert(tq_transform_padded_dimension(5) == 8);
	assert(tq_transform_prepare(&config, &state, errmsg, sizeof(errmsg)));
	assert(state.dimension == 5);
	assert(state.padded_dimension == 8);
	assert(state.sign_count == 8);
	assert(state.permutation_count == 8);
}

static void
test_tq_transform_same_seed_same_input_same_output(void)
{
	TqTransformConfig config = default_transform_config(5, UINT64_C(0xABCDEF));
	TqTransformState state;
	const float	input[5] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
	float		left[8];
	float		right[8];
	char		errmsg[256];

	memset(&state, 0, sizeof(state));
	memset(left, 0, sizeof(left));
	memset(right, 0, sizeof(right));

	assert(tq_transform_prepare(&config, &state, errmsg, sizeof(errmsg)));
	assert(tq_transform_apply(&state, input, left, 8, errmsg, sizeof(errmsg)));
	assert(tq_transform_apply(&state, input, right, 8, errmsg, sizeof(errmsg)));
	assert_float_array_close(left, right, 8, 1e-6f);
}

static void
test_tq_transform_different_seed_materially_changes_output(void)
{
	TqTransformConfig config_a = default_transform_config(8, UINT64_C(11));
	TqTransformConfig config_b = default_transform_config(8, UINT64_C(12));
	TqTransformState state_a;
	TqTransformState state_b;
	const float	input[8] = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, 0.5f, 0.25f};
	float		output_a[8];
	float		output_b[8];
	char		errmsg[256];

	memset(&state_a, 0, sizeof(state_a));
	memset(&state_b, 0, sizeof(state_b));
	memset(output_a, 0, sizeof(output_a));
	memset(output_b, 0, sizeof(output_b));

	assert(tq_transform_prepare(&config_a, &state_a, errmsg, sizeof(errmsg)));
	assert(tq_transform_prepare(&config_b, &state_b, errmsg, sizeof(errmsg)));
	assert(tq_transform_apply(&state_a, input, output_a, 8, errmsg, sizeof(errmsg)));
	assert(tq_transform_apply(&state_b, input, output_b, 8, errmsg, sizeof(errmsg)));
	assert(sum_abs_diff(output_a, output_b, 8) > 0.5f);
}

static void
test_tq_transform_metadata_roundtrip_prepare(void)
{
	TqTransformMetadata metadata = default_transform_metadata(5, UINT64_C(99));
	TqTransformState state;
	char		errmsg[256];

	memset(&state, 0, sizeof(state));

	assert(metadata.contract_version == TQ_TRANSFORM_CONTRACT_VERSION);
	assert(metadata.kind == TQ_TRANSFORM_HADAMARD);
	assert(metadata.input_dimension == 5);
	assert(metadata.output_dimension == 8);
	assert(metadata.seed == UINT64_C(99));
	assert(tq_transform_prepare_metadata(&metadata, &state, errmsg, sizeof(errmsg)));
	assert(state.dimension == metadata.input_dimension);
	assert(state.padded_dimension == metadata.output_dimension);
}

static void
test_tq_transform_apply_uses_full_padded_contract(void)
{
	TqTransformConfig config = default_transform_config(5, UINT64_C(99));
	TqTransformState state;
	const float	input[5] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
	float		transformed[8];
	float		reference[8];
	char		errmsg[256];

	memset(&state, 0, sizeof(state));
	memset(transformed, 0, sizeof(transformed));
	memset(reference, 0, sizeof(reference));

	assert(tq_transform_prepare(&config, &state, errmsg, sizeof(errmsg)));
	assert(tq_transform_apply(&state, input, transformed, 8, errmsg, sizeof(errmsg)));
	assert(tq_transform_apply_reference(&state, input, reference, 8, errmsg, sizeof(errmsg)));
	assert_float_array_close(transformed, reference, 8, 1e-6f);
}

static void
test_tq_transform_inverse_roundtrip(void)
{
	TqTransformConfig config = default_transform_config(5, UINT64_C(99));
	TqTransformState state;
	const float	input[5] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
	float		transformed[8];
	float		recovered[5];
	char		errmsg[256];

	memset(&state, 0, sizeof(state));
	memset(transformed, 0, sizeof(transformed));
	memset(recovered, 0, sizeof(recovered));

	assert(tq_transform_prepare(&config, &state, errmsg, sizeof(errmsg)));
	assert(tq_transform_apply(&state, input, transformed, 8, errmsg, sizeof(errmsg)));
	assert(tq_transform_inverse(&state, transformed, 8, recovered, 5, errmsg, sizeof(errmsg)));
	assert_float_array_close(recovered, input, 5, 1e-5f);
}

static void
test_tq_transform_apply_rejects_truncated_output_buffer(void)
{
	TqTransformConfig config = default_transform_config(5, UINT64_C(123));
	TqTransformState state;
	const float	input[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
	float		output[5];
	char		errmsg[256];

	memset(&state, 0, sizeof(state));
	memset(output, 0, sizeof(output));

	assert(tq_transform_prepare(&config, &state, errmsg, sizeof(errmsg)));
	assert(!tq_transform_apply(&state, input, output, 5, errmsg, sizeof(errmsg)));
	assert(strstr(errmsg, "output buffer is too small") != NULL);
}

static void
test_tq_transform_norm_behavior_is_stable(void)
{
	TqTransformConfig config = default_transform_config(5, UINT64_C(1234));
	TqTransformState state;
	const float	input[5] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
	float		transformed[8];
	char		errmsg[256];

	memset(&state, 0, sizeof(state));
	memset(transformed, 0, sizeof(transformed));

	assert(tq_transform_prepare(&config, &state, errmsg, sizeof(errmsg)));
	assert(tq_transform_apply_reference(&state, input, transformed, 8, errmsg, sizeof(errmsg)));
	assert_float_close(sum_squared(transformed, 8), sum_squared(input, 5), 1e-3f);
}

static void
test_tq_transform_zero_vector_stays_zero(void)
{
	TqTransformConfig config = default_transform_config(7, UINT64_C(555));
	TqTransformState state;
	const float	input[7] = {0};
	float		output[8];
	char		errmsg[256];

	memset(&state, 0, sizeof(state));
	memset(output, 0, sizeof(output));

	assert(tq_transform_prepare(&config, &state, errmsg, sizeof(errmsg)));
	assert(tq_transform_apply(&state, input, output, 8, errmsg, sizeof(errmsg)));
	assert_float_array_close(output, input, 7, 1e-6f);
	assert_float_close(output[7], 0.0f, 1e-6f);
}

static void
test_tq_transform_normalized_input_keeps_unit_norm(void)
{
	TqTransformConfig config = default_transform_config(8, UINT64_C(8080));
	TqTransformState state;
	const float	inv_sqrt2 = 0.70710678f;
	const float	input[8] = {inv_sqrt2, -inv_sqrt2, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	float		transformed[8];
	char		errmsg[256];

	memset(&state, 0, sizeof(state));
	memset(transformed, 0, sizeof(transformed));

	assert(tq_transform_prepare(&config, &state, errmsg, sizeof(errmsg)));
	assert(tq_transform_apply_reference(&state, input, transformed, 8, errmsg, sizeof(errmsg)));
	assert_float_close(sum_squared(transformed, 8), 1.0f, 1e-4f);
}

static void
test_tq_mse_packed_length_calculations(void)
{
	TqMseCodecConfig config5 = default_mse_config(5, 4);
	TqMseCodecConfig config7 = default_mse_config(7, 2);
	size_t		packed_bytes = 0;
	char		errmsg[256];

	assert(tq_mse_packed_bytes(&config5, &packed_bytes, errmsg, sizeof(errmsg)));
	assert(packed_bytes == 3);

	assert(tq_mse_packed_bytes(&config7, &packed_bytes, errmsg, sizeof(errmsg)));
	assert(packed_bytes == 2);
}

static void
test_tq_mse_encode_decode_determinism(void)
{
	TqMseCodecConfig config = default_mse_config(5, 4);
	const float	input[5] = {-0.95f, -0.20f, 0.10f, 0.62f, 0.99f};
	uint8_t		left[3];
	uint8_t		right[3];
	float		decoded_left[5];
	float		decoded_right[5];
	char		errmsg[256];

	memset(left, 0, sizeof(left));
	memset(right, 0, sizeof(right));
	memset(decoded_left, 0, sizeof(decoded_left));
	memset(decoded_right, 0, sizeof(decoded_right));

	assert(tq_mse_encode(&config, input, left, sizeof(left), errmsg, sizeof(errmsg)));
	assert(tq_mse_encode(&config, input, right, sizeof(right), errmsg, sizeof(errmsg)));
	assert(memcmp(left, right, sizeof(left)) == 0);

	assert(tq_mse_decode(&config, left, sizeof(left), decoded_left, 5, errmsg, sizeof(errmsg)));
	assert(tq_mse_decode(&config, right, sizeof(right), decoded_right, 5, errmsg, sizeof(errmsg)));
	assert_float_array_close(decoded_left, decoded_right, 5, 1e-6f);
}

static void
test_tq_mse_reconstruction_error_bound(void)
{
	TqMseCodecConfig config = default_mse_config(5, 4);
	const float	input[5] = {-0.95f, -0.20f, 0.10f, 0.62f, 0.99f};
	uint8_t		packed[3];
	float		decoded[5];
	char		errmsg[256];

	memset(packed, 0, sizeof(packed));
	memset(decoded, 0, sizeof(decoded));

	assert(tq_mse_encode(&config, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_mse_decode(&config, packed, sizeof(packed), decoded, 5, errmsg, sizeof(errmsg)));
	assert(sum_abs_diff(input, decoded, 5) < 0.35f);
}

static void
test_tq_mse_invalid_parameter_handling(void)
{
	TqMseCodecConfig invalid_bits = default_mse_config(5, 1);
	TqMseCodecConfig invalid_range = default_mse_config(5, 4);
	TqMseCodecConfig valid = default_mse_config(5, 4);
	const float	input[5] = {0};
	uint8_t		packed[3];
	char		errmsg[256];

	invalid_range.min_value = 1.0f;
	invalid_range.max_value = 1.0f;

	assert(!tq_mse_encode(&invalid_bits, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_mse codec config: bits must be between 2 and 8") == 0);

	assert(!tq_mse_encode(&invalid_range, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_mse codec config: max_value must be greater than min_value") == 0);

	assert(!tq_mse_encode(&valid, input, packed, 2, errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_mse buffer: packed output buffer is too small") == 0);
}

static void
test_tq_mse_lut_shape_and_known_values(void)
{
	TqMseCodecConfig config = default_mse_config(2, 2);
	const float	query[2] = {0.0f, 0.5f};
	TqMseLut	lut;
	float		expected[8] = {
		1.0f, 0.08376885f, 0.08376885f, 1.0f,
		2.25f, 0.62319732f, 0.04434036f, 0.25f
	};
	char		errmsg[256];

	memset(&lut, 0, sizeof(lut));

	assert(tq_mse_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(lut.dimension == 2);
	assert(lut.level_count == 4);
	assert(lut.values != NULL);
	assert_float_array_close(lut.values, expected, 8, 1e-5f);

	tq_mse_lut_reset(&lut);
}

static void
test_tq_mse_scalar_scoring_tiny_corpus(void)
{
	TqMseCodecConfig config = default_mse_config(2, 2);
	const float	query[2] = {0.0f, 0.5f};
	const float	corpus0[2] = {0.0f, 0.5f};
	const float	corpus1[2] = {-1.0f, 1.0f};
	const float	corpus2[2] = {1.0f, -1.0f};
	uint8_t		packed0[1];
	uint8_t		packed1[1];
	uint8_t		packed2[1];
	TqMseLut	lut;
	float		score0 = 0.0f;
	float		score1 = 0.0f;
	float		score2 = 0.0f;
	char		errmsg[256];

	memset(packed0, 0, sizeof(packed0));
	memset(packed1, 0, sizeof(packed1));
	memset(packed2, 0, sizeof(packed2));
	memset(&lut, 0, sizeof(lut));

	assert(tq_mse_encode(&config, corpus0, packed0, sizeof(packed0), errmsg, sizeof(errmsg)));
	assert(tq_mse_encode(&config, corpus1, packed1, sizeof(packed1), errmsg, sizeof(errmsg)));
	assert(tq_mse_encode(&config, corpus2, packed2, sizeof(packed2), errmsg, sizeof(errmsg)));
	assert(tq_mse_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));

	assert(tq_mse_score_packed_l2(&config, &lut, packed0, sizeof(packed0), &score0, errmsg, sizeof(errmsg)));
	assert(tq_mse_score_packed_l2(&config, &lut, packed1, sizeof(packed1), &score1, errmsg, sizeof(errmsg)));
	assert(tq_mse_score_packed_l2(&config, &lut, packed2, sizeof(packed2), &score2, errmsg, sizeof(errmsg)));

	assert(score0 < score1);
	assert(score0 < score2);
	assert(score1 < score2);

	tq_mse_lut_reset(&lut);
}

typedef enum TqMseTestDistribution
{
	TQ_MSE_TEST_NORMALIZED = 0,
	TQ_MSE_TEST_HEAVY_TAIL = 1,
	TQ_MSE_TEST_SKEWED = 2
} TqMseTestDistribution;

static float
clamp_float(float value, float lower, float upper)
{
	if (value < lower)
		return lower;
	if (value > upper)
		return upper;
	return value;
}

static float
lcg_float_unit(uint32_t *state)
{
	return (float) (lcg_next(state) & UINT32_C(0x00FFFFFF)) / 16777215.0f;
}

static float
lcg_gaussian(uint32_t *state, float stddev)
{
	const float	two_pi = 6.28318530718f;
	float		u1 = lcg_float_unit(state);
	float		u2 = lcg_float_unit(state);
	float		radius = 0.0f;

	if (u1 < 1e-6f)
		u1 = 1e-6f;

	radius = sqrtf(-2.0f * logf(u1));
	return cosf(two_pi * u2) * radius * stddev;
}

static void
fill_tq_mse_test_vector(TqMseTestDistribution distribution,
						  uint32_t dimension,
						  uint32_t *state,
						  float *output)
{
	uint32_t	dim = 0;

	for (dim = 0; dim < dimension; dim++)
	{
		float		value = 0.0f;

		if (distribution == TQ_MSE_TEST_NORMALIZED)
			value = lcg_gaussian(state, 0.35f);
		else if (distribution == TQ_MSE_TEST_HEAVY_TAIL)
		{
			if ((lcg_next(state) & UINT32_C(15)) == 0)
				value = lcg_gaussian(state, 0.85f);
			else
				value = lcg_gaussian(state, 0.18f);
		}
		else
		{
			float		u = lcg_float_unit(state);
			float		base = ((float) (dim % 5) * 0.08f) - 0.18f;

			value = base + (u * u * u * 0.72f);
		}

		output[dim] = clamp_float(value, -1.0f, 1.0f);
	}

	if (distribution == TQ_MSE_TEST_NORMALIZED)
	{
		float		norm = sqrtf(sum_squared(output, dimension));

		if (norm > 0.0f)
		{
			for (dim = 0; dim < dimension; dim++)
				output[dim] /= norm;
		}
	}
}

static float
legacy_tq_mse_decode_value(const TqMseCodecConfig *config, float value)
{
	uint32_t	level_count = UINT32_C(1) << config->bits;
	float		step = (config->max_value - config->min_value) / (float) (level_count - 1);
	float		position = 0.0f;
	long		code = 0;

	if (value <= config->min_value)
		return config->min_value;

	if (value >= config->max_value)
		return config->max_value;

	position = (value - config->min_value) / step;
	code = lroundf(position);
	if (code < 0)
		code = 0;
	if ((uint32_t) code >= level_count)
		code = (long) level_count - 1;

	return config->min_value + ((float) code * step);
}

static float
tq_mse_average_reconstruction_error(const TqMseCodecConfig *config,
									 TqMseTestDistribution distribution,
									 bool use_legacy_codec)
{
	enum { SAMPLE_COUNT = 96 };
	uint32_t	state = UINT32_C(0xC0DEC0DE) + (uint32_t) distribution;
	float		input[32];
	float		decoded[32];
	uint8_t		packed[32];
	float		total_error = 0.0f;
	char		errmsg[256];
	int			sample = 0;

	assert(config->dimension <= 32);
	memset(packed, 0, sizeof(packed));

	for (sample = 0; sample < SAMPLE_COUNT; sample++)
	{
		uint32_t	dim = 0;

		fill_tq_mse_test_vector(distribution, config->dimension, &state, input);
		memset(decoded, 0, sizeof(decoded));

		if (use_legacy_codec)
		{
			for (dim = 0; dim < config->dimension; dim++)
				decoded[dim] = legacy_tq_mse_decode_value(config, input[dim]);
		}
		else
		{
			assert(tq_mse_encode(config, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));
			assert(tq_mse_decode(config, packed, sizeof(packed), decoded, config->dimension, errmsg, sizeof(errmsg)));
		}

		for (dim = 0; dim < config->dimension; dim++)
		{
			float		diff = input[dim] - decoded[dim];

			total_error += diff * diff;
		}
	}

	return total_error / (float) (SAMPLE_COUNT * (int) config->dimension);
}

static void
insert_ranked_score(float score, int id, float *scores, int *ids, int limit)
{
	int			pos = 0;

	for (pos = 0; pos < limit; pos++)
	{
		if (score < scores[pos] || (fabsf(score - scores[pos]) <= 1e-6f && id < ids[pos]))
		{
			int			shift = 0;

			for (shift = limit - 1; shift > pos; shift--)
			{
				scores[shift] = scores[shift - 1];
				ids[shift] = ids[shift - 1];
			}
			scores[pos] = score;
			ids[pos] = id;
			return;
		}
	}
}

static float
tq_mse_average_recall_at_k(const TqMseCodecConfig *config,
							 TqMseTestDistribution distribution,
							 bool use_legacy_codec)
{
	enum { ROW_COUNT = 160, QUERY_COUNT = 20, TOP_K = 10 };
	uint32_t	row_state = UINT32_C(0x13572468) + (uint32_t) distribution;
	uint32_t	query_state = UINT32_C(0x24681357) + (uint32_t) distribution;
	float		corpus[ROW_COUNT][16];
	float		query[16];
	float		exact_scores[TOP_K];
	float		approx_scores[TOP_K];
	int			exact_ids[TOP_K];
	int			approx_ids[TOP_K];
	uint8_t		packed[ROW_COUNT][16];
	TqMseLut	lut;
	float		total_recall = 0.0f;
	char		errmsg[256];
	int			row = 0;
	int			query_index = 0;

	assert(config->dimension <= 16);
	memset(&lut, 0, sizeof(lut));

	for (row = 0; row < ROW_COUNT; row++)
	{
		fill_tq_mse_test_vector(distribution, config->dimension, &row_state, corpus[row]);
		if (!use_legacy_codec)
			assert(tq_mse_encode(config, corpus[row], packed[row], sizeof(packed[row]), errmsg, sizeof(errmsg)));
	}

	for (query_index = 0; query_index < QUERY_COUNT; query_index++)
	{
		int			top_index = 0;

		fill_tq_mse_test_vector(distribution, config->dimension, &query_state, query);
		for (top_index = 0; top_index < TOP_K; top_index++)
		{
			exact_scores[top_index] = INFINITY;
			approx_scores[top_index] = INFINITY;
			exact_ids[top_index] = INT32_MAX;
			approx_ids[top_index] = INT32_MAX;
		}

		assert(tq_mse_lut_build(config, query, &lut, errmsg, sizeof(errmsg)));

		for (row = 0; row < ROW_COUNT; row++)
		{
			float		exact_score = 0.0f;
			float		approx_score = 0.0f;
			uint32_t	dim = 0;

			for (dim = 0; dim < config->dimension; dim++)
			{
				float		diff = query[dim] - corpus[row][dim];

				exact_score += diff * diff;
			}

			if (use_legacy_codec)
			{
				for (dim = 0; dim < config->dimension; dim++)
				{
					float		diff = query[dim] - legacy_tq_mse_decode_value(config, corpus[row][dim]);

					approx_score += diff * diff;
				}
			}
			else
			{
				assert(tq_mse_score_packed_l2(config, &lut, packed[row], sizeof(packed[row]), &approx_score, errmsg, sizeof(errmsg)));
			}

			insert_ranked_score(exact_score, row, exact_scores, exact_ids, TOP_K);
			insert_ranked_score(approx_score, row, approx_scores, approx_ids, TOP_K);
		}

		for (top_index = 0; top_index < TOP_K; top_index++)
		{
			int			match_index = 0;

			for (match_index = 0; match_index < TOP_K; match_index++)
			{
				if (exact_ids[top_index] == approx_ids[match_index])
				{
					total_recall += 1.0f;
					break;
				}
			}
		}

		tq_mse_lut_reset(&lut);
	}

	return total_recall / (float) (QUERY_COUNT * TOP_K);
}

static void
test_tq_mse_codebook_reduces_reconstruction_error_on_representative_distributions(void)
{
	TqMseCodecConfig config = default_mse_config(16, 4);
	float		legacy_normalized = tq_mse_average_reconstruction_error(&config, TQ_MSE_TEST_NORMALIZED, true);
	float		legacy_heavy_tail = tq_mse_average_reconstruction_error(&config, TQ_MSE_TEST_HEAVY_TAIL, true);
	float		legacy_skewed = tq_mse_average_reconstruction_error(&config, TQ_MSE_TEST_SKEWED, true);
	float		current_normalized = tq_mse_average_reconstruction_error(&config, TQ_MSE_TEST_NORMALIZED, false);
	float		current_heavy_tail = tq_mse_average_reconstruction_error(&config, TQ_MSE_TEST_HEAVY_TAIL, false);
	float		current_skewed = tq_mse_average_reconstruction_error(&config, TQ_MSE_TEST_SKEWED, false);

	assert(current_normalized < legacy_normalized * 0.95f);
	assert(current_heavy_tail < legacy_heavy_tail * 0.90f);
	assert(current_skewed < legacy_skewed * 0.95f);
}

static void
test_tq_mse_codebook_improves_or_preserves_ann_recall(void)
{
	TqMseCodecConfig config = default_mse_config(16, 4);
	float		legacy_normalized = tq_mse_average_recall_at_k(&config, TQ_MSE_TEST_NORMALIZED, true);
	float		legacy_heavy_tail = tq_mse_average_recall_at_k(&config, TQ_MSE_TEST_HEAVY_TAIL, true);
	float		legacy_skewed = tq_mse_average_recall_at_k(&config, TQ_MSE_TEST_SKEWED, true);
	float		current_normalized = tq_mse_average_recall_at_k(&config, TQ_MSE_TEST_NORMALIZED, false);
	float		current_heavy_tail = tq_mse_average_recall_at_k(&config, TQ_MSE_TEST_HEAVY_TAIL, false);
	float		current_skewed = tq_mse_average_recall_at_k(&config, TQ_MSE_TEST_SKEWED, false);

	assert(current_normalized >= legacy_normalized + 0.02f);
	assert(current_heavy_tail >= legacy_heavy_tail + 0.01f);
	assert(current_skewed >= legacy_skewed);
}

static void
test_tq_prod_packed_length_calculations(void)
{
	TqProdCodecConfig config5 = default_prod_config(5, 4);
	TqProdCodecConfig config17 = default_prod_config(17, 4);
	TqProdPackedLayout layout;
	char		errmsg[256];

	memset(&layout, 0, sizeof(layout));

	config5.qjl_dimension = 5;
	config17.qjl_dimension = 9;

	assert(tq_prod_packed_layout(&config5, &layout, errmsg, sizeof(errmsg)));
	assert(layout.idx_bytes == 2);
	assert(layout.qjl_bytes == 1);
	assert(layout.gamma_bytes == 4);
	assert(layout.total_bytes == 7);

	assert(tq_prod_packed_layout(&config17, &layout, errmsg, sizeof(errmsg)));
	assert(layout.idx_bytes == 7);
	assert(layout.qjl_bytes == 2);
	assert(layout.gamma_bytes == 4);
	assert(layout.total_bytes == 13);
}

static void
test_tq_prod_encode_decode_determinism(void)
{
	TqProdCodecConfig config = default_prod_config(5, 4);
	const float	input[5] = {-0.95f, -0.20f, 0.10f, 0.62f, 0.99f};
	uint8_t		left[7];
	uint8_t		right[7];
	float		decoded_left[5];
	float		decoded_right[5];
	float		gamma_left = 0.0f;
	float		gamma_right = 0.0f;
	char		errmsg[256];

	memset(left, 0, sizeof(left));
	memset(right, 0, sizeof(right));
	memset(decoded_left, 0, sizeof(decoded_left));
	memset(decoded_right, 0, sizeof(decoded_right));

	assert(tq_prod_encode(&config, input, left, sizeof(left), errmsg, sizeof(errmsg)));
	assert(tq_prod_encode(&config, input, right, sizeof(right), errmsg, sizeof(errmsg)));
	assert(memcmp(left, right, sizeof(left)) == 0);

	assert(tq_prod_read_gamma(&config, left, sizeof(left), &gamma_left, errmsg, sizeof(errmsg)));
	assert(tq_prod_read_gamma(&config, right, sizeof(right), &gamma_right, errmsg, sizeof(errmsg)));
	assert(gamma_left > 0.0f);
	assert_float_close(gamma_left, gamma_right, 1e-6f);

	assert(tq_prod_decode(&config, left, sizeof(left), decoded_left, 5, errmsg, sizeof(errmsg)));
	assert(tq_prod_decode(&config, right, sizeof(right), decoded_right, 5, errmsg, sizeof(errmsg)));
	assert_float_array_close(decoded_left, decoded_right, 5, 1e-6f);
}

static void
test_tq_prod_invalid_parameter_handling(void)
{
	TqProdCodecConfig invalid_dim = default_prod_config(0, 4);
	TqProdCodecConfig invalid_bits = default_prod_config(5, 1);
	TqProdCodecConfig valid = default_prod_config(5, 4);
	const float	input[5] = {0};
	uint8_t		packed[7];
	char		errmsg[256];

	assert(!tq_prod_encode(&invalid_dim, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_prod codec config: dimension must be positive") == 0);

	assert(!tq_prod_encode(&invalid_bits, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_prod codec config: bits must be between 2 and 8") == 0);

	assert(!tq_prod_encode(&valid, input, packed, 2, errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_prod buffer: packed output buffer is too small") == 0);
}

static void
test_tq_prod_score_decomposition(void)
{
	TqProdCodecConfig config = default_prod_config(2, 4);
	const float	input[2] = {0.5f, -1.0f};
	const float	query[2] = {0.25f, -0.5f};
	uint8_t		packed[6];
	TqProdLut	lut;
	float		decoded[2];
	float		mse_contribution = 0.0f;
	float		qjl_contribution = 0.0f;
	float		combined = 0.0f;
	float		helper_score = 0.0f;
	char		errmsg[256];

	memset(packed, 0, sizeof(packed));
	memset(&lut, 0, sizeof(lut));
	memset(decoded, 0, sizeof(decoded));

	assert(tq_prod_encode(&config, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(lut.dimension == 2);
	assert(lut.level_count == 8);
	assert(lut.qjl_values != NULL);

	assert(tq_prod_score_decompose_ip(&config, &lut, packed, sizeof(packed),
									  &mse_contribution, &qjl_contribution, &combined,
									  errmsg, sizeof(errmsg)));
	assert(tq_prod_decode(&config, packed, sizeof(packed), decoded, 2, errmsg, sizeof(errmsg)));
	helper_score = dot_product(query, decoded, 2);
	assert_float_close(combined, mse_contribution + qjl_contribution, 1e-6f);
	assert_float_close(combined, helper_score, 1e-6f);
	assert(fabsf(qjl_contribution) > 1e-4f);

	tq_prod_lut_reset(&lut);
}

static void
test_tq_prod_scalar_score_matches_decode_helper(void)
{
	TqProdCodecConfig config = default_prod_config(5, 4);
	const float	input[5] = {-0.95f, -0.20f, 0.10f, 0.62f, 0.99f};
	const float	query[5] = {0.50f, -0.25f, 0.75f, -0.10f, 0.20f};
	uint8_t		packed[7];
	float		decoded[5];
	TqProdLut	lut;
	float		packed_score = 0.0f;
	float		helper_score = 0.0f;
	char		errmsg[256];

	memset(packed, 0, sizeof(packed));
	memset(decoded, 0, sizeof(decoded));
	memset(&lut, 0, sizeof(lut));

	assert(tq_prod_encode(&config, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_prod_decode(&config, packed, sizeof(packed), decoded, 5, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_prod_score_packed_ip(&config, &lut, packed, sizeof(packed), &packed_score, errmsg, sizeof(errmsg)));

	helper_score = dot_product(query, decoded, 5);
	assert_float_close(packed_score, helper_score, 1e-5f);

	tq_prod_lut_reset(&lut);
}

static void
test_tq_prod_stability_seeded_random_corpus(void)
{
	TqProdCodecConfig config = default_prod_config(8, 4);
	float		query[8];
	uint32_t	rng_state = UINT32_C(123456789);
	float		signed_error_sum = 0.0f;
	float		abs_error_sum = 0.0f;
	size_t		vec = 0;
	char		errmsg[256];

	for (vec = 0; vec < 8; vec++)
		query[vec] = lcg_float_signed(&rng_state);

	for (vec = 0; vec < 16; vec++)
	{
		float		input[8];
		uint8_t		packed[8];
		float		decoded[8];
		TqProdLut	lut;
		float		exact_score = 0.0f;
		float		approx_score = 0.0f;
		size_t		dim = 0;

		memset(packed, 0, sizeof(packed));
		memset(decoded, 0, sizeof(decoded));
		memset(&lut, 0, sizeof(lut));

		for (dim = 0; dim < 8; dim++)
			input[dim] = lcg_float_signed(&rng_state);

		assert(tq_prod_encode(&config, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));
		assert(tq_prod_decode(&config, packed, sizeof(packed), decoded, 8, errmsg, sizeof(errmsg)));
		assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_packed_ip(&config, &lut, packed, sizeof(packed), &approx_score, errmsg, sizeof(errmsg)));

		exact_score = dot_product(query, input, 8);
		signed_error_sum += (approx_score - exact_score);
		abs_error_sum += fabsf(approx_score - exact_score);

		tq_prod_lut_reset(&lut);
	}

	assert(fabsf(signed_error_sum / 16.0f) < 0.08f);
	assert((abs_error_sum / 16.0f) < 0.22f);
}

typedef enum TqProdTestDistribution
{
	TQ_PROD_TEST_NORMALIZED = 0,
	TQ_PROD_TEST_VARIED_NORMS = 1,
	TQ_PROD_TEST_HEAVY_TAIL = 2
} TqProdTestDistribution;

static void
fill_tq_prod_test_vector(TqProdTestDistribution distribution,
						 uint32_t dimension,
						 uint32_t *state,
						 float *output)
{
	uint32_t	dim = 0;

	for (dim = 0; dim < dimension; dim++)
	{
		float		value = 0.0f;

		if (distribution == TQ_PROD_TEST_NORMALIZED)
			value = lcg_gaussian(state, 0.35f);
		else if (distribution == TQ_PROD_TEST_VARIED_NORMS)
		{
			float		scale = 0.25f + (lcg_float_unit(state) * 3.25f);

			value = lcg_float_signed(state) * scale;
		}
		else
		{
			if ((lcg_next(state) & UINT32_C(15)) == 0)
				value = lcg_gaussian(state, 0.85f);
			else
				value = lcg_gaussian(state, 0.18f);
		}

		output[dim] = clamp_float(value, -3.0f, 3.0f);
	}

	if (distribution == TQ_PROD_TEST_NORMALIZED)
	{
		float		norm = sqrtf(sum_squared(output, dimension));

		if (norm > 0.0f)
		{
			for (dim = 0; dim < dimension; dim++)
				output[dim] /= norm;
		}
	}
}

static float
legacy_tq_prod_max_abs(const float *input, uint32_t dimension)
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
legacy_tq_prod_decode_value(const TqProdCodecConfig *config, float gamma, float value)
{
	uint32_t	max_code = (UINT32_C(1) << (config->bits - 1)) - 1;
	float		magnitude = fabsf(value);
	long		code = 0;
	float		decoded_magnitude = 0.0f;

	if (gamma <= 0.0f || max_code == 0)
		return 0.0f;

	if (magnitude >= gamma)
		code = (long) max_code;
	else
		code = lroundf((magnitude / gamma) * (float) max_code);

	if (code < 0)
		code = 0;
	if ((uint32_t) code > max_code)
		code = (long) max_code;

	decoded_magnitude = gamma * ((float) code / (float) max_code);
	return value >= 0.0f ? decoded_magnitude : -decoded_magnitude;
}

static float
legacy_tq_prod_score_ip(const TqProdCodecConfig *config,
						  const float *query,
						  const float *input)
{
	float		gamma = legacy_tq_prod_max_abs(input, config->dimension);
	float		score = 0.0f;
	uint32_t	dim = 0;

	for (dim = 0; dim < config->dimension; dim++)
		score += query[dim] * legacy_tq_prod_decode_value(config, gamma, input[dim]);

	return score;
}

static float
tq_prod_average_signed_ip_error(const TqProdCodecConfig *config,
								TqProdTestDistribution distribution)
{
	enum { SAMPLE_COUNT = 48 };
	uint32_t	input_state = UINT32_C(0x5A7B9C11) + (uint32_t) distribution;
	uint32_t	query_state = UINT32_C(0x5A7B9D11) + (uint32_t) distribution;
	float		input[16];
	float		query[16];
	uint8_t		packed[32];
	TqProdLut	lut;
	float		total_error = 0.0f;
	char		errmsg[256];
	int			sample = 0;

	assert(config->dimension <= 16);
	memset(&lut, 0, sizeof(lut));

	for (sample = 0; sample < SAMPLE_COUNT; sample++)
	{
		float		exact_score = 0.0f;
		float		approx_score = 0.0f;

		fill_tq_prod_test_vector(distribution, config->dimension, &input_state, input);
		fill_tq_prod_test_vector(distribution, config->dimension, &query_state, query);
		exact_score = dot_product(query, input, config->dimension);

		memset(packed, 0, sizeof(packed));
		assert(tq_prod_encode(config, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));
		assert(tq_prod_lut_build(config, query, &lut, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_packed_ip(config, &lut, packed, sizeof(packed), &approx_score, errmsg, sizeof(errmsg)));
		tq_prod_lut_reset(&lut);

		total_error += (approx_score - exact_score);
	}

	return total_error / (float) SAMPLE_COUNT;
}

static float
tq_prod_average_recall_at_k(const TqProdCodecConfig *config,
							TqProdTestDistribution distribution,
							bool use_legacy_codec)
{
	enum { ROW_COUNT = 160, QUERY_COUNT = 20, TOP_K = 10 };
	uint32_t	row_state = UINT32_C(0x74185296) + (uint32_t) distribution;
	uint32_t	query_state = UINT32_C(0x15926374) + (uint32_t) distribution;
	float		corpus[ROW_COUNT][16];
	float		query[16];
	float		exact_scores[TOP_K];
	float		approx_scores[TOP_K];
	int			exact_ids[TOP_K];
	int			approx_ids[TOP_K];
	uint8_t		packed[ROW_COUNT][24];
	TqProdLut	lut;
	float		total_recall = 0.0f;
	char		errmsg[256];
	int			row = 0;
	int			query_index = 0;

	assert(config->dimension <= 16);
	memset(&lut, 0, sizeof(lut));

	for (row = 0; row < ROW_COUNT; row++)
	{
		fill_tq_prod_test_vector(distribution, config->dimension, &row_state, corpus[row]);
		if (!use_legacy_codec)
			assert(tq_prod_encode(config, corpus[row], packed[row], sizeof(packed[row]), errmsg, sizeof(errmsg)));
	}

	for (query_index = 0; query_index < QUERY_COUNT; query_index++)
	{
		int			top_index = 0;

		fill_tq_prod_test_vector(distribution, config->dimension, &query_state, query);
		for (top_index = 0; top_index < TOP_K; top_index++)
		{
			exact_scores[top_index] = -INFINITY;
			approx_scores[top_index] = -INFINITY;
			exact_ids[top_index] = INT32_MAX;
			approx_ids[top_index] = INT32_MAX;
		}

		assert(tq_prod_lut_build(config, query, &lut, errmsg, sizeof(errmsg)));

		for (row = 0; row < ROW_COUNT; row++)
		{
			float		exact_score = dot_product(query, corpus[row], config->dimension);
			float		approx_score = 0.0f;

			if (use_legacy_codec)
				approx_score = legacy_tq_prod_score_ip(config, query, corpus[row]);
			else
				assert(tq_prod_score_packed_ip(config, &lut, packed[row], sizeof(packed[row]), &approx_score, errmsg, sizeof(errmsg)));

			insert_ranked_score(-exact_score, row, exact_scores, exact_ids, TOP_K);
			insert_ranked_score(-approx_score, row, approx_scores, approx_ids, TOP_K);
		}

		for (top_index = 0; top_index < TOP_K; top_index++)
		{
			int			match_index = 0;

			for (match_index = 0; match_index < TOP_K; match_index++)
			{
				if (exact_ids[top_index] == approx_ids[match_index])
				{
					total_recall += 1.0f;
					break;
				}
			}
		}

		tq_prod_lut_reset(&lut);
	}

	return total_recall / (float) (QUERY_COUNT * TOP_K);
}

static void
test_tq_prod_unbiased_estimator_has_low_signed_error_on_seeded_corpora(void)
{
	TqProdCodecConfig config = default_prod_config(8, 4);
	float		normalized_error = tq_prod_average_signed_ip_error(&config, TQ_PROD_TEST_NORMALIZED);
	float		heavy_tail_error = tq_prod_average_signed_ip_error(&config, TQ_PROD_TEST_HEAVY_TAIL);

	assert(fabsf(normalized_error) < 0.08f);
	assert(fabsf(heavy_tail_error) < 0.08f);
}

static void
test_tq_prod_calibrated_estimator_improves_recall_on_representative_corpora(void)
{
	TqProdCodecConfig config = default_prod_config(16, 4);
	float		legacy_normalized = tq_prod_average_recall_at_k(&config, TQ_PROD_TEST_NORMALIZED, true);
	float		current_normalized = tq_prod_average_recall_at_k(&config, TQ_PROD_TEST_NORMALIZED, false);
	float		legacy_varied = tq_prod_average_recall_at_k(&config, TQ_PROD_TEST_VARIED_NORMS, true);
	float		current_varied = tq_prod_average_recall_at_k(&config, TQ_PROD_TEST_VARIED_NORMS, false);

	assert(current_normalized >= legacy_normalized - 0.01f);
	assert(current_varied >= legacy_varied);
}

static void
test_tq_vector_copy_from_pgvector_struct(void)
{
	const float	values[4] = {1.0f, 0.0f, -0.5f, 0.25f};
	Vector	   *vector = alloc_test_vector(4, values);
	float		out[4];
	uint32_t	dimension = 0;
	char		errmsg[256];

	memset(out, 0, sizeof(out));

	assert(tq_vector_copy_from_pgvector(vector, out, 4, &dimension, errmsg, sizeof(errmsg)));
	assert(dimension == 4);
	assert_float_array_close(out, values, 4, 1e-6f);

	free(vector);
}

static void
test_tq_vector_copy_from_halfvec_struct(void)
{
	const float	values[4] = {1.0f, -0.5f, 0.25f, 0.125f};
	HalfVector *vector = alloc_test_halfvec(4, values);
	float		out[4];
	uint32_t	dimension = 0;
	char		errmsg[256];

	memset(out, 0, sizeof(out));

	assert(tq_vector_copy_from_halfvec(vector, out, 4, &dimension, errmsg, sizeof(errmsg)));
	assert(dimension == 4);
	assert_float_array_close(out, values, 4, 1e-3f);

	free(vector);
}

static void
test_tq_vector_copy_from_halfvec_datum_typed(void)
{
	const float	values[3] = {0.5f, -0.25f, 0.125f};
	HalfVector *vector = alloc_test_halfvec(3, values);
	float		out[3];
	uint32_t	dimension = 0;
	char		errmsg[256];

	memset(out, 0, sizeof(out));

	assert(tq_vector_dimension_from_datum_typed(PointerGetDatum(vector),
												TQ_VECTOR_INPUT_HALFVEC,
												&dimension,
												errmsg,
												sizeof(errmsg)));
	assert(dimension == 3);
	assert(tq_vector_copy_from_datum_typed(PointerGetDatum(vector),
										  TQ_VECTOR_INPUT_HALFVEC,
										  out,
										  3,
										  &dimension,
										  errmsg,
										  sizeof(errmsg)));
	assert(dimension == 3);
	assert_float_array_close(out, values, 3, 1e-3f);

	free(vector);
}

static void
test_tq_vector_copy_struct_validation_messages_are_consistent(void)
{
	const float	vector_values[3] = {1.0f, -0.5f, 0.25f};
	const float	halfvec_values[3] = {0.5f, -0.25f, 0.125f};
	Vector	   *vector = alloc_test_vector(3, vector_values);
	HalfVector *halfvec = alloc_test_halfvec(3, halfvec_values);
	float		out[3];
	uint32_t	dimension = 0;
	char		errmsg[256];

	memset(out, 0, sizeof(out));
	memset(errmsg, 0, sizeof(errmsg));
	assert(!tq_vector_copy_from_pgvector(NULL, out, 3, &dimension, errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_pgvector conversion: vector, output, and dimension must be non-null") == 0);

	memset(errmsg, 0, sizeof(errmsg));
	assert(!tq_vector_copy_from_halfvec(NULL, out, 3, &dimension, errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_pgvector conversion: halfvec, output, and dimension must be non-null") == 0);

	memset(errmsg, 0, sizeof(errmsg));
	assert(!tq_vector_copy_from_pgvector(vector, out, 2, &dimension, errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_pgvector conversion: output buffer is too small") == 0);

	memset(errmsg, 0, sizeof(errmsg));
	assert(!tq_vector_copy_from_halfvec(halfvec, out, 2, &dimension, errmsg, sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_pgvector conversion: output buffer is too small") == 0);

	free(vector);
	free(halfvec);
}

static void
test_tq_vector_typed_validation_messages_are_consistent(void)
{
	uint32_t	dimension = 0;
	uint8_t		raw[64];
	char		errmsg[256];

	memset(raw, 0, sizeof(raw));
	memset(errmsg, 0, sizeof(errmsg));

	assert(!tq_vector_dimension_from_datum_typed(PointerGetDatum(NULL),
												 TQ_VECTOR_INPUT_VECTOR,
												 &dimension,
												 errmsg,
												 sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_pgvector conversion: input value must be non-null with positive dimension") == 0);

	memset(errmsg, 0, sizeof(errmsg));
	assert(!tq_vector_dimension_from_datum_typed(PointerGetDatum(NULL),
												 TQ_VECTOR_INPUT_HALFVEC,
												 &dimension,
												 errmsg,
												 sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_pgvector conversion: input value must be non-null with positive dimension") == 0);

	memset(errmsg, 0, sizeof(errmsg));
	assert(!tq_vector_copy_raw_datum_typed(PointerGetDatum(NULL),
										   TQ_VECTOR_INPUT_VECTOR,
										   raw,
										   sizeof(raw),
										   &dimension,
										   errmsg,
										   sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_pgvector conversion: input value must be non-null with positive dimension") == 0);

	memset(errmsg, 0, sizeof(errmsg));
	assert(!tq_vector_copy_raw_datum_typed(PointerGetDatum(NULL),
										   TQ_VECTOR_INPUT_HALFVEC,
										   raw,
										   sizeof(raw),
										   &dimension,
										   errmsg,
										   sizeof(errmsg)));
	assert(strcmp(errmsg,
				  "invalid tq_pgvector conversion: input value must be non-null with positive dimension") == 0);
}

static void
test_tq_metric_distance_from_ip_score_modes(void)
{
	float		distance = 0.0f;
	char		errmsg[256];

	assert(tq_metric_distance_from_ip_score(TQ_DISTANCE_COSINE, 0.75f,
											&distance, errmsg, sizeof(errmsg)));
	assert_float_close(distance, 0.25f, 1e-6f);

	assert(tq_metric_distance_from_ip_score(TQ_DISTANCE_IP, 0.75f,
											&distance, errmsg, sizeof(errmsg)));
	assert_float_close(distance, -0.75f, 1e-6f);

	assert(tq_metric_distance_from_ip_score(TQ_DISTANCE_L2, 0.75f,
											&distance, errmsg, sizeof(errmsg)));
	assert_float_close(distance, 0.5f, 1e-6f);
}

static void
test_tq_candidate_budget_helper(void)
{
	assert(tq_scan_candidate_capacity(0, 4, 3) == 0);
	assert(tq_scan_candidate_capacity(5, 1, 1) == 1);
	assert(tq_scan_candidate_capacity(5, 2, 3) == 5);
	assert(tq_scan_candidate_capacity(100, 4, 8) == 32);
	assert(tq_scan_candidate_capacity(100, 200, 20) == 100);
}

static void
test_tq_streaming_candidate_budget_helper(void)
{
	assert(tq_streaming_candidate_capacity(0, 0) == 1);
	assert(tq_streaming_candidate_capacity(1, 1) == 1);
	assert(tq_streaming_candidate_capacity(4, 8) == 32);
	assert(tq_streaming_candidate_capacity(200, 20) == 4000);
}

static void
test_tq_planner_cost_helper_prefers_flat_for_small_tables(void)
{
	TqPlannerCostEstimate flat;
	TqPlannerCostEstimate ivf;

	memset(&flat, 0, sizeof(flat));
	memset(&ivf, 0, sizeof(ivf));

	assert(tq_estimate_ordered_scan_cost(4.0, 32.0, 10.0, 1.0, 0, 1, 4,
										 0, 0,
										 0.005, 0.0025, 4.0, 0.01, &flat));
	assert(tq_estimate_ordered_scan_cost(4.0, 32.0, 10.0, 1.0, 4, 1, 4,
										 0, 0,
										 0.005, 0.0025, 4.0, 0.01, &ivf));
	assert_float_close((float) flat.scanned_fraction, 1.0f, 1e-6f);
	assert(ivf.scanned_fraction < flat.scanned_fraction);
	assert(flat.total_cost < ivf.total_cost);
}

static void
test_tq_planner_cost_helper_prefers_ivf_for_large_tables_and_low_probes(void)
{
	TqPlannerCostEstimate flat;
	TqPlannerCostEstimate ivf;

	memset(&flat, 0, sizeof(flat));
	memset(&ivf, 0, sizeof(ivf));

	assert(tq_estimate_ordered_scan_cost(512.0, 4096.0, 100.0, 1.0, 0, 1, 4,
										 0, 0,
										 0.005, 0.0025, 4.0, 0.01, &flat));
	assert(tq_estimate_ordered_scan_cost(512.0, 4096.0, 100.0, 1.0, 16, 1, 4,
										 0, 0,
										 0.005, 0.0025, 4.0, 0.01, &ivf));
	assert(ivf.scanned_fraction < 0.1);
	assert(ivf.total_cost < flat.total_cost);
}

static void
test_tq_planner_cost_helper_high_probes_remove_ivf_advantage(void)
{
	TqPlannerCostEstimate flat;
	TqPlannerCostEstimate ivf;

	memset(&flat, 0, sizeof(flat));
	memset(&ivf, 0, sizeof(ivf));

	assert(tq_estimate_ordered_scan_cost(512.0, 4096.0, 100.0, 1.0, 0, 16, 4,
										 0, 0,
										 0.005, 0.0025, 4.0, 0.01, &flat));
	assert(tq_estimate_ordered_scan_cost(512.0, 4096.0, 100.0, 1.0, 16, 16, 4,
										 0, 0,
										 0.005, 0.0025, 4.0, 0.01, &ivf));
	assert_float_close((float) ivf.scanned_fraction, 1.0f, 1e-6f);
	assert(flat.total_cost < ivf.total_cost);
}

static void
test_tq_planner_cost_helper_visit_budgets_limit_ivf_work(void)
{
	TqPlannerCostEstimate unbounded;
	TqPlannerCostEstimate bounded_codes;
	TqPlannerCostEstimate bounded_pages;

	memset(&unbounded, 0, sizeof(unbounded));
	memset(&bounded_codes, 0, sizeof(bounded_codes));
	memset(&bounded_pages, 0, sizeof(bounded_pages));

	assert(tq_estimate_ordered_scan_cost(512.0, 4096.0, 100.0, 1.0, 16, 16, 4,
										 0, 0,
										 0.005, 0.0025, 4.0, 0.01, &unbounded));
	assert(tq_estimate_ordered_scan_cost(512.0, 4096.0, 100.0, 1.0, 16, 16, 4,
										 256, 0,
										 0.005, 0.0025, 4.0, 0.01, &bounded_codes));
	assert(tq_estimate_ordered_scan_cost(512.0, 4096.0, 100.0, 1.0, 16, 16, 4,
										 0, 32,
										 0.005, 0.0025, 4.0, 0.01, &bounded_pages));

	assert(bounded_codes.effective_probe_count < unbounded.effective_probe_count);
	assert(bounded_codes.visited_tuples < unbounded.visited_tuples);
	assert(bounded_codes.pages_fetched < unbounded.pages_fetched);
	assert(bounded_codes.total_cost < unbounded.total_cost);

	assert(bounded_pages.effective_probe_count < unbounded.effective_probe_count);
	assert(bounded_pages.visited_tuples < unbounded.visited_tuples);
	assert(bounded_pages.pages_fetched < unbounded.pages_fetched);
	assert(bounded_pages.total_cost < unbounded.total_cost);
}

static void
test_tq_planner_cost_helper_accounts_for_filter_selectivity(void)
{
	TqPlannerCostEstimate broad_filter;
	TqPlannerCostEstimate selective_filter;

	memset(&broad_filter, 0, sizeof(broad_filter));
	memset(&selective_filter, 0, sizeof(selective_filter));

	assert(tq_estimate_ordered_scan_cost(512.0, 4096.0, 16.0, 1.0, 16, 2, 4,
										 0, 0,
										 0.005, 0.0025, 4.0, 0.01, &broad_filter));
	assert(tq_estimate_ordered_scan_cost(512.0, 4096.0, 16.0, 0.01, 16, 2, 4,
										 0, 0,
										 0.005, 0.0025, 4.0, 0.01, &selective_filter));
	assert(selective_filter.qual_selectivity < broad_filter.qual_selectivity);
	assert(selective_filter.candidate_bound > broad_filter.candidate_bound);
	assert(selective_filter.total_cost > broad_filter.total_cost);
}

static void
test_tq_candidate_heap_behavior(void)
{
	TqCandidateHeap heap;
	TqCandidateEntry entry;

	memset(&heap, 0, sizeof(heap));
	memset(&entry, 0, sizeof(entry));

	assert(tq_candidate_heap_init(&heap, 3));
	assert(tq_candidate_heap_push(&heap, 0.40f, 10, 1, NULL, NULL, 0));
	assert(tq_candidate_heap_push(&heap, 0.20f, 10, 2, NULL, NULL, 0));
	assert(tq_candidate_heap_push(&heap, 0.30f, 10, 3, NULL, NULL, 0));
	assert(tq_candidate_heap_push(&heap, 0.90f, 10, 4, NULL, NULL, 0));
	assert(tq_candidate_heap_push(&heap, 0.10f, 10, 5, NULL, NULL, 0));

	assert(heap.count == 3);
	assert(tq_candidate_heap_pop_best(&heap, &entry));
	assert_float_close(entry.score, 0.10f, 1e-6f);
	assert(entry.tid.block_number == 10);
	assert(entry.tid.offset_number == 5);

	assert(tq_candidate_heap_pop_best(&heap, &entry));
	assert_float_close(entry.score, 0.20f, 1e-6f);
	assert(entry.tid.offset_number == 2);

	assert(tq_candidate_heap_pop_best(&heap, &entry));
	assert_float_close(entry.score, 0.30f, 1e-6f);
	assert(entry.tid.offset_number == 3);

	assert(!tq_candidate_heap_pop_best(&heap, &entry));
	tq_candidate_heap_reset(&heap);
}

static void
test_tq_prod_query_score_dispatch_matches_scalar(void)
{
	TqProdCodecConfig config = default_prod_config(8, 4);
	float		query[8];
	uint32_t	rng_state = UINT32_C(424242);
	size_t		vec = 0;
	char		errmsg[256];

	for (vec = 0; vec < 8; vec++)
		query[vec] = lcg_float_signed(&rng_state);

	for (vec = 0; vec < 12; vec++)
	{
		float		input[8];
		uint8_t		packed[12];
		TqProdLut	lut;
		float		scalar_score = 0.0f;
		float		dispatch_score = 0.0f;
		TqProdScoreKernel used_kernel = TQ_PROD_SCORE_SCALAR;
		size_t		dim = 0;

		memset(packed, 0, sizeof(packed));
		memset(&lut, 0, sizeof(lut));

		for (dim = 0; dim < 8; dim++)
			input[dim] = lcg_float_signed(&rng_state);

		assert(tq_prod_encode(&config, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));
		assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
		assert(tq_prod_score_packed_ip(&config, &lut, packed, sizeof(packed), &scalar_score,
									   errmsg, sizeof(errmsg)));
		assert(tq_prod_score_query_dispatch(&config, query, 8, packed, sizeof(packed),
											TQ_PROD_SCORE_AUTO, &dispatch_score, &used_kernel,
											errmsg, sizeof(errmsg)));
		assert_float_close(dispatch_score, scalar_score, 1e-5f);
		assert(used_kernel == tq_prod_score_preferred_kernel());

		if (tq_simd_avx2_runtime_available())
		{
			float avx2_score = 0.0f;
			TqProdScoreKernel explicit_kernel = TQ_PROD_SCORE_SCALAR;

			assert(tq_prod_score_query_dispatch(&config, query, 8, packed, sizeof(packed),
												TQ_PROD_SCORE_AVX2, &avx2_score, &explicit_kernel,
												errmsg, sizeof(errmsg)));
			assert(explicit_kernel == TQ_PROD_SCORE_AVX2);
			assert_float_close(avx2_score, scalar_score, 1e-5f);
		}

		if (tq_simd_avx512_runtime_available())
		{
			float avx512_score = 0.0f;
			TqProdScoreKernel explicit_kernel = TQ_PROD_SCORE_SCALAR;

			assert(tq_prod_score_query_dispatch(&config, query, 8, packed, sizeof(packed),
												TQ_PROD_SCORE_AVX512, &avx512_score, &explicit_kernel,
												errmsg, sizeof(errmsg)));
			assert(explicit_kernel == TQ_PROD_SCORE_AVX512);
			assert_float_close(avx512_score, scalar_score, 1e-5f);
		}

		if (tq_simd_neon_runtime_available())
		{
			float neon_score = 0.0f;
			TqProdScoreKernel explicit_kernel = TQ_PROD_SCORE_SCALAR;

			assert(tq_prod_score_query_dispatch(&config, query, 8, packed, sizeof(packed),
												TQ_PROD_SCORE_NEON, &neon_score, &explicit_kernel,
												errmsg, sizeof(errmsg)));
			assert(explicit_kernel == TQ_PROD_SCORE_NEON);
			assert_float_close(neon_score, scalar_score, 1e-5f);
		}

		tq_prod_lut_reset(&lut);
	}
}

static void
test_tq_prod_query_score_dispatch_disabled_fallback(void)
{
	TqProdCodecConfig config = default_prod_config(4, 4);
	const float	query[4] = {0.25f, -0.5f, 0.75f, -0.125f};
	const float	input[4] = {0.5f, -0.25f, 1.0f, 0.0f};
	uint8_t		packed[7];
	float		score = 0.0f;
	TqProdScoreKernel used_kernel = TQ_PROD_SCORE_AVX2;
	char		errmsg[256];

	memset(packed, 0, sizeof(packed));

	assert(tq_prod_encode(&config, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	tq_simd_force_disable(true);
	assert(tq_prod_score_query_dispatch(&config, query, 4, packed, sizeof(packed),
										TQ_PROD_SCORE_AUTO, &score, &used_kernel,
										errmsg, sizeof(errmsg)));
	assert(used_kernel == TQ_PROD_SCORE_SCALAR);
	tq_simd_force_disable(false);
}

static void
test_tq_prod_query_score_dispatch_unavailable_kernel_errors(void)
{
	TqProdCodecConfig config = default_prod_config(4, 4);
	const float	query[4] = {0.25f, -0.5f, 0.75f, -0.125f};
	const float	input[4] = {0.5f, -0.25f, 1.0f, 0.0f};
	uint8_t		packed[7];
	float		score = 0.0f;
	TqProdScoreKernel used_kernel = TQ_PROD_SCORE_SCALAR;
	TqProdScoreKernel requested = TQ_PROD_SCORE_SCALAR;
	char		errmsg[256];

	memset(packed, 0, sizeof(packed));
	memset(errmsg, 0, sizeof(errmsg));

	assert(tq_prod_encode(&config, input, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	if (!tq_simd_avx512_runtime_available())
		requested = TQ_PROD_SCORE_AVX512;
	else if (!tq_simd_avx2_runtime_available())
		requested = TQ_PROD_SCORE_AVX2;
	else if (!tq_simd_neon_runtime_available())
		requested = TQ_PROD_SCORE_NEON;
	else
		return;

	assert(!tq_prod_score_query_dispatch(&config, query, 4, packed, sizeof(packed),
										 requested, &score, &used_kernel,
										 errmsg, sizeof(errmsg)));
	assert(strstr(errmsg, "not available") != NULL);
}

static void
test_tq_prod_score_kernel_names_known(void)
{
	assert(strcmp(tq_prod_score_kernel_name(TQ_PROD_SCORE_SCALAR), "scalar") == 0);
	assert(strcmp(tq_prod_score_kernel_name(TQ_PROD_SCORE_AVX2), "avx2") == 0);
	assert(strcmp(tq_prod_score_kernel_name(TQ_PROD_SCORE_AVX512), "avx512") == 0);
	assert(strcmp(tq_prod_score_kernel_name(TQ_PROD_SCORE_NEON), "neon") == 0);
	assert(strcmp(tq_prod_score_kernel_name(TQ_PROD_SCORE_AUTO), "auto") == 0);
}

static void
test_tq_router_assignment_tiny_clusters(void)
{
	TqRouterModel model;
	const float	vectors[16] = {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.95f, 0.05f, 0.0f, 0.0f,
		0.05f, 0.95f, 0.0f, 0.0f
	};
	uint32_t	list_id = UINT32_MAX;
	char		errmsg[256];

	memset(&model, 0, sizeof(model));

	assert(tq_router_train_first(vectors, 4, 4, 2, &model, errmsg, sizeof(errmsg)));
	assert(model.list_count == 2);
	assert(tq_router_assign_best(&model, vectors + 8, &list_id, NULL, errmsg, sizeof(errmsg)));
	assert(list_id == 0);
	assert(tq_router_assign_best(&model, vectors + 12, &list_id, NULL, errmsg, sizeof(errmsg)));
	assert(list_id == 1);

	tq_router_reset(&model);
}

static void
test_tq_router_probe_selection_order(void)
{
	TqRouterModel model;
	const float	vectors[12] = {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f
	};
	const float	query[4] = {0.8f, 0.2f, 0.0f, 0.0f};
	uint32_t	probes[3];
	uint32_t	selected = 0;
	char		errmsg[256];

	memset(&model, 0, sizeof(model));
	memset(probes, 0xFF, sizeof(probes));

	assert(tq_router_train_first(vectors, 3, 4, 3, &model, errmsg, sizeof(errmsg)));
	assert(tq_router_select_probes(&model, query, 2, probes, 3, &selected, errmsg, sizeof(errmsg)));
	assert(selected == 2);
	assert(probes[0] == 0);
	assert(probes[1] == 1);

	tq_router_reset(&model);
}

static float
tq_router_assignment_loss(const TqRouterModel *model,
						  const float *vectors,
						  size_t vector_count,
						  uint32_t dimension)
{
	size_t		vector_index = 0;
	float		total_loss = 0.0f;
	char		errmsg[256];

	for (vector_index = 0; vector_index < vector_count; vector_index++)
	{
		const float *vector = vectors + (vector_index * (size_t) dimension);
		uint32_t	list_id = UINT32_MAX;
		size_t		dimension_index = 0;
		float		loss = 0.0f;

		assert(tq_router_assign_best(model, vector, &list_id, NULL, errmsg, sizeof(errmsg)));
		for (dimension_index = 0; dimension_index < dimension; dimension_index++)
		{
			float diff =
				vector[dimension_index] -
				model->centroids[(list_id * (size_t) dimension) + dimension_index];
			loss += diff * diff;
		}
		total_loss += loss;
	}

	return total_loss;
}

static void
test_tq_router_kmeans_deterministic_training_metadata(void)
{
	const float	vectors[16] = {
		1.0f, 0.0f,
		0.95f, 0.05f,
		0.0f, 1.0f,
		0.05f, 0.95f,
		-1.0f, 0.0f,
		-0.95f, -0.05f,
		0.0f, -1.0f,
		-0.05f, -0.95f
	};
	TqRouterTrainingConfig config = {
		.seed = 7,
		.sample_count = 8,
		.max_iterations = 6
	};
	TqRouterModel left;
	TqRouterModel right;
	char		errmsg[256];

	memset(&left, 0, sizeof(left));
	memset(&right, 0, sizeof(right));

	assert(tq_router_train_kmeans(vectors, 8, 2, 4, &config, &left, errmsg, sizeof(errmsg)));
	assert(tq_router_train_kmeans(vectors, 8, 2, 4, &config, &right, errmsg, sizeof(errmsg)));

	assert(left.metadata.algorithm == TQ_ROUTER_ALGORITHM_KMEANS);
	assert(left.metadata.seed == config.seed);
	assert(left.metadata.sample_count == 8);
	assert(left.metadata.trained_vector_count == 8);
	assert(left.metadata.completed_iterations > 0);
	assert(left.metadata.completed_iterations <= config.max_iterations);
	assert_float_array_close(left.centroids, right.centroids, 8, 1e-6f);

	tq_router_reset(&left);
	tq_router_reset(&right);
}

static void
test_tq_router_kmeans_assignment_coverage_across_lists(void)
{
	const float	vectors[18] = {
		1.0f, 0.0f,
		0.96f, 0.04f,
		0.92f, 0.08f,
		0.0f, 1.0f,
		0.04f, 0.96f,
		0.08f, 0.92f,
		-1.0f, 0.0f,
		-0.96f, -0.04f,
		-0.92f, -0.08f
	};
	TqRouterTrainingConfig config = {
		.seed = 11,
		.sample_count = 9,
		.max_iterations = 8
	};
	TqRouterModel model;
	uint32_t	assignments[3] = {0, 0, 0};
	size_t		index = 0;
	char		errmsg[256];

	memset(&model, 0, sizeof(model));

	assert(tq_router_train_kmeans(vectors, 9, 2, 3, &config, &model, errmsg, sizeof(errmsg)));
	for (index = 0; index < 9; index++)
	{
		uint32_t	list_id = UINT32_MAX;

		assert(tq_router_assign_best(&model,
									 vectors + (index * 2),
									 &list_id,
									 NULL,
									 errmsg,
									 sizeof(errmsg)));
		assert(list_id < 3);
		assignments[list_id] += 1;
	}

	assert(assignments[0] > 0);
	assert(assignments[1] > 0);
	assert(assignments[2] > 0);

	tq_router_reset(&model);
}

static void
test_tq_router_kmeans_beats_first_k_on_clustered_objective(void)
{
	const float	vectors[24] = {
		0.99f, 0.01f,
		0.97f, 0.03f,
		0.95f, 0.05f,
		0.93f, 0.07f,
		0.03f, 0.97f,
		0.01f, 0.99f,
		-0.97f, 0.03f,
		-0.99f, 0.01f,
		-0.95f, 0.05f,
		0.03f, -0.97f,
		0.01f, -0.99f,
		0.05f, -0.95f
	};
	TqRouterTrainingConfig config = {
		.seed = 19,
		.sample_count = 12,
		.max_iterations = 8
	};
	TqRouterModel baseline;
	TqRouterModel trained;
	float		baseline_loss = 0.0f;
	float		trained_loss = 0.0f;
	char		errmsg[256];

	memset(&baseline, 0, sizeof(baseline));
	memset(&trained, 0, sizeof(trained));

	assert(tq_router_train_first(vectors, 12, 2, 4, &baseline, errmsg, sizeof(errmsg)));
	assert(tq_router_train_kmeans(vectors, 12, 2, 4, &config, &trained, errmsg, sizeof(errmsg)));

	baseline_loss = tq_router_assignment_loss(&baseline, vectors, 12, 2);
	trained_loss = tq_router_assignment_loss(&trained, vectors, 12, 2);

	assert(trained_loss < baseline_loss);

	tq_router_reset(&baseline);
	tq_router_reset(&trained);
}

static void
test_tq_router_kmeans_handles_fewer_rows_than_lists(void)
{
	const float	vectors[4] = {
		1.0f, 0.0f,
		0.0f, 1.0f
	};
	const float	query[2] = {1.0f, 0.0f};
	TqRouterTrainingConfig config = {
		.seed = 23,
		.sample_count = 2,
		.max_iterations = 4
	};
	TqRouterModel model;
	uint32_t	probes[4] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
	uint32_t	selected = 0;
	char		errmsg[256];

	memset(&model, 0, sizeof(model));

	assert(tq_router_train_kmeans(vectors, 2, 2, 4, &config, &model, errmsg, sizeof(errmsg)));
	assert(model.list_count == 4);
	assert(tq_router_select_probes(&model, query, 4, probes, 4, &selected, errmsg, sizeof(errmsg)));
	assert(selected == 4);

	tq_router_reset(&model);
}

static void
test_tq_batch_page_scan_tiny_corpus(void)
{
	TqProdCodecConfig config = default_prod_config(2, 4);
	TqProdLut lut;
	TqBatchPageParams params = {
		.lane_count = 4,
		.code_bytes = 6,
		.list_id = 0,
		.next_block = TQ_INVALID_BLOCK_NUMBER
	};
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	const float	query[2] = {1.0f, 0.0f};
	const float	vec0[2] = {1.0f, 0.0f};
	const float	vec1[2] = {0.0f, 1.0f};
	const float	vec2[2] = {-1.0f, 0.0f};
	uint8_t		packed[6];
	uint16_t	lane = 0;
	TqCandidateHeap heap;
	TqCandidateEntry entry;
	char		errmsg[256];

	memset(&lut, 0, sizeof(lut));
	memset(page, 0, sizeof(page));
	memset(packed, 0, sizeof(packed));
	memset(&heap, 0, sizeof(heap));
	memset(&entry, 0, sizeof(entry));

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_candidate_heap_init(&heap, 3));

	assert(tq_prod_encode(&config, vec0, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &(TqTid){.block_number = 1, .offset_number = 1}, &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	assert(tq_prod_encode(&config, vec1, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &(TqTid){.block_number = 1, .offset_number = 2}, &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	assert(tq_prod_encode(&config, vec2, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &(TqTid){.block_number = 1, .offset_number = 3}, &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	assert(tq_batch_page_scan_prod_cosine(page, sizeof(page), &config, true, &lut,
										  query, 2, false, 0, &heap, NULL, errmsg, sizeof(errmsg)));

	assert(tq_candidate_heap_pop_best(&heap, &entry));
	assert(entry.tid.offset_number == 1);
	assert(tq_candidate_heap_pop_best(&heap, &entry));
	assert(entry.tid.offset_number == 2);
	assert(tq_candidate_heap_pop_best(&heap, &entry));
	assert(entry.tid.offset_number == 3);

	tq_candidate_heap_reset(&heap);
	tq_prod_lut_reset(&lut);
}

static void
test_tq_batch_page_scan_normalized_metric_orders_align(void)
{
	TqProdCodecConfig config = default_prod_config(2, 4);
	TqProdLut lut;
	TqBatchPageParams params = {
		.lane_count = 4,
		.code_bytes = 6,
		.list_id = 0,
		.next_block = TQ_INVALID_BLOCK_NUMBER
	};
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	const float	query[2] = {1.0f, 0.0f};
	const float	vec0[2] = {1.0f, 0.0f};
	const float	vec1[2] = {0.70710678f, 0.70710678f};
	const float	vec2[2] = {0.0f, 1.0f};
	uint8_t		packed[6];
	uint16_t	lane = 0;
	TqCandidateHeap cosine_heap;
	TqCandidateHeap ip_heap;
	TqCandidateHeap l2_heap;
	TqCandidateEntry cosine_entry;
	TqCandidateEntry ip_entry;
	TqCandidateEntry l2_entry;
	char		errmsg[256];

	memset(&lut, 0, sizeof(lut));
	memset(page, 0, sizeof(page));
	memset(packed, 0, sizeof(packed));
	memset(&cosine_heap, 0, sizeof(cosine_heap));
	memset(&ip_heap, 0, sizeof(ip_heap));
	memset(&l2_heap, 0, sizeof(l2_heap));
	memset(&cosine_entry, 0, sizeof(cosine_entry));
	memset(&ip_entry, 0, sizeof(ip_entry));
	memset(&l2_entry, 0, sizeof(l2_entry));

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_candidate_heap_init(&cosine_heap, 3));
	assert(tq_candidate_heap_init(&ip_heap, 3));
	assert(tq_candidate_heap_init(&l2_heap, 3));

	assert(tq_prod_encode(&config, vec0, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &(TqTid){.block_number = 1, .offset_number = 1}, &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	assert(tq_prod_encode(&config, vec1, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &(TqTid){.block_number = 1, .offset_number = 2}, &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	assert(tq_prod_encode(&config, vec2, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &(TqTid){.block_number = 1, .offset_number = 3}, &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, true, TQ_DISTANCE_COSINE, &lut,
								   query, 2, false, 0, &cosine_heap, NULL, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, true, TQ_DISTANCE_IP, &lut,
								   query, 2, false, 0, &ip_heap, NULL, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, true, TQ_DISTANCE_L2, &lut,
								   query, 2, false, 0, &l2_heap, NULL, errmsg, sizeof(errmsg)));

	assert(tq_candidate_heap_pop_best(&cosine_heap, &cosine_entry));
	assert(tq_candidate_heap_pop_best(&ip_heap, &ip_entry));
	assert(tq_candidate_heap_pop_best(&l2_heap, &l2_entry));
	assert(cosine_entry.tid.offset_number == 1);
	assert(ip_entry.tid.offset_number == 1);
	assert(l2_entry.tid.offset_number == 1);

	assert(tq_candidate_heap_pop_best(&cosine_heap, &cosine_entry));
	assert(tq_candidate_heap_pop_best(&ip_heap, &ip_entry));
	assert(tq_candidate_heap_pop_best(&l2_heap, &l2_entry));
	assert(cosine_entry.tid.offset_number == 2);
	assert(ip_entry.tid.offset_number == 2);
	assert(l2_entry.tid.offset_number == 2);

	assert(tq_candidate_heap_pop_best(&cosine_heap, &cosine_entry));
	assert(tq_candidate_heap_pop_best(&ip_heap, &ip_entry));
	assert(tq_candidate_heap_pop_best(&l2_heap, &l2_entry));
	assert(cosine_entry.tid.offset_number == 3);
	assert(ip_entry.tid.offset_number == 3);
	assert(l2_entry.tid.offset_number == 3);

	tq_candidate_heap_reset(&cosine_heap);
	tq_candidate_heap_reset(&ip_heap);
	tq_candidate_heap_reset(&l2_heap);
	tq_prod_lut_reset(&lut);
}

static void
test_tq_batch_page_scan_cosine_is_scale_invariant(void)
{
	TqProdCodecConfig config = default_prod_config(2, 4);
	TqProdLut lut;
	TqBatchPageParams params = {
		.lane_count = 4,
		.code_bytes = 6,
		.list_id = 0,
		.next_block = TQ_INVALID_BLOCK_NUMBER
	};
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	const float	query[2] = {1.0f, 0.0f};
	const float	better_cosine[2] = {0.8f, 0.6f};
	const float	worse_but_larger[2] = {3.0f, 4.0f};
	const float	orthogonal[2] = {0.0f, 1.0f};
	uint8_t		packed[6];
	uint16_t	lane = 0;
	TqCandidateHeap heap;
	TqCandidateEntry entry;
	char		errmsg[256];

	memset(&lut, 0, sizeof(lut));
	memset(page, 0, sizeof(page));
	memset(packed, 0, sizeof(packed));
	memset(&heap, 0, sizeof(heap));
	memset(&entry, 0, sizeof(entry));

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_candidate_heap_init(&heap, 3));

	assert(tq_prod_encode(&config, better_cosine, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &(TqTid){.block_number = 1, .offset_number = 1}, &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	assert(tq_prod_encode(&config, worse_but_larger, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &(TqTid){.block_number = 1, .offset_number = 2}, &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	assert(tq_prod_encode(&config, orthogonal, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &(TqTid){.block_number = 1, .offset_number = 3}, &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, false, TQ_DISTANCE_COSINE, &lut,
								   query, 2, false, 0, &heap, NULL, errmsg, sizeof(errmsg)));

	assert(tq_candidate_heap_pop_best(&heap, &entry));
	assert(entry.tid.offset_number == 1);
	assert(tq_candidate_heap_pop_best(&heap, &entry));
	assert(entry.tid.offset_number == 2);
	assert(tq_candidate_heap_pop_best(&heap, &entry));
	assert(entry.tid.offset_number == 3);

	tq_candidate_heap_reset(&heap);
	tq_prod_lut_reset(&lut);
}

static void
test_tq_batch_page_scan_non_normalized_metrics_use_compatibility_fallback(void)
{
	TqProdCodecConfig config = default_prod_config(2, 4);
	TqProdLut lut;
	TqScanStats stats;
	TqBatchPageParams params = {
		.lane_count = 4,
		.code_bytes = 6,
		.list_id = 0,
		.next_block = TQ_INVALID_BLOCK_NUMBER
	};
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	const float	query[2] = {1.0f, 0.0f};
	const float	nearby[2] = {1.1f, 0.0f};
	const float	large_aligned[2] = {3.0f, 0.0f};
	const float	far_off_axis[2] = {0.0f, 2.0f};
	uint8_t		packed[6];
	uint16_t	lane = 0;
	TqCandidateHeap ip_heap;
	TqCandidateHeap l2_heap;
	TqCandidateEntry ip_entry;
	TqCandidateEntry l2_entry;
	char		errmsg[256];

	memset(&lut, 0, sizeof(lut));
	memset(&stats, 0, sizeof(stats));
	memset(page, 0, sizeof(page));
	memset(packed, 0, sizeof(packed));
	memset(&ip_heap, 0, sizeof(ip_heap));
	memset(&l2_heap, 0, sizeof(l2_heap));
	memset(&ip_entry, 0, sizeof(ip_entry));
	memset(&l2_entry, 0, sizeof(l2_entry));

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_candidate_heap_init(&ip_heap, 3));
	assert(tq_candidate_heap_init(&l2_heap, 3));

	assert(tq_prod_encode(&config, nearby, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &(TqTid){.block_number = 1, .offset_number = 1}, &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	assert(tq_prod_encode(&config, large_aligned, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &(TqTid){.block_number = 1, .offset_number = 2}, &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	assert(tq_prod_encode(&config, far_off_axis, packed, sizeof(packed), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &(TqTid){.block_number = 1, .offset_number = 3}, &lane, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_code(page, sizeof(page), lane, packed, sizeof(packed), errmsg, sizeof(errmsg)));

	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, false, TQ_DISTANCE_IP, &lut,
								   query, 2, false, 0, &ip_heap, NULL, errmsg, sizeof(errmsg)));
	tq_scan_stats_snapshot(&stats);
	assert(!stats.faithful_fast_path);
	assert(stats.compatibility_fallback);
	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, false, TQ_DISTANCE_L2, &lut,
								   query, 2, false, 0, &l2_heap, NULL, errmsg, sizeof(errmsg)));
	tq_scan_stats_snapshot(&stats);
	assert(!stats.faithful_fast_path);
	assert(stats.compatibility_fallback);

	assert(tq_candidate_heap_pop_best(&ip_heap, &ip_entry));
	assert(ip_entry.tid.offset_number >= 1);
	assert(ip_entry.tid.offset_number <= 3);

	assert(tq_candidate_heap_pop_best(&l2_heap, &l2_entry));
	assert(l2_entry.tid.offset_number >= 1);
	assert(l2_entry.tid.offset_number <= 3);
	assert(tq_candidate_heap_pop_best(&l2_heap, &l2_entry));
	assert(l2_entry.tid.offset_number >= 1);
	assert(l2_entry.tid.offset_number <= 3);
	assert(tq_candidate_heap_pop_best(&l2_heap, &l2_entry));
	assert(l2_entry.tid.offset_number >= 1);
	assert(l2_entry.tid.offset_number <= 3);

	tq_candidate_heap_reset(&ip_heap);
	tq_candidate_heap_reset(&l2_heap);
	tq_prod_lut_reset(&lut);
}

static void
test_tq_metadata_slot_compare_orders_supported_kinds(void)
{
	uint8_t		left[TQ_METADATA_SLOT_BYTES];
	uint8_t		right[TQ_METADATA_SLOT_BYTES];
	pg_uuid_t	left_uuid;
	pg_uuid_t	right_uuid;
	int			cmp = 0;
	char		errmsg[256];

	memset(left, 0, sizeof(left));
	memset(right, 0, sizeof(right));
	memset(&left_uuid, 0, sizeof(left_uuid));
	memset(&right_uuid, 0, sizeof(right_uuid));

	assert(tq_metadata_encode_datum(TQ_METADATA_KIND_BOOL, BoolGetDatum(false),
									left, errmsg, sizeof(errmsg)));
	assert(tq_metadata_encode_datum(TQ_METADATA_KIND_BOOL, BoolGetDatum(true),
									right, errmsg, sizeof(errmsg)));
	assert(tq_metadata_slot_compare(TQ_METADATA_KIND_BOOL, left, right, &cmp,
									errmsg, sizeof(errmsg)));
	assert(cmp < 0);

	assert(tq_metadata_encode_datum(TQ_METADATA_KIND_INT4, Int32GetDatum(7),
									left, errmsg, sizeof(errmsg)));
	assert(tq_metadata_encode_datum(TQ_METADATA_KIND_INT4, Int32GetDatum(7),
									right, errmsg, sizeof(errmsg)));
	assert(tq_metadata_slot_compare(TQ_METADATA_KIND_INT4, left, right, &cmp,
									errmsg, sizeof(errmsg)));
	assert(cmp == 0);

	assert(tq_metadata_encode_datum(TQ_METADATA_KIND_DATE, DateADTGetDatum(100),
									left, errmsg, sizeof(errmsg)));
	assert(tq_metadata_encode_datum(TQ_METADATA_KIND_DATE, DateADTGetDatum(140),
									right, errmsg, sizeof(errmsg)));
	assert(tq_metadata_slot_compare(TQ_METADATA_KIND_DATE, left, right, &cmp,
									errmsg, sizeof(errmsg)));
	assert(cmp < 0);

	left_uuid.data[UUID_LEN - 1] = 1;
	right_uuid.data[UUID_LEN - 1] = 9;
	assert(tq_metadata_encode_datum(TQ_METADATA_KIND_UUID, UUIDPGetDatum(&left_uuid),
									left, errmsg, sizeof(errmsg)));
	assert(tq_metadata_encode_datum(TQ_METADATA_KIND_UUID, UUIDPGetDatum(&right_uuid),
									right, errmsg, sizeof(errmsg)));
	assert(tq_metadata_slot_compare(TQ_METADATA_KIND_UUID, left, right, &cmp,
									errmsg, sizeof(errmsg)));
	assert(cmp < 0);
}

static void
test_tq_meta_page_roundtrip(void)
{
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	TqMetaPageFields written = {
		.dimension = 1536,
		.transform_output_dimension = 2048,
		.codec = TQ_CODEC_PROD,
		.distance = TQ_DISTANCE_COSINE,
		.bits = 4,
		.lane_count = 8,
		.transform = TQ_TRANSFORM_HADAMARD,
		.transform_version = TQ_TRANSFORM_CONTRACT_VERSION,
		.normalized = true,
		.list_count = 128,
		.directory_root_block = 7,
		.centroid_root_block = 9,
		.delta_head_block = TQ_INVALID_BLOCK_NUMBER,
		.delta_tail_block = TQ_INVALID_BLOCK_NUMBER,
		.delta_live_count = 0,
		.delta_batch_page_count = 0,
		.exact_key_head_block = TQ_INVALID_BLOCK_NUMBER,
		.exact_key_tail_block = TQ_INVALID_BLOCK_NUMBER,
		.exact_key_page_count = 0,
		.transform_seed = UINT64_C(0x1122334455667788),
		.algorithm_version = TQ_ALGORITHM_VERSION,
		.quantizer_version = TQ_QUANTIZER_VERSION,
		.residual_sketch_version = TQ_RESIDUAL_SKETCH_VERSION,
		.residual_bits_per_dimension = 1,
		.residual_sketch_dimension = 512,
		.estimator_version = TQ_ESTIMATOR_VERSION
	};
	TqMetaPageFields readback;
	char		errmsg[256];

	memset(page, 0xCC, sizeof(page));
	memset(&readback, 0, sizeof(readback));

	assert(tq_meta_page_init(page, sizeof(page), &written, errmsg, sizeof(errmsg)));
	assert(tq_meta_page_read(page, sizeof(page), &readback, errmsg, sizeof(errmsg)));
	assert(readback.dimension == written.dimension);
	assert(readback.transform_output_dimension == written.transform_output_dimension);
	assert(readback.codec == written.codec);
	assert(readback.distance == written.distance);
	assert(readback.bits == written.bits);
	assert(readback.lane_count == written.lane_count);
	assert(readback.transform == written.transform);
	assert(readback.transform_version == written.transform_version);
	assert(readback.normalized == written.normalized);
	assert(readback.list_count == written.list_count);
	assert(readback.directory_root_block == written.directory_root_block);
	assert(readback.centroid_root_block == written.centroid_root_block);
	assert(readback.delta_head_block == written.delta_head_block);
	assert(readback.delta_tail_block == written.delta_tail_block);
	assert(readback.delta_live_count == written.delta_live_count);
	assert(readback.delta_batch_page_count == written.delta_batch_page_count);
	assert(readback.exact_key_head_block == written.exact_key_head_block);
	assert(readback.exact_key_tail_block == written.exact_key_tail_block);
	assert(readback.exact_key_page_count == written.exact_key_page_count);
	assert(readback.transform_seed == written.transform_seed);
	assert(readback.algorithm_version == written.algorithm_version);
	assert(readback.quantizer_version == written.quantizer_version);
	assert(readback.residual_sketch_version == written.residual_sketch_version);
	assert(readback.residual_bits_per_dimension == written.residual_bits_per_dimension);
	assert(readback.residual_sketch_dimension == written.residual_sketch_dimension);
	assert(readback.estimator_version == written.estimator_version);
}

static void
test_tq_meta_page_rejects_old_format_version(void)
{
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	TqMetaPageFields written = {
		.dimension = 5,
		.transform_output_dimension = 8,
		.codec = TQ_CODEC_PROD,
		.distance = TQ_DISTANCE_COSINE,
		.bits = 4,
		.lane_count = 8,
		.transform = TQ_TRANSFORM_HADAMARD,
		.transform_version = TQ_TRANSFORM_CONTRACT_VERSION,
		.normalized = true,
		.list_count = 0,
		.directory_root_block = TQ_INVALID_BLOCK_NUMBER,
		.centroid_root_block = TQ_INVALID_BLOCK_NUMBER,
		.delta_head_block = TQ_INVALID_BLOCK_NUMBER,
		.delta_tail_block = TQ_INVALID_BLOCK_NUMBER,
		.delta_live_count = 0,
		.delta_batch_page_count = 0,
		.exact_key_head_block = TQ_INVALID_BLOCK_NUMBER,
		.exact_key_tail_block = TQ_INVALID_BLOCK_NUMBER,
		.exact_key_page_count = 0,
		.transform_seed = UINT64_C(7),
		.algorithm_version = TQ_ALGORITHM_VERSION,
		.quantizer_version = TQ_QUANTIZER_VERSION,
		.residual_sketch_version = TQ_RESIDUAL_SKETCH_VERSION,
		.residual_bits_per_dimension = 1,
		.residual_sketch_dimension = 8,
		.estimator_version = TQ_ESTIMATOR_VERSION
	};
	TqMetaPageFields readback;
	char		errmsg[256];

	memset(page, 0, sizeof(page));
	memset(&readback, 0, sizeof(readback));

	assert(tq_meta_page_init(page, sizeof(page), &written, errmsg, sizeof(errmsg)));
	page[8] = 2;
	page[9] = 0;
	page[10] = 0;
	page[11] = 0;
	assert(!tq_meta_page_read(page, sizeof(page), &readback, errmsg, sizeof(errmsg)));
	assert(strstr(errmsg, "unsupported format version") != NULL);
}

static void
test_tq_batch_summary_page_roundtrip_with_metadata_synopsis(void)
{
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	TqBatchSummaryPageHeaderView header;
	TqBatchPageSummary written;
	TqBatchPageSummary readback;
	uint8_t		representative_code[16];
	uint8_t		readback_code[16];
	char		errmsg[256];

	memset(page, 0, sizeof(page));
	memset(&header, 0, sizeof(header));
	memset(&written, 0, sizeof(written));
	memset(&readback, 0, sizeof(readback));
	memset(representative_code, 0xAB, sizeof(representative_code));
	memset(readback_code, 0, sizeof(readback_code));

	written.representative_lane = 3;
	written.residual_radius = 0.125f;
	written.null_any_mask = UINT16_C(0x0003);
	written.null_all_mask = UINT16_C(0x0002);
	written.all_same_mask = UINT16_C(0x0005);
	memset(written.same_values, 0x11, sizeof(written.same_values));
	memset(written.min_values, 0x22, sizeof(written.min_values));
	memset(written.max_values, 0x33, sizeof(written.max_values));

	assert(tq_batch_summary_page_init(page, sizeof(page), 16, 4,
									  TQ_INVALID_BLOCK_NUMBER,
									  errmsg, sizeof(errmsg)));
	assert(tq_batch_summary_page_set_entry(page, sizeof(page), 0, 77,
										   &written,
										   representative_code,
										   sizeof(representative_code),
										   errmsg, sizeof(errmsg)));
	assert(tq_batch_summary_page_read_header(page, sizeof(page), &header,
											errmsg, sizeof(errmsg)));
	assert(header.entry_count == 1);
	assert(tq_batch_summary_page_get_entry(page, sizeof(page), 0,
										   &header.next_block,
										   &readback,
										   readback_code,
										   sizeof(readback_code),
										   errmsg, sizeof(errmsg)));
	assert(readback.representative_lane == written.representative_lane);
	assert(fabsf(readback.residual_radius - written.residual_radius) < 1e-6f);
	assert(readback.null_any_mask == written.null_any_mask);
	assert(readback.null_all_mask == written.null_all_mask);
	assert(readback.all_same_mask == written.all_same_mask);
	assert(memcmp(readback.same_values, written.same_values,
				  sizeof(written.same_values)) == 0);
	assert(memcmp(readback.min_values, written.min_values,
				  sizeof(written.min_values)) == 0);
	assert(memcmp(readback.max_values, written.max_values,
				  sizeof(written.max_values)) == 0);
	assert(memcmp(readback_code, representative_code,
				  sizeof(representative_code)) == 0);
}

static void
test_tq_list_dir_entry_roundtrip(void)
{
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	TqListDirEntry written = {
		.list_id = 11,
		.head_block = 101,
		.tail_block = 109,
		.live_count = 17,
		.dead_count = 3,
		.batch_page_count = 4,
		.summary_head_block = 211,
		.free_lane_hint = 5
	};
	TqListDirEntry readback;
	char		errmsg[256];

	memset(page, 0x00, sizeof(page));
	memset(&readback, 0, sizeof(readback));

	assert(tq_list_dir_page_init(page, sizeof(page), 4, 88, errmsg, sizeof(errmsg)));
	assert(tq_list_dir_page_set_entry(page, sizeof(page), 0, &written, errmsg, sizeof(errmsg)));
	assert(tq_list_dir_page_get_entry(page, sizeof(page), 0, &readback, errmsg, sizeof(errmsg)));
	assert(readback.list_id == written.list_id);
	assert(readback.head_block == written.head_block);
	assert(readback.tail_block == written.tail_block);
	assert(readback.live_count == written.live_count);
	assert(readback.dead_count == written.dead_count);
	assert(readback.batch_page_count == written.batch_page_count);
	assert(readback.summary_head_block == written.summary_head_block);
	assert(readback.free_lane_hint == written.free_lane_hint);
}

static void
test_tq_batch_page_header_init(void)
{
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	TqBatchPageParams params = {
		.lane_count = 8,
		.code_bytes = 774,
		.list_id = 3,
		.next_block = 42
	};
	TqBatchPageHeaderView header;
	char		errmsg[256];

	memset(page, 0x00, sizeof(page));
	memset(&header, 0, sizeof(header));

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_read_header(page, sizeof(page), &header, errmsg, sizeof(errmsg)));
	assert(header.lane_count == 8);
	assert(header.code_bytes == 774);
	assert(header.list_id == 3);
	assert(header.next_block == 42);
	assert(header.occupied_count == 0);
	assert(header.live_count == 0);
}

static void
test_tq_batch_page_capacity_checks(void)
{
	assert(tq_batch_page_required_bytes(8, 774) < (size_t) TQ_DEFAULT_BLOCK_SIZE);
	assert(tq_batch_page_can_fit(TQ_DEFAULT_BLOCK_SIZE, 8, 774));
	assert(!tq_batch_page_can_fit(TQ_DEFAULT_BLOCK_SIZE, 16, 774));
}

static void
test_tq_batch_page_lane_occupancy_and_tids(void)
{
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	TqBatchPageParams params = {
		.lane_count = 4,
		.code_bytes = 32,
		.list_id = 0,
		.next_block = TQ_INVALID_BLOCK_NUMBER
	};
	TqBatchPageHeaderView header;
	TqTid		tid0 = {.block_number = 10, .offset_number = 1};
	TqTid		tid1 = {.block_number = 10, .offset_number = 2};
	TqTid		tid2 = {.block_number = 10, .offset_number = 3};
	TqTid		readback;
	uint16_t	lane_index = 0;
	char		errmsg[256];

	memset(page, 0x00, sizeof(page));
	memset(&header, 0, sizeof(header));
	memset(&readback, 0, sizeof(readback));

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));

	assert(tq_batch_page_append_lane(page, sizeof(page), &tid0, &lane_index, errmsg, sizeof(errmsg)));
	assert(lane_index == 0);
	assert(tq_batch_page_append_lane(page, sizeof(page), &tid1, &lane_index, errmsg, sizeof(errmsg)));
	assert(lane_index == 1);
	assert(tq_batch_page_append_lane(page, sizeof(page), &tid2, &lane_index, errmsg, sizeof(errmsg)));
	assert(lane_index == 2);

	assert(tq_batch_page_read_header(page, sizeof(page), &header, errmsg, sizeof(errmsg)));
	assert(header.occupied_count == 3);
	assert(header.live_count == 3);

	assert(tq_batch_page_get_tid(page, sizeof(page), 0, &readback, errmsg, sizeof(errmsg)));
	assert(readback.block_number == tid0.block_number);
	assert(readback.offset_number == tid0.offset_number);

	assert(tq_batch_page_get_tid(page, sizeof(page), 1, &readback, errmsg, sizeof(errmsg)));
	assert(readback.block_number == tid1.block_number);
	assert(readback.offset_number == tid1.offset_number);

	assert(tq_batch_page_get_tid(page, sizeof(page), 2, &readback, errmsg, sizeof(errmsg)));
	assert(readback.block_number == tid2.block_number);
	assert(readback.offset_number == tid2.offset_number);
}

static void
test_tq_batch_page_live_bitmap_and_iterator(void)
{
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	TqBatchPageParams params = {
		.lane_count = 4,
		.code_bytes = 32,
		.list_id = 0,
		.next_block = TQ_INVALID_BLOCK_NUMBER
	};
	TqBatchPageHeaderView header;
	TqTid		tid = {.block_number = 22, .offset_number = 1};
	uint16_t	lane_index = 0;
	bool		is_live = false;
	char		errmsg[256];

	memset(page, 0x00, sizeof(page));
	memset(&header, 0, sizeof(header));

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &tid, &lane_index, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &tid, &lane_index, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &tid, &lane_index, errmsg, sizeof(errmsg)));

	assert(tq_batch_page_mark_dead(page, sizeof(page), 1, errmsg, sizeof(errmsg)));

	assert(tq_batch_page_is_live(page, sizeof(page), 0, &is_live, errmsg, sizeof(errmsg)));
	assert(is_live);
	assert(tq_batch_page_is_live(page, sizeof(page), 1, &is_live, errmsg, sizeof(errmsg)));
	assert(!is_live);
	assert(tq_batch_page_is_live(page, sizeof(page), 2, &is_live, errmsg, sizeof(errmsg)));
	assert(is_live);

	assert(tq_batch_page_read_header(page, sizeof(page), &header, errmsg, sizeof(errmsg)));
	assert(header.occupied_count == 3);
	assert(header.live_count == 2);

	assert(tq_batch_page_next_live_lane(page, sizeof(page), -1, &lane_index, errmsg, sizeof(errmsg)));
	assert(lane_index == 0);
	assert(tq_batch_page_next_live_lane(page, sizeof(page), (int) lane_index, &lane_index, errmsg, sizeof(errmsg)));
	assert(lane_index == 2);
	assert(!tq_batch_page_next_live_lane(page, sizeof(page), (int) lane_index, &lane_index, errmsg, sizeof(errmsg)));
}

static void
test_tq_batch_page_tail_append_decisions(void)
{
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	TqBatchPageParams params = {
		.lane_count = 2,
		.code_bytes = 32,
		.list_id = 0,
		.next_block = TQ_INVALID_BLOCK_NUMBER
	};
	TqTid		tid = {.block_number = 55, .offset_number = 1};
	uint16_t	lane_index = 0;
	bool		has_capacity = false;
	char		errmsg[256];

	memset(page, 0x00, sizeof(page));

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_has_capacity(page, sizeof(page), &has_capacity, errmsg, sizeof(errmsg)));
	assert(has_capacity);

	assert(tq_batch_page_append_lane(page, sizeof(page), &tid, &lane_index, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_has_capacity(page, sizeof(page), &has_capacity, errmsg, sizeof(errmsg)));
	assert(has_capacity);

	assert(tq_batch_page_append_lane(page, sizeof(page), &tid, &lane_index, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_has_capacity(page, sizeof(page), &has_capacity, errmsg, sizeof(errmsg)));
	assert(!has_capacity);
}

static void
test_tq_batch_page_reclaim_decisions(void)
{
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	TqBatchPageParams params = {
		.lane_count = 3,
		.code_bytes = 32,
		.list_id = 0,
		.next_block = TQ_INVALID_BLOCK_NUMBER
	};
	TqTid		tid = {.block_number = 77, .offset_number = 1};
	uint16_t	lane_index = 0;
	bool		should_reclaim = false;
	char		errmsg[256];

	memset(page, 0x00, sizeof(page));

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_should_reclaim(page, sizeof(page), &should_reclaim, errmsg, sizeof(errmsg)));
	assert(!should_reclaim);

	assert(tq_batch_page_append_lane(page, sizeof(page), &tid, &lane_index, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &tid, &lane_index, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_should_reclaim(page, sizeof(page), &should_reclaim, errmsg, sizeof(errmsg)));
	assert(!should_reclaim);

	assert(tq_batch_page_mark_dead(page, sizeof(page), 0, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_mark_dead(page, sizeof(page), 0, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_should_reclaim(page, sizeof(page), &should_reclaim, errmsg, sizeof(errmsg)));
	assert(!should_reclaim);

	assert(tq_batch_page_mark_dead(page, sizeof(page), 1, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_should_reclaim(page, sizeof(page), &should_reclaim, errmsg, sizeof(errmsg)));
	assert(should_reclaim);
}

static void
test_tq_batch_page_compaction_reuses_dead_lanes(void)
{
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	TqBatchPageParams params = {
		.lane_count = 4,
		.code_bytes = 16,
		.list_id = 0,
		.next_block = TQ_INVALID_BLOCK_NUMBER
	};
	TqTid		tid0 = {.block_number = 91, .offset_number = 1};
	TqTid		tid1 = {.block_number = 91, .offset_number = 2};
	TqTid		tid2 = {.block_number = 91, .offset_number = 3};
	TqTid		readback;
	TqBatchPageHeaderView header;
	uint16_t	lane_index = 0;
	bool		is_live = false;
	char		errmsg[256];

	memset(page, 0x00, sizeof(page));
	memset(&readback, 0, sizeof(readback));
	memset(&header, 0, sizeof(header));

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &tid0, &lane_index, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &tid1, &lane_index, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &tid2, &lane_index, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_mark_dead(page, sizeof(page), 1, errmsg, sizeof(errmsg)));

	assert(tq_batch_page_compact(page, sizeof(page), errmsg, sizeof(errmsg)));
	assert(tq_batch_page_read_header(page, sizeof(page), &header, errmsg, sizeof(errmsg)));
	assert(header.occupied_count == 2);
	assert(header.live_count == 2);

	assert(tq_batch_page_get_tid(page, sizeof(page), 0, &readback, errmsg, sizeof(errmsg)));
	assert(readback.block_number == tid0.block_number);
	assert(readback.offset_number == tid0.offset_number);
	assert(tq_batch_page_get_tid(page, sizeof(page), 1, &readback, errmsg, sizeof(errmsg)));
	assert(readback.block_number == tid2.block_number);
	assert(readback.offset_number == tid2.offset_number);

	assert(tq_batch_page_is_live(page, sizeof(page), 0, &is_live, errmsg, sizeof(errmsg)));
	assert(is_live);
	assert(tq_batch_page_is_live(page, sizeof(page), 1, &is_live, errmsg, sizeof(errmsg)));
	assert(is_live);
	assert(!tq_batch_page_next_live_lane(page, sizeof(page), 1, &lane_index, errmsg, sizeof(errmsg)));
}

static void
test_tq_batch_page_soa_filter_int4_roundtrip(void)
{
	TqProdCodecConfig config = default_prod_config(32, 4);
	TqProdPackedLayout layout;
	TqBatchPageParams params = {
		.lane_count = 16,
		.list_id = 0,
		.next_block = TQ_INVALID_BLOCK_NUMBER,
		.dimension = 32,
		.int4_attribute_count = 1
	};
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	TqTid		tid = {.block_number = 42, .offset_number = 7};
	uint8_t		nibbles[32];
	uint16_t	lane_index = 0;
	int32_t		filter_value = 0;
	bool		has_filter = false;
	const float   *gammas = NULL;
	const uint8_t *page_nibbles = NULL;
	uint32_t	dimension = 0;
	uint16_t	lane_count = 0;
	char		errmsg[256];
	uint32_t	d;

	memset(&layout, 0, sizeof(layout));
	memset(page, 0, sizeof(page));
	memset(nibbles, 0, sizeof(nibbles));

	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	params.code_bytes = (uint32_t) layout.total_bytes;

	for (d = 0; d < 32; d++)
		nibbles[d] = (uint8_t) (d & 0x0Fu);

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_is_soa(page, sizeof(page)));
	assert(tq_batch_page_append_lane(page, sizeof(page), &tid, &lane_index, errmsg, sizeof(errmsg)));
	assert(lane_index == 0);
	assert(tq_batch_page_set_filter_int4(page, sizeof(page), lane_index, 77, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_set_nibble_and_gamma(page, sizeof(page), lane_index, nibbles, 32, 0.25f,
											  errmsg, sizeof(errmsg)));

	assert(tq_batch_page_has_filter_int4(page, sizeof(page), &has_filter, errmsg, sizeof(errmsg)));
	assert(has_filter);
	assert(tq_batch_page_get_filter_int4(page, sizeof(page), lane_index, &filter_value, errmsg, sizeof(errmsg)));
	assert(filter_value == 77);
	assert(tq_batch_page_get_nibble_ptr(page, sizeof(page), &page_nibbles, &dimension, &lane_count,
										errmsg, sizeof(errmsg)));
	assert(tq_batch_page_get_gamma_ptr(page, sizeof(page), &gammas, errmsg, sizeof(errmsg)));
	assert(dimension == 32);
	assert(lane_count == 16);
	for (d = 0; d < 32; d++)
		assert((page_nibbles[(size_t) d * (lane_count / 2u)] & 0x0Fu) == (uint8_t) (d & 0x0Fu));
	assert(fabsf(gammas[0] - 0.25f) < 1e-6f);
}

static void
test_tq_batch_page_scan_filtered_soa_stays_code_domain(void)
{
	TqProdCodecConfig config = default_prod_config(32, 4);
	TqProdPackedLayout layout;
	TqProdLut lut;
	TqBatchPageParams params = {
		.lane_count = 16,
		.list_id = 0,
		.next_block = TQ_INVALID_BLOCK_NUMBER,
		.dimension = 32,
		.int4_attribute_count = 1
	};
	uint8_t		page[TQ_DEFAULT_BLOCK_SIZE];
	uint8_t		packed[256];
	float		query[32];
	TqCandidateHeap heap;
	TqCandidateEntry entry;
	TqScanStats stats;
	uint16_t	lane = 0;
	size_t		i = 0;
	char		errmsg[256];

	memset(&layout, 0, sizeof(layout));
	memset(&lut, 0, sizeof(lut));
	memset(page, 0, sizeof(page));
	memset(packed, 0, sizeof(packed));
	memset(query, 0, sizeof(query));
	memset(&heap, 0, sizeof(heap));
	memset(&entry, 0, sizeof(entry));
	memset(&stats, 0, sizeof(stats));

	seeded_unit_vector(77u, query, 32);
	assert(tq_prod_packed_layout(&config, &layout, errmsg, sizeof(errmsg)));
	assert(tq_prod_lut_build(&config, query, &lut, errmsg, sizeof(errmsg)));
	assert(tq_candidate_heap_init(&heap, 16));
	params.code_bytes = (uint32_t) layout.total_bytes;

	assert(tq_batch_page_init(page, sizeof(page), &params, errmsg, sizeof(errmsg)));
	assert(tq_batch_page_is_soa(page, sizeof(page)));

	for (i = 0; i < 16; i++)
	{
		float		input[32];
		float		gamma = 0.0f;
		uint8_t		nibbles[32];

		seeded_unit_vector((uint32_t) (1200 + i), input, 32);
		memset(packed, 0, sizeof(packed));
		memset(nibbles, 0, sizeof(nibbles));

		assert(tq_prod_encode(&config, input, packed, layout.total_bytes, errmsg, sizeof(errmsg)));
		assert(tq_prod_extract_nibbles(&config, packed, layout.total_bytes, nibbles, 32, errmsg, sizeof(errmsg)));
		assert(tq_prod_read_gamma(&config, packed, layout.total_bytes, &gamma, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_append_lane(page, sizeof(page),
										 &(TqTid){.block_number = 1, .offset_number = (uint16_t) (i + 1)},
										 &lane, errmsg, sizeof(errmsg)));
		assert(tq_batch_page_set_filter_int4(page, sizeof(page), lane, (i < 8) ? 1 : 2,
											 errmsg, sizeof(errmsg)));
		assert(tq_batch_page_set_nibble_and_gamma(page, sizeof(page), lane, nibbles, 32, gamma,
											  errmsg, sizeof(errmsg)));
	}

	tq_scan_stats_begin(TQ_SCAN_MODE_FLAT, 1);
	assert(tq_batch_page_scan_prod(page, sizeof(page), &config, true, TQ_DISTANCE_COSINE, &lut,
								   query, 32, true, 1, &heap, NULL, errmsg, sizeof(errmsg)));
	tq_scan_stats_snapshot(&stats);

	assert(stats.score_mode == TQ_SCAN_SCORE_MODE_CODE_DOMAIN);
	assert(stats.faithful_fast_path);
	assert(!stats.compatibility_fallback);
	assert(stats.decoded_vector_count == 8u);
	assert(stats.visited_code_count == 8u);
	assert(heap.count == 8u);

	while (tq_candidate_heap_pop_best(&heap, &entry))
		assert(entry.tid.offset_number >= 1 && entry.tid.offset_number <= 8);

	tq_candidate_heap_reset(&heap);
	tq_prod_lut_reset(&lut);
}

int
main(void)
{
	test_tq_init_amroutine_flags();
	test_tq_validate_option_config_valid();
	test_tq_validate_option_config_invalid_bits();
	test_tq_validate_option_config_invalid_lists();
	test_tq_validate_option_config_invalid_transform();
	test_tq_validate_option_config_invalid_lanes();
	test_tq_compute_code_bytes_prod_default();
	test_tq_resolve_lane_count_default_prod();
	test_tq_resolve_lane_count_small_dimensions();
	test_tq_resolve_lane_count_impossible();
	test_tq_transform_prepare_and_dimension_padding();
	test_tq_transform_same_seed_same_input_same_output();
	test_tq_transform_different_seed_materially_changes_output();
	test_tq_transform_metadata_roundtrip_prepare();
	test_tq_transform_apply_uses_full_padded_contract();
	test_tq_transform_inverse_roundtrip();
	test_tq_transform_apply_rejects_truncated_output_buffer();
	test_tq_transform_norm_behavior_is_stable();
	test_tq_transform_zero_vector_stays_zero();
	test_tq_transform_normalized_input_keeps_unit_norm();
	test_tq_mse_packed_length_calculations();
	test_tq_mse_encode_decode_determinism();
	test_tq_mse_reconstruction_error_bound();
	test_tq_mse_invalid_parameter_handling();
	test_tq_mse_lut_shape_and_known_values();
	test_tq_mse_scalar_scoring_tiny_corpus();
	test_tq_mse_codebook_reduces_reconstruction_error_on_representative_distributions();
	test_tq_mse_codebook_improves_or_preserves_ann_recall();
	test_tq_prod_packed_length_calculations();
	test_tq_prod_encode_decode_determinism();
	test_tq_prod_invalid_parameter_handling();
	test_tq_prod_score_decomposition();
	test_tq_prod_scalar_score_matches_decode_helper();
	test_tq_prod_stability_seeded_random_corpus();
	test_tq_prod_unbiased_estimator_has_low_signed_error_on_seeded_corpora();
	test_tq_prod_calibrated_estimator_improves_recall_on_representative_corpora();
	test_tq_vector_copy_from_pgvector_struct();
	test_tq_vector_copy_from_halfvec_struct();
	test_tq_vector_copy_from_halfvec_datum_typed();
	test_tq_vector_copy_struct_validation_messages_are_consistent();
	test_tq_vector_typed_validation_messages_are_consistent();
	test_tq_candidate_budget_helper();
	test_tq_streaming_candidate_budget_helper();
	test_tq_planner_cost_helper_prefers_flat_for_small_tables();
	test_tq_planner_cost_helper_prefers_ivf_for_large_tables_and_low_probes();
	test_tq_planner_cost_helper_high_probes_remove_ivf_advantage();
	test_tq_planner_cost_helper_visit_budgets_limit_ivf_work();
	test_tq_planner_cost_helper_accounts_for_filter_selectivity();
	test_tq_candidate_heap_behavior();
	test_tq_metric_distance_from_ip_score_modes();
	test_tq_prod_query_score_dispatch_matches_scalar();
	test_tq_prod_query_score_dispatch_disabled_fallback();
	test_tq_prod_query_score_dispatch_unavailable_kernel_errors();
	test_tq_prod_score_kernel_names_known();
	test_tq_router_assignment_tiny_clusters();
	test_tq_router_probe_selection_order();
	test_tq_router_kmeans_deterministic_training_metadata();
	test_tq_router_kmeans_assignment_coverage_across_lists();
	test_tq_router_kmeans_beats_first_k_on_clustered_objective();
	test_tq_router_kmeans_handles_fewer_rows_than_lists();
	test_tq_batch_page_scan_tiny_corpus();
	test_tq_batch_page_scan_normalized_metric_orders_align();
	test_tq_batch_page_scan_cosine_is_scale_invariant();
	test_tq_batch_page_scan_non_normalized_metrics_use_compatibility_fallback();
	test_tq_metadata_slot_compare_orders_supported_kinds();
	test_tq_meta_page_roundtrip();
	test_tq_meta_page_rejects_old_format_version();
	test_tq_batch_summary_page_roundtrip_with_metadata_synopsis();
	test_tq_list_dir_entry_roundtrip();
	test_tq_batch_page_header_init();
	test_tq_batch_page_capacity_checks();
	test_tq_batch_page_lane_occupancy_and_tids();
	test_tq_batch_page_live_bitmap_and_iterator();
	test_tq_batch_page_tail_append_decisions();
	test_tq_batch_page_reclaim_decisions();
	test_tq_batch_page_compaction_reuses_dead_lanes();
	test_tq_batch_page_soa_filter_int4_roundtrip();
	test_tq_batch_page_scan_filtered_soa_stays_code_domain();
	return 0;
}
