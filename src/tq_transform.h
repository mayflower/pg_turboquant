#ifndef TQ_TRANSFORM_H
#define TQ_TRANSFORM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/tq_options.h"

#define TQ_TRANSFORM_CONTRACT_VERSION UINT16_C(1)

typedef struct TqTransformConfig
{
	TqTransformKind kind;
	uint32_t		dimension;
	uint64_t		seed;
} TqTransformConfig;

typedef struct TqTransformMetadata
{
	uint16_t		contract_version;
	TqTransformKind kind;
	uint32_t		input_dimension;
	uint32_t		output_dimension;
	uint64_t		seed;
} TqTransformMetadata;

typedef struct TqTransformState
{
	TqTransformKind kind;
	uint32_t		dimension;
	uint32_t		padded_dimension;
	uint64_t		seed;
	uint32_t		permutation_count;
	uint32_t		sign_count;
	uint32_t	   *permutation;
	int8_t		   *signs;
} TqTransformState;

extern uint32_t tq_transform_padded_dimension(uint32_t dimension);
extern void tq_transform_reset(TqTransformState *state);
extern bool tq_transform_metadata_init(const TqTransformConfig *config,
									   TqTransformMetadata *metadata,
									   char *errmsg,
									   size_t errmsg_len);
extern bool tq_transform_prepare_metadata(const TqTransformMetadata *metadata,
										  TqTransformState *state,
										  char *errmsg,
										  size_t errmsg_len);
extern bool tq_transform_prepare(const TqTransformConfig *config,
								 TqTransformState *state,
								 char *errmsg,
								 size_t errmsg_len);
extern bool tq_transform_apply(const TqTransformState *state,
							   const float *input,
							   float *output,
							   size_t output_len,
							   char *errmsg,
							   size_t errmsg_len);
extern bool tq_transform_inverse(const TqTransformState *state,
								 const float *input,
								 size_t input_len,
								 float *output,
								 size_t output_len,
								 char *errmsg,
								 size_t errmsg_len);
extern bool tq_transform_apply_reference(const TqTransformState *state,
										 const float *input,
										 float *output,
										 size_t output_len,
										 char *errmsg,
										 size_t errmsg_len);
extern bool tq_transform_inverse_reference(const TqTransformState *state,
										   const float *input,
										   float *output,
										   size_t output_len,
										   char *errmsg,
										   size_t errmsg_len);

#endif
