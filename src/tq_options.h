#ifndef TQ_OPTIONS_H
#define TQ_OPTIONS_H

#include <stdbool.h>
#include <stddef.h>

#define TQ_DEFAULT_BLOCK_SIZE 8192
#define TQ_PAGE_HEADER_BYTES 24
#define TQ_PAGE_SPECIAL_BYTES 16
#define TQ_PAGE_RESERVED_BYTES 128
#define TQ_TID_BYTES 6

typedef enum TqCodecKind
{
	TQ_CODEC_PROD = 0,
	TQ_CODEC_MSE = 1
} TqCodecKind;

typedef enum TqTransformKind
{
	TQ_TRANSFORM_HADAMARD = 0
} TqTransformKind;

typedef enum TqRouterAlgorithmKind
{
	TQ_ROUTER_ALGORITHM_FIRST_K = 1,
	TQ_ROUTER_ALGORITHM_KMEANS = 2
} TqRouterAlgorithmKind;

typedef struct TqOptionConfig
{
	int			bits;
	int			lists;
	int			router_samples;
	int			router_iterations;
	int			router_restarts;
	int			router_seed;
	int			qjl_sketch_dim;
	bool		normalized;
	const char *transform_name;
	const char *lanes_name;
} TqOptionConfig;

typedef struct TqLaneConfig
{
	int			block_size;
	int			dimension;
	int			qjl_dimension;
	int			bits;
	TqCodecKind codec;
	bool		normalized;
	int			page_header_bytes;
	int			special_space_bytes;
	int			reserve_bytes;
	int			tid_bytes;
} TqLaneConfig;

extern bool tq_validate_option_config(const TqOptionConfig *config,
									  char *errmsg,
									  size_t errmsg_len);
extern bool tq_parse_transform_name(const char *transform_name,
									TqTransformKind *transform_kind,
									char *errmsg,
									size_t errmsg_len);
extern bool tq_validate_lanes_name(const char *lanes_name,
								   char *errmsg,
								   size_t errmsg_len);
extern bool tq_compute_code_bytes(const TqLaneConfig *config,
								  size_t *code_bytes,
								  char *errmsg,
								  size_t errmsg_len);
extern bool tq_resolve_lane_count(const TqLaneConfig *config,
								  int *lane_count,
								  char *errmsg,
								  size_t errmsg_len);

#endif
