#ifndef TQ_ROUTER_H
#define TQ_ROUTER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/tq_options.h"

#define TQ_ROUTER_MAX_LIST_WEIGHT 0.25f
#define TQ_ROUTER_COEFF_VAR_WEIGHT 0.15f

typedef struct TqRouterTrainingConfig
{
	uint32_t	seed;
	uint32_t	sample_count;
	uint32_t	max_iterations;
	uint32_t	restart_count;
} TqRouterTrainingConfig;

typedef struct TqRouterTrainingMetadata
{
	TqRouterAlgorithmKind algorithm;
	uint32_t	seed;
	uint32_t	sample_count;
	uint32_t	max_iterations;
	uint32_t	completed_iterations;
	uint32_t	trained_vector_count;
	uint32_t	restart_count;
	uint32_t	selected_restart;
	float		mean_distortion;
	float		max_list_over_avg;
	float		coeff_var;
	float		balance_penalty;
	float		selection_score;
} TqRouterTrainingMetadata;

typedef struct TqRouterModel
{
	uint32_t	dimension;
	uint32_t	list_count;
	float	   *centroids;
	TqRouterTrainingMetadata metadata;
} TqRouterModel;

extern void tq_router_reset(TqRouterModel *model);
extern bool tq_router_train_first(const float *vectors,
								  size_t vector_count,
								  uint32_t dimension,
								  uint32_t list_count,
								  TqRouterModel *model,
								  char *errmsg,
								  size_t errmsg_len);
extern bool tq_router_train_kmeans(const float *vectors,
								   size_t vector_count,
								   uint32_t dimension,
								   uint32_t list_count,
								   const TqRouterTrainingConfig *config,
								   TqRouterModel *model,
								   char *errmsg,
								   size_t errmsg_len);
extern bool tq_router_assign_best(const TqRouterModel *model,
								  const float *vector,
								  uint32_t *list_id,
								  float *score,
								  char *errmsg,
								  size_t errmsg_len);
extern bool tq_router_select_probes(const TqRouterModel *model,
									const float *query,
									uint32_t probes,
									uint32_t *out_list_ids,
									size_t out_capacity,
									uint32_t *selected_count,
									char *errmsg,
									size_t errmsg_len);

#endif
