#include "src/tq_router.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct TqRouterProbeScore
{
	uint32_t	list_id;
	float		score;
} TqRouterProbeScore;

static void
tq_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg == NULL || errmsg_len == 0)
		return;

	snprintf(errmsg, errmsg_len, "%s", message);
}

static float
tq_router_squared_l2_distance(const float *left, const float *right, uint32_t dimension)
{
	float		sum = 0.0f;
	uint32_t	i = 0;

	for (i = 0; i < dimension; i++)
	{
		float		diff = left[i] - right[i];

		sum += diff * diff;
	}

	return sum;
}

static uint32_t
tq_router_lcg_next(uint32_t *state)
{
	*state = (*state * UINT32_C(1664525)) + UINT32_C(1013904223);
	return *state;
}

static bool
tq_router_validate_training_inputs(const float *vectors,
								   size_t vector_count,
								   uint32_t dimension,
								   uint32_t list_count,
								   const TqRouterTrainingConfig *config,
								   TqRouterModel *model,
								   char *errmsg,
								   size_t errmsg_len)
{
	if (vectors == NULL || config == NULL || model == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: vectors, config, and model must be non-null");
		return false;
	}

	if (vector_count == 0 || dimension == 0 || list_count == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: vector_count, dimension, and list_count must be positive");
		return false;
	}

	if (config->sample_count == 0 || config->max_iterations == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: sample_count and max_iterations must be positive");
		return false;
	}

	return true;
}

static bool
tq_router_allocate_model(uint32_t dimension,
						 uint32_t list_count,
						 TqRouterModel *model,
						 char *errmsg,
						 size_t errmsg_len)
{
	tq_router_reset(model);
	model->centroids = (float *) calloc((size_t) list_count * (size_t) dimension, sizeof(float));
	if (model->centroids == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: out of memory");
		return false;
	}

	model->dimension = dimension;
	model->list_count = list_count;
	return true;
}

static bool
tq_router_build_sample(const float *vectors,
						 size_t vector_count,
						 uint32_t dimension,
						 const TqRouterTrainingConfig *config,
						 float **sampled_vectors,
						 uint32_t *sample_count,
						 char *errmsg,
						 size_t errmsg_len)
{
	uint32_t	target_count = 0;
	uint32_t	state = config->seed;
	size_t	   *indices = NULL;
	float	   *sample = NULL;
	size_t		i = 0;

	target_count = config->sample_count < vector_count
		? config->sample_count
		: (uint32_t) vector_count;
	*sample_count = target_count;

	indices = (size_t *) malloc(sizeof(size_t) * vector_count);
	sample = (float *) malloc(sizeof(float) * (size_t) target_count * (size_t) dimension);
	if (indices == NULL || sample == NULL)
	{
		free(indices);
		free(sample);
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: out of memory");
		return false;
	}

	for (i = 0; i < vector_count; i++)
		indices[i] = i;

	for (i = 0; i < target_count; i++)
	{
		size_t		remaining = vector_count - i;
		size_t		swap_index = i + (size_t) (tq_router_lcg_next(&state) % remaining);
		size_t		chosen_index = 0;
		size_t		dimension_offset = 0;

		{
			size_t tmp = indices[i];
			indices[i] = indices[swap_index];
			indices[swap_index] = tmp;
		}

		chosen_index = indices[i];
		dimension_offset = chosen_index * (size_t) dimension;
		memcpy(sample + (i * (size_t) dimension),
			   vectors + dimension_offset,
			   sizeof(float) * (size_t) dimension);
	}

	free(indices);
	*sampled_vectors = sample;
	return true;
}

static size_t
tq_router_choose_weighted_index(const double *weights,
								  const bool *chosen,
								  size_t count,
								  uint32_t *state)
{
	double		total = 0.0;
	double		target = 0.0;
	size_t		i = 0;

	for (i = 0; i < count; i++)
	{
		if (!chosen[i] && weights[i] > 0.0)
			total += weights[i];
	}

	if (total <= 0.0)
	{
		for (i = 0; i < count; i++)
		{
			if (!chosen[i])
				return i;
		}
		return 0;
	}

	target = ((double) tq_router_lcg_next(state) / (double) UINT32_MAX) * total;
	for (i = 0; i < count; i++)
	{
		if (chosen[i] || weights[i] <= 0.0)
			continue;
		target -= weights[i];
		if (target <= 0.0)
			return i;
	}

	for (i = count; i > 0; i--)
	{
		size_t		index = i - 1;

		if (!chosen[index])
			return index;
	}

	return 0;
}

static void
tq_router_initialize_kmeanspp(const float *sampled_vectors,
							  uint32_t sample_count,
							  uint32_t dimension,
							  uint32_t list_count,
							  const TqRouterTrainingConfig *config,
							  TqRouterModel *model)
{
	bool	   *chosen = NULL;
	double	   *weights = NULL;
	uint32_t	state = config->seed;
	size_t		centroid_index = 0;
	size_t		first_index = 0;

	chosen = (bool *) calloc(sample_count, sizeof(bool));
	weights = (double *) calloc(sample_count, sizeof(double));
	if (chosen == NULL || weights == NULL)
	{
		free(chosen);
		free(weights);
		return;
	}

	first_index = config->seed % sample_count;
	memcpy(model->centroids,
		   sampled_vectors + (first_index * (size_t) dimension),
		   sizeof(float) * (size_t) dimension);
	chosen[first_index] = true;

	for (centroid_index = 1; centroid_index < list_count; centroid_index++)
	{
		size_t		sample_index = 0;
		size_t		selected_index = 0;

		for (sample_index = 0; sample_index < sample_count; sample_index++)
		{
			const float *sample = sampled_vectors + (sample_index * (size_t) dimension);
			double		best_distance = 0.0;
			bool		have_best = false;
			size_t		existing_index = 0;

			for (existing_index = 0; existing_index < centroid_index; existing_index++)
			{
				const float *centroid = model->centroids + (existing_index * (size_t) dimension);
				double		current_distance = (double) tq_router_squared_l2_distance(sample, centroid, dimension);

				if (!have_best || current_distance < best_distance)
				{
					have_best = true;
					best_distance = current_distance;
				}
			}

			weights[sample_index] = chosen[sample_index] ? 0.0 : best_distance;
		}

		selected_index = tq_router_choose_weighted_index(weights, chosen, sample_count, &state);
		memcpy(model->centroids + (centroid_index * (size_t) dimension),
			   sampled_vectors + (selected_index * (size_t) dimension),
			   sizeof(float) * (size_t) dimension);
		chosen[selected_index] = true;
	}

	free(chosen);
	free(weights);
}

static void
tq_router_reseed_empty_centroid(const float *sampled_vectors,
								const uint32_t *assignments,
								uint32_t sample_count,
								uint32_t dimension,
								uint32_t centroid_index,
								uint32_t list_count,
								float *centroids)
{
	size_t		sample_index = 0;
	double		best_distance = -1.0;
	size_t		best_index = 0;

	for (sample_index = 0; sample_index < sample_count; sample_index++)
	{
		const float *sample = sampled_vectors + (sample_index * (size_t) dimension);
		const float *assigned_centroid = centroids + ((size_t) assignments[sample_index] * (size_t) dimension);
		double		distance = (double) tq_router_squared_l2_distance(sample, assigned_centroid, dimension);

		if (distance > best_distance)
		{
			best_distance = distance;
			best_index = sample_index;
		}
	}

	memcpy(centroids + ((size_t) centroid_index * (size_t) dimension),
		   sampled_vectors + (best_index * (size_t) dimension),
		   sizeof(float) * (size_t) dimension);
	(void) list_count;
}

static int
tq_router_probe_compare(const void *left, const void *right)
{
	const TqRouterProbeScore *lhs = (const TqRouterProbeScore *) left;
	const TqRouterProbeScore *rhs = (const TqRouterProbeScore *) right;

	if (lhs->score > rhs->score)
		return -1;
	if (lhs->score < rhs->score)
		return 1;
	if (lhs->list_id < rhs->list_id)
		return -1;
	if (lhs->list_id > rhs->list_id)
		return 1;
	return 0;
}

void
tq_router_reset(TqRouterModel *model)
{
	if (model == NULL)
		return;

	free(model->centroids);
	memset(model, 0, sizeof(*model));
}

bool
tq_router_train_first(const float *vectors,
					  size_t vector_count,
					  uint32_t dimension,
					  uint32_t list_count,
					  TqRouterModel *model,
					  char *errmsg,
					  size_t errmsg_len)
{
	size_t		i = 0;
	size_t		source_index = 0;

	if (vectors == NULL || model == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: vectors and model must be non-null");
		return false;
	}

	if (vector_count == 0 || dimension == 0 || list_count == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: vector_count, dimension, and list_count must be positive");
		return false;
	}

	tq_router_reset(model);
	model->centroids = (float *) calloc((size_t) list_count * (size_t) dimension, sizeof(float));
	if (model->centroids == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: out of memory");
		return false;
	}

	model->dimension = dimension;
	model->list_count = list_count;
	model->metadata.algorithm = TQ_ROUTER_ALGORITHM_FIRST_K;
	model->metadata.seed = 0;
	model->metadata.sample_count = (uint32_t) vector_count;
	model->metadata.max_iterations = 0;
	model->metadata.completed_iterations = 0;
	model->metadata.trained_vector_count = (uint32_t) vector_count;

	for (i = 0; i < list_count; i++)
	{
		source_index = i < vector_count ? i : (vector_count - 1);
		memcpy(model->centroids + (i * (size_t) dimension),
			   vectors + (source_index * (size_t) dimension),
			   sizeof(float) * (size_t) dimension);
	}

	return true;
}

bool
tq_router_train_kmeans(const float *vectors,
					   size_t vector_count,
					   uint32_t dimension,
					   uint32_t list_count,
					   const TqRouterTrainingConfig *config,
					   TqRouterModel *model,
					   char *errmsg,
					   size_t errmsg_len)
{
	float	   *sampled_vectors = NULL;
	uint32_t	sample_count = 0;
	uint32_t   *assignments = NULL;
	float	   *sums = NULL;
	uint32_t   *counts = NULL;
	uint32_t	iteration = 0;
	bool		have_changes = true;

	if (!tq_router_validate_training_inputs(vectors, vector_count, dimension, list_count,
											config, model, errmsg, errmsg_len))
		return false;

	if (!tq_router_allocate_model(dimension, list_count, model, errmsg, errmsg_len))
		return false;

	if (!tq_router_build_sample(vectors, vector_count, dimension, config,
								&sampled_vectors, &sample_count, errmsg, errmsg_len))
	{
		tq_router_reset(model);
		return false;
	}

	assignments = (uint32_t *) malloc(sizeof(uint32_t) * (size_t) sample_count);
	sums = (float *) calloc((size_t) list_count * (size_t) dimension, sizeof(float));
	counts = (uint32_t *) calloc(list_count, sizeof(uint32_t));
	if (assignments == NULL || sums == NULL || counts == NULL)
	{
		free(sampled_vectors);
		free(assignments);
		free(sums);
		free(counts);
		tq_router_reset(model);
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: out of memory");
		return false;
	}

	tq_router_initialize_kmeanspp(sampled_vectors, sample_count, dimension, list_count,
								  config, model);

	for (iteration = 0; iteration < config->max_iterations && have_changes; iteration++)
	{
		uint32_t	sample_index = 0;

		memset(sums, 0, sizeof(float) * (size_t) list_count * (size_t) dimension);
		memset(counts, 0, sizeof(uint32_t) * (size_t) list_count);
		have_changes = false;

		for (sample_index = 0; sample_index < sample_count; sample_index++)
		{
			const float *sample = sampled_vectors + ((size_t) sample_index * (size_t) dimension);
			uint32_t	best_list = 0;
			float		best_distance = 0.0f;
			bool		have_best = false;
			uint32_t	list_index = 0;
			float	   *sum_dst = NULL;
			uint32_t	dimension_index = 0;

			for (list_index = 0; list_index < list_count; list_index++)
			{
				const float *centroid = model->centroids + ((size_t) list_index * (size_t) dimension);
				float		current_distance = tq_router_squared_l2_distance(sample, centroid, dimension);

				if (!have_best || current_distance < best_distance)
				{
					have_best = true;
					best_distance = current_distance;
					best_list = list_index;
				}
			}

			if (iteration == 0 || assignments[sample_index] != best_list)
			{
				assignments[sample_index] = best_list;
				have_changes = true;
			}

			sum_dst = sums + ((size_t) best_list * (size_t) dimension);
			for (dimension_index = 0; dimension_index < dimension; dimension_index++)
				sum_dst[dimension_index] += sample[dimension_index];
			counts[best_list] += 1;
		}

		for (sample_index = 0; sample_index < list_count; sample_index++)
		{
			float	   *centroid = model->centroids + ((size_t) sample_index * (size_t) dimension);
			uint32_t	dimension_index = 0;

			if (counts[sample_index] == 0)
			{
				tq_router_reseed_empty_centroid(sampled_vectors, assignments, sample_count,
												dimension, sample_index, list_count,
												model->centroids);
				continue;
			}

			for (dimension_index = 0; dimension_index < dimension; dimension_index++)
			{
				centroid[dimension_index] =
					sums[((size_t) sample_index * (size_t) dimension) + (size_t) dimension_index]
					/ (float) counts[sample_index];
			}
		}
	}

	model->metadata.algorithm = TQ_ROUTER_ALGORITHM_KMEANS;
	model->metadata.seed = config->seed;
	model->metadata.sample_count = sample_count;
	model->metadata.max_iterations = config->max_iterations;
	model->metadata.completed_iterations = iteration;
	model->metadata.trained_vector_count = (uint32_t) vector_count;

	free(sampled_vectors);
	free(assignments);
	free(sums);
	free(counts);
	return true;
}

bool
tq_router_assign_best(const TqRouterModel *model,
					  const float *vector,
					  uint32_t *list_id,
					  float *score,
					  char *errmsg,
					  size_t errmsg_len)
{
	uint32_t	i = 0;
	uint32_t	best_list = 0;
	float		best_score = 0.0f;
	bool		have_best = false;

	if (model == NULL || vector == NULL || list_id == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: model, vector, and list_id must be non-null");
		return false;
	}

	if (model->dimension == 0 || model->list_count == 0 || model->centroids == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: model is not initialized");
		return false;
	}

	for (i = 0; i < model->list_count; i++)
	{
		float		current_score = -tq_router_squared_l2_distance(
			vector,
			model->centroids + (i * (size_t) model->dimension),
			model->dimension);

		if (!have_best || current_score > best_score)
		{
			have_best = true;
			best_score = current_score;
			best_list = i;
		}
	}

	*list_id = best_list;
	if (score != NULL)
		*score = best_score;

	return true;
}

bool
tq_router_select_probes(const TqRouterModel *model,
						const float *query,
						uint32_t probes,
						uint32_t *out_list_ids,
						size_t out_capacity,
						uint32_t *selected_count,
						char *errmsg,
						size_t errmsg_len)
{
	TqRouterProbeScore *scores = NULL;
	uint32_t	count = 0;
	uint32_t	i = 0;

	if (model == NULL || query == NULL || out_list_ids == NULL || selected_count == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: model, query, outputs, and selected_count must be non-null");
		return false;
	}

	if (model->dimension == 0 || model->list_count == 0 || model->centroids == NULL || probes == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: model must be initialized and probes must be positive");
		return false;
	}

	count = probes < model->list_count ? probes : model->list_count;
	if (out_capacity < count)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: output buffer is too small");
		return false;
	}

	scores = (TqRouterProbeScore *) calloc(model->list_count, sizeof(TqRouterProbeScore));
	if (scores == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: out of memory");
		return false;
	}

	for (i = 0; i < model->list_count; i++)
	{
		scores[i].list_id = i;
		scores[i].score = -tq_router_squared_l2_distance(
			query,
			model->centroids + (i * (size_t) model->dimension),
			model->dimension);
	}

	qsort(scores, model->list_count, sizeof(TqRouterProbeScore), tq_router_probe_compare);

	for (i = 0; i < count; i++)
		out_list_ids[i] = scores[i].list_id;

	*selected_count = count;
	free(scores);
	return true;
}
