#include "src/tq_router.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TQ_UNIT_TEST
#undef snprintf
#endif

typedef struct TqRouterObjective
{
	float		mean_distortion;
	float		max_list_over_avg;
	float		coeff_var;
	float		balance_penalty;
	float		selection_score;
} TqRouterObjective;

static void tq_router_reseed_empty_centroid(const float *sampled_vectors,
											const uint32_t *assignments,
											uint32_t sample_count,
											uint32_t dimension,
											uint32_t centroid_index,
											uint32_t list_count,
											float *centroids);

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
							  uint32_t seed,
							  TqRouterModel *model)
{
	bool	   *chosen = NULL;
	double	   *weights = NULL;
	uint32_t	state = seed;
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

	first_index = seed % sample_count;
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

static uint32_t
tq_router_restart_seed(const TqRouterTrainingConfig *config, uint32_t restart_index)
{
	uint32_t	state = config->seed;
	uint32_t	index = 0;

	for (index = 0; index < restart_index; index++)
		(void) tq_router_lcg_next(&state);

	return state;
}

static bool
tq_router_run_kmeans_restart(const float *sampled_vectors,
							 uint32_t sample_count,
							 uint32_t dimension,
							 uint32_t list_count,
							 uint32_t restart_seed,
							 uint32_t max_iterations,
							 TqRouterModel *model,
							 uint32_t *completed_iterations,
							 char *errmsg,
							 size_t errmsg_len)
{
	uint32_t   *assignments = NULL;
	float	   *sums = NULL;
	uint32_t   *counts = NULL;
	uint32_t	iteration = 0;
	bool		have_changes = true;

	assignments = (uint32_t *) malloc(sizeof(uint32_t) * (size_t) sample_count);
	sums = (float *) calloc((size_t) list_count * (size_t) dimension, sizeof(float));
	counts = (uint32_t *) calloc(list_count, sizeof(uint32_t));
	if (assignments == NULL || sums == NULL || counts == NULL)
	{
		free(assignments);
		free(sums);
		free(counts);
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: out of memory");
		return false;
	}

	tq_router_initialize_kmeanspp(sampled_vectors, sample_count, dimension, list_count,
								  restart_seed, model);

	for (iteration = 0; iteration < max_iterations && have_changes; iteration++)
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

	free(assignments);
	free(sums);
	free(counts);

	*completed_iterations = iteration;
	return true;
}

static bool
tq_router_evaluate_objective(const TqRouterModel *model,
							 const float *vectors,
							 size_t vector_count,
							 uint32_t dimension,
							 TqRouterObjective *objective,
							 char *errmsg,
							 size_t errmsg_len)
{
	uint32_t   *counts = NULL;
	size_t		vector_index = 0;
	double		total_distortion = 0.0;
	double		avg_list_size = 0.0;
	double		max_list_size = 0.0;
	double		variance = 0.0;
	uint32_t	list_index = 0;

	if (objective == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: objective output must be non-null");
		return false;
	}

	counts = (uint32_t *) calloc(model->list_count, sizeof(uint32_t));
	if (counts == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: out of memory");
		return false;
	}

	for (vector_index = 0; vector_index < vector_count; vector_index++)
	{
		const float *vector = vectors + (vector_index * (size_t) dimension);
		uint32_t	list_id = 0;
		uint32_t	dimension_index = 0;
		double		distortion = 0.0;

		if (!tq_router_assign_best(model, vector, &list_id, NULL, errmsg, errmsg_len))
		{
			free(counts);
			return false;
		}

		counts[list_id] += 1;
		for (dimension_index = 0; dimension_index < dimension; dimension_index++)
		{
			double		diff =
				(double) vector[dimension_index] -
				(double) model->centroids[(list_id * (size_t) dimension) + (size_t) dimension_index];

			distortion += diff * diff;
		}
		total_distortion += distortion;
	}

	avg_list_size = (double) vector_count / (double) model->list_count;
	for (list_index = 0; list_index < model->list_count; list_index++)
	{
		double		current = (double) counts[list_index];
		double		centered = current - avg_list_size;

		if (current > max_list_size)
			max_list_size = current;
		variance += centered * centered;
	}
	variance /= (double) model->list_count;

	objective->mean_distortion = vector_count == 0
		? 0.0f
		: (float) (total_distortion / (double) vector_count);
	objective->max_list_over_avg = avg_list_size <= 0.0
		? 0.0f
		: (float) (max_list_size / avg_list_size);
	objective->coeff_var = avg_list_size <= 0.0
		? 0.0f
		: (float) (sqrt(variance) / avg_list_size);
	objective->balance_penalty =
		(TQ_ROUTER_MAX_LIST_WEIGHT * (objective->max_list_over_avg > 1.0f
									  ? (objective->max_list_over_avg - 1.0f)
									  : 0.0f))
		+ (TQ_ROUTER_COEFF_VAR_WEIGHT * objective->coeff_var);
	objective->selection_score =
		objective->mean_distortion * (1.0f + objective->balance_penalty);

	free(counts);
	return true;
}

static bool
tq_router_objective_is_better(const TqRouterObjective *candidate,
							  const TqRouterObjective *best,
							  uint32_t candidate_restart,
							  uint32_t best_restart)
{
	const float	epsilon = 1e-6f;

	if (candidate->selection_score + epsilon < best->selection_score)
		return true;
	if (candidate->selection_score > best->selection_score + epsilon)
		return false;
	if (candidate->balance_penalty + epsilon < best->balance_penalty)
		return true;
	if (candidate->balance_penalty > best->balance_penalty + epsilon)
		return false;
	if (candidate->mean_distortion + epsilon < best->mean_distortion)
		return true;
	if (candidate->mean_distortion > best->mean_distortion + epsilon)
		return false;
	return candidate_restart < best_restart;
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

static bool
tq_router_probe_is_better(const TqRouterProbeScore *left, const TqRouterProbeScore *right)
{
	return tq_router_probe_compare(left, right) < 0;
}

static void
tq_router_probe_swap(TqRouterProbeScore *left, TqRouterProbeScore *right)
{
	TqRouterProbeScore tmp = *left;

	*left = *right;
	*right = tmp;
}

static void
tq_router_worst_heap_sift_up(TqRouterProbeScore *heap, size_t index)
{
	while (index > 0)
	{
		size_t parent = (index - 1) / 2;

		if (!tq_router_probe_is_better(&heap[parent], &heap[index]))
			break;

		tq_router_probe_swap(&heap[parent], &heap[index]);
		index = parent;
	}
}

static void
tq_router_worst_heap_sift_down(TqRouterProbeScore *heap, size_t count, size_t index)
{
	for (;;)
	{
		size_t worst = index;
		size_t left = (index * 2) + 1;
		size_t right = left + 1;

		if (left < count && tq_router_probe_is_better(&heap[worst], &heap[left]))
			worst = left;
		if (right < count && tq_router_probe_is_better(&heap[worst], &heap[right]))
			worst = right;
		if (worst == index)
			break;

		tq_router_probe_swap(&heap[index], &heap[worst]);
		index = worst;
	}
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
	model->metadata.restart_count = 1;
	model->metadata.selected_restart = 0;
	model->metadata.mean_distortion = 0.0f;
	model->metadata.max_list_over_avg = 0.0f;
	model->metadata.coeff_var = 0.0f;
	model->metadata.balance_penalty = 0.0f;
	model->metadata.selection_score = 0.0f;

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
	uint32_t	restart_count = 0;
	uint32_t	restart_index = 0;
	uint32_t	best_restart = 0;
	TqRouterObjective best_objective;
	bool		have_best = false;

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

	restart_count = config->restart_count == 0 ? 1 : config->restart_count;
	memset(&best_objective, 0, sizeof(best_objective));

	for (restart_index = 0; restart_index < restart_count; restart_index++)
	{
		TqRouterModel candidate;
		TqRouterObjective objective;
		uint32_t	completed_iterations = 0;
		uint32_t	restart_seed = tq_router_restart_seed(config, restart_index);

		memset(&candidate, 0, sizeof(candidate));
		memset(&objective, 0, sizeof(objective));
		if (!tq_router_allocate_model(dimension, list_count, &candidate, errmsg, errmsg_len))
		{
			free(sampled_vectors);
			tq_router_reset(model);
			return false;
		}

		if (!tq_router_run_kmeans_restart(sampled_vectors, sample_count, dimension, list_count,
										  restart_seed, config->max_iterations, &candidate,
										  &completed_iterations, errmsg, errmsg_len)
			|| !tq_router_evaluate_objective(&candidate, vectors, vector_count, dimension,
											 &objective, errmsg, errmsg_len))
		{
			tq_router_reset(&candidate);
			free(sampled_vectors);
			tq_router_reset(model);
			return false;
		}

		if (!have_best
			|| tq_router_objective_is_better(&objective, &best_objective,
											 restart_index, best_restart))
		{
			memcpy(model->centroids,
				   candidate.centroids,
				   sizeof(float) * (size_t) list_count * (size_t) dimension);
			best_objective = objective;
			best_restart = restart_index;
			model->metadata.completed_iterations = completed_iterations;
			have_best = true;
		}

		tq_router_reset(&candidate);
	}

	model->metadata.algorithm = TQ_ROUTER_ALGORITHM_KMEANS;
	model->metadata.seed = config->seed;
	model->metadata.sample_count = sample_count;
	model->metadata.max_iterations = config->max_iterations;
	model->metadata.trained_vector_count = (uint32_t) vector_count;
	model->metadata.restart_count = restart_count;
	model->metadata.selected_restart = best_restart;
	model->metadata.mean_distortion = best_objective.mean_distortion;
	model->metadata.max_list_over_avg = best_objective.max_list_over_avg;
	model->metadata.coeff_var = best_objective.coeff_var;
	model->metadata.balance_penalty = best_objective.balance_penalty;
	model->metadata.selection_score = best_objective.selection_score;

	free(sampled_vectors);
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
tq_router_rank_probes(const TqRouterModel *model,
					  const float *query,
					  TqRouterProbeScore *out_scores,
					  size_t out_capacity,
					  char *errmsg,
					  size_t errmsg_len)
{
	uint32_t	i = 0;

	if (model == NULL || query == NULL || out_scores == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: model, query, and output scores must be non-null");
		return false;
	}

	if (model->dimension == 0 || model->list_count == 0 || model->centroids == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: model must be initialized");
		return false;
	}

	if (out_capacity == 0)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: output buffer must be non-empty");
		return false;
	}

	if (out_capacity >= model->list_count)
	{
		for (i = 0; i < model->list_count; i++)
		{
			out_scores[i].list_id = i;
			out_scores[i].score = -tq_router_squared_l2_distance(
				query,
				model->centroids + (i * (size_t) model->dimension),
				model->dimension);
		}

		qsort(out_scores, model->list_count, sizeof(TqRouterProbeScore), tq_router_probe_compare);
		return true;
	}

	for (i = 0; i < model->list_count; i++)
	{
		TqRouterProbeScore current;

		current.list_id = i;
		current.score = -tq_router_squared_l2_distance(
			query,
			model->centroids + (i * (size_t) model->dimension),
			model->dimension);

		if ((size_t) i < out_capacity)
		{
			out_scores[i] = current;
			tq_router_worst_heap_sift_up(out_scores, (size_t) i);
			continue;
		}

		if (tq_router_probe_is_better(&current, &out_scores[0]))
		{
			out_scores[0] = current;
			tq_router_worst_heap_sift_down(out_scores, out_capacity, 0);
		}
	}

	qsort(out_scores, out_capacity, sizeof(TqRouterProbeScore), tq_router_probe_compare);
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

	scores = (TqRouterProbeScore *) calloc(count, sizeof(TqRouterProbeScore));
	if (scores == NULL)
	{
		tq_set_error(errmsg, errmsg_len,
					 "invalid turboquant router: out of memory");
		return false;
	}

	if (!tq_router_rank_probes(model, query, scores, count, errmsg, errmsg_len))
	{
		free(scores);
		return false;
	}

	for (i = 0; i < count; i++)
		out_list_ids[i] = scores[i].list_id;

	*selected_count = count;
	free(scores);
	return true;
}
