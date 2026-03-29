#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "src/tq_router.h"

static void
assert_float_close(float actual, float expected, float tolerance)
{
	assert(fabsf(actual - expected) <= tolerance);
}

static float
max_list_over_avg(const TqRouterModel *model,
				  const float *vectors,
				  size_t vector_count,
				  uint32_t dimension)
{
	uint32_t	counts[8] = {0};
	size_t		index = 0;
	float		avg = (float) vector_count / (float) model->list_count;
	float		max_count = 0.0f;
	char		errmsg[256];

	assert(model->list_count <= 8);
	for (index = 0; index < vector_count; index++)
	{
		uint32_t	list_id = UINT32_MAX;

		assert(tq_router_assign_best(model,
									 vectors + (index * (size_t) dimension),
									 &list_id,
									 NULL,
									 errmsg,
									 sizeof(errmsg)));
		counts[list_id] += 1;
	}

	for (index = 0; index < model->list_count; index++)
	{
		if ((float) counts[index] > max_count)
			max_count = (float) counts[index];
	}

	return avg <= 0.0f ? 0.0f : (max_count / avg);
}

static float
assignment_loss(const TqRouterModel *model,
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
		uint32_t	dimension_index = 0;
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
test_seeded_restarts_are_deterministic(void)
{
	const float	vectors[36] = {
		1.18f, 0.02f,
		0.59f, -0.01f,
		0.72f, -0.01f,
		1.10f, -0.01f,
		0.67f, 0.01f,
		1.27f, 0.00f,
		0.39f, 0.02f,
		0.87f, -0.01f,
		4.12f, 4.14f,
		4.09f, 4.12f,
		3.94f, 4.07f,
		4.12f, 4.06f,
		-4.01f, 3.88f,
		-4.02f, 4.03f,
		-3.88f, 4.14f,
		-4.01f, 4.11f,
		-0.07f, -3.91f,
		0.01f, -4.15f
	};
	TqRouterTrainingConfig config = {
		.seed = 17,
		.sample_count = 18,
		.max_iterations = 8,
		.restart_count = 4
	};
	TqRouterModel left;
	TqRouterModel right;
	char		errmsg[256];

	memset(&left, 0, sizeof(left));
	memset(&right, 0, sizeof(right));

	assert(tq_router_train_kmeans(vectors, 18, 2, 4, &config, &left, errmsg, sizeof(errmsg)));
	assert(tq_router_train_kmeans(vectors, 18, 2, 4, &config, &right, errmsg, sizeof(errmsg)));
	assert(left.metadata.restart_count == 4);
	assert(left.metadata.selected_restart == right.metadata.selected_restart);
	assert_float_close(left.metadata.balance_penalty, right.metadata.balance_penalty, 1e-6f);
	assert_float_close(left.metadata.selection_score, right.metadata.selection_score, 1e-6f);
	for (size_t i = 0; i < 8; i++)
		assert_float_close(left.centroids[i], right.centroids[i], 1e-6f);

	tq_router_reset(&left);
	tq_router_reset(&right);
}

static void
test_balanced_selection_reduces_max_list_ratio_on_skew_fixture(void)
{
	const float	vectors[36] = {
		1.18f, 0.02f,
		0.59f, -0.01f,
		0.72f, -0.01f,
		1.10f, -0.01f,
		0.67f, 0.01f,
		1.27f, 0.00f,
		0.39f, 0.02f,
		0.87f, -0.01f,
		4.12f, 4.14f,
		4.09f, 4.12f,
		3.94f, 4.07f,
		4.12f, 4.06f,
		-4.01f, 3.88f,
		-4.02f, 4.03f,
		-3.88f, 4.14f,
		-4.01f, 4.11f,
		-0.07f, -3.91f,
		0.01f, -4.15f
	};
	TqRouterTrainingConfig baseline_config = {
		.seed = 17,
		.sample_count = 18,
		.max_iterations = 8,
		.restart_count = 1
	};
	TqRouterTrainingConfig balanced_config = {
		.seed = 17,
		.sample_count = 18,
		.max_iterations = 8,
		.restart_count = 4
	};
	TqRouterModel baseline;
	TqRouterModel balanced;
	float		baseline_ratio = 0.0f;
	float		balanced_ratio = 0.0f;
	float		baseline_loss = 0.0f;
	float		balanced_loss = 0.0f;
	char		errmsg[256];

	memset(&baseline, 0, sizeof(baseline));
	memset(&balanced, 0, sizeof(balanced));

	assert(tq_router_train_kmeans(vectors, 18, 2, 4, &baseline_config, &baseline, errmsg, sizeof(errmsg)));
	assert(tq_router_train_kmeans(vectors, 18, 2, 4, &balanced_config, &balanced, errmsg, sizeof(errmsg)));

	baseline_ratio = max_list_over_avg(&baseline, vectors, 18, 2);
	balanced_ratio = max_list_over_avg(&balanced, vectors, 18, 2);
	baseline_loss = assignment_loss(&baseline, vectors, 18, 2);
	balanced_loss = assignment_loss(&balanced, vectors, 18, 2);

	assert(balanced.metadata.selected_restart < balanced.metadata.restart_count);
	assert(balanced_ratio < baseline_ratio);
	assert(balanced_loss <= (baseline_loss * 1.20f));

	tq_router_reset(&baseline);
	tq_router_reset(&balanced);
}

static void
test_empty_cluster_handling_still_works(void)
{
	const float	vectors[4] = {
		1.0f, 0.0f,
		0.0f, 1.0f
	};
	const float	query[2] = {1.0f, 0.0f};
	TqRouterTrainingConfig config = {
		.seed = 29,
		.sample_count = 2,
		.max_iterations = 4,
		.restart_count = 3
	};
	TqRouterModel model;
	uint32_t	probes[4] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
	uint32_t	selected = 0;
	char		errmsg[256];

	memset(&model, 0, sizeof(model));

	assert(tq_router_train_kmeans(vectors, 2, 2, 4, &config, &model, errmsg, sizeof(errmsg)));
	assert(model.metadata.restart_count == 3);
	assert(tq_router_select_probes(&model, query, 4, probes, 4, &selected, errmsg, sizeof(errmsg)));
	assert(selected == 4);

	tq_router_reset(&model);
}

int
main(void)
{
	test_seeded_restarts_are_deterministic();
	test_balanced_selection_reduces_max_list_ratio_on_skew_fixture();
	test_empty_cluster_handling_still_works();
	return 0;
}
