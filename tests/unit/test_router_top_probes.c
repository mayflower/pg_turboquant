#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "src/tq_router.h"

static uint32_t
lcg_next(uint32_t *state)
{
	*state = (*state * UINT32_C(1664525)) + UINT32_C(1013904223);
	return *state;
}

static float
seeded_component(uint32_t *state)
{
	return ((float) (lcg_next(state) % 2001u) / 1000.0f) - 1.0f;
}

static void
fill_seeded_points(float *values,
					 uint32_t list_count,
					 uint32_t dimension,
					 uint32_t seed)
{
	uint32_t state = seed;
	uint32_t list_id = 0;

	for (list_id = 0; list_id < list_count; list_id++)
	{
		uint32_t dim = 0;

		for (dim = 0; dim < dimension; dim++)
			values[(size_t) list_id * (size_t) dimension + (size_t) dim] = seeded_component(&state);
	}
}

static void
assert_probe_prefix_matches(const TqRouterProbeScore *full_scores,
							  const TqRouterProbeScore *top_scores,
							  uint32_t probe_count)
{
	uint32_t index = 0;

	for (index = 0; index < probe_count; index++)
	{
		assert(top_scores[index].list_id == full_scores[index].list_id);
		assert(fabsf(top_scores[index].score - full_scores[index].score) < 1e-6f);
	}
}

static void
test_top_probe_prefix_matches_full_sort_on_seeded_fixture(void)
{
	const uint32_t dimension = 3;
	const uint32_t list_count = 8;
	const uint32_t probe_count = 3;
	float *centroids = (float *) calloc((size_t) list_count * (size_t) dimension, sizeof(float));
	const float query[3] = {0.3f, -0.7f, 0.5f};
	TqRouterModel model;
	TqRouterProbeScore *full_scores = (TqRouterProbeScore *) calloc(list_count, sizeof(TqRouterProbeScore));
	TqRouterProbeScore *top_scores = (TqRouterProbeScore *) calloc(probe_count, sizeof(TqRouterProbeScore));
	char errmsg[256];

	assert(centroids != NULL);
	assert(full_scores != NULL);
	assert(top_scores != NULL);

	memset(&model, 0, sizeof(model));
	fill_seeded_points(centroids, list_count, dimension, 17u);

	model.dimension = dimension;
	model.list_count = list_count;
	model.centroids = centroids;

	assert(tq_router_rank_probes(&model,
								 query,
								 full_scores,
								 list_count,
								 errmsg,
								 sizeof(errmsg)));
	assert(tq_router_rank_probes(&model,
								 query,
								 top_scores,
								 probe_count,
								 errmsg,
								 sizeof(errmsg)));
	assert_probe_prefix_matches(full_scores, top_scores, probe_count);

	free(centroids);
	free(full_scores);
	free(top_scores);
}

static void
test_top_probe_prefix_preserves_tie_break_by_list_id(void)
{
	const uint32_t dimension = 1;
	const uint32_t list_count = 4;
	const uint32_t probe_count = 2;
	float *centroids = (float *) calloc((size_t) list_count * (size_t) dimension, sizeof(float));
	const float query[1] = {0.0f};
	TqRouterModel model;
	TqRouterProbeScore *full_scores = (TqRouterProbeScore *) calloc(list_count, sizeof(TqRouterProbeScore));
	TqRouterProbeScore *top_scores = (TqRouterProbeScore *) calloc(probe_count, sizeof(TqRouterProbeScore));
	char errmsg[256];

	assert(centroids != NULL);
	assert(full_scores != NULL);
	assert(top_scores != NULL);

	memset(&model, 0, sizeof(model));

	centroids[0] = 1.0f;
	centroids[1] = -1.0f;
	centroids[2] = 1.0f;
	centroids[3] = -1.0f;

	model.dimension = dimension;
	model.list_count = list_count;
	model.centroids = centroids;

	assert(tq_router_rank_probes(&model,
								 query,
								 full_scores,
								 list_count,
								 errmsg,
								 sizeof(errmsg)));
	assert(tq_router_rank_probes(&model,
								 query,
								 top_scores,
								 probe_count,
								 errmsg,
								 sizeof(errmsg)));
	assert(top_scores[0].list_id == 0u);
	assert(top_scores[1].list_id == 1u);
	assert_probe_prefix_matches(full_scores, top_scores, probe_count);

	free(centroids);
	free(full_scores);
	free(top_scores);
}

static void
test_top_probe_prefix_matches_full_sort_across_seeded_randomized_inputs(void)
{
	uint32_t seed = 0;

	for (seed = 1; seed <= 64u; seed++)
	{
		uint32_t dimension = 1u + (seed % 7u);
		uint32_t list_count = 4u + (seed % 29u);
		uint32_t probe_count = 1u + (seed % list_count);
		float *centroids = (float *) calloc((size_t) list_count * (size_t) dimension, sizeof(float));
		float *query = (float *) calloc(dimension, sizeof(float));
		TqRouterProbeScore *full_scores = (TqRouterProbeScore *) calloc(list_count, sizeof(TqRouterProbeScore));
		TqRouterProbeScore *top_scores = (TqRouterProbeScore *) calloc(probe_count, sizeof(TqRouterProbeScore));
		TqRouterModel model;
		char errmsg[256];
		uint32_t dim = 0;

		assert(centroids != NULL);
		assert(query != NULL);
		assert(full_scores != NULL);
		assert(top_scores != NULL);

		memset(&model, 0, sizeof(model));
		fill_seeded_points(centroids, list_count, dimension, seed * 17u);
		{
			uint32_t state = seed * 97u;

			for (dim = 0; dim < dimension; dim++)
				query[dim] = seeded_component(&state);
		}

		model.dimension = dimension;
		model.list_count = list_count;
		model.centroids = centroids;

		assert(tq_router_rank_probes(&model,
									 query,
									 full_scores,
									 list_count,
									 errmsg,
									 sizeof(errmsg)));
		assert(tq_router_rank_probes(&model,
									 query,
									 top_scores,
									 probe_count,
									 errmsg,
									 sizeof(errmsg)));
		assert_probe_prefix_matches(full_scores, top_scores, probe_count);

		free(centroids);
		free(query);
		free(full_scores);
		free(top_scores);
	}
}

int
main(void)
{
	test_top_probe_prefix_matches_full_sort_on_seeded_fixture();
	test_top_probe_prefix_preserves_tie_break_by_list_id();
	test_top_probe_prefix_matches_full_sort_across_seeded_randomized_inputs();
	return 0;
}
