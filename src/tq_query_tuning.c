#include "src/tq_query_tuning.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>

static size_t
tq_positive_value(int value)
{
	if (value <= 0)
		return 1;

	return (size_t) value;
}

size_t
tq_streaming_candidate_capacity(int probes, int oversample_factor)
{
	size_t		probe_count = tq_positive_value(probes);
	size_t		oversample = tq_positive_value(oversample_factor);

	if (probe_count > (SIZE_MAX / oversample))
		return SIZE_MAX;

	return probe_count * oversample;
}

static double
tq_positive_double(double value, double fallback)
{
	if (value <= 0.0 || !isfinite(value))
		return fallback;

	return value;
}

size_t
tq_scan_candidate_capacity(size_t live_count, int probes, int oversample_factor)
{
	size_t		capacity = tq_streaming_candidate_capacity(probes, oversample_factor);

	if (live_count == 0)
		return 0;

	if (capacity == 0)
		capacity = 1;

	if (capacity > live_count)
		capacity = live_count;

	return capacity;
}

double
tq_scan_cost_multiplier(int probes, int oversample_factor)
{
	size_t		probe_count = tq_positive_value(probes);
	size_t		oversample = tq_positive_value(oversample_factor);

	return (double) (probe_count * oversample);
}

bool
tq_estimate_ordered_scan_cost(double index_pages,
							  double index_tuples,
							  double output_rows,
							  unsigned int list_count,
							  int probes,
							  int oversample_factor,
							  double cpu_index_tuple_cost,
							  double cpu_operator_cost,
							  double random_page_cost,
							  double cpu_tuple_cost,
							  TqPlannerCostEstimate *estimate)
{
	double		tuples = tq_positive_double(index_tuples, 1.0);
	double		pages = tq_positive_double(index_pages, 1.0);
	double		probe_count = (double) tq_positive_value(probes);
	double		oversample = (double) tq_positive_value(oversample_factor);
	double		selectivity = 1.0;
	double		scanned_fraction = 1.0;
	double		scanned_tuples = 0.0;
	double		candidate_bound = 0.0;
	double		pages_fetched = 0.0;
	double		startup_cost = 0.0;
	double		total_cost = 0.0;
	double		heap_fetches = 0.0;
	double		router_penalty = 0.0;

	if (estimate == NULL)
		return false;

	if (output_rows > 0.0 && isfinite(output_rows))
		selectivity = output_rows / tuples;
	if (selectivity <= 0.0)
		selectivity = 1.0 / tuples;
	if (selectivity > 1.0)
		selectivity = 1.0;

	if (list_count > 0)
	{
		scanned_fraction = probe_count / (double) list_count;
		if (scanned_fraction > 1.0)
			scanned_fraction = 1.0;
	}

	scanned_tuples = tuples * scanned_fraction;
	candidate_bound = (double) tq_streaming_candidate_capacity(probes, oversample_factor);
	if (candidate_bound > scanned_tuples)
		candidate_bound = scanned_tuples;
	pages_fetched = pages * scanned_fraction;
	if (pages_fetched < 1.0)
		pages_fetched = 1.0;

	heap_fetches = candidate_bound * selectivity;
	if (heap_fetches < 1.0)
		heap_fetches = 1.0;

	if (list_count > 0)
		router_penalty = 2.0 + ((double) list_count * 0.10);

	startup_cost = 1.0 + router_penalty + (candidate_bound * cpu_operator_cost * 0.5);
	total_cost = startup_cost
		+ (pages_fetched * random_page_cost * 0.15)
		+ (scanned_tuples * (cpu_index_tuple_cost + (cpu_operator_cost * 2.0)))
		+ (heap_fetches * (cpu_tuple_cost + (random_page_cost * 0.05)))
		+ (oversample * cpu_operator_cost * 0.25);

	estimate->startup_cost = startup_cost;
	estimate->total_cost = total_cost;
	estimate->scanned_fraction = scanned_fraction;
	estimate->pages_fetched = pages_fetched;
	estimate->candidate_bound = candidate_bound;
	estimate->selectivity = selectivity;
	return true;
}
