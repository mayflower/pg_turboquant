#include "src/tq_query_tuning.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static size_t
tq_positive_value(int value)
{
	if (value <= 0)
		return 1;

	return (size_t) value;
}

static void
tq_tuning_set_error(char *errmsg, size_t errmsg_len, const char *message)
{
	if (errmsg == NULL || errmsg_len == 0)
		return;

	snprintf(errmsg, errmsg_len, "%s", message);
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

static double
tq_effective_ivf_probe_count(double tuples,
							   double pages,
							   unsigned int list_count,
							   int probes,
							   int max_visited_codes,
							   int max_visited_pages)
{
	double		nominal_probe_count = 1.0;
	double		effective_probe_count = 1.0;
	double		tuples_per_list = 1.0;
	double		pages_per_list = 1.0;
	double		code_limited_count = 0.0;
	double		page_limited_count = 0.0;

	if (list_count == 0)
		return 1.0;

	nominal_probe_count = (double) tq_positive_value(probes);
	if (nominal_probe_count > (double) list_count)
		nominal_probe_count = (double) list_count;
	effective_probe_count = nominal_probe_count;
	tuples_per_list = tq_positive_double(tuples / (double) list_count, 1.0);
	pages_per_list = tq_positive_double(pages / (double) list_count, 1.0);

	if (max_visited_codes > 0)
	{
		code_limited_count = ceil((double) max_visited_codes / tuples_per_list);
		if (code_limited_count < 1.0)
			code_limited_count = 1.0;
		if (code_limited_count < effective_probe_count)
			effective_probe_count = code_limited_count;
	}

	if (max_visited_pages > 0)
	{
		page_limited_count = ceil((double) max_visited_pages / pages_per_list);
		if (page_limited_count < 1.0)
			page_limited_count = 1.0;
		if (page_limited_count < effective_probe_count)
			effective_probe_count = page_limited_count;
	}

	if (effective_probe_count > nominal_probe_count)
		effective_probe_count = nominal_probe_count;
	if (effective_probe_count < 1.0)
		effective_probe_count = 1.0;

	return effective_probe_count;
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
tq_adaptive_probe_budget_enabled(unsigned int list_count,
								 int max_visited_codes,
								 int max_visited_pages)
{
	return list_count > 0 && (max_visited_codes > 0 || max_visited_pages > 0);
}

bool
tq_choose_probe_budget(const uint32_t *ranked_live_counts,
					   const uint32_t *ranked_page_counts,
					   size_t ranked_count,
					   int nominal_probes,
					   int max_visited_codes,
					   int max_visited_pages,
					   TqProbeBudgetResult *result,
					   char *errmsg,
					   size_t errmsg_len)
{
	size_t		nominal_count = 0;
	size_t		code_budget = 0;
	size_t		page_budget = 0;
	size_t		index = 0;

	if (ranked_live_counts == NULL || result == NULL)
	{
		tq_tuning_set_error(errmsg, errmsg_len,
							"invalid turboquant probe budgeting: live counts and result must be non-null");
		return false;
	}

	if (nominal_probes <= 0)
	{
		tq_tuning_set_error(errmsg, errmsg_len,
							"invalid turboquant probe budgeting: nominal probes must be positive");
		return false;
	}

	if (max_visited_codes < 0 || max_visited_pages < 0)
	{
		tq_tuning_set_error(errmsg, errmsg_len,
							"invalid turboquant probe budgeting: visit budgets must be non-negative");
		return false;
	}

	if (max_visited_pages > 0 && ranked_page_counts == NULL)
	{
		tq_tuning_set_error(errmsg, errmsg_len,
							"invalid turboquant probe budgeting: page counts are required when a page budget is configured");
		return false;
	}

	memset(result, 0, sizeof(*result));
	nominal_count = tq_positive_value(nominal_probes);
	if (nominal_count > ranked_count)
		nominal_count = ranked_count;

	code_budget = (size_t) max_visited_codes;
	page_budget = (size_t) max_visited_pages;
	result->nominal_probe_count = nominal_count;
	result->max_visited_codes = code_budget;
	result->max_visited_pages = page_budget;
	result->adaptive_enabled = tq_adaptive_probe_budget_enabled((unsigned int) ranked_count,
																max_visited_codes,
																max_visited_pages);

	for (index = 0; index < nominal_count; index++)
	{
		size_t next_live = (size_t) ranked_live_counts[index];
		size_t next_pages = ranked_page_counts != NULL ? (size_t) ranked_page_counts[index] : 0;

		if (index > 0 && result->adaptive_enabled)
		{
			if ((code_budget > 0 && result->selected_live_count >= code_budget)
				|| (page_budget > 0 && result->selected_page_count >= page_budget)
				|| (code_budget > 0 && result->selected_live_count + next_live > code_budget)
				|| (page_budget > 0 && result->selected_page_count + next_pages > page_budget))
				break;
		}

		result->effective_probe_count += 1;
		result->selected_live_count += next_live;
		result->selected_page_count += next_pages;
	}

	return true;
}

bool
tq_estimate_ordered_scan_cost(double index_pages,
							  double index_tuples,
							  double output_rows,
							  unsigned int list_count,
							  int probes,
							  int oversample_factor,
							  int max_visited_codes,
							  int max_visited_pages,
							  double cpu_index_tuple_cost,
							  double cpu_operator_cost,
							  double random_page_cost,
							  double cpu_tuple_cost,
							  TqPlannerCostEstimate *estimate)
{
	double		tuples = tq_positive_double(index_tuples, 1.0);
	double		pages = tq_positive_double(index_pages, 1.0);
	double		oversample = (double) tq_positive_value(oversample_factor);
	double		selectivity = 1.0;
	double		scanned_fraction = 1.0;
	double		scanned_tuples = 0.0;
	double		candidate_bound = 0.0;
	double		pages_fetched = 0.0;
	double		effective_probe_count = 1.0;
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
		effective_probe_count = tq_effective_ivf_probe_count(tuples,
															 pages,
															 list_count,
															 probes,
															 max_visited_codes,
															 max_visited_pages);
		scanned_fraction = effective_probe_count / (double) list_count;
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
		router_penalty = 2.0
			+ ((double) list_count * 0.10)
			+ (effective_probe_count * cpu_operator_cost * 0.5);

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
	estimate->visited_tuples = scanned_tuples;
	estimate->effective_probe_count = effective_probe_count;
	return true;
}
