#include "src/tq_query_tuning.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef TQ_UNIT_TEST
#undef snprintf
#endif

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

typedef struct TqProbeSelectionCandidate
{
	size_t		ranked_index;
	double		score;
	double		utility;
	double		cost_fraction;
	size_t		live_count;
	size_t		page_count;
} TqProbeSelectionCandidate;

static double
tq_probe_score_weight(double score, double best_score, double worst_score)
{
	double		span = best_score - worst_score;

	if (!isfinite(score) || !isfinite(best_score) || !isfinite(worst_score))
		return 1.0;

	if (fabs(span) <= 1e-12)
		return 1.0;

	return 1.0 + ((score - worst_score) / span);
}

static double
tq_probe_cost_fraction(size_t live_count,
					   size_t page_count,
					   size_t code_budget,
					   size_t page_budget)
{
	double		fraction = 0.0;

	if (code_budget > 0)
		fraction = (double) live_count / (double) code_budget;

	if (page_budget > 0)
	{
		double page_fraction = (double) page_count / (double) page_budget;

		if (page_fraction > fraction)
			fraction = page_fraction;
	}

	return fraction;
}

static int
tq_probe_selection_candidate_compare(const void *left, const void *right)
{
	const TqProbeSelectionCandidate *lhs = (const TqProbeSelectionCandidate *) left;
	const TqProbeSelectionCandidate *rhs = (const TqProbeSelectionCandidate *) right;

	if (lhs->utility > rhs->utility)
		return -1;
	if (lhs->utility < rhs->utility)
		return 1;
	if (lhs->score > rhs->score)
		return -1;
	if (lhs->score < rhs->score)
		return 1;
	if (lhs->cost_fraction < rhs->cost_fraction)
		return -1;
	if (lhs->cost_fraction > rhs->cost_fraction)
		return 1;
	if (lhs->ranked_index < rhs->ranked_index)
		return -1;
	if (lhs->ranked_index > rhs->ranked_index)
		return 1;
	return 0;
}

static int
tq_probe_selected_index_compare(const void *left, const void *right)
{
	const size_t *lhs = (const size_t *) left;
	const size_t *rhs = (const size_t *) right;

	if (*lhs < *rhs)
		return -1;
	if (*lhs > *rhs)
		return 1;
	return 0;
}

static bool
tq_fraction_meets_near_exhaustive_threshold(size_t selected_count, size_t total_count)
{
	if (selected_count == 0 || total_count == 0)
		return false;

	return selected_count >= ((total_count * (size_t) 7) + (size_t) 9) / (size_t) 10;
}

bool
tq_should_use_near_exhaustive_scan(size_t selected_live_count,
								   size_t total_live_count,
								   size_t selected_page_count,
								   size_t total_page_count)
{
	return tq_fraction_meets_near_exhaustive_threshold(selected_live_count, total_live_count)
		|| tq_fraction_meets_near_exhaustive_threshold(selected_page_count, total_page_count);
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
tq_select_cost_aware_probes(const double *ranked_scores,
							const uint32_t *ranked_live_counts,
							const uint32_t *ranked_page_counts,
							size_t ranked_count,
							int nominal_probes,
							int max_visited_codes,
							int max_visited_pages,
							size_t *selected_indexes,
							size_t selected_capacity,
							size_t *selected_count,
							TqProbeBudgetResult *result,
							char *errmsg,
							size_t errmsg_len)
{
	size_t nominal_count = 0;
	size_t code_budget = 0;
	size_t page_budget = 0;
	size_t index = 0;
	size_t chosen_count = 0;

	if (ranked_live_counts == NULL
		|| ranked_page_counts == NULL
		|| selected_indexes == NULL
		|| selected_count == NULL
		|| result == NULL)
	{
		tq_tuning_set_error(errmsg, errmsg_len,
							"invalid turboquant probe selection: counts, outputs, and result must be non-null");
		return false;
	}

	if (nominal_probes <= 0)
	{
		tq_tuning_set_error(errmsg, errmsg_len,
							"invalid turboquant probe selection: nominal probes must be positive");
		return false;
	}

	if (max_visited_codes < 0 || max_visited_pages < 0)
	{
		tq_tuning_set_error(errmsg, errmsg_len,
							"invalid turboquant probe selection: visit budgets must be non-negative");
		return false;
	}

	memset(result, 0, sizeof(*result));
	*selected_count = 0;
	nominal_count = tq_positive_value(nominal_probes);
	if (nominal_count > ranked_count)
		nominal_count = ranked_count;
	if (selected_capacity < nominal_count)
	{
		tq_tuning_set_error(errmsg, errmsg_len,
							"invalid turboquant probe selection: selected index buffer is too small");
		return false;
	}

	code_budget = (size_t) max_visited_codes;
	page_budget = (size_t) max_visited_pages;
	result->nominal_probe_count = nominal_count;
	result->max_visited_codes = code_budget;
	result->max_visited_pages = page_budget;
	result->adaptive_enabled = tq_adaptive_probe_budget_enabled((unsigned int) ranked_count,
																max_visited_codes,
																max_visited_pages);

	if (!result->adaptive_enabled || nominal_count == 0)
	{
		for (index = 0; index < nominal_count; index++)
		{
			selected_indexes[index] = index;
			result->effective_probe_count += 1;
			result->selected_live_count += (size_t) ranked_live_counts[index];
			result->selected_page_count += (size_t) ranked_page_counts[index];
		}
		*selected_count = nominal_count;
		return true;
	}

	if (ranked_scores == NULL)
	{
		tq_tuning_set_error(errmsg, errmsg_len,
							"invalid turboquant probe selection: ranked scores are required when adaptive selection is enabled");
		return false;
	}

	{
		TqProbeSelectionCandidate *candidates = NULL;
		double best_score = ranked_scores[0];
		double worst_score = ranked_scores[0];
		size_t selected_live = 0;
		size_t selected_pages = 0;

		candidates = (TqProbeSelectionCandidate *) calloc(ranked_count, sizeof(TqProbeSelectionCandidate));
		if (candidates == NULL)
		{
			tq_tuning_set_error(errmsg, errmsg_len,
								"invalid turboquant probe selection: out of memory");
			return false;
		}

		for (index = 1; index < ranked_count; index++)
		{
			if (ranked_scores[index] > best_score)
				best_score = ranked_scores[index];
			if (ranked_scores[index] < worst_score)
				worst_score = ranked_scores[index];
		}

		for (index = 0; index < ranked_count; index++)
		{
			double score_weight = tq_probe_score_weight(ranked_scores[index], best_score, worst_score);
			double cost_fraction = tq_probe_cost_fraction((size_t) ranked_live_counts[index],
														 (size_t) ranked_page_counts[index],
														 code_budget,
														 page_budget);

			candidates[index].ranked_index = index;
			candidates[index].score = ranked_scores[index];
			candidates[index].cost_fraction = cost_fraction;
			candidates[index].live_count = (size_t) ranked_live_counts[index];
			candidates[index].page_count = (size_t) ranked_page_counts[index];
			candidates[index].utility = score_weight / (1.0 + cost_fraction);
		}

		qsort(candidates, ranked_count, sizeof(candidates[0]), tq_probe_selection_candidate_compare);

		for (index = 0; index < ranked_count && chosen_count < nominal_count; index++)
		{
			size_t next_live = candidates[index].live_count;
			size_t next_pages = candidates[index].page_count;
			bool fits_code = code_budget == 0 || selected_live + next_live <= code_budget;
			bool fits_pages = page_budget == 0 || selected_pages + next_pages <= page_budget;

			if (!fits_code || !fits_pages)
				continue;

			selected_indexes[chosen_count++] = candidates[index].ranked_index;
			selected_live += next_live;
			selected_pages += next_pages;
		}

		if (chosen_count == 0 && nominal_count > 0)
		{
			selected_indexes[0] = candidates[0].ranked_index;
			chosen_count = 1;
			selected_live = candidates[0].live_count;
			selected_pages = candidates[0].page_count;
		}

		qsort(selected_indexes, chosen_count, sizeof(selected_indexes[0]), tq_probe_selected_index_compare);
		for (index = 0; index < chosen_count; index++)
		{
			result->effective_probe_count += 1;
			result->selected_live_count += (size_t) ranked_live_counts[selected_indexes[index]];
			result->selected_page_count += (size_t) ranked_page_counts[selected_indexes[index]];
		}
		*selected_count = chosen_count;
		free(candidates);
	}

	return true;
}

bool
tq_estimate_ordered_scan_cost(double index_pages,
							  double index_tuples,
							  double output_rows,
							  double qual_selectivity,
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
	double		filter_selectivity = 1.0;
	double		scanned_fraction = 1.0;
	double		scanned_tuples = 0.0;
	double		candidate_bound = 0.0;
	double		required_candidates = 0.0;
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
	if (qual_selectivity > 0.0 && isfinite(qual_selectivity))
		filter_selectivity = qual_selectivity;
	if (filter_selectivity <= 0.0)
		filter_selectivity = 1.0 / tuples;
	if (filter_selectivity > 1.0)
		filter_selectivity = 1.0;

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
	required_candidates = ceil(((output_rows > 1.0) ? output_rows : 1.0) / filter_selectivity);
	candidate_bound = (double) tq_streaming_candidate_capacity(probes, oversample_factor);
	if (required_candidates > candidate_bound)
		candidate_bound = required_candidates;
	if (candidate_bound > scanned_tuples)
		candidate_bound = scanned_tuples;
	pages_fetched = pages * scanned_fraction;
	if (pages_fetched < 1.0)
		pages_fetched = 1.0;

	heap_fetches = candidate_bound;
	if (heap_fetches < 1.0)
		heap_fetches = 1.0;

	if (list_count > 0)
		router_penalty = 3.0
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
	estimate->qual_selectivity = filter_selectivity;
	estimate->visited_tuples = scanned_tuples;
	estimate->effective_probe_count = effective_probe_count;
	return true;
}
