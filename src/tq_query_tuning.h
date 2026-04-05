#ifndef TQ_QUERY_TUNING_H
#define TQ_QUERY_TUNING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define TQ_DEFAULT_PROBES 8
#define TQ_DEFAULT_OVERSAMPLE_FACTOR 8
#define TQ_MIN_TUNING_VALUE 1
#define TQ_MAX_TUNING_VALUE 1024

typedef struct TqPlannerCostEstimate
{
	double		startup_cost;
	double		total_cost;
	double		scanned_fraction;
	double		pages_fetched;
	double		candidate_bound;
	double		selectivity;
	double		qual_selectivity;
	double		visited_tuples;
	double		effective_probe_count;
} TqPlannerCostEstimate;

typedef struct TqProbeBudgetResult
{
	size_t		nominal_probe_count;
	size_t		effective_probe_count;
	size_t		selected_live_count;
	size_t		selected_page_count;
	size_t		max_visited_codes;
	size_t		max_visited_pages;
	bool		adaptive_enabled;
} TqProbeBudgetResult;

extern bool tq_should_use_near_exhaustive_scan(size_t selected_live_count,
											   size_t total_live_count,
											   size_t selected_page_count,
											   size_t total_page_count);
extern size_t tq_scan_candidate_capacity(size_t live_count,
										 int probes,
										 int oversample_factor);
extern size_t tq_streaming_candidate_capacity(int probes,
											 int oversample_factor);
extern double tq_scan_cost_multiplier(int probes, int oversample_factor);
extern bool tq_adaptive_probe_budget_enabled(unsigned int list_count,
											 int max_visited_codes,
											 int max_visited_pages);
extern bool tq_choose_probe_budget(const uint32_t *ranked_live_counts,
								   const uint32_t *ranked_page_counts,
								   size_t ranked_count,
								   int nominal_probes,
								   int max_visited_codes,
								   int max_visited_pages,
								   TqProbeBudgetResult *result,
								   char *errmsg,
								   size_t errmsg_len);
extern bool tq_select_cost_aware_probes(const double *ranked_scores,
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
										size_t errmsg_len);
extern bool tq_estimate_ordered_scan_cost(double index_pages,
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
										  TqPlannerCostEstimate *estimate);

#endif
