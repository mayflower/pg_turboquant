#ifndef TQ_QUERY_TUNING_H
#define TQ_QUERY_TUNING_H

#include <stdbool.h>
#include <stddef.h>

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
} TqPlannerCostEstimate;

extern size_t tq_scan_candidate_capacity(size_t live_count,
										 int probes,
										 int oversample_factor);
extern size_t tq_streaming_candidate_capacity(int probes,
											 int oversample_factor);
extern double tq_scan_cost_multiplier(int probes, int oversample_factor);
extern bool tq_estimate_ordered_scan_cost(double index_pages,
										  double index_tuples,
										  double output_rows,
										  unsigned int list_count,
										  int probes,
										  int oversample_factor,
										  double cpu_index_tuple_cost,
										  double cpu_operator_cost,
										  double random_page_cost,
										  double cpu_tuple_cost,
										  TqPlannerCostEstimate *estimate);

#endif
