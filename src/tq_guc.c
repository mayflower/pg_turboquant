#include "postgres.h"

#include <limits.h>

#ifndef TQ_UNIT_TEST
#include "utils/guc.h"
#endif

#include "src/tq_guc.h"
#include "src/tq_query_tuning.h"

int			tq_guc_probes = TQ_DEFAULT_PROBES;
int			tq_guc_oversample_factor = TQ_DEFAULT_OVERSAMPLE_FACTOR;
int			tq_guc_max_visited_codes = 0;
int			tq_guc_max_visited_pages = 0;
int			tq_guc_iterative_scan = 0;
int			tq_guc_min_rows_after_filter = 0;
int			tq_guc_delta_merge_live_count_threshold = 256;
int			tq_guc_delta_merge_page_count_threshold = 8;
int			tq_guc_delta_merge_live_percent_threshold = 10;
int			tq_guc_maintenance_dead_tuple_percent_threshold = 10;
int			tq_guc_maintenance_reclaimable_page_threshold = 4;
int			tq_guc_decode_rescore_factor = 1;
int			tq_guc_decode_rescore_extra_candidates = -1;
bool		tq_guc_enable_summary_bounds = true;
bool		tq_guc_shadow_decode_diagnostics = false;
bool		tq_guc_force_decode_score_diagnostics = false;

void		_PG_init(void);

#ifndef TQ_UNIT_TEST
static const struct config_enum_entry tq_iterative_scan_options[] = {
	{"off", 0, false},
	{"strict_order", 1, false},
	{"relaxed_order", 2, false},
	{NULL, 0, false}
};

void
_PG_init(void)
{
	DefineCustomIntVariable("turboquant.probes",
							"TurboQuant query breadth budget.",
							"Flat mode uses probes with oversample_factor to bound candidate retention; IVF mode will use it for list probes.",
							&tq_guc_probes,
							TQ_DEFAULT_PROBES,
							TQ_MIN_TUNING_VALUE,
							TQ_MAX_TUNING_VALUE,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("turboquant.oversample_factor",
							"TurboQuant candidate oversampling multiplier.",
							"Controls how many approximate candidates are retained relative to the breadth budget.",
							&tq_guc_oversample_factor,
							TQ_DEFAULT_OVERSAMPLE_FACTOR,
							TQ_MIN_TUNING_VALUE,
							TQ_MAX_TUNING_VALUE,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("turboquant.max_visited_codes",
							"TurboQuant IVF code-visit budget.",
							"When greater than zero, IVF probing stops adding lower-ranked lists once their cumulative live-code work would exceed this budget, except for the first mandatory list.",
							&tq_guc_max_visited_codes,
							0,
							0,
							INT_MAX,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("turboquant.max_visited_pages",
							"TurboQuant IVF page-visit budget.",
							"When greater than zero, IVF probing also stops adding lower-ranked lists once their cumulative page count would exceed this budget, except for the first mandatory list.",
							&tq_guc_max_visited_pages,
							0,
							0,
							INT_MAX,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomEnumVariable("turboquant.iterative_scan",
							 "TurboQuant filtered ordered-scan completion policy.",
							 "When enabled for filtered IVF scans, TurboQuant keeps expanding lower-ranked lists in router order until enough rows survive the filters or the work budget is exhausted.",
							 &tq_guc_iterative_scan,
							 0,
							 tq_iterative_scan_options,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomIntVariable("turboquant.min_rows_after_filter",
							"TurboQuant minimum surviving rows target for filtered IVF scans.",
							"When iterative filtered completion is enabled, TurboQuant continues probing until at least this many rows survive the metadata filters, or until the configured work budget is exhausted.",
							&tq_guc_min_rows_after_filter,
							0,
							0,
							INT_MAX,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("turboquant.delta_merge_live_count_threshold",
							"TurboQuant recommended delta-merge live-row threshold.",
							"When the built-in delta tier reaches at least this many live rows, tq_index_metadata marks delta merge as recommended.",
							&tq_guc_delta_merge_live_count_threshold,
							256,
							1,
							INT_MAX,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("turboquant.delta_merge_page_count_threshold",
							"TurboQuant recommended delta-merge page-depth threshold.",
							"When the built-in delta tier reaches at least this many batch pages, tq_index_metadata marks delta merge as recommended.",
							&tq_guc_delta_merge_page_count_threshold,
							8,
							1,
							INT_MAX,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("turboquant.delta_merge_live_percent_threshold",
							"TurboQuant recommended delta-merge live-fraction threshold.",
							"When the built-in delta tier holds at least this percentage of live rows, tq_index_metadata marks delta merge as recommended.",
							&tq_guc_delta_merge_live_percent_threshold,
							10,
							1,
							100,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("turboquant.maintenance_dead_tuple_percent_threshold",
							"TurboQuant recommended dead-tuple compaction threshold.",
							"When dead rows reach at least this percentage of stored tuples, tq_index_metadata marks compaction as recommended.",
							&tq_guc_maintenance_dead_tuple_percent_threshold,
							10,
							1,
							100,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("turboquant.maintenance_reclaimable_page_threshold",
							"TurboQuant recommended reclaimable-page compaction threshold.",
							"When at least this many reclaimable pages accumulate, tq_index_metadata marks compaction as recommended.",
							&tq_guc_maintenance_reclaimable_page_threshold,
							4,
							1,
							INT_MAX,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("turboquant.decode_rescore_factor",
							"TurboQuant internal decode rescoring expansion factor.",
							"When greater than one on normalized tq_prod ordered scans, TurboQuant retains a broader code-domain preheap and then rescoring only those candidates with decoded-vector distances before the final candidate cut.",
							&tq_guc_decode_rescore_factor,
							1,
							1,
							64,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomIntVariable("turboquant.decode_rescore_extra_candidates",
							"TurboQuant SQL rerank boundary-band expansion.",
							"When decode rescoring is enabled, values above zero retain this many extra approximate candidates before SQL exact reranking; -1 enables the built-in auto band and 0 disables any extra band.",
							&tq_guc_decode_rescore_extra_candidates,
							-1,
							-1,
							INT_MAX,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomBoolVariable("turboquant.enable_summary_bounds",
							 "Use persistent per-page summary side structures for IVF optimistic bounds.",
							 "When enabled, TurboQuant derives bound ordering and pruning from persisted summary pages instead of pre-reading each selected batch data page.",
							 &tq_guc_enable_summary_bounds,
							 true,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomBoolVariable("turboquant.shadow_decode_diagnostics",
							 "Record a decode-scored shadow heap alongside the active code-domain heap.",
							 "Diagnostic-only setting for benchmarking and debugging code-domain ranking quality; when enabled, TurboQuant keeps a second candidate heap scored from decoded vectors in the same scan path.",
							 &tq_guc_shadow_decode_diagnostics,
							 false,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomBoolVariable("turboquant.force_decode_score_diagnostics",
							 "Use decode-scored active ranking instead of code-domain ranking.",
							 "Diagnostic-only setting for benchmarking and debugging code-domain ranking quality; when enabled, the primary TurboQuant candidate heap is filled from decoded-vector scores even on code-domain-capable normalized tq_prod scans.",
							 &tq_guc_force_decode_score_diagnostics,
							 false,
							 PGC_USERSET,
							 0,
							 NULL,
							 NULL,
							 NULL);
}
#else
void
_PG_init(void)
{
}
#endif
