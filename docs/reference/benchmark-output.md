# Benchmark output schema

The canonical benchmark driver is `scripts/benchmark_suite.py`. Its JSON output is intentionally explicit about environment, scenario matrix, query API, SIMD, and index capabilities.

## Top-level fields

- `profile`
- `corpora`
- `methods`
- `environment`
- `scenario_matrix`
- `scenarios`
- `report`
- `artifacts`

## Scenario fields

- `corpus`
- `method`
- `query_mode`
- `ground_truth`
- `metrics`
- `index`
- `query_knobs`
- `query_api`
- `index_metadata`
- `ordered_ios_observation`
- `simd`
- `scan_stats`

## Key metric fields

- `recall_at_10`
- `recall_at_100`
- `p50_ms`
- `p95_ms`
- `build_seconds`
- `index_size_bytes`
- `candidate_slots_bound`
- `build_wal_bytes`
- `insert_wal_bytes`
- `maintenance_wal_bytes`
- `concurrent_insert_rows_per_second`
- `sealed_baseline_build_wal_bytes`

## Capability flags

`index_metadata.capabilities` currently reports:

- `ordered_scan`
- `bitmap_scan`
- `index_only_scan`
- `multicolumn`
- `include_columns`

## Query API metadata

The suite records which helper was used, for example:

- `tq_rerank_candidates`
- `tq_bitmap_cosine_filter`

It also records candidate and final limits so query shape can be reproduced exactly.

## Scan observability metadata

`scan_stats` records backend-local work counters captured from `tq_last_scan_stats()` after the benchmark query. Important fields currently include:

- `mode`
- `score_mode`
- `configured_probe_count`
- `nominal_probe_count`
- `effective_probe_count`
- `max_visited_codes`
- `max_visited_pages`
- `selected_list_count`
- `selected_live_count`
- `visited_page_count`
- `visited_code_count`
- `candidate_heap_count`
- `decoded_vector_count`
- `page_prune_count`
- `early_stop_count`

## Ordered IOS observation

For ordered TurboQuant scenarios, `ordered_ios_observation` records the observed planner/runtime shape for covered vector-key queries under `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)`.

Important fields include:

- `capability_claimed`
- `explain_analyze_buffers_captured`
- `observed_plan_node_type`
- `observed_index_only_scan`
- `sample_count`
- `query_sample_count`
- `query_sample_indexes`
- `samples`
- `plan_json_samples`
- `heap_fetch_samples`
- `heap_fetch_min`
- `heap_fetch_max`
- `visibility_map_context`

Each `samples` entry is a compact per-`EXPLAIN` record with:

- `query_index`
- `repetition`
- `observed_plan_node_type`
- `observed_index_only_scan`
- `heap_fetches`

Each `plan_json_samples` entry carries the matching:

- `query_index`
- `repetition`
- `plan_json`

`visibility_map_context` captures the heap visibility state used to interpret heap-fetch behavior:

- `captured`
- `source`
- `heap_relation`
- `heap_relpages`
- `heap_relallvisible`
- `heap_all_visible_fraction`
- `vm_all_visible_pages`
- `vm_all_frozen_pages`
- `vm_pages`

When available, the harness prefers `pg_visibility_map(...)` for page-level evidence and sets `source` to `pg_visibility`, while still retaining summary counters. If only `pg_visibility_map_summary(...)` is available, it uses that summary; otherwise it falls back to `pg_class` visibility counters.

The nested `filtered_query` object uses the same shape for a measurement-only filtered ordered query so filtered and unfiltered IOS behavior can be compared separately.
