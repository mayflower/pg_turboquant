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
- `simd`

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
