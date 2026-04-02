# SQL API

This page summarizes the SQL-visible surface that matters for operating and benchmarking the extension.

## Access method

```sql
USING turboquant
```

## Supported opclasses

### `vector`

- `tq_cosine_ops`
- `tq_ip_ops`
- `tq_l2_ops`

### `halfvec`

- `tq_halfvec_cosine_ops`
- `tq_halfvec_ip_ops`
- `tq_halfvec_l2_ops`

## Helper functions

### `tq_rerank_candidates(...)`

Returns approximate candidates reranked exactly in SQL over the candidate set.

Typical call:

```sql
SELECT *
FROM tq_rerank_candidates(
  'docs'::regclass,
  'id',
  'embedding',
  '[1,0,0,0]'::vector(4),
  'cosine',
  50,
  10
);
```

### `tq_approx_candidates(...)`

Returns the approximate candidate set without exact reranking.

### `tq_recommended_query_knobs(candidate_limit, final_limit)`

Returns recommended query settings for the requested candidate pool and final limit.

The helper keeps the SQL surface index-agnostic: it recommends `probes` and `oversample_factor`, then derives a generic `max_visited_codes` budget from the resolved oversampling so skewed IVF workloads can be costed and bounded by predicted visited work instead of raw `probes / lists`.

The current return columns are:

- `probes`
- `oversample_factor`
- `max_visited_codes`
- `max_visited_pages`

### `tq_bitmap_cosine_filter(...)`

Bitmap-oriented helper used for filtered cosine workloads where ordered ANN scans are not the only plan shape.

### `tq_index_metadata(regclass)`

Returns JSON metadata for a TurboQuant index, including:

- format version
- metric and opclass
- transform metadata
- router metadata
- live/dead counts
- page counts
- capability flags
- cheap heap estimate fields

### `tq_index_heap_stats(regclass)`

Returns exact heap statistics for a TurboQuant index.

This helper is intentionally expensive. Use it when you want an exact heap row count and do not want that cost hidden behind the normal metadata API.

### `tq_runtime_simd_features()`

Returns the compiled and runtime-visible SIMD surface and selected score kernel.

### `tq_last_scan_stats()`

Returns backend-local JSON for the most recent TurboQuant scan in the current session.

The current JSON includes:

Probe budget and scan work:

- `configured_probe_count`, `nominal_probe_count`, `effective_probe_count`
- `max_visited_codes`, `max_visited_pages`
- `selected_list_count`, `selected_live_count`, `selected_page_count`
- `visited_page_count`, `visited_code_count`
- `candidate_heap_capacity`, `candidate_heap_count`, `candidate_heap_insert_count`, `candidate_heap_reject_count`, `candidate_heap_replace_count`

Scoring and pruning:

- `score_mode` — `code_domain` (faithful fast path) or `decode_score` (fallback)
- `score_kernel` — `avx2`, `neon`, or `scalar`
- `scan_orchestration` — `ivf_bounded_pages` or `ivf_near_exhaustive` (IVF only)
- `page_bound_mode` — `safe_summary_pruning` or `none`
- `page_prune_count`, `early_stop_count`
- `near_exhaustive_crossover` — boolean, whether scan crossed the near-exhaustive threshold
- `decoded_vector_count` — zero on the faithful fast path

### `tq_last_shadow_decode_candidate_tids()`

Returns backend-local shadow decode candidate CTIDs for the most recent TurboQuant scan in the current session.

This is an expert diagnostic used by the benchmark harness. The raw `_core` helper is not part of the public API.

## Supported plan shapes

- ordered ANN index scans
- bitmap support with explicit heap recheck semantics

## Explicitly unsupported in the current release

- index-only scans
- multicolumn turboquant indexes
- `INCLUDE` columns
- internal heap reranking inside the access method
