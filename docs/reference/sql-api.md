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

### `tq_runtime_simd_features()`

Returns the compiled and runtime-visible SIMD surface and selected score kernel.

## Supported plan shapes

- ordered ANN index scans
- bitmap support with explicit heap recheck semantics

## Explicitly unsupported in the current release

- index-only scans
- multicolumn turboquant indexes
- `INCLUDE` columns
- internal heap reranking inside the access method
