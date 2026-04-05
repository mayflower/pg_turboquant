# Install and use it in PostgreSQL

This guide covers the practical PostgreSQL workflow after the extension has been built and installed into the server with `make install`.

## 1. Enable the extensions

Connect to the target database and enable pgvector plus `pg_turboquant`:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;
```

Confirm that PostgreSQL can see the access method:

```sql
SELECT amname
FROM pg_am
WHERE amname = 'turboquant';
```

## 2. Create a table with embeddings

This example uses a small `vector(4)` column so the queries are easy to read:

```sql
CREATE TABLE docs (
  id bigint PRIMARY KEY,
  title text NOT NULL,
  embedding vector(4) NOT NULL
);

INSERT INTO docs (id, title, embedding) VALUES
  (1, 'alpha', '[1,0,0,0]'),
  (2, 'beta',  '[0.9,0.1,0,0]'),
  (3, 'gamma', '[0,1,0,0]'),
  (4, 'delta', '[0,0,1,0]');
```

## 3. Create a TurboQuant index

For normalized cosine embeddings, start with a flat index:

```sql
CREATE INDEX docs_embedding_tq_idx
ON docs
USING turboquant (embedding tq_cosine_ops)
WITH (
  bits = 4,
  lists = 0,
  transform = 'hadamard',
  normalized = true
);
```

Use `lists = 0` for flat mode. Move to IVF mode later by choosing `lists > 0`.

Example IVF index:

```sql
CREATE INDEX docs_embedding_tq_ivf_idx
ON docs
USING turboquant (embedding tq_cosine_ops)
WITH (
  bits = 4,
  lists = 128,
  transform = 'hadamard',
  normalized = true
);
```

Only set `normalized = true` when the stored vectors are already unit-normalized for cosine search.

## 4. Run an approximate nearest-neighbor query

The normal PostgreSQL query shape is an ordered `LIMIT` query over the embedding operator:

```sql
SELECT id, title, embedding <=> '[1,0,0,0]'::vector(4) AS distance
FROM docs
ORDER BY embedding <=> '[1,0,0,0]'::vector(4), id
LIMIT 3;
```

For IVF indexes, tune the scan breadth per session:

```sql
SET turboquant.probes = 4;
SET turboquant.oversample_factor = 4;
```

You can inspect the plan with:

```sql
EXPLAIN (COSTS OFF)
SELECT id, title
FROM docs
ORDER BY embedding <=> '[1,0,0,0]'::vector(4), id
LIMIT 3;
```

## 5. Use SQL-side exact reranking

`pg_turboquant` keeps approximate retrieval in the index and exact reranking in SQL. The helper below returns candidates reranked exactly against the original vectors:

```sql
SELECT *
FROM tq_rerank_candidates(
  'docs'::regclass,
  'id',
  'embedding',
  '[1,0,0,0]'::vector(4),
  'cosine',
  20,
  5
);
```

If you want the raw approximate candidate set without exact reranking:

```sql
SELECT *
FROM tq_approx_candidates(
  'docs'::regclass,
  'id',
  'embedding',
  '[1,0,0,0]'::vector(4),
  'cosine',
  20,
  5
);
```

## 6. Inspect the index

The extension exposes stable JSON metadata for debugging and operations:

```sql
SELECT tq_index_metadata('docs_embedding_tq_idx'::regclass);
```

This includes the format version, codec, transform, routing mode, live and dead counts, page counts, and capability flags.

You can also inspect runtime SIMD selection:

```sql
SELECT tq_runtime_simd_features();
```

## 7. Maintain the index

Normal PostgreSQL maintenance still applies:

```sql
VACUUM ANALYZE docs;
```

TurboQuant also exposes a lightweight maintenance entrypoint for the built-in delta tier and maintenance counters:

```sql
SELECT tq_maintain_index('docs_embedding_tq_idx'::regclass);
```

If you need a full rebuild after heavy churn or configuration changes, rebuild with:

```sql
REINDEX INDEX docs_embedding_tq_idx;
```

## Notes

- Use `tq_cosine_ops`, `tq_ip_ops`, or `tq_l2_ops` to match the metric you query with.
- Use the `tq_halfvec_*_ops` opclasses when the column type is `halfvec`.
- Fixed-width metadata and payload columns are supported on multicolumn TurboQuant indexes; varlena / text metadata predicates are still outside the ANN fast path.
- Internal heap reranking remains outside the access method.
