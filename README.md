# pg_turboquant

`pg_turboquant` is a PostgreSQL extension that adds a custom ANN index access method, `turboquant`, for compact nearest-neighbor search over `vector` and `halfvec`.

It is designed around PostgreSQL's storage and executor constraints rather than treating ANN indexing as an external service. The project combines structured transforms, a faithful TurboQuant `v2` payload for normalized cosine and inner-product retrieval, lane-adaptive row-major batch pages, ordered ANN scans, bitmap support for filtered workloads, SQL-side exact reranking helpers, and a reproducible benchmark harness with machine-readable microbench regression gates.

## Why it exists

- PostgreSQL users already storing embeddings with pgvector often want a denser index format.
- `pg_turboquant` keeps ANN inside PostgreSQL while optimizing for compact storage and cache-friendly scoring.
- The access method is explicit about PostgreSQL boundaries: MVCC still lives in the executor, exact reranking still lives in SQL, and v1 still uses generic WAL.

## Algorithm contract

- Faithful fast path:
  normalized cosine and inner-product retrieval use the paper-faithful `Qprod` payload with structured rotation, `b - 1` stage-1 scalar codes, a residual 1-bit QJL sketch, and stored residual norm `gamma`.
- Compatibility fallback:
  L2 and non-normalized scans still work, but they fall back to decoded-vector scoring rather than claiming faithful TurboQuant semantics.
- Rebuild boundary:
  the `v2` rewrite is a format bump. Older indexes must be rebuilt with `REINDEX` or recreated.

## Current capabilities

- Custom access method: `USING turboquant`
- Input types: `vector`, `halfvec`
- Metrics: cosine, inner product, L2
- Modes:
  - flat scan with `lists = 0`
  - IVF-routed scan with `lists > 0`
  - bitmap-filter support for predicate-heavy workloads
- Fast-path scope:
  - normalized cosine and inner product run on the faithful `v2` code-domain path
  - L2 and non-normalized scans use explicit compatibility fallback scoring
- SQL helpers:
  - `tq_rerank_candidates(...)`
  - `tq_approx_candidates(...)`
  - `tq_recommended_query_knobs(...)`
  - `tq_index_metadata(...)`
  - `tq_last_scan_stats()`

## Quick start

Build and install against PostgreSQL 16 or 17 with PGXS:

```sh
make
make install
```

Enable the required extensions in your database:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;
```

Create an index:

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

Run approximate retrieval with SQL-side reranking:

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

## Benchmark highlights

### Comparative retrieval (amd64, PG 16, bge-small-en-v1.5, v2 Qprod/QJL codec, 200 queries per dataset)

**KILT NQ** (2.5K passages, flat scan):

| Method | Recall@10 | P95 Latency (ms) | Footprint |
|---|---:|---:|---:|
| `pg_turboquant_approx` | 0.950 | 2.6 | 1.2 MB |
| `pg_turboquant_rerank` | 0.950 | 2.7 | 1.2 MB |
| `pgvector_hnsw_approx` | 0.950 | 3.7 | 5.1 MB |
| `pgvector_hnsw_rerank` | 0.950 | 2.6 | 5.1 MB |
| `pgvector_ivfflat_approx` | 0.950 | 3.5 | 4.4 MB |
| `pgvector_ivfflat_rerank` | 0.950 | 2.8 | 4.4 MB |

**KILT HotpotQA** (10K passages, IVF scan):

| Method | Recall@10 | P95 Latency (ms) | Footprint |
|---|---:|---:|---:|
| `pg_turboquant_approx` | 0.363 | 4.6 | 6.2 MB |
| `pg_turboquant_rerank` | 1.000 | 6.8 | 6.2 MB |
| `pgvector_hnsw_approx` | 0.585 | 7.5 | 21.6 MB |
| `pgvector_hnsw_rerank` | 1.000 | 7.4 | 21.6 MB |
| `pgvector_ivfflat_approx` | 0.585 | 7.9 | 17.6 MB |
| `pgvector_ivfflat_rerank` | 1.000 | 7.4 | 17.6 MB |

**PopQA** (4.9K passages, flat scan):

| Method | Recall@10 | P95 Latency (ms) | Footprint |
|---|---:|---:|---:|
| `pg_turboquant_approx` | 1.000 | 6.5 | 2.5 MB |
| `pg_turboquant_rerank` | 1.000 | 4.3 | 2.5 MB |
| `pgvector_hnsw_approx` | 1.000 | 6.5 | 10.0 MB |
| `pgvector_hnsw_rerank` | 1.000 | 4.4 | 10.0 MB |
| `pgvector_ivfflat_approx` | 1.000 | 6.4 | 8.2 MB |
| `pgvector_ivfflat_rerank` | 1.000 | 4.3 | 8.2 MB |

Those results are environment-specific. The benchmark harness keeps recall, latency, footprint, WAL, and concurrent-write measurements separate so tradeoffs remain visible instead of being collapsed into a single score.

## Documentation

The public docs follow Diataxis:

- Tutorial: [docs/tutorials/getting-started.md](docs/tutorials/getting-started.md)
- How-to:
  - [docs/how-to/install.md](docs/how-to/install.md)
  - [docs/how-to/install-and-use-in-postgres.md](docs/how-to/install-and-use-in-postgres.md)
  - [docs/how-to/run-benchmarks.md](docs/how-to/run-benchmarks.md)
- Reference:
  - [docs/reference/sql-api.md](docs/reference/sql-api.md)
  - [docs/reference/index-options.md](docs/reference/index-options.md)
  - [docs/reference/benchmark-output.md](docs/reference/benchmark-output.md)
- Explanation:
  - [docs/explanation/architecture.md](docs/explanation/architecture.md)
  - [docs/explanation/benchmark-results.md](docs/explanation/benchmark-results.md)
  - [docs/adrs](docs/adrs)

The docs hub lives at [docs/README.md](docs/README.md).

## Compatibility and scope

- PostgreSQL: 16 and 17
- pgvector: required for `vector` and `halfvec`
- Current support boundary:
  - single-column indexes only
  - no index-only scans
  - no multicolumn support
  - no `INCLUDE` columns
  - exact reranking stays outside the access method

## Development

Canonical commands:

```sh
make
make install
make unitcheck
make installcheck
make tapcheck
```

The benchmark harness lives in `scripts/benchmark_suite.py`. The RAG evaluation harness lives under `benchmarks/rag/`.

For scan observability, `tq_last_scan_stats()` exposes backend-local JSON for the most recent TurboQuant scan, including score mode, SIMD kernel, scan orchestration, and page pruning counters. `tq_index_metadata(...)` reports the algorithm version, quantizer family, residual sketch kind, and whether the index is eligible for the faithful fast path.
