# pg_turboquant

`pg_turboquant` is a PostgreSQL extension that adds a custom ANN index access method, `turboquant`, for compact nearest-neighbor search over `vector` and `halfvec`.

It is designed around PostgreSQL's storage and executor constraints rather than treating ANN indexing as an external service. The project combines structured transforms, compact TurboQuant-style codecs, lane-adaptive batch pages, ordered ANN scans, bitmap support for filtered workloads, SQL-side exact reranking helpers, and a reproducible benchmark harness.

## Why it exists

- PostgreSQL users already storing embeddings with pgvector often want a denser index format.
- `pg_turboquant` keeps ANN inside PostgreSQL while optimizing for compact storage and cache-friendly scoring.
- The access method is explicit about PostgreSQL boundaries: MVCC still lives in the executor, exact reranking still lives in SQL, and v1 still uses generic WAL.

## Current capabilities

- Custom access method: `USING turboquant`
- Input types: `vector`, `halfvec`
- Metrics: cosine, inner product, L2
- Modes:
  - flat scan with `lists = 0`
  - IVF-routed scan with `lists > 0`
  - bitmap-filter support for predicate-heavy workloads
- SQL helpers:
  - `tq_rerank_candidates(...)`
  - `tq_approx_candidates(...)`
  - `tq_recommended_query_knobs(...)`
  - `tq_index_metadata(...)`

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
  lists = 128,
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

Representative checked-in retrieval results:

| Workload | Method | Recall@10 | P95 Latency (ms) | Footprint (bytes) |
|---|---|---:|---:|---:|
| KILT NQ | `pg_turboquant_approx` | 0.946667 | 8.836190 | 1,277,952 |
| KILT NQ | `pgvector_hnsw_approx` | 0.946667 | 28.845381 | 5,079,040 |
| KILT NQ | `pgvector_ivfflat_approx` | 0.946667 | 49.492140 | 4,399,104 |
| PopQA mini | `pg_turboquant_rerank` | 1.000000 | 4.796433 | 24,576 |
| PopQA mini | `pgvector_hnsw_rerank` | 1.000000 | 5.408724 | 73,728 |
| PopQA mini | `pgvector_ivfflat_rerank` | 1.000000 | 5.975884 | 81,920 |
| HotpotQA fixed-q50 | `pg_turboquant_approx` | 1.000000 | 64.991898 | 5,873,664 |
| HotpotQA fixed-q50 | `pgvector_ivfflat_approx` | 1.000000 | 8.329492 | 17,563,648 |

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
  - [docs/PRD.md](docs/PRD.md)
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
