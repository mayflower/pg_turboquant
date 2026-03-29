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

The repository ships a benchmark harness and committed benchmark evidence. A few representative results from the checked-in artifacts:

- KILT NQ retrieval:
  `pg_turboquant_approx` reached `recall@10 = 0.946667` with `1,277,952` bytes and `8.84 ms` p95 retrieval latency.
  The same run recorded `5,079,040` bytes / `28.85 ms` for pgvector HNSW and `4,399,104` bytes / `49.49 ms` for pgvector IVFFlat at the same recall.
- PopQA mini retrieval:
  `pg_turboquant_rerank` used `24,576` bytes and `4.80 ms` p95 latency, versus `73,728` bytes / `5.41 ms` for pgvector HNSW rerank and `81,920` bytes / `5.98 ms` for pgvector IVFFlat rerank.
- Planner tuning evidence:
  a committed medium-profile IVF run recorded `candidate_slots_bound` rising from `4` to `16` and `recall_at_10` rising from `0.291667` to `0.733333` when `turboquant.probes` increased from `1` to `4`.
- HotpotQA fixed-q50 retrieval:
  TurboQuant kept the smallest footprint and matched top-line recall, but pgvector IVFFlat was materially faster on latency. The project does not claim universal latency wins.

Those results are environment-specific. The benchmark harness keeps recall, latency, footprint, WAL, and concurrent-write measurements separate so tradeoffs remain visible instead of being collapsed into a single score.

## Documentation

The public docs follow Diataxis:

- Tutorial: [docs/tutorials/getting-started.md](docs/tutorials/getting-started.md)
- How-to:
  - [docs/how-to/install.md](docs/how-to/install.md)
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
- Upgrade chain: `0.1.0 -> 0.1.1 -> 0.1.2 -> 0.1.3 -> 0.1.4`
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
