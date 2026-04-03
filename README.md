# pg_turboquant

`pg_turboquant` is a PostgreSQL extension that adds a custom ANN index access method, `turboquant`, for compact nearest-neighbor search over `vector` and `halfvec`.

It is designed around PostgreSQL's storage and executor constraints rather than treating ANN indexing as an external service. The project combines structured transforms, a faithful TurboQuant `v2` payload for normalized cosine and inner-product retrieval, SoA batch pages with 4-bit packed dimension-major nibbles for zero-copy SIMD scoring, NEON TBL and AVX2 VPSHUFB block-16 kernels with global-scale int16 accumulation, ordered ANN scans, bitmap support for filtered workloads, SQL-side exact reranking helpers, and a reproducible benchmark harness with machine-readable microbench regression gates.

## Why it exists

- PostgreSQL users already storing embeddings with pgvector often want a denser index format.
- `pg_turboquant` keeps ANN inside PostgreSQL while optimizing for compact storage and cache-friendly scoring.
- The access method is explicit about PostgreSQL boundaries: MVCC still lives in the executor, exact reranking still lives in SQL, and v1 still uses generic WAL.

## Algorithm contract

- Faithful fast path:
  normalized cosine and inner-product retrieval use the paper-faithful `Qprod` payload with structured rotation, `b - 1` stage-1 scalar codes, a residual 1-bit QJL sketch, and stored residual norm `gamma`. The fast path scores via a global-scale quantized LUT16 with NEON TBL or AVX2 VPSHUFB block-16 kernels, accumulating in int16 with periodic drain to int32 (Faiss FastScan-style).
- Page format:
  when LUT16 is supported (bits=4, dimension divisible by 8), batch pages use an SoA layout with 4-bit packed dimension-major nibbles, enabling the SIMD kernel to read directly from the page buffer with no per-scan transpose. Pages that exceed the 8 KB budget for SoA fall back to the legacy AoS interleaved layout.
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
./scripts/bootstrap_dev.sh
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

### Comparative retrieval (arm64 Apple Silicon, PG 16, harrier-oss-v1-270m 640d, 200 queries, SoA page format)

**Knowledge Base RAG** (2.8K passages, flat scan, microsoft/harrier-oss-v1-270m):

| Method | P50 Latency (ms) | P95 Latency (ms) | Index Size |
|---|---:|---:|---:|
| `pg_turboquant` | **3.54** | 5.17 | **2.8 MB** |
| `pgvector_hnsw` | 3.99 | 4.76 | 10.7 MB |
| `pgvector_ivfflat` | 3.77 | 4.65 | 7.6 MB |

turboquant is **12% faster at p50** while being **3.8x smaller** than HNSW.

### Comparative retrieval (arm64, PG 16, bge-small-en-v1.5 384d, 200 queries per dataset)

| Dataset | turboquant P95 | HNSW P95 | tq/HNSW | tq Footprint | HNSW Footprint |
|---|---:|---:|---:|---:|---:|
| KILT NQ (2.5K, flat) | 1.66 ms | 1.43 ms | 1.16x | 1.2 MB | 5.1 MB |
| KILT HotpotQA (10K, IVF) | 1.92 ms | 2.87 ms | **0.67x** | 6.5 MB | 21.6 MB |
| PopQA (4.9K, flat) | 2.87 ms | 2.94 ms | **0.98x** | 2.5 MB | 10.0 MB |

turboquant **beats HNSW on IVF datasets** (1.5x faster on HotpotQA) and reaches **parity on flat scans**, while maintaining a **3-4x smaller footprint** across all datasets.

Results are environment-specific. The benchmark harness keeps recall, latency, footprint, WAL, and concurrent-write measurements separate so tradeoffs remain visible instead of being collapsed into a single score.

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
  - [docs/reference/compatibility.md](docs/reference/compatibility.md)
- Explanation:
  - [docs/explanation/architecture.md](docs/explanation/architecture.md)
  - [docs/explanation/benchmark-results.md](docs/explanation/benchmark-results.md)
  - [docs/adrs](docs/adrs)

The docs hub lives at [docs/README.md](docs/README.md).

## Compatibility and scope

- PostgreSQL: 16 and 17
- pgvector: required for `vector` and `halfvec`
- Tested pgvector contract: pinned development and CI reference `v0.8.1`
- Current support boundary:
  - one `vector`/`halfvec` ANN key plus up to eight stored `int4` metadata attributes
  - exact `int4` filter keys support equality and `ANY(int4[])` predicates inside the ordered scan
  - `INCLUDE`-style `int4` payload columns are returned through index tuples for stage-1/index-only-style retrieval
  - the production fast lane still assumes normalized cosine/IP, `transform = 'hadamard'`, and `lanes = auto`
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

For scan observability, `tq_last_scan_stats()` exposes backend-local JSON for the most recent TurboQuant scan, including score mode, SIMD kernel, scan orchestration, and page pruning counters. `tq_index_metadata(...)` reports the algorithm version, quantizer family, residual sketch kind, and whether the index is eligible for the faithful fast path. It now carries only cheap heap estimates; use `tq_index_heap_stats(...)` when you intentionally want an exact heap row count.
