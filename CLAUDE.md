# pg_turboquant

PostgreSQL extension implementing a custom index access method (`turboquant`) for approximate nearest-neighbor search over vector embeddings. Compresses vectors using scalar quantization (2-8 bits/dim) with Hadamard transforms for storage-efficient ANN retrieval. Depends on pgvector for `vector` and `halfvec` types.

## Build

Requires PostgreSQL 16 or 17 with development headers (PGXS).

```sh
# Auto-detects pg_config; override with PG_CONFIG=/path/to/pg_config
make              # build extension
make install      # install extension + pgvector dependency
```

Compiler flags: `-Wall -Werror` enforced. C11 for unit tests. Links `-lm`.

## Test

Three-layer test suite, run in this order:

```sh
make unitcheck      # standalone C unit tests (tests/unit/)
make installcheck   # SQL regression tests via pg_regress (test/sql/ vs test/expected/)
make tapcheck       # TAP tests in Perl (t/*.pl) - cluster lifecycle, WAL, recovery
```

- Unit tests use `assert.h`, compiled with `-DTQ_UNIT_TEST=1`
- SQL tests: DROP/CREATE EXTENSION, DDL, queries against deterministic fixtures
- TAP tests: `PostgreSQL::Test::Cluster` framework, require Perl + IPC::Run
- Python tests (pytest): `tests/test_benchmark_suite.py`, `tests/test_rag_*.py`

## Code layout

```
src/                  C extension source (one module per .c/.h pair)
sql/                  Extension SQL DDL (CREATE TYPE, opclass, functions)
test/sql/             SQL regression test inputs
test/expected/        Expected regression output
t/                    TAP tests (Perl)
tests/unit/           C unit tests
tests/perf/           Performance microbenchmarks
scripts/              Build helpers, benchmark harness, test runners
docs/                 Documentation (Diataxis: tutorials, how-to, reference, explanation)
docs/adrs/            Architecture Decision Records
benchmarks/rag/       RAG evaluation suite
third_party/          Fetched dependencies (pgvector, pg test libs, perl5)
docker/               Dockerfile.dev
```

Key source modules: `tq_am.c` (access method), `tq_codec_prod.c` / `tq_codec_mse.c` (codecs + LUT16 quantization), `tq_page.c` (page layout: AoS + SoA with 4-bit packed nibbles), `tq_scan.c` (scoring + block-16 transpose + zero-copy SoA path), `tq_router.c` (IVF routing), `tq_transform.c` (Hadamard), `tq_simd_avx2.c` (NEON TBL + AVX2 VPSHUFB block-16 kernels), `tq_wal.c` (generic WAL + SoA append).

## Code style

- **Indentation**: tabs (4-space width); spaces for md/yml/json
- **C naming**: `tq_` prefix functions (snake_case), `Tq` prefix types (PascalCase), `TQ_` prefix enums/macros (UPPER_SNAKE_CASE)
- **Headers**: `#ifndef TQ_MODULE_H` include guards; `postgres.h` first, then system, then local
- **Modules**: one .c/.h pair per concern; no `static inline` in headers
- `-Wall -Werror` — fix all warnings, do not suppress them

## CI

GitHub Actions (`.github/workflows/ci.yml`): matrix of PostgreSQL 16 + 17 on ubuntu-latest. Runs `make && make install && make unitcheck && make installcheck && make tapcheck`.

## Key conventions

- Access method returns approximate candidates only; exact reranking happens in SQL (no internal heap reranking)
- Routing is immutable after build; use REINDEX to refresh
- AVX2/NEON are optional optimizations; scalar path is source of truth
- Block-16 SIMD kernels: NEON TBL (arm64) and AVX2 VPSHUFB (x86_64) with global-scale int16 accumulation
- SoA page format: 4-bit packed dimension-major nibbles for zero-copy SIMD scoring; falls back to AoS when SoA doesn't fit
- LUT16 global-scale quantization: single scale per base/QJL component (not per-dimension) enables pure integer accumulation
- Generic WAL (not custom resource manager); SoA pages use `tq_wal_append_batch_soa`
- Append-only inserts after build; deletes via dead bitmaps
