# PRD — `pg_turboquant` v1

- Status: Draft for implementation
- Date: 2026-03-27
- Scope: PostgreSQL extension, custom ANN index access method
- Primary implementation language: C with SQL migration files and test harnesses
- Target development workflow: Codex CLI driving small, test-first increments

## Rewrite note

As of 2026-03-30, the primary codec contract is being rewritten around faithful TurboQuant `v2` semantics for normalized cosine and inner-product retrieval. ADR-0012 supersedes the earlier product assumption that the main packed path could remain a merely TurboQuant-inspired `tq_prod`/`tq_mse` split. This PRD remains the repository baseline for PostgreSQL boundaries, testing, and maintenance policy, but the packed algorithm contract should now be read through ADR-0012.

## 1. Executive summary

`pg_turboquant` is a PostgreSQL extension that introduces a new index access method, `turboquant`, for approximate nearest-neighbor retrieval over `vector` and `halfvec` embeddings stored in PostgreSQL. The current rewrite targets a faithful TurboQuant `v2` packed path for normalized cosine and inner-product retrieval while preserving the surrounding PostgreSQL execution model, maintenance story, and access-method boundaries.

The core v1 product thesis is:

1. **A dedicated access method is required.** A faithful TurboQuant `v2` compressed layout cannot be expressed cleanly as an opclass bolted onto `ivfflat` or `hnsw`.
2. **The physical layout must respect PostgreSQL's page model.** On default builds the effective lane count is small, often 8 for realistic embedding sizes and bit widths.
3. **Exact reranking belongs outside the access method.** The access method returns candidate TIDs in approximate distance order; SQL or the executor does any exact second-stage reranking.
4. **The index format must stay production-safe.** v1 uses structured transforms only, append-only page mutation patterns, dead bitmaps for deletes, immutable IVF routing after build, and rebuilds via `REINDEX` when maintenance requires it.
5. **Faithfulness is narrow and explicit.** Normalized cosine and inner product are the primary faithful fast path; L2 and non-normalized retrieval stay available as compatibility fallbacks until separately redesigned.

The result should be a compact, PostgreSQL-native ANN index that offers materially better code density than a per-tuple float representation while preserving correctness boundaries imposed by PostgreSQL.

## 2. Problem statement

Teams already storing embeddings in PostgreSQL with pgvector want a more storage-efficient index for approximate retrieval. Existing approaches either:

- store comparatively large vector payloads,
- carry tuple-oriented overheads that reduce SIMD and cache efficiency,
- or require training and maintenance paths that do not map cleanly onto PostgreSQL extension constraints.

TurboQuant-style compression is attractive because it promises:

- very small per-vector codes,
- fast asymmetric scoring using query-side precomputation,
- a clean flat-scan mode for small corpora,
- and an IVF-routed mode for larger corpora.

However, directly transplanting a research design into PostgreSQL is unsafe unless the implementation is shaped around:

- 8 KB pages by default,
- MVCC visibility separation between access method and executor,
- WAL and recovery semantics,
- build, insert, vacuum, and reindex behavior,
- and dependency realities around pgvector types.

## 3. Product goals

### 3.1 Functional goals

1. Provide a loadable extension named `pg_turboquant`.
2. Register a custom index access method named `turboquant`.
3. Support ordered ANN scans for:
   - cosine distance
   - inner product
   - L2 distance
4. Support both:
   - `vector`
   - `halfvec`
5. Support two routing modes under the same access method:
   - `lists = 0` → flat TurboQuant scan
   - `lists > 0` → IVF-routed TurboQuant scan
6. Support a compact on-disk format with lane-adaptive micro-batches.
7. Support append-only inserts after initial build.
8. Support deletes via tombstoning / dead bitmaps.
9. Support cleanup through `VACUUM` and full rebuild via `REINDEX`.
10. Provide deterministic test hooks and a benchmark harness.

### 3.2 Non-functional goals

1. Be safe within PostgreSQL's extension model and page system.
2. Preserve clear architectural boundaries with MVCC and executor responsibilities.
3. Offer a scalar path that is always correct, plus AVX2 acceleration where available.
4. Keep v1 deterministic enough for unit testing and regression tests.
5. Be documented well enough that Codex CLI can implement it incrementally.

## 4. Non-goals for v1

The following are explicitly out of scope for v1:

1. Internal exact reranking inside the access method.
2. Dense-matrix “exact” transforms in the on-disk index format.
3. Online router refresh or centroid migration in place.
4. Multi-column indices.
5. Bitmap scans.
6. Index-only scans.
7. GPU offload.
8. Distributed search across shards.
9. Adaptive retraining without rebuild.
10. Peak-optimized AVX-512 or ARM NEON kernels.
11. Production-ready concurrency tuning for `CREATE INDEX CONCURRENTLY`.
12. Custom WAL resource managers; v1 should use PostgreSQL's generic WAL path.

## 5. Primary users and use cases

### 5.1 Primary users

- PostgreSQL engineers already using pgvector
- teams with medium-to-large embedding tables who need better storage density
- systems programmers extending PostgreSQL
- benchmark-driven database engineers evaluating retrieval quality vs. footprint

### 5.2 Primary use cases

1. Build a compact ANN index on a table of normalized embeddings and run cosine top-k queries.
2. Use flat mode for smaller tables without training an IVF router.
3. Use IVF mode for larger tables with tunable probe counts.
4. Insert new rows over time without retraining the quantizer.
5. Delete rows and reclaim wholly dead pages via normal maintenance.
6. Run exact reranking in SQL over a small candidate set when final ranking quality matters.

## 6. Assumptions and hard constraints

### 6.1 PostgreSQL constraints

1. PostgreSQL page size is a build-time constant and is commonly 8192 bytes.
2. Custom access methods must implement the required `IndexAmRoutine` hooks.
3. The executor, not the access method, is responsible for heap visibility checks.
4. Bitmap scans discard order, so they are not the right primary execution model for ordered ANN.
5. Index relations do not get the same kind of row-level large-value TOAST strategy as table attributes; the design must not rely on giant opaque objects stored “inside the index”.
6. Crash safety and WAL semantics matter from the first functional slice.

### 6.2 Storage constraints

1. Lane count must be derived from real page budget, not assumed.
2. On default 8 KB builds, realistic lane counts will often be 8 for `d=1536`, `b=4`, `tq_prod`, normalized data.
3. Page-local packing should favor SIMD traversal over tuple-by-tuple payloads.

### 6.3 Dependency constraints

1. `vector` and `halfvec` come from pgvector.
2. The project must not assume system-wide public pgvector headers are present.
3. v1 needs a pinned pgvector source dependency or compatibility layer strategy.

### 6.4 Platform assumptions

1. Primary development target: Linux, x86_64.
2. Primary PostgreSQL target: PG16 first; PG17 support is a stretch objective once PG16 is stable.
3. AVX2 may be used when present; scalar fallback is mandatory.

## 7. v1 product decisions

The following decisions are fixed unless superseded by a new ADR:

1. `turboquant` is a **dedicated access method**.
2. v1 uses **lane-adaptive micro-batches** computed from page budget.
3. v1 uses **structured transforms only** in the persistent format.
4. v1 does **not** rerank from heap tuples inside the AM.
5. v1 keeps **flat and IVF** under the same AM.
6. v1 uses **append-only mutation**, **dead bitmaps**, and **immutable IVF routing after build**.
7. v1 exposes **dedicated opclasses** named `tq_*_ops`.
8. v1 supports `vector` first and extends to `halfvec` before feature freeze.
9. v1 uses **generic WAL** rather than a custom resource manager.

## 8. Functional requirements

## FR-1 Extension installation

- The extension must install with `CREATE EXTENSION pg_turboquant;`.
- The extension must register the `turboquant` access method.
- The extension must install catalog objects required for supported opclasses.

Acceptance:
- `CREATE EXTENSION pg_turboquant;` succeeds.
- Catalog queries find `turboquant`.

## FR-2 Index syntax

Users must be able to create an index with syntax similar to:

```sql
CREATE INDEX docs_emb_tq
ON docs
USING turboquant (embedding tq_cosine_ops)
WITH (
  bits = 4,
  lists = 1024,
  lanes = auto,
  transform = 'hadamard',
  normalized = true
);
```

Acceptance:
- The extension parses supported reloptions.
- Unsupported combinations error with clear messages.

## FR-3 Supported type and metric matrix

v1 must support at least:

| Input type | Metric | Opclass |
|---|---|---|
| `vector` | cosine | `tq_cosine_ops` |
| `vector` | inner product | `tq_ip_ops` |
| `vector` | L2 | `tq_l2_ops` |
| `halfvec` | cosine | `tq_halfvec_cosine_ops` |
| `halfvec` | inner product | `tq_halfvec_ip_ops` |
| `halfvec` | L2 | `tq_halfvec_l2_ops` |

Acceptance:
- Each opclass can be used to create an index once implemented.
- At least one query per opclass passes regression tests.

## FR-4 Flat scan mode

When `lists = 0`, the index must build without IVF training and scan all batch pages in approximate order.

Acceptance:
- `CREATE INDEX ... WITH (lists = 0)` succeeds.
- `ORDER BY ... LIMIT k` can choose an index scan.
- Top-1 on a deterministic small corpus is correct in regression tests.

## FR-5 IVF mode

When `lists > 0`, the build must train or derive an IVF router and assign vectors to lists. Query execution must use `turboquant.probes` to control the number of scanned lists.

Acceptance:
- `CREATE INDEX ... WITH (lists > 0)` succeeds.
- Query behavior changes as probes are adjusted.
- Regression tests confirm correctness on easy clustered datasets.

## FR-6 Insert path

After build, inserts must append to the appropriate flat tail page or IVF list tail page.

Acceptance:
- Rows inserted after index creation are discoverable by queries.
- No in-place router refresh is attempted.

## FR-7 Delete and vacuum behavior

Deletes must be represented as tombstones / dead bits. Vacuum may reclaim pages that become fully dead, but v1 does not compact or migrate live entries across pages.

Acceptance:
- Deleted rows are not returned after visibility rules are applied.
- Maintenance paths do not corrupt page layout.
- Tests cover `VACUUM` and later search behavior.

## FR-8 Maintenance

The supported maintenance story is:

- `VACUUM` for cleanup of tombstones and fully dead pages
- `REINDEX INDEX` / `REINDEX INDEX CONCURRENTLY` as the approved rebuild path when router drift or fragmentation requires a refresh

Acceptance:
- `REINDEX INDEX` succeeds on a populated index.
- Documentation clearly states rebuild as the official refresh strategy.

## FR-9 Query tuning

v1 must provide bounded, documented query knobs, for example:

- `turboquant.probes`
- `turboquant.oversample_factor`

Acceptance:
- Values validate correctly.
- EXPLAIN and regression tests exercise the knobs.

## 9. Non-functional requirements

## NFR-1 Storage density

For normalized `tq_prod` at `d=1536`, `b=4`, the design target is to keep the per-vector code payload around:

- `idx`: `1536 * 3 bits = 576 bytes`
- `qjl`: `1536 * 1 bits = 192 bytes`
- `gamma`: `4 bytes`
- `TID`: `6 bytes`
- page-local overhead and alignment on top

This is an order-of-magnitude smaller than keeping the full 1536-d float32 vector inside the index.

Acceptance:
- Page math unit tests verify expected code size calculations.
- Build output logs or debug assertions confirm lane selection.

## NFR-2 Determinism

Given the same seed, input, and options, codec and transform outputs must be deterministic.

Acceptance:
- Unit tests compare repeated runs and fixed seeds.

## NFR-3 Correctness before speed

The scalar path is the source of truth. SIMD kernels must match scalar results within documented tolerances.

Acceptance:
- Unit tests compare scalar vs. AVX2 score outputs.
- SIMD can be disabled at build or runtime for debugging.

## NFR-4 Recovery safety

Index page writes must be WAL-safe and survive restart.

Acceptance:
- TAP tests cover restart/recovery scenarios.
- v1 uses generic WAL unless a later ADR changes that.

## NFR-5 Codebase maintainability

The implementation must be decomposed into narrow modules with clear responsibilities and test entry points.

Acceptance:
- Directory layout follows the module split in Section 12.
- AGENTS tracker stays current.

## 10. High-level architecture

## 10.1 Extension surface

Top-level objects:

- extension: `pg_turboquant`
- access method: `turboquant`
- opclasses:
  - `tq_cosine_ops`
  - `tq_ip_ops`
  - `tq_l2_ops`
  - `tq_halfvec_cosine_ops`
  - `tq_halfvec_ip_ops`
  - `tq_halfvec_l2_ops`
- GUCs:
  - `turboquant.probes`
  - `turboquant.oversample_factor`
  - optional debug GUCs if needed for tests

## 10.2 Internal modules

Recommended module split:

- `src/tq_am.c` — handler, AM routine, planner hooks, reloption plumbing
- `src/tq_build.c` — build path, sampling, router training, page writing
- `src/tq_insert.c` — post-build append path
- `src/tq_scan.c` — scan state, candidate heap, flat and IVF scans
- `src/tq_codec_mse.c` — TQ-MSE encoder/decoder
- `src/tq_codec_prod.c` — TQ-Prod encoder/decoder
- `src/tq_transform.c` — structured transforms and query preparation
- `src/tq_page.c` — meta/list/batch page read/write helpers
- `src/tq_router.c` — IVF centroid handling and routing
- `src/tq_wal.c` — generic WAL wrappers
- `src/tq_pgvector_compat.c` — access shims for `vector` / `halfvec`
- `src/tq_simd_avx2.c` — optional optimized kernels
- `src/tq_debug.c` — strictly optional test/debug support

## 10.3 Persistent page types

### Meta page

Stores:

- format version
- dimension
- codec kind (`mse` or `prod`)
- distance family
- bit width
- lane count
- transform kind
- transform seeds / parameters
- normalization flag
- list count
- root directory references
- build-time metadata and checksums as needed

### Centroid pages

Used only when `lists > 0`:

- IVF centroids
- centroid count
- dimensional metadata
- references from meta page

### List directory pages

For IVF mode:

- list id
- head page
- tail page
- live/dead counters
- free-lane hint if useful

### Batch pages

Primary storage for codes:

- page-local header
- live bitmap
- TID array
- gamma array (prod)
- optional norm array
- bit-packed code payloads
- opaque tail for list chaining and bookkeeping

## 10.4 Lane-adaptive micro-batches

Lane count is not hard-coded. It is derived from page budget.

Conceptually:

```text
usable_page_bytes = block_size - page_header - opaque_header - reserve
lane_count = max lane value in {16, 8, 4, 2, 1}
             such that lane_count * bytes_per_code <= usable_page_bytes
```

Where:

- `bytes_per_code` depends on dimension, codec, bit width, presence of `qjl`, `gamma`, norms, and alignment.
- For default 8 KB PostgreSQL builds and realistic high-dimensional embeddings, lane count will frequently be 8.

## 10.5 Query execution model

### Flat mode

1. Normalize query if metric requires it.
2. Prepare query-side transformed representation and LUTs.
3. Scan all batch pages.
4. Score lanes in SIMD or scalar blocks.
5. Maintain top-k approximate candidates by TID.
6. Return TIDs in approximate distance order.

### IVF mode

1. Prepare query transform and router-side representation.
2. Score centroids.
3. Pick top `probes` lists.
4. Scan those list chains only.
5. Score and rank candidates.
6. Return TIDs in approximate order.

### Exact rerank

Exact reranking is intentionally **outside** the AM.

Official documented pattern:

```sql
WITH candidates AS MATERIALIZED (
  SELECT id, embedding
  FROM docs
  ORDER BY embedding <=> $1
  LIMIT 128
)
SELECT id
FROM candidates
ORDER BY embedding <=> $1
LIMIT 10;
```

## 11. Build and maintenance flows

## 11.1 Build

1. Validate options.
2. Initialize meta page.
3. If `lists > 0`, draw a sample and train/derive router centroids.
4. Run `table_index_build_scan()`.
5. Convert pgvector inputs into an internal float view.
6. Normalize if configured.
7. Apply structured transform.
8. Encode with TQ-MSE or TQ-Prod path.
9. Route to flat or IVF output stream.
10. Write batch pages with generic WAL protection.

## 11.2 Insert

1. Read index metadata.
2. Convert input datum to internal float view.
3. Normalize and transform as needed.
4. Encode.
5. Choose list id if IVF.
6. Append to tail page or allocate a new page.
7. Write via generic WAL.

## 11.3 Delete / vacuum

1. Mark corresponding lane as dead in the live bitmap.
2. Do not bit-shift or compact live payload in place.
3. During cleanup, reclaim fully dead pages where possible.
4. Do not rebalance or reroute live entries.

## 11.4 Refresh / rebuild

The official refresh story is `REINDEX`, not online router mutation.

## 12. Testing strategy

The project must use three layers of tests:

### 12.1 Unit tests (`tests/unit/`)

Purpose:

- page math
- reloption validation
- page serialization / deserialization
- transform determinism
- codec packing / unpacking
- scalar vs. AVX2 parity
- candidate ranking kernels

Expected command:
- `make unitcheck`

### 12.2 SQL regression tests (`test/sql/`, `test/expected/`)

Purpose:

- extension install
- catalog presence
- index creation
- query semantics on deterministic fixtures
- maintenance commands
- opclass coverage

Expected command:
- `make installcheck`

### 12.3 TAP tests (`t/`)

Purpose:

- restart / recovery durability
- basic WAL survivability
- multi-step lifecycle checks too awkward for `pg_regress`

Expected command:
- `prove -I $(pg_config --pgxs | xargs dirname)/../test/perl t/*.pl`
- or a repo-level wrapper target

## 13. Benchmark strategy

v1 needs a reproducible benchmark harness, but benchmark wins are **not** release gates unless correctness is already green.

The harness should compare:

- exact baseline on heap vectors
- `turboquant` flat mode
- `turboquant` IVF mode
- optional pgvector baselines if environment allows

Suggested benchmark datasets:

- synthetic normalized Gaussian vectors
- clustered synthetic vectors
- one medium real-world embedding dataset if license allows in local development only

Metrics:

- build time
- index size
- median / p95 query latency
- recall@k
- top-1 accuracy on deterministic fixtures
- WAL volume for insert-heavy tests

## 14. Observability and debug hooks

Minimal acceptable observability:

- clear option validation errors
- optional debug logging under a guarded GUC
- compile-time assertions for page layout
- helper comments and test fixtures for page math

Debug hooks should remain limited and must not become a permanent user-facing API unless intentionally productized.

## 15. Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Page budget miscalculation | corrupt or impossible layouts | unit tests for code size and lane count |
| pgvector ABI drift | build/runtime mismatch | pin pgvector source dependency |
| SIMD path divergence | wrong ranking or crashes | scalar oracle tests and parity tests |
| IVF drift after many inserts | degraded recall | explicit rebuild story via `REINDEX` |
| MVCC layering violations | incorrect visibility / unsafe reads | no internal heap rerank |
| transform memory blow-up | poor latency and huge metadata | structured transforms only |
| WAL bugs | corruption after restart | generic WAL + TAP restart tests |

## 16. Exit criteria for v1

v1 is ready to tag only when all of the following are true:

1. All prompts through the final hardening prompt are complete.
2. `make unitcheck`, `make installcheck`, and TAP tests pass on the target environment.
3. Both flat and IVF modes work for at least one supported metric.
4. `vector` and `halfvec` each have at least one passing indexed query path.
5. The documented exact rerank SQL pattern is validated in tests or examples.
6. Restart/recovery tests pass.
7. The benchmark harness runs and emits results.
8. `AGENTS.md` tracker is fully updated.

## 17. Milestone mapping to prompt sequence

- `00_environment_setup.md` → bootstrap build/test environment
- `01_access_method_skeleton.md` → loadable AM skeleton
- `02_reloptions_and_page_budget.md` → page math and reloptions
- `03_page_formats_and_dead_bitmap.md` → persistent layout
- `04_structured_transforms.md` → transform engine
- `05_tq_mse_codec.md` → MSE codec
- `06_tq_prod_codec.md` → Prod codec
- `07_flat_build_and_scan.md` → first end-to-end searchable slice
- `08_planner_gucs_and_simd_parity.md` → query controls and parity
- `09_ivf_router_and_probe_scan.md` → routed search
- `10_insert_delete_vacuum_and_reindex.md` → maintenance slice
- `11_additional_opclasses_and_halfvec.md` → broaden surface
- `12_wal_durability_and_benchmark_harness.md` → hardening and benchmark closure
