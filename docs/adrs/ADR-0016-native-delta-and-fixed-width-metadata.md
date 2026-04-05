# ADR-0016 — Add fixed-width metadata v2 and a native delta lifecycle inside the access method

- Status: Accepted
- Date: 2026-04-04

## Context

`ADR-0006` intentionally chose append-only payload pages, dead bitmaps, and `REINDEX` as the major cleanup path. That was the right v1 simplification, but the repository now targets a narrower production RAG fast lane where the missing capabilities are no longer primarily SIMD or codec mechanics:

1. metadata filtering must handle more than compact `int4` columns,
2. steady inserts need a mutable hot tier inside the index,
3. ordinary maintenance must restore health without making frequent `REINDEX` the default operator workflow.

The current code already has the necessary substrate:

- compact batch pages with reusable/free-page discovery,
- detached free blocks and tail truncation,
- list-directory metadata,
- compaction and summary refresh helpers,
- filtered completion and stage-1 payload plumbing on the query side.

What is missing is a stricter on-disk metadata contract and an explicit delta + merge lifecycle.

## Decision

The accepted post-v1 production RAG contract is:

1. Stored metadata after the embedding key may use a fixed-width descriptor layer rather than compact `int4` only.
2. Supported stored metadata types in the first production pass are fixed-width and nullable only:
   - `bool`
   - `int2`
   - `int4`
   - `int8`
   - `date`
   - `timestamptz`
   - `uuid`
3. Ordered scan filters must support `IS NULL`, `=`, and `ANY(...)` for those fixed-width key types.
4. INCLUDE-style stage-1 payloads may use the same fixed-width descriptor layer and remain index-local.
5. IVF indexes gain a native mutable delta segment inside the AM:
   - post-build inserts land in the delta tier,
   - ordered reads merge routed IVF candidates with delta candidates,
   - maintenance may fold delta into routed lists without a background worker.
6. The AM exposes an explicit maintenance surface and health metadata for delta, dead-lane churn, and fragmentation.
7. `REINDEX` remains a valid major refresh path, but it is no longer the only intended steady-state repair mechanism.

## Consequences

### Positive

- Production RAG filters can stay inside one ordered ANN path for a broader fixed-width metadata surface.
- Stage-1 payload retrieval remains index-local for practical RAG identifiers and versioning fields.
- Write-heavy RAG workloads gain a native hot tier without pushing all lifecycle policy into SQL adapters.
- Operators get an ordinary maintenance/compaction workflow before resorting to full rebuilds.

### Negative

- Batch-page layout and WAL mutation rules become more complex.
- The repository moves beyond the original strict `ADR-0006` “append-only plus REINDEX” simplification.
- Planner-visible behavior may still lag the storage/runtime surface in some cases even after the storage contract broadens.

## Supersedes / refines

This ADR refines `ADR-0006` for the production RAG lane:

- append-only writes are no longer the only intended steady-state write path,
- in-place health recovery is no longer limited to dead-bit cleanup,
- `REINDEX` is demoted from the default refresh mechanism to the exceptional major refresh path.

The accepted narrow fast-lane constraints from `ADR-0014` still apply:

- normalized dense retrieval,
- cosine / inner-product as the primary optimized path,
- `transform = 'hadamard'`,
- `lanes = auto`.
