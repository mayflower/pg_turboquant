# ADR-0015 — Prioritize the production RAG retrieval surface over broader algorithm expansion

- Status: Proposed
- Date: 2026-04-03

## Context

Prompt `16` closed the old “multicolumn unsupported” boundary with a narrow real filtered ordered-scan path, but that surface is still smaller than a production RAG service typically needs:

- the current ordered filtered path supports one ANN key plus one persisted `int4` equality filter,
- filtered pages currently fall off the SoA/block-16 storage path,
- `INCLUDE` and index-only payload retrieval are still unsupported,
- steady-state write/delete behavior still depends on append-only base storage plus rebuild-oriented maintenance boundaries from `ADR-0006`.

The largest remaining gap is no longer raw ANN scoring mechanics. It is the production retrieval surface:

- filtering,
- stage-1 payload shape,
- lifecycle/compaction under churn.

## Proposed decision

Prioritize production RAG work in this order:

1. richer metadata filtering without losing the SoA/block-16 path,
2. small covering payloads for stage-1 retrieval,
3. delta + merge + compaction plus an explicit filtered top-k completion/work-budget contract.

For the filtering slice, prefer compact sidecars or packed metadata blocks that preserve the existing fast page shape where possible.

For the payload slice, prefer narrow retrieval payloads such as ids, version-ish metadata, and offsets rather than storing passage text inside ANN pages.

For the lifecycle slice, treat the current compaction/reuse primitives as substrate only. Any accepted move away from the existing `ADR-0006` rebuild boundary must land through a new ADR instead of being introduced implicitly.

## Consequences

### Positive

- The roadmap aligns with real RAG bottlenecks: filters, heap I/O, and churn.
- Future work gets a clear boundary between “fast ANN AM” and “operable retrieval index.”
- The repository can make narrower, more credible production claims.

### Negative

- Some work that is attractive in isolation, such as more SIMD specialization or broader transform experimentation, moves behind retrieval-surface gaps.
- Fully general metadata filtering remains a staged roadmap rather than a single prompt-sized change.

## Follow-up implications

- The next implementation slices should be tests-first and incremental: first preserve SoA for the existing narrow filter contract, then widen the filter surface.
- Covering payloads and ingestion-policy work should be scoped so they do not silently violate `ADR-0004` or `ADR-0006`.
