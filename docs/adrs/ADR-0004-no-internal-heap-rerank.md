# ADR-0004 — Do not do exact heap reranking inside the access method

- Status: Accepted
- Date: 2026-03-27

## Context

An access method does not own MVCC visibility and should not casually fetch heap rows to perform second-stage exact ranking on raw embeddings. Doing so risks layering violations and complicated correctness problems.

## Decision

The `turboquant` AM returns candidate TIDs in approximate order only. Exact reranking is performed outside the AM using normal SQL patterns over a candidate set.

## Consequences

### Positive

- Preserves PostgreSQL architectural boundaries
- Keeps the AM simpler and safer
- Makes correctness easier to reason about

### Negative

- Users wanting best final ranking must issue a two-stage SQL query
- Some benchmark comparisons must account for two-phase execution

### Follow-up implications

- Documentation must include the official rerank SQL pattern.
- No internal AM API should be introduced for heap-based reranking.
