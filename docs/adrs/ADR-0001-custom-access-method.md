# ADR-0001 — Use a dedicated PostgreSQL access method

- Status: Accepted
- Date: 2026-03-27

## Context

TurboQuant-style storage wants compact, page-oriented, code-dense payloads with query-side scoring. Existing pgvector access methods (`ivfflat`, `hnsw`) are not the right abstraction boundary for a fundamentally different physical layout and scan kernel.

Trying to force the format into an existing AM would retain tuple-oriented overheads and fight both the PostgreSQL AM API and the storage objectives.

## Decision

Build `pg_turboquant` as a **new index access method**, `turboquant`, with its own `IndexAmRoutine`, page layouts, reloptions, and opclasses.

## Consequences

### Positive

- Full control over on-disk layout and scan behavior
- Clean separation from pgvector implementation details
- Can expose flat and IVF modes under one consistent AM

### Negative

- More implementation work than a new opclass
- WAL, vacuum, scan, and build paths must all be owned by this project

### Follow-up implications

- The repository must prioritize AM scaffolding and page I/O tests early.
- Debugging and recovery semantics become first-class concerns.
