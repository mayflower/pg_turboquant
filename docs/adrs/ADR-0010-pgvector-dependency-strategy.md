# ADR-0010 — Treat pgvector as a pinned source dependency until a stable public-header story exists

- Status: Accepted
- Date: 2026-03-27

## Context

`pg_turboquant` needs to work with `vector` and `halfvec`, which are defined by pgvector. External extensions cannot safely assume that pgvector installs all required public headers into the PostgreSQL server include tree.

## Decision

Use a pinned pgvector source dependency strategy for development and CI. The repository should keep either:

- a vendored source tree under `third_party/pgvector`, or
- a bootstrap script that fetches a pinned pgvector revision into `third_party/pgvector`.

Wrap all direct interactions with pgvector internals behind a small adapter layer, such as `tq_pgvector_compat.*`.

## Consequences

### Positive

- Deterministic builds
- Explicit compatibility boundary
- Easier future upgrades

### Negative

- Additional dependency management work
- Periodic need to validate against new pgvector releases

### Follow-up implications

- The environment setup prompt must provision this dependency strategy.
- Compatibility tests should avoid leaking pgvector internals across the whole codebase.
