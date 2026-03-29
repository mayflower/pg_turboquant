# ADR-0003 — Only structured transforms are allowed in the persistent index format

- Status: Accepted
- Date: 2026-03-27

## Context

Dense transform matrices become huge for realistic embedding dimensions. Persisting them inside a PostgreSQL index would create large metadata footprints and poor query-start behavior.

The design needs a transform strategy that is compact, deterministic, and practical inside an index.

## Decision

v1 stores **structured transforms only** in the on-disk index format, represented by seeds and small parameter blocks rather than dense matrices.

The product default is a Hadamard-style seeded transform family.

## Consequences

### Positive

- Tiny metadata footprint
- Fast query startup
- Easy deterministic tests

### Negative

- This is not a perfect reproduction of a dense-matrix research setup
- Empirical validation becomes part of the benchmark plan

### Follow-up implications

- Transform tests must cover determinism, dimensionality, and norm behavior.
- Any future dense-matrix support requires a new ADR.
