# ADR-0012 — Adopt faithful TurboQuant v2 as the primary access-method contract

- Status: Superseded by ADR-0013 for the primary packed payload details
- Date: 2026-03-30

## Context

The original `pg_turboquant` implementation used `tq_prod` and `tq_mse` as pragmatic TurboQuant-inspired codecs. That design was useful for bringing up the access method, but it diverged from the public TurboQuant story in three important ways:

- the primary packed path stored a per-vector float scalar
- the primary packed path did not model a residual 1-bit correction stage as part of the on-disk contract
- the first-stage quantizer was a house codec rather than the structured scalar-quantizer story TurboQuant relies on after rotation

The project now needs a breaking rewrite that makes the primary implementation faithful to the public TurboQuant contract for the cases where that contract is actually defined and testable inside PostgreSQL.

ADR-0003 still applies: the persistent transform must remain structured and compact rather than storing dense matrices. ADR-0005 also still applies at the access-method boundary: flat and IVF remain modes under one AM even if the faithful rollout lands flat-path validation first.

## Decision

`pg_turboquant` now treats **faithful TurboQuant v2** as the primary algorithm contract for normalized cosine and inner-product retrieval.

The primary `v2` packed representation was originally framed as:

- structured random orthogonal transform metadata
- first-stage scalar quantizer codes
- residual 1-bit sketch bits
- no per-vector full-precision quantization constants such as `gamma`

That last point does not match the public paper's `Qprod` contract and is superseded by ADR-0013.

The primary `v2` query path is:

- transform the query with the persisted structured transform
- build deterministic first-stage lookup tables from the fixed quantizer tables
- apply residual-bit correction in code domain
- score normalized cosine and inner-product candidates through the faithful `v2` estimator

The persistent metadata contract must expose enough versioning to make rebuild requirements explicit:

- algorithm version
- quantizer version/family
- residual-sketch version/kind
- estimator version/mode
- transform version/seed

Existing indexes created under the earlier payload contract are not upgraded in place. Format changes require rebuild through `REINDEX` or index recreation.

`tq_prod` and `tq_mse` remain in the tree only as transitional implementation names and reference scaffolding. They are no longer the product contract; the faithful `v2` packed path is.

## Consequences

### Positive

- The default packed path is materially closer to the public TurboQuant design.
- The primary payload no longer spends bytes on per-vector float constants.
- The access method can report whether a scan is on the faithful fast path or a compatibility fallback.
- Scalar correctness has a clearer oracle for future SIMD and IVF work.

### Negative

- This is a breaking on-disk-format change and requires rebuilds.
- The codebase temporarily carries legacy naming and fallback seams during the rewrite.
- L2 and non-normalized retrieval remain compatibility paths until a faithful distance contract is defined.
- IVF routing still needs a separate validation pass against the faithful flat-path oracle.

### Follow-up implications

- Unit and SQL tests must treat normalized cosine/IP as the faithful fast path and surface fallbacks explicitly.
- Benchmark artifacts must report estimator mode and fallback usage rather than collapsing all scans into one label.
- Future pruning work must use same-space bounds; the previous mixed-space summary heuristic is not sufficient for safe faithful pruning.
