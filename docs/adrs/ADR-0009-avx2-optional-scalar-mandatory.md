# ADR-0009 — AVX2 is optional; scalar correctness is mandatory

- Status: Accepted
- Date: 2026-03-27

## Context

SIMD acceleration is valuable for score kernels, but PostgreSQL extensions must remain portable and debuggable. Architecture-specific code must not become the sole source of truth.

## Decision

Implement a scalar path first and treat it as the correctness oracle. Add AVX2 acceleration as an optional optimization layer with parity tests.

## Consequences

### Positive

- Easier debugging
- Broader portability
- Safe fallback on machines without AVX2

### Negative

- More code paths to maintain
- Benchmark harness must compare parity and performance

### Follow-up implications

- Unit tests must compare scalar vs. AVX2 outputs.
- Build scripts should permit disabling SIMD.
