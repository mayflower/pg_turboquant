# ADR-0013 — Align the primary faithful path with the public Qprod/QJL contract

- Status: Accepted
- Date: 2026-03-30
- Supersedes: ADR-0012 where it prohibited per-vector `gamma`

## Context

ADR-0012 moved the project toward a more faithful TurboQuant rewrite, but it made one important assumption that does not match the public paper contract: it treated the primary faithful packed path as stage-1 scalar codes plus residual 1-bit bits **without** a per-vector `gamma`.

The public TurboQuant paper's `Qprod` construction is stricter:

- first-stage scalar quantization at `b - 1` bits
- residual `r = x - x_tilde`
- one-bit QJL sketch `sign(S r)`
- explicit `gamma = ||r||_2`
- unbiased inner-product estimator built from the stage-1 term plus the QJL residual correction

The zero-overhead blog framing is useful context, but the repository's "100% TurboQuant/QJL" target must follow the public paper contract the code can actually justify and test.

ADR-0003 still applies: the persisted transform must remain structured and compact. This ADR does **not** authorize dense persisted transforms. It only restores the paper's residual-QJL payload elements and estimator semantics.

## Decision

The primary faithful TurboQuant path in `pg_turboquant` is the public-paper `Qprod` contract for normalized cosine and inner-product retrieval.

The primary packed representation is:

- structured random orthogonal transform metadata
- first-stage scalar codes at `b - 1` bits
- residual 1-bit QJL sketch bits
- per-vector float32 `gamma = ||r||_2`

The primary estimator is:

- stage-1 inner-product contribution from the first-stage reconstruction
- plus the residual QJL correction scaled by `sqrt(pi / 2) / d`
- multiplied by the stored `gamma`

The access method must surface this contract explicitly in metadata:

- a new algorithm/format version
- quantizer family/version naming that matches the paper-facing story
- residual sketch kind/version naming that explicitly says `QJL`
- estimator mode/version naming that explicitly says `Qprod`

Existing indexes built under the earlier gamma-free rewrite are not upgraded in place. They must be rebuilt.

## Consequences

### Positive

- The repo's "faithful" claim matches the public paper instead of an inferred blog-only variant.
- The packed path now has a testable unbiased-IP story rather than a heuristic residual correction.
- SIMD and scan kernels have a clearer scalar oracle: they must reproduce the same `Qprod` score decomposition.

### Negative

- The packed payload is larger than the gamma-free rewrite.
- This is another breaking on-disk-format bump and requires rebuilds.
- The repository loses the simpler "no float constants" story from ADR-0012.

### Follow-up implications

- Any docs or benchmark copy that describe the faithful path as gamma-free must be corrected.
- Page-budget math, SQL metadata, and benchmark artifacts must all reflect the restored `gamma` bytes.
- Later pruning work still needs same-space bounds; restoring `Qprod` does not make the current ordering summaries safe.
