# ADR-0005 — Support flat and IVF routing under the same access method

- Status: Accepted
- Date: 2026-03-27

## Context

TurboQuant-style compression is useful even without an IVF router. For small and medium corpora, a compact flat scan can be attractive. For larger corpora, an IVF router improves query latency.

Building two separate access methods would fragment the project and duplicate infrastructure.

## Decision

Keep both modes under one AM:

- `lists = 0` → flat mode
- `lists > 0` → IVF mode

## Consequences

### Positive

- Single extension surface
- Shared page format and scan framework
- Easier migration path from small to larger datasets

### Negative

- The build and scan code must branch on routing mode
- Testing must cover both modes

### Follow-up implications

- Prompt sequencing must land flat mode before IVF mode.
- Reloption validation must make the routing mode explicit.
