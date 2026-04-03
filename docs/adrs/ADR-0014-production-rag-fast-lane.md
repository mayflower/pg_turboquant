# ADR-0014 — Narrow the production RAG fast lane before broadening feature coverage

- Status: Proposed
- Date: 2026-04-03

## Context

`pg_turboquant` already supports multiple distances, fallback scoring paths, several benchmark modes, and a growing set of planner/query helpers. But the strongest implementation path is still narrower than that surface suggests:

- normalized cosine and inner-product retrieval on the faithful `Qprod`/QJL path,
- structured Hadamard transform metadata,
- `lanes = auto`,
- page-local SoA/block-16 fast scoring when the codec shape allows it.

For production RAG, that narrow path matters more than adding another partially supported configuration. Release criteria and follow-on implementation work should optimize for the path the repository can currently justify, test, and benchmark honestly.

## Proposed decision

Treat the production-facing v1 fast lane as:

1. normalized dense retrieval,
2. cosine / inner-product queries,
3. faithful `Qprod`/QJL code-domain scoring,
4. structured Hadamard transform metadata only,
5. `lanes = auto`,
6. release gates and published benchmark claims anchored to that configuration first.

Operational guidance should assume normalization is enforced at ingest and query time for this lane.

Other supported modes remain available, but they should be documented and benchmarked as compatibility or secondary paths rather than as the primary product contract.

## Consequences

### Positive

- Performance work can focus on the path most likely to matter for production dense retrieval.
- Benchmark and release claims become easier to defend.
- The roadmap stops treating “every metric and every embedding shape” as equally urgent.

### Negative

- Some currently supported but weaker modes become explicitly second-class for release gating.
- Documentation must be precise about what is “supported” versus what is “the primary production lane.”

## Follow-up implications

- Filtering, covering payloads, and ingestion/compaction work should target this fast lane first.
- Broadening to learned dense transforms still requires a separate accepted ADR that supersedes `ADR-0003`.
