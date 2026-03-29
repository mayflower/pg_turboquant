# ADR-0006 — Use append-only pages, dead bitmaps, immutable router, and rebuild by REINDEX

- Status: Accepted
- Date: 2026-03-27

## Context

In-place reorganization of compressed bit-packed payloads is expensive and complex. IVF router drift is also hard to correct online without moving many stored codes.

PostgreSQL already has robust rebuild workflows.

## Decision

v1 storage and maintenance policy is:

1. append-only writes for new entries,
2. dead bitmaps for deletes,
3. no in-place compaction of live payload during normal operation,
4. immutable IVF router after build,
5. refresh and major cleanup via `REINDEX`.

## Consequences

### Positive

- Simpler and safer storage mutation rules
- Vacuum logic stays tractable
- No fragile online centroid migration path

### Negative

- Fragmentation may accumulate over time
- Router quality can drift under heavy write workloads

### Follow-up implications

- Vacuum should reclaim fully dead pages when feasible.
- Documentation must set correct expectations around rebuilds.
