# ADR-0002 — Use lane-adaptive micro-batches derived from page budget

- Status: Accepted
- Date: 2026-03-27

## Context

Research systems often discuss batch sizes in terms of SIMD throughput, but PostgreSQL stores index data on fixed-size pages. On default builds, the usable payload after headers is far smaller than a naive “64 or 128 vectors per page” assumption.

For realistic embedding dimensions and bit widths, a page often fits only a small number of compressed codes.

## Decision

The index will use **lane-adaptive micro-batches**. Lane count is derived from actual page budget and stored in index metadata.

The implementation target is to choose the largest supported lane count in a small set such as `{16, 8, 4, 2, 1}` that fits the real page budget for the codec and dimension.

## Consequences

### Positive

- Physically valid layouts on real PostgreSQL installations
- Predictable SIMD-friendly scoring loops
- Same code can scale to non-default block sizes

### Negative

- Some code paths must handle small lane counts
- Performance claims must be benchmarked, not assumed

### Follow-up implications

- Page math must be unit tested before any real page writer lands.
- Batch-page serialization must not assume a fixed lane count.
