# ADR-0007 — Use dedicated `tq_*_ops` opclasses and support `vector` plus `halfvec`

- Status: Accepted
- Date: 2026-03-27

## Context

PostgreSQL allows opclass names to repeat across access methods, but reusing pgvector naming would create coupling and ambiguity. The project also needs a clean story for both `vector` and `halfvec`.

## Decision

Define dedicated opclasses:

- `tq_cosine_ops`
- `tq_ip_ops`
- `tq_l2_ops`
- `tq_halfvec_cosine_ops`
- `tq_halfvec_ip_ops`
- `tq_halfvec_l2_ops`

## Consequences

### Positive

- Clear ownership and upgrade boundaries
- Easier documentation
- Cleaner catalog introspection

### Negative

- Slightly more verbose DDL for users
- More catalog objects to maintain

### Follow-up implications

- Regression tests must cover at least one query per opclass family.
- Documentation must explain why names differ from pgvector's AM-specific opclasses.
