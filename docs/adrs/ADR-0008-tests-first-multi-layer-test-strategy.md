# ADR-0008 — Implement with tests first using unit, SQL regression, and TAP layers

- Status: Accepted
- Date: 2026-03-27

## Context

This project combines low-level page management, compression logic, PostgreSQL AM hooks, and SQL-visible behavior. No single test style is sufficient.

## Decision

Every implementation increment must begin by adding tests first. The repository will use three layers:

1. unit tests for pure C logic,
2. `pg_regress` for SQL behavior,
3. TAP tests for restart and recovery flows.

## Consequences

### Positive

- Better isolation of bugs
- Easier Codex CLI iteration
- Safer refactoring of codec and page logic

### Negative

- More harness setup work up front
- Contributors must maintain multiple test styles

### Follow-up implications

- `AGENTS.md` must track prompt status and test evidence.
- Prompts must explicitly say “tests first”.
