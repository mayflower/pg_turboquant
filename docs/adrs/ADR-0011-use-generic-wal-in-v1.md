# ADR-0011 — Use PostgreSQL generic WAL in v1

- Status: Accepted
- Date: 2026-03-27

## Context

A custom WAL resource manager would significantly increase implementation complexity for a first release. The project still needs restart-safe index mutation.

## Decision

v1 will use PostgreSQL's generic WAL facilities for page modifications. A custom WAL resource manager is out of scope unless later evidence shows generic WAL is insufficient.

## Consequences

### Positive

- Lower implementation complexity
- Faster path to crash-safe persistence
- Better fit for an incremental extension project

### Negative

- Potentially less specialized WAL efficiency than a custom solution
- WAL code still needs careful testing

### Follow-up implications

- TAP tests for restart and durability are mandatory.
- Page helper APIs should centralize write paths so WAL concerns stay localized.
