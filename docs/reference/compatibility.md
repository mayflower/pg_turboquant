# Compatibility and upgrade policy

## Supported matrix

- PostgreSQL 16
- PostgreSQL 17
- pgvector v0.8.1

`pg_turboquant` remains a separate access method. It depends on pgvector for the `vector` and `halfvec` types, but it does not aim to share pgvector’s internal access-method surface.

## pgvector compatibility contract

- Development and CI pin pgvector to `v0.8.1`.
- The repository keeps pgvector interactions behind `tq_pgvector_compat.*`.
- The public SQL opclass surface uses `tq_*` wrapper support functions instead of binding directly to pgvector-owned function names.
- When changing PostgreSQL or pgvector versions, rerun the full build, unit, installcheck, and TAP suites before treating the combination as supported.

## Install contract

- `make install` installs only `pg_turboquant`.
- pgvector provisioning belongs in `./scripts/bootstrap_dev.sh`, CI, package management, or deployment tooling.
- `CREATE EXTENSION vector;` must succeed before `CREATE EXTENSION pg_turboquant;`.

## Upgrade and rebuild policy

- The extension is still pre-1.0 and currently exposes a single public install version, `0.1.0`.
- On-disk format changes are compatibility events. If the page format changes, existing indexes must be rebuilt with `REINDEX` or recreated.
- In-place extension upgrade scripts are not currently part of the public contract. Unsupported historical upgrade paths should be treated as rebuild-required rather than silently accepted.

## Operational caveats

- The extension is marked `superuser = true`.
- Custom access methods and format changes can be deployment blockers in managed PostgreSQL environments. Validate that extension installation, `CREATE ACCESS METHOD`, and `REINDEX` are allowed before planning production rollout.
