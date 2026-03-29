# Install the extension

This guide covers the supported installation path for local development and benchmarking.

## Prerequisites

- PostgreSQL 16 or 17 server development headers and PGXS
- `make`
- `pg_config`
- `pgvector` available to the target PostgreSQL instance

## Build and install

```sh
make
make install
```

If PostgreSQL is not on your default path, provide `PG_CONFIG` explicitly:

```sh
PG_CONFIG=/path/to/pg_config make
PG_CONFIG=/path/to/pg_config make install
```

## Enable in the database

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;
```

If you want a full PostgreSQL workflow with table DDL, index creation, and query examples, continue with [install-and-use-in-postgres.md](install-and-use-in-postgres.md).

## Validate the install

```sql
\dx pg_turboquant

SELECT amname
FROM pg_am
WHERE amname = 'turboquant';
```

You can also run the smoke function shipped with the extension:

```sql
SELECT tq_smoke();
```

## Recommended local verification

```sh
make unitcheck
make installcheck
make tapcheck
```

## Common failure points

- `CREATE EXTENSION vector` fails:
  pgvector is not installed in the target PostgreSQL instance.
- `make installcheck` fails on macOS with only `libpq` installed:
  you need a full server toolchain, not just the client library.
- Queries fail on `halfvec` or `vector` operators:
  ensure the correct `tq_*_ops` opclass is used for the column type and metric.
