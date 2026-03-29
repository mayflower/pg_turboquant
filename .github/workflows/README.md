CI validates the repository against the supported PostgreSQL versions.

Current support policy:

- PostgreSQL 16: required
- PostgreSQL 17: required

The primary workflow should keep the build, install, `make unitcheck`, `make installcheck`, and `make tapcheck` path green on both versions.
