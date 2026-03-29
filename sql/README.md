This directory contains versioned extension install and upgrade scripts.

Current packaging contract:

- `pg_turboquant--0.1.0.sql` remains the historical install baseline
- `pg_turboquant--0.1.4.sql` is the current default install target
- `pg_turboquant--0.1.0--0.1.1.sql`, `pg_turboquant--0.1.1--0.1.2.sql`, `pg_turboquant--0.1.2--0.1.3.sql`, and `pg_turboquant--0.1.3--0.1.4.sql` are the tested upgrade path chain

On-disk index rebuild requirements are driven by page-format changes, not by every extension SQL version bump.
