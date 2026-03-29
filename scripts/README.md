This directory contains bootstrap, test-wrapper, and benchmark helper scripts.

## Benchmark suite

`benchmark_suite.py` is the canonical benchmark and acceptance driver for the repository. It supports deterministic matrix runs across corpus presets and index methods.

Example:

```sh
uv run python scripts/benchmark_suite.py \
  --host 127.0.0.1 \
  --port 5432 \
  --dbname postgres \
  --profile quick \
  --corpus normalized_dense,clustered \
  --methods turboquant_flat,turboquant_ivf,pgvector_ivfflat,pgvector_hnsw \
  --output benchmark-suite.json
```

Profiles:

- `tiny`
- `quick`
- `medium`
- `full`

Corpus presets:

- `normalized_dense`
- `non_normalized_varied_norms`
- `clustered`
- `mixed_live_dead`

Methods:

- `turboquant_flat`
- `turboquant_ivf`
- `pgvector_ivfflat`
- `pgvector_hnsw`

The compatibility wrapper `benchmark_smoke.py` now forwards to a one-scenario `benchmark_suite.py` run for users that still rely on the older entry point.

The suite now evaluates through the SQL helper `tq_rerank_candidates(...)` and records a `query_api` object plus attached `index_metadata` in the JSON output. Use `--rerank-candidate-limit` to request a larger approximate candidate pool before the helper reranks down to the final benchmark limit.
