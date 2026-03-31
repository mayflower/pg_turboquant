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
- `hotpot_skewed`
- `hotpot_overlap`

Methods:

- `turboquant_flat`
- `turboquant_ivf`
- `pgvector_ivfflat`
- `pgvector_hnsw`

The compatibility wrapper `benchmark_smoke.py` now forwards to a one-scenario `benchmark_suite.py` run for users that still rely on the older entry point.

The suite records a `query_api` object plus attached `index_metadata` in the JSON output. Non-TurboQuant ordered evaluation still uses `tq_rerank_candidates(...)`; the TurboQuant ordered timing path now uses a single-batch inline equivalent so `tq_last_scan_stats()` reflects the measured ANN scan. Use `--rerank-candidate-limit` to request a larger approximate candidate pool before reranking down to the final benchmark limit.

Use `--microbench` to attach a JSON-only prod score microbenchmark section to the benchmark payload. For a standalone run, `scripts/prod_score_microbench.py` builds and runs `tests/perf/test_prod_score_microbench` and emits only machine-readable JSON.

Prompt-08 regression harness additions:

- `microbenchmarks.results` keeps the raw row-level timings and counters.
- `microbenchmarks.comparisons` derives stable pairwise comparisons for scalar vs AVX2, scalar vs NEON, float LUT vs quantized LUT reference, and block-local page selection vs global-heap-only page selection.
- `microbenchmarks.regression_gates` adds directional machine-readable statuses (`pass`, `warn`, `not_applicable`) with explicit checks instead of collapsing the pack to one noisy threshold.
- `microbenchmarks.interpretation_notes` is the short reader guide carried through into the generated report artifacts.

Current prompt-00 contract lock for the Qprod/QJL speed pack:

- `tq_prod` is the repository's current Qprod/QJL-style scorer, not a pre-Qprod magnitude/sign redesign target.
- Normalized `tq_prod` cosine/IP scans keep the existing code-domain fast path as the baseline contract for this pack.
- Decode-scored paths remain a compatibility and diagnostics seam; baseline prompt `00` only freezes parity and observability around the current scorer before later kernel work.
- The prod score microbenchmark now records used kernel plus code/page throughput and candidate-heap insert/replace/reject counters in machine-readable JSON so later prompts can compare kernel changes without changing query semantics.
- The prompt-08 comparison layer now keeps visited code/page counts and heap-churn deltas beside throughput ratios so regressions can be judged on preserved work as well as speed.

Current prompt-06 near-exhaustive crossover contract:

- IVF ordered scans now switch from bounded-page orchestration to `ivf_near_exhaustive` once the selected live rows or selected batch pages reach at least 70% of the corresponding ranked-list totals for that query.
- The benchmark JSON exposes this through `scan_stats.scan_orchestration` plus `scan_stats.near_exhaustive_crossover`, alongside the existing selected/visited live/page counters.
- The threshold is intentionally conservative: overlap-heavy workloads where probe selection is already close to a full scan skip page-bound computation and sorting, while narrower skewed probes stay on the bounded-page pruning path.

When `--report` is enabled the suite also emits:

- `benchmark-report.json`
- `benchmark-report.md`
- `benchmark-report.html`

The report is intentionally factual and configuration-specific. It lists measured recall, p50/p95 latency, footprint, scan-work fields such as `visited_code_count` and `selected_page_count`, plus the observed `score_kernel`, without making blanket claims outside the selected corpus/profile/knob matrix.

When `--microbench --report` is enabled, the report also adds a `Microbenchmark Regression` section that mirrors the JSON comparison/gate rows. Read it as a directional regression surface:

- kernel gates can be `not_applicable` when the requested SIMD path is unavailable on the current machine
- LUT and heap-selection gates keep equal-workload counters in view, so a small wall-clock wobble does not erase a real work-reduction signal
