# RAG Benchmarks

This directory isolates the RAG-facing benchmark harness from the PGXS extension build.

## Layered benchmark strategy

- `VSBT`: keep the existing systems-level ANN benchmark path for build time, index size, latency, WAL, and recall tracking.
- `BERGEN`: use BERGEN as the primary RAG benchmark harness for retrieval and end-to-end QA experiments against PostgreSQL-backed retrievers.
- `BEIR` and `LoTTE`: keep these as optional retrieval-only regression gates once the BERGEN integration is in place.

## Primary dependency

The primary RAG harness is [BERGEN](https://github.com/naver/bergen). This repository does not bundle BERGEN into the core extension tree. Instead, the bootstrap script below clones a pinned BERGEN checkout into `benchmarks/rag/vendor/bergen` and installs its Python dependencies into an isolated virtual environment.

Current pinned BERGEN ref:

- upstream repository: `https://github.com/naver/bergen.git`
- pinned tag: `v0.1`

## Bootstrap

Use the isolated bootstrap script:

```sh
./benchmarks/rag/bootstrap_bergen.sh
```

Useful flags:

- `--dry-run` prints the deterministic setup plan without creating files or installing packages
- `--env-dir /path/to/.venv` overrides the default uv-managed environment location

The bootstrap flow is intentionally separate from `make`, `make install`, `make unitcheck`, and the PostgreSQL TAP wrappers so BERGEN-side experiments cannot mutate the core extension build environment.

## Ingestion pipeline

Use the shared ingestion pipeline entrypoint to validate or inspect a campaign config:

```sh
uv run python benchmarks/rag/run_ingestion_pipeline.py --config benchmarks/rag/configs/campaign.json
```

Campaign configs live in JSON and record:

- dataset name and dataset version
- embedding model and embedding dimension
- chunking strategy, size, and overlap
- normalization settings
- the set of PostgreSQL backends to build for the campaign

The ingestion module keeps schema setup, manifest recording, passage IDs, embedding literals, and backend index-build SQL identical across `pg_turboquant`, `pgvector_hnsw`, and `pgvector_ivfflat`.

## Retrieval-only evaluation

Before any generator is involved, the retrieval evaluation layer computes deterministic:

- Recall@k
- MRR@k
- nDCG@k
- hit rate
- evidence coverage
- latency p50/p95/p99
- throughput

The export surface writes:

- JSON payloads for machine-readable regression checks
- CSV metric rows for lightweight tooling
- compact Markdown tables for human inspection

## Exact rerank mode

The PostgreSQL benchmark path now also supports an explicit two-stage rerank contract:

1. ANN candidate generation from the selected backend
2. exact SQL reranking over the same embeddings inside PostgreSQL

This stays outside the access method and is recorded as a separate benchmark stage so pre-rerank and post-rerank retrieval metrics can be compared directly.

## Primary dataset pack

The repository now carries a declarative primary dataset pack for:

- `kilt_nq`
- `kilt_hotpotqa`
- `kilt_triviaqa`
- `popqa`
- `asqa`

Each dataset config records:

- the benchmark source identifier and split
- the default retrieval and rerank top-k values
- whether evidence handling is enabled
- the answer-metric family expected later in end-to-end evaluation
- the stable fields used to derive passage IDs across runs

## Fixed end-to-end mode

The benchmark harness now also supports a fixed-generator end-to-end mode:

- retrieved PostgreSQL contexts are cached first
- a single fixed generator configuration is applied across backends
- prompts and retrieved contexts are exported for auditability
- end-to-end outputs stay separate from retrieval-only result artifacts

## Operational metrics

The RAG harness now tracks operational measurements separately from answer-quality and retrieval-quality metrics:

- retrieval latency in milliseconds
- rerank latency in milliseconds when exact SQL rerank is enabled
- generator latency in milliseconds for fixed-generator runs
- total end-to-end latency in milliseconds
- retrieved context token budget
- prompt token budget and prompt context count
- optional approximate query-cost fields, each with an explicit unit

Where per-query distributions exist, exports summarize them with `p50`, `p95`, and `p99` rather than collapsing retrieval-only and full end-to-end latency into one generic field.

## HotpotQA multihop overlay

For `kilt_hotpotqa`, the repository now also exposes an optional multihop retrieval overlay:

- it checks whether all declared supporting passages are present in top-k
- it reports `multihop_support_coverage@k` alongside ordinary retrieval metrics
- it emits a machine-readable diagnostic artifact listing which supporting passages were missed per query and per k

This overlay is intentionally separate from the generic retrieval metric pipeline so non-multihop datasets keep the same base evaluation contract.

## Decision Matrix

Use the RAG harnesses with this split:

| Harness | Use when | Current repo contract |
|---|---|---|
| `BERGEN` | end-to-end QA runs or primary PostgreSQL-backed RAG experiments | primary harness for ingestion, retrieval, rerank, and fixed-generator evaluation |
| `BEIR` | retrieval-only regression gate on small standard-style corpora | separate adapter-backed smoke and regression path |
| `LoTTE` | future retrieval-only long-form/task-routing checks | interface scaffold only unless a local dataset path is explicitly added |

The BEIR regression path is intentionally separate from BERGEN orchestration. It reuses the PostgreSQL retriever adapter for retrieval-only checks, while LoTTE remains an optional extension point so future support can land without rewriting the BEIR gate.

## Comparative campaign reporting

The repository now also carries a comparative campaign/report layer for the first serious `pg_turboquant` versus `pgvector` RAG pass:

- campaign plans enumerate the six core method variants:
  `pg_turboquant` approx, `pg_turboquant` approx + rerank, `pgvector` HNSW, `pgvector` HNSW + rerank, `pgvector` IVFFlat, and `pgvector` IVFFlat + rerank
- retrieval-only and end-to-end tables are emitted separately
- JSON plus CSV artifacts remain the reproducible source of truth for every table in the Markdown report
- narrative findings stay split across retrieval quality, answer quality, latency, and footprint
- metric-validity caveats are called out explicitly in the generated report

The repo now also includes a local comparative campaign entrypoint for fixture-backed smoke runs:

```sh
uv run python benchmarks/rag/run_comparative_campaign.py \
  --config benchmarks/rag/configs/comparative/toy_campaign.json \
  --output-dir benchmarks/rag/results/toy-campaign
```

This path is intentionally local and reproducible. It uses saved fixture payloads rather than BERGEN live execution, so the report layer can be exercised in-repo before a full live runner is added.

## Live comparative campaign

The repository now also carries a BERGEN-backed live comparative runner for real PostgreSQL retrieval and end-to-end runs:

```sh
uv run python benchmarks/rag/run_live_campaign.py \
  --output-dir benchmarks/rag/results/live-campaign \
  --dry-run \
  --datasets kilt_nq kilt_hotpotqa popqa
```

The dry-run resolves the comparative plan, dataset configs, BERGEN retriever/generator names, and the six required backend variants without touching PostgreSQL or importing heavy BERGEN runtime dependencies.

For a real run, point the CLI at an existing PostgreSQL corpus plus the three index families:

```sh
uv run python benchmarks/rag/run_live_campaign.py \
  --output-dir benchmarks/rag/results/live-campaign \
  --datasets kilt_nq kilt_hotpotqa popqa \
  --dsn "$RAG_PG_DSN" \
  --table-name "$RAG_TABLE_NAME" \
  --turboquant-index-name "$RAG_TURBOQUANT_INDEX_NAME" \
  --hnsw-index-name "$RAG_HNSW_INDEX_NAME" \
  --ivfflat-index-name "$RAG_IVFFLAT_INDEX_NAME"
```

### Live prerequisites

- Run `./benchmarks/rag/bootstrap_bergen.sh` first, or otherwise provide a valid BERGEN checkout via `BERGEN_ROOT`.
- Install a PostgreSQL driver available to the same Python environment: `psycopg` or `psycopg2`.
- Ensure the BERGEN environment also has `datasets`, `torch`, `transformers`, and `PyYAML`.
- The PostgreSQL passage table must already exist and contain embeddings built with one fixed dense model across all three backend indexes.
- The table's ID column must align with the selected dataset's stable evidence IDs.
  Exact match is preferred.
  For the bundled KILT-style configs and `popqa`, the live runner also accepts composite IDs whose leading component before `:` is the stable ID recorded by the dataset config.
- The live runner does not mix BEIR orchestration into this path.
  BEIR remains in `benchmarks/rag/regression_gate.py`, while the live campaign uses BERGEN-style datasets plus the PostgreSQL adapter stack in `benchmarks/rag/bergen_adapter/`.

### Live environment variables

- `BERGEN_ROOT`: optional path to the vendored BERGEN checkout. Defaults to `benchmarks/rag/vendor/bergen`.
- `RAG_PG_DSN`: PostgreSQL DSN for the live campaign.
- `RAG_TABLE_NAME`: passage table containing `id`, `text`, and embedding columns.
- `RAG_ID_COLUMN`: optional ID column name. Defaults to `passage_id`.
- `RAG_TEXT_COLUMN`: optional passage text column name. Defaults to `passage_text`.
- `RAG_EMBEDDING_COLUMN`: optional embedding column name. Defaults to `embedding`.
- `RAG_QUERY_VECTOR_CAST`: optional PostgreSQL cast for the query vector literal. Defaults to `vector`.
- `RAG_METRIC`: optional metric name. Defaults to `cosine`.
- `RAG_TURBOQUANT_INDEX_NAME`: turboquant index name for the live campaign.
- `RAG_HNSW_INDEX_NAME`: pgvector HNSW index name for the live campaign.
- `RAG_IVFFLAT_INDEX_NAME`: pgvector IVFFlat index name for the live campaign.
- `RAG_RETRIEVER_NAME`: BERGEN retriever config name used to encode live query embeddings. Defaults to `bge-small-en-v1.5`.
- `RAG_GENERATOR_NAME`: BERGEN generator config name for end-to-end runs. Defaults to `oracle_answer`.
- `RAG_PROMPT_NAME`: BERGEN prompt config name for generator-backed runs. Defaults to `basic`.
- `RAG_GENERATION_TOP_K`: optional number of retrieved contexts passed to the generator. Defaults to `5`.
- `RAG_QUERY_LIMIT`: optional per-dataset query cap for small live slices.

### Live artifacts

The live runner writes:

- root-level aggregated comparative artifacts through `benchmarks/rag/campaign_report.py`
- per-scenario retrieval artifacts under `scenarios/<dataset>/<method>/retrieval/`
- per-scenario end-to-end artifacts under `scenarios/<dataset>/<method>/end_to_end/`
- a saved `live-campaign-config.json` with the resolved dataset pack, backend knobs, and redacted DSN metadata

The existing fixture-backed smoke path remains available as the fallback contract:

```sh
uv run python benchmarks/rag/run_comparative_campaign.py \
  --config benchmarks/rag/configs/comparative/toy_campaign.json \
  --output-dir benchmarks/rag/results/toy-campaign
```

## Expected layout

- `benchmarks/rag/bootstrap_bergen.sh` — isolated Python environment bootstrapper
- `benchmarks/rag/bergen_adapter/` — backend-neutral PostgreSQL retriever adapter used by later BERGEN integrations
  Current concrete backend coverage includes `pg_turboquant`, `pgvector_hnsw`, and `pgvector_ivfflat`, each behind the same adapter contract.
- `benchmarks/rag/configs/datasets/` — declarative primary dataset configs for BERGEN-facing runs
- `benchmarks/rag/configs/comparative/` — local comparative campaign configs and fixture payloads
- `benchmarks/rag/configs/regression/` — local BEIR smoke configs and fixtures for retrieval-only regression gating
- `benchmarks/rag/campaign_report.py` — comparative campaign planning, artifact emission, and narrative report generation
- `benchmarks/rag/live_campaign.py` — testable BERGEN-backed live campaign orchestration helpers
- `benchmarks/rag/dataset_pack.py` — schema validation and benchmark-plan resolution for the primary dataset pack
- `benchmarks/rag/end_to_end.py` — fixed-generator prompt building, cache consumption, and end-to-end exports
- `benchmarks/rag/ingestion_pipeline.py` — shared ingestion, manifest, embedding, and backend index-build helpers
- `benchmarks/rag/multihop_eval.py` — optional HotpotQA-style multihop support-coverage metrics and machine-readable diagnostics
- `benchmarks/rag/operational_metrics.py` — deterministic token-budget estimation plus stage-latency and cost-summary aggregation
- `benchmarks/rag/regression_gate.py` — BEIR-style retrieval-only regression helpers plus optional LoTTE harness selection scaffolding
- `benchmarks/rag/rerank_eval.py` — explicit exact SQL rerank planning and dual-stage export helpers
- `benchmarks/rag/retrieval_eval.py` — reusable retrieval-only metrics and export helpers
- `benchmarks/rag/run_comparative_campaign.py` — CLI entrypoint for local fixture-backed comparative RAG campaign runs
- `benchmarks/rag/run_live_campaign.py` — CLI entrypoint for BERGEN-backed live comparative RAG campaign runs
- `benchmarks/rag/run_ingestion_pipeline.py` — small CLI entrypoint for campaign config inspection and future ingestion runs
- `benchmarks/rag/requirements-bergen.txt` — local requirements entrypoint that delegates to the vendored BERGEN checkout
- `benchmarks/rag/vendor/` — pinned upstream checkouts created by the bootstrap script
- `benchmarks/rag/results/` — future benchmark outputs
- `benchmarks/rag/configs/` — future BERGEN adapter and dataset configs
