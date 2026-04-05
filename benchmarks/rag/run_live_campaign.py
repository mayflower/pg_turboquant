#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.rag.runtime_env import maybe_reexec_into_bergen_venv

    maybe_reexec_into_bergen_venv(script_path=Path(__file__).resolve())
    from benchmarks.rag.live_campaign import (
        DEFAULT_DATASETS,
        DEFAULT_GENERATION_TOP_K,
        build_live_campaign_runtime,
        run_live_campaign,
    )
else:
    from .live_campaign import (
        DEFAULT_DATASETS,
        DEFAULT_GENERATION_TOP_K,
        build_live_campaign_runtime,
        run_live_campaign,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BERGEN-backed live RAG benchmark against PostgreSQL")
    parser.add_argument("--output-dir", required=True, help="Directory for generated campaign artifacts")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset IDs to include. Supported: kilt_nq kilt_hotpotqa popqa kilt_triviaqa",
    )
    parser.add_argument(
        "--generator-name",
        default=os.environ.get("RAG_GENERATOR_NAME", "oracle_answer"),
        help="BERGEN generator config name",
    )
    parser.add_argument(
        "--retriever-name",
        default=os.environ.get("RAG_RETRIEVER_NAME", "bge-small-en-v1.5"),
        help="BERGEN retriever config name used for live query encoding",
    )
    parser.add_argument(
        "--prompt-name",
        default=os.environ.get("RAG_PROMPT_NAME", "basic"),
        help="BERGEN prompt config name for end-to-end generation",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved live campaign plan and exit")

    parser.add_argument("--dsn", default=os.environ.get("RAG_PG_DSN"), help="PostgreSQL DSN")
    parser.add_argument("--table-name", default=os.environ.get("RAG_TABLE_NAME"), help="Passage table name")
    parser.add_argument("--id-column", default=os.environ.get("RAG_ID_COLUMN", "passage_id"))
    parser.add_argument("--text-column", default=os.environ.get("RAG_TEXT_COLUMN", "passage_text"))
    parser.add_argument("--embedding-column", default=os.environ.get("RAG_EMBEDDING_COLUMN", "embedding"))
    parser.add_argument("--query-vector-cast", default=os.environ.get("RAG_QUERY_VECTOR_CAST", "vector"))
    parser.add_argument("--metric", default=os.environ.get("RAG_METRIC", "cosine"))
    parser.add_argument(
        "--turboquant-index-name",
        default=os.environ.get("RAG_TURBOQUANT_INDEX_NAME"),
        help="Index name for the pg_turboquant backend",
    )
    parser.add_argument(
        "--hnsw-index-name",
        default=os.environ.get("RAG_HNSW_INDEX_NAME"),
        help="Index name for the pgvector HNSW backend",
    )
    parser.add_argument(
        "--ivfflat-index-name",
        default=os.environ.get("RAG_IVFFLAT_INDEX_NAME"),
        help="Index name for the pgvector IVFFlat backend",
    )
    parser.add_argument(
        "--generation-top-k",
        type=int,
        default=int(os.environ.get("RAG_GENERATION_TOP_K", str(DEFAULT_GENERATION_TOP_K))),
        help="How many retrieved contexts to pass into the generator",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=int(os.environ["RAG_QUERY_LIMIT"]) if "RAG_QUERY_LIMIT" in os.environ else None,
        help="Optional limit for the number of live queries per dataset",
    )
    parser.add_argument(
        "--bergen-root",
        default=os.environ.get("BERGEN_ROOT"),
        help="Path to the vendored BERGEN checkout",
    )
    parser.add_argument(
        "--no-backend-isolation",
        action="store_true",
        help="Reuse the shared source table directly instead of cloning one backend-isolated table per index family",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = build_live_campaign_runtime(
        dataset_ids=args.datasets,
        generator_name=args.generator_name,
        retriever_name=args.retriever_name,
        prompt_name=args.prompt_name,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "plan": runtime["plan"],
                    "generator_name": runtime["generator_name"],
                    "retriever_name": runtime["retriever_name"],
                    "prompt_name": runtime["prompt_name"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    required = {
        "dsn": args.dsn,
        "table_name": args.table_name,
        "turboquant_index_name": args.turboquant_index_name,
        "hnsw_index_name": args.hnsw_index_name,
        "ivfflat_index_name": args.ivfflat_index_name,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise SystemExit(f"missing required live-run arguments: {', '.join(missing)}")

    result = run_live_campaign(
        output_dir=Path(args.output_dir),
        runtime=runtime,
        dsn=args.dsn,
        table_name=args.table_name,
        id_column=args.id_column,
        text_column=args.text_column,
        embedding_column=args.embedding_column,
        metric=args.metric,
        query_vector_cast=args.query_vector_cast,
        turboquant_index_name=args.turboquant_index_name,
        hnsw_index_name=args.hnsw_index_name,
        ivfflat_index_name=args.ivfflat_index_name,
        query_limit=args.query_limit,
        generation_top_k=args.generation_top_k,
        bergen_root=args.bergen_root,
        prompt_name=args.prompt_name,
        backend_isolation=not args.no_backend_isolation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
