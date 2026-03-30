#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.rag.runtime_env import maybe_reexec_into_bergen_venv

    if "--apply" in sys.argv:
        maybe_reexec_into_bergen_venv(script_path=Path(__file__).resolve())
    from benchmarks.rag.ingestion_pipeline import (
        build_embedder,
        hf_auth_kwargs,
        load_campaign_config,
        manifest_payload,
        run_hf_ingestion,
    )
else:
    from .ingestion_pipeline import build_embedder, hf_auth_kwargs, load_campaign_config, manifest_payload, run_hf_ingestion


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a RAG ingestion campaign config")
    parser.add_argument("--config", required=True, help="Path to JSON campaign config")
    parser.add_argument("--apply", action="store_true", help="Load the configured HF subset into PostgreSQL")
    parser.add_argument("--dsn", help="PostgreSQL DSN for --apply mode")
    parser.add_argument("--limit", type=int, help="Override the configured source subset limit")
    args = parser.parse_args()

    config = load_campaign_config(Path(args.config))
    if not args.apply:
        print(json.dumps(manifest_payload(config), indent=2, sort_keys=True))
        return 0

    if not args.dsn:
        raise SystemExit("--dsn is required with --apply")

    import psycopg  # type: ignore

    embedder = build_embedder(config.embedding.model, config.embedding.normalized)
    with psycopg.connect(args.dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_turboquant")
            cursor.execute(f"DROP TABLE IF EXISTS {config.schema['passages_table']} CASCADE")
            cursor.execute(f"DROP TABLE IF EXISTS {config.schema['documents_table']} CASCADE")
        payload = run_hf_ingestion(
            connection,
            config,
            embedder=embedder,
            limit_override=args.limit,
        )

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
