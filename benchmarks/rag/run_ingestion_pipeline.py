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
    from benchmarks.rag.ingestion_pipeline import hf_auth_kwargs, load_campaign_config, manifest_payload, run_hf_ingestion
else:
    from .ingestion_pipeline import hf_auth_kwargs, load_campaign_config, manifest_payload, run_hf_ingestion


def build_embedder(model_name: str, normalized: bool):
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    auth_kwargs = hf_auth_kwargs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **auth_kwargs)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        **auth_kwargs,
    ).to(device)
    model.eval()

    def embed(texts):
        vectors = []
        batch_size = 32
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = list(texts[start : start + batch_size])
                batch = tokenizer(
                    chunk,
                    padding="longest",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                embeddings = outputs[0][:, 0]
                if normalized:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                vectors.extend(embeddings.detach().cpu().float().tolist())
        return vectors

    return embed


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
