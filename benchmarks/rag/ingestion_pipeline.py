from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .bergen_adapter.adapter import vector_literal


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    version: str
    source_path: str


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    dimension: int
    normalized: bool


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class CampaignConfig:
    dataset: DatasetConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    schema: dict[str, str]
    backends: list[dict[str, Any]]


def hf_auth_kwargs(env: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if env is None else env
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        token = source.get(key)
        if token:
            return {"token": token}
    return {}


def load_campaign_config(path: str | Path) -> CampaignConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return CampaignConfig(
        dataset=DatasetConfig(**payload["dataset"]),
        embedding=EmbeddingConfig(**payload["embedding"]),
        chunking=ChunkingConfig(**payload["chunking"]),
        schema=dict(payload["schema"]),
        backends=[dict(item) for item in payload["backends"]],
    )


def parse_hf_source_path(source_path: str) -> dict[str, Any]:
    match = re.fullmatch(r"hf://(?P<path>[^\[]+)\[(?P<split>[^\]]+)\](?:\[:(?P<limit>\d+)\])?", source_path)
    if match is None:
        raise ValueError(f"unsupported HF source path: {source_path}")

    path = match.group("path")
    dataset_name: str
    subset_name: str | None
    if path == "kilt_tasks/nq":
        dataset_name = "kilt_tasks"
        subset_name = "nq"
    elif path == "kilt_tasks/hotpotqa":
        dataset_name = "kilt_tasks"
        subset_name = "hotpotqa"
    else:
        dataset_name = path
        subset_name = None

    limit = match.group("limit")
    return {
        "dataset_name": dataset_name,
        "subset_name": subset_name,
        "split_name": match.group("split"),
        "limit": int(limit) if limit is not None else None,
    }


def build_hf_corpus(
    config: CampaignConfig,
    *,
    dataset_loader: Callable[[str, str | None, str], Sequence[dict[str, Any]]] | None = None,
    limit_override: int | None = None,
) -> list[dict[str, Any]]:
    if not config.dataset.source_path.startswith("hf://"):
        raise ValueError(f"unsupported source path for HF ingestion: {config.dataset.source_path}")

    parsed = parse_hf_source_path(config.dataset.source_path)
    if dataset_loader is None:
        dataset_loader = _default_hf_dataset_loader

    rows = list(
        dataset_loader(
            parsed["dataset_name"],
            parsed["subset_name"],
            parsed["split_name"],
        )
    )
    limit = limit_override if limit_override is not None else parsed["limit"]
    if limit is not None:
        rows = rows[:limit]

    dataset_key = (parsed["dataset_name"], parsed["subset_name"])
    if dataset_key == ("akariasai/PopQA", None):
        return _build_popqa_corpus(rows)
    if dataset_key == ("kilt_tasks", "nq"):
        return _build_kilt_nq_corpus(rows)
    if dataset_key == ("kilt_tasks", "hotpotqa"):
        return _build_kilt_nq_corpus(rows)
    raise ValueError(
        f"unsupported HF ingestion dataset source: {parsed['dataset_name']}"
        + (f"/{parsed['subset_name']}" if parsed["subset_name"] else "")
    )


def _default_hf_dataset_loader(
    dataset_name: str,
    subset_name: str | None,
    split_name: str,
) -> Sequence[dict[str, Any]]:
    import datasets  # type: ignore

    auth_kwargs = hf_auth_kwargs()
    if subset_name is None:
        return datasets.load_dataset(dataset_name, **auth_kwargs)[split_name]
    return datasets.load_dataset(dataset_name, subset_name, **auth_kwargs)[split_name]


def _build_popqa_corpus(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    seen_document_ids: set[str] = set()
    for item in rows:
        document_id = str(item["subj_id"])
        if document_id in seen_document_ids:
            continue
        seen_document_ids.add(document_id)
        subject = str(item["subj"])
        property_name = str(item["prop"])
        object_name = str(item["obj"])
        question = str(item["question"])
        corpus.append(
            {
                "document_id": document_id,
                "title": subject,
                "passages": [
                    {
                        "passage_id": f"{document_id}:0",
                        "text": (
                            f"{subject} | property: {property_name} | "
                            f"answer: {object_name} | question form: {question}"
                        ),
                    }
                ],
            }
        )
    return corpus


def _build_kilt_nq_corpus(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    seen_document_ids: set[str] = set()
    for item in rows:
        question = str(item["input"])
        for output in item.get("output", []):
            answer = output.get("answer")
            if not answer:
                continue
            for provenance in output.get("provenance", []):
                wiki_id = provenance.get("wikipedia_id")
                if wiki_id is None:
                    continue
                document_id = str(wiki_id)
                if document_id in seen_document_ids:
                    continue
                seen_document_ids.add(document_id)
                corpus.append(
                    {
                        "document_id": document_id,
                        "title": f"wiki:{document_id}",
                        "passages": [
                            {
                                "passage_id": f"{document_id}:0",
                                "text": (
                                    f"Wikipedia id {document_id} | "
                                    f"question: {question} | answer: {answer}"
                                ),
                            }
                        ],
                    }
                )
    return corpus


def manifest_payload(config: CampaignConfig) -> dict[str, Any]:
    return {
        "dataset_name": config.dataset.name,
        "dataset_version": config.dataset.version,
        "embedding_model": config.embedding.model,
        "dimension": config.embedding.dimension,
        "normalization": config.embedding.normalized,
        "chunking": {
            "strategy": config.chunking.strategy,
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
        },
    }


def ensure_schema(connection: Any, config: CampaignConfig) -> None:
    documents_table = config.schema["documents_table"]
    passages_table = config.schema["passages_table"]
    with connection.cursor() as cursor:
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {documents_table} ("
            "document_id text PRIMARY KEY, "
            "title text NOT NULL)"
        )
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {passages_table} ("
            "passage_id text PRIMARY KEY, "
            "document_id text NOT NULL, "
            "chunk_index integer NOT NULL, "
            "passage_text text NOT NULL, "
            "embedding vector NOT NULL)"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS rag_campaign_manifest ("
            "dataset_name text NOT NULL, "
            "dataset_version text NOT NULL, "
            "embedding_model text NOT NULL, "
            "dimension integer NOT NULL, "
            "normalization boolean NOT NULL, "
            "chunking_json jsonb NOT NULL, "
            "PRIMARY KEY (dataset_name, dataset_version, embedding_model)"
            ")"
        )


def upsert_manifest(connection: Any, config: CampaignConfig) -> dict[str, Any]:
    payload = manifest_payload(config)
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO rag_campaign_manifest ("
            "dataset_name, dataset_version, embedding_model, dimension, normalization, chunking_json"
            ") VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (dataset_name, dataset_version, embedding_model) "
            "DO UPDATE SET "
            "dimension = EXCLUDED.dimension, "
            "normalization = EXCLUDED.normalization, "
            "chunking_json = EXCLUDED.chunking_json",
            (
                payload["dataset_name"],
                payload["dataset_version"],
                payload["embedding_model"],
                payload["dimension"],
                payload["normalization"],
                json.dumps(payload["chunking"]),
            ),
        )
    return payload


def ingest_corpus(
    connection: Any,
    config: CampaignConfig,
    corpus: Sequence[dict[str, Any]],
    embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]],
) -> None:
    documents_table = config.schema["documents_table"]
    passages_table = config.schema["passages_table"]

    passage_records: list[tuple[str, str, int, str]] = []
    document_records: list[tuple[str, str]] = []
    for item in corpus:
        document_records.append((item["document_id"], item.get("title", item["document_id"])))
        for chunk_index, passage in enumerate(item["passages"]):
            passage_records.append(
                (
                    passage["passage_id"],
                    item["document_id"],
                    chunk_index,
                    passage["text"],
                )
            )

    vectors = embedder([record[3] for record in passage_records])
    if len(vectors) != len(passage_records):
        raise ValueError("embedder returned unexpected vector count")

    with connection.cursor() as cursor:
        for document_id, title in document_records:
            cursor.execute(
                f"INSERT INTO {documents_table} (document_id, title) VALUES (%s, %s) "
                "ON CONFLICT (document_id) DO UPDATE SET title = EXCLUDED.title",
                (document_id, title),
            )

        for record, vector in zip(passage_records, vectors):
            cursor.execute(
                f"INSERT INTO {passages_table} (passage_id, document_id, chunk_index, passage_text, embedding) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (passage_id) DO UPDATE SET "
                "document_id = EXCLUDED.document_id, "
                "chunk_index = EXCLUDED.chunk_index, "
                "passage_text = EXCLUDED.passage_text, "
                "embedding = EXCLUDED.embedding",
                (
                    record[0],
                    record[1],
                    record[2],
                    record[3],
                    vector_literal(vector),
                ),
            )


def build_backend_indexes(connection: Any, config: CampaignConfig) -> None:
    passages_table = config.schema["passages_table"]
    for backend in config.backends:
        index_kind = backend["kind"]
        index_name = backend["index_name"]
        metric = backend["metric"]
        with connection.cursor() as cursor:
            cursor.execute(build_index_sql(passages_table, index_kind, index_name, metric, backend))


def prepare_fixed_embedding_column(connection: Any, config: CampaignConfig) -> None:
    passages_table = config.schema["passages_table"]
    with connection.cursor() as cursor:
        cursor.execute(
            f"ALTER TABLE {passages_table} "
            f"ALTER COLUMN embedding TYPE vector({config.embedding.dimension}) "
            f"USING embedding::vector({config.embedding.dimension})"
        )


def build_index_sql(
    passages_table: str,
    index_kind: str,
    index_name: str,
    metric: str,
    backend: dict[str, Any],
) -> str:
    if index_kind == "pg_turboquant":
        opclass = {
            "cosine": "tq_cosine_ops",
            "inner_product": "tq_ip_ops",
            "l2": "tq_l2_ops",
        }[metric]
        options = render_with_options(backend.get("options", {}))
        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {passages_table} USING turboquant (embedding {opclass}){options}"
        )
    if index_kind == "pgvector_hnsw":
        opclass = {
            "cosine": "vector_cosine_ops",
            "inner_product": "vector_ip_ops",
            "l2": "vector_l2_ops",
        }[metric]
        options = render_with_options(backend.get("options", {}))
        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {passages_table} USING hnsw (embedding {opclass}){options}"
        )
    if index_kind == "pgvector_ivfflat":
        opclass = {
            "cosine": "vector_cosine_ops",
            "inner_product": "vector_ip_ops",
            "l2": "vector_l2_ops",
        }[metric]
        options = render_with_options(backend.get("options", {}))
        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {passages_table} USING ivfflat (embedding {opclass}){options}"
        )
    raise ValueError(f"unsupported backend kind: {index_kind}")


def render_with_options(options: dict[str, Any]) -> str:
    if not options:
        return ""
    rendered = ", ".join(f"{key} = {render_option_value(value)}" for key, value in options.items())
    return f" WITH ({rendered})"


def render_option_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f"'{value}'"
    return str(value)


def run_campaign(
    connection: Any,
    config: CampaignConfig,
    corpus: Sequence[dict[str, Any]],
    embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]],
) -> dict[str, Any]:
    ensure_schema(connection, config)
    payload = upsert_manifest(connection, config)
    ingest_corpus(connection, config, corpus, embedder)
    build_backend_indexes(connection, config)
    if hasattr(connection, "commit"):
        connection.commit()
    return payload


def run_hf_ingestion(
    connection: Any,
    config: CampaignConfig,
    *,
    embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    dataset_loader: Callable[[str, str | None, str], Sequence[dict[str, Any]]] | None = None,
    limit_override: int | None = None,
) -> dict[str, Any]:
    corpus = build_hf_corpus(
        config,
        dataset_loader=dataset_loader,
        limit_override=limit_override,
    )
    ensure_schema(connection, config)
    prepare_fixed_embedding_column(connection, config)
    payload = upsert_manifest(connection, config)
    ingest_corpus(connection, config, corpus, embedder)
    build_backend_indexes(connection, config)
    if hasattr(connection, "commit"):
        connection.commit()
    return payload
