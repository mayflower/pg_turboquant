from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from .bergen_adapter import PassageTable, PostgresRetrieverAdapter, RetrievalRequest
from .bergen_adapter.pgvector_backends import (
    MODE_APPROX as PGVECTOR_MODE_APPROX,
    MODE_APPROX_RERANK as PGVECTOR_MODE_APPROX_RERANK,
    PgvectorHnswBackend,
    PgvectorIvfflatBackend,
)
from .bergen_adapter.turboquant_backend import (
    MODE_APPROX as TURBOQUANT_MODE_APPROX,
    MODE_APPROX_RERANK as TURBOQUANT_MODE_APPROX_RERANK,
    PgTurboquantBackend,
)
from .campaign_report import build_comparative_campaign_plan, run_comparative_campaign
from .dataset_pack import load_primary_dataset_pack
from .end_to_end import export_end_to_end_run
from .ingestion_pipeline import CampaignConfig, build_embedder, load_campaign_config, run_hf_ingestion
from .operational_metrics import QueryOperationalMetrics
from .rerank_eval import export_two_stage_retrieval_run
from .retrieval_eval import QueryEvaluation, compute_retrieval_metrics, export_retrieval_run


DEFAULT_DATASETS = ("kilt_nq", "kilt_hotpotqa", "popqa")
SUPPORTED_DATASETS = frozenset({"kilt_nq", "kilt_hotpotqa", "kilt_triviaqa", "popqa"})

DEFAULT_TURBOQUANT_PROBES = 8
DEFAULT_TURBOQUANT_OVERSAMPLING = 4
DEFAULT_TURBOQUANT_MAX_VISITED_CODES = 4096
DEFAULT_TURBOQUANT_MAX_VISITED_PAGES = 0
DEFAULT_HNSW_EF_SEARCH = 80
DEFAULT_IVFFLAT_PROBES = 8
DEFAULT_GENERATION_TOP_K = 5
BACKEND_FAMILIES = ("pg_turboquant", "pgvector_hnsw", "pgvector_ivfflat")
LIVE_CONFIG_PATHS = {
    "kilt_nq": Path(__file__).resolve().parent / "configs" / "live" / "kilt_nq_small_live.json",
    "kilt_hotpotqa": Path(__file__).resolve().parent / "configs" / "live" / "kilt_hotpotqa_ivf_live.json",
    "popqa": Path(__file__).resolve().parent / "configs" / "live" / "popqa_small_live.json",
}


@dataclass(frozen=True)
class QuerySample:
    query_id: str
    question: str
    answers: list[str]
    relevant_ids: list[str]
    evidence_ids: list[str]


def build_live_campaign_runtime(
    *,
    dataset_ids: Sequence[str] | None = None,
    generator_name: str = "oracle_answer",
    retriever_name: str = "bge-small-en-v1.5",
    prompt_name: str = "basic",
) -> dict[str, object]:
    pack = load_primary_dataset_pack()
    selected_dataset_ids = list(dataset_ids or DEFAULT_DATASETS)
    invalid = sorted(set(selected_dataset_ids) - SUPPORTED_DATASETS)
    if invalid:
        raise ValueError(f"unsupported live campaign dataset(s): {', '.join(invalid)}")

    dataset_configs = {dataset_id: pack[dataset_id] for dataset_id in selected_dataset_ids}
    plan = build_comparative_campaign_plan(
        dataset_ids=selected_dataset_ids,
        generator_id=generator_name,
    )
    return {
        "plan": plan,
        "dataset_configs": dataset_configs,
        "generator_name": generator_name,
        "retriever_name": retriever_name,
        "prompt_name": prompt_name,
    }


def run_live_campaign(
    *,
    output_dir: str | Path,
    runtime: dict[str, object],
    dsn: str,
    table_name: str,
    id_column: str,
    text_column: str,
    embedding_column: str,
    metric: str,
    turboquant_index_name: str,
    hnsw_index_name: str,
    ivfflat_index_name: str,
    query_vector_cast: str = "vector",
    query_limit: int | None = None,
    generation_top_k: int = DEFAULT_GENERATION_TOP_K,
    connect_fn: Callable[[str], Any] | None = None,
    dataset_loader: Callable[[str, dict[str, Any], int | None], list[QuerySample]] | None = None,
    query_encoder: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
    generator_runner: Callable[[QuerySample, list[dict[str, str]], dict[str, Any]], dict[str, str]]
    | None = None,
    bergen_root: str | Path | None = None,
    prompt_name: str | None = None,
    turboquant_probes: int = DEFAULT_TURBOQUANT_PROBES,
    turboquant_oversampling: int = DEFAULT_TURBOQUANT_OVERSAMPLING,
    turboquant_max_visited_codes: int = DEFAULT_TURBOQUANT_MAX_VISITED_CODES,
    turboquant_max_visited_pages: int = DEFAULT_TURBOQUANT_MAX_VISITED_PAGES,
    hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
    ivfflat_probes: int = DEFAULT_IVFFLAT_PROBES,
    backend_isolation: bool = True,
    clock_fn: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = dict(runtime["plan"])
    dataset_configs = {
        dataset_id: dict(config) for dataset_id, config in dict(runtime["dataset_configs"]).items()
    }
    resolved_prompt_name = prompt_name or str(runtime["prompt_name"])

    if connect_fn is None:
        connect_fn = _default_connect_fn()
    if dataset_loader is None:
        dataset_loader = _build_default_dataset_loader()
    if query_encoder is None:
        query_encoder = _build_default_query_encoder(
            bergen_root=bergen_root,
            retriever_name=str(runtime["retriever_name"]),
        )
    if generator_runner is None:
        generator_runner = _build_default_generator_runner(
            bergen_root=bergen_root,
            generator_name=str(runtime["generator_name"]),
            prompt_name=resolved_prompt_name,
        )

    run_config = {
        "campaign_kind": plan["campaign_kind"],
        "datasets": list(plan["datasets"]),
        "generator_name": runtime["generator_name"],
        "retriever_name": runtime["retriever_name"],
        "prompt_name": resolved_prompt_name,
        "dsn_redacted": _redact_dsn(dsn),
        "table": {
            "table_name": table_name,
            "id_column": id_column,
            "text_column": text_column,
            "embedding_column": embedding_column,
            "query_vector_cast": query_vector_cast,
            "metric": metric,
        },
        "indexes": {
            "pg_turboquant": turboquant_index_name,
            "pgvector_hnsw": hnsw_index_name,
            "pgvector_ivfflat": ivfflat_index_name,
        },
        "ann_defaults": {
            "turboquant": {
                "probes": turboquant_probes,
                "oversampling": turboquant_oversampling,
                "max_visited_codes": turboquant_max_visited_codes,
                "max_visited_pages": turboquant_max_visited_pages,
            },
            "pgvector_hnsw": {"ef_search": hnsw_ef_search},
            "pgvector_ivfflat": {"probes": ivfflat_probes},
        },
        "backend_isolation": backend_isolation,
        "query_limit": query_limit,
        "generation_top_k": generation_top_k,
        "dataset_configs": dataset_configs,
    }
    (output_dir / "live-campaign-config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    passage_table = PassageTable(
        table_name=table_name,
        id_column=id_column,
        text_column=text_column,
        embedding_column=embedding_column,
        query_vector_cast=query_vector_cast,
    )

    answer_metrics_fn = _build_answer_metrics_computer(bergen_root)
    retrieval_payloads: dict[tuple[str, str], dict[str, object]] = {}
    end_to_end_payloads: dict[tuple[str, str], dict[str, object]] = {}
    dataset_source_layouts: dict[str, dict[str, object]] = {}

    for dataset_id in plan["datasets"]:
        dataset_config = dataset_configs[dataset_id]
        dataset_samples = dataset_loader(dataset_id, dataset_config, query_limit)
        if not dataset_samples:
            raise ValueError(f"dataset loader returned no queries for dataset: {dataset_id}")

        dataset_layout = _prepare_dataset_source_layout(
            dataset_id=dataset_id,
            output_dir=output_dir,
            dsn=dsn,
            connect_fn=connect_fn,
            metric=metric,
            query_vector_cast=query_vector_cast,
            passage_table=passage_table,
        )
        dataset_source_layouts[dataset_id] = _serialize_dataset_source_layout(dataset_layout)
        isolation_layout = (
            _prepare_backend_isolated_layout(
                dsn=dsn,
                connect_fn=connect_fn,
                source_table=dataset_layout["passage_table"],
                output_dir=output_dir / dataset_id,
                turboquant_index_name=dataset_layout["index_names"]["pg_turboquant"],
                hnsw_index_name=dataset_layout["index_names"]["pgvector_hnsw"],
                ivfflat_index_name=dataset_layout["index_names"]["pgvector_ivfflat"],
            )
            if backend_isolation
            else {}
        )

        dataset_top_k = int(dataset_config["retrieval_profile"]["top_k_default"])
        rerank_top_k = int(dataset_config["retrieval_profile"].get("rerank_top_k_default") or dataset_top_k)
        eval_ks = _metric_ks(dataset_top_k)

        for variant in plan["method_variants"]:
            method_id = str(variant["method_id"])
            backend_family = str(variant["backend_family"])
            scenario_table = _resolve_scenario_passage_table(
                dataset_layout["passage_table"], isolation_layout, backend_family
            )
            scenario_index_names = _resolve_scenario_index_names(
                isolation_layout=isolation_layout,
                turboquant_index_name=dataset_layout["index_names"]["pg_turboquant"],
                hnsw_index_name=dataset_layout["index_names"]["pgvector_hnsw"],
                ivfflat_index_name=dataset_layout["index_names"]["pgvector_ivfflat"],
            )
            scenario_root = output_dir / "scenarios" / dataset_id / method_id
            retrieval_root = scenario_root / "retrieval"
            end_to_end_root = scenario_root / "end_to_end"
            retrieval_root.mkdir(parents=True, exist_ok=True)
            end_to_end_root.mkdir(parents=True, exist_ok=True)

            retrieval_result = _run_retrieval_scenario(
                dsn=dsn,
                connect_fn=connect_fn,
                passage_table=scenario_table,
                dataset_config=dataset_config,
                dataset_samples=dataset_samples,
                method_id=method_id,
                metric=metric,
                query_encoder=query_encoder,
                dataset_top_k=dataset_top_k,
                rerank_top_k=rerank_top_k,
                eval_ks=eval_ks,
                retrieval_root=retrieval_root,
                clock_fn=clock_fn,
                turboquant_index_name=scenario_index_names["pg_turboquant"],
                hnsw_index_name=scenario_index_names["pgvector_hnsw"],
                ivfflat_index_name=scenario_index_names["pgvector_ivfflat"],
                turboquant_probes=turboquant_probes,
                turboquant_oversampling=turboquant_oversampling,
                turboquant_max_visited_codes=turboquant_max_visited_codes,
                turboquant_max_visited_pages=turboquant_max_visited_pages,
                hnsw_ef_search=hnsw_ef_search,
                ivfflat_probes=ivfflat_probes,
            )
            retrieval_payloads[(dataset_id, method_id)] = retrieval_result

            end_to_end_result = _run_end_to_end_scenario(
                dataset_config=dataset_config,
                dataset_samples=dataset_samples,
                method_id=method_id,
                generation_top_k=generation_top_k,
                retrieval_result=retrieval_result,
                end_to_end_root=end_to_end_root,
                generator_runner=generator_runner,
                answer_metrics_fn=answer_metrics_fn,
                clock_fn=clock_fn,
            )
            end_to_end_payloads[(dataset_id, method_id)] = end_to_end_result

    if dataset_source_layouts:
        run_config["dataset_source_layouts"] = dataset_source_layouts
    (output_dir / "live-campaign-config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    artifacts = run_comparative_campaign(
        output_dir=output_dir,
        plan=plan,
        retrieval_runner=lambda scenario: retrieval_payloads[(scenario["dataset_id"], scenario["method_id"])],
        end_to_end_runner=lambda scenario, _retrieval: end_to_end_payloads[
            (scenario["dataset_id"], scenario["method_id"])
        ],
    )
    return {"plan": plan, "artifacts": artifacts, "config_path": "live-campaign-config.json"}


def _resolve_scenario_passage_table(
    passage_table: PassageTable,
    isolation_layout: dict[str, dict[str, str]],
    backend_family: str,
) -> PassageTable:
    entry = isolation_layout.get(backend_family)
    if entry is None:
        return passage_table
    return PassageTable(
        table_name=entry["table_name"],
        id_column=passage_table.id_column,
        text_column=passage_table.text_column,
        embedding_column=passage_table.embedding_column,
        query_vector_cast=passage_table.query_vector_cast,
    )


def _resolve_scenario_index_names(
    *,
    isolation_layout: dict[str, dict[str, str]],
    turboquant_index_name: str,
    hnsw_index_name: str,
    ivfflat_index_name: str,
) -> dict[str, str]:
    return {
        "pg_turboquant": isolation_layout.get("pg_turboquant", {}).get("index_name", turboquant_index_name),
        "pgvector_hnsw": isolation_layout.get("pgvector_hnsw", {}).get("index_name", hnsw_index_name),
        "pgvector_ivfflat": isolation_layout.get("pgvector_ivfflat", {}).get("index_name", ivfflat_index_name),
    }


def _prepare_dataset_source_layout(
    *,
    dataset_id: str,
    output_dir: Path,
    dsn: str,
    connect_fn: Callable[[str], Any],
    metric: str,
    query_vector_cast: str,
    passage_table: PassageTable,
) -> dict[str, object]:
    live_config_path = LIVE_CONFIG_PATHS.get(dataset_id)
    if live_config_path is None:
        raise ValueError(f"missing live ingestion config for dataset: {dataset_id}")

    source_config = load_campaign_config(live_config_path)
    suffix = output_dir.name.replace("-", "_")
    dataset_suffix = f"{dataset_id}_source"
    documents_table = _isolated_relation_name(source_config.schema["documents_table"], dataset_suffix, suffix)
    passages_table = _isolated_relation_name(source_config.schema["passages_table"], dataset_suffix, suffix)

    backend_index_names: dict[str, str] = {}
    rewritten_backends: list[dict[str, Any]] = []
    for backend in source_config.backends:
        backend_family = str(backend["kind"])
        index_name = _isolated_relation_name(
            str(backend["index_name"]),
            f"{dataset_id}_{backend_family}_source_idx",
            suffix,
        )
        rewritten = dict(backend)
        rewritten["index_name"] = index_name
        rewritten["metric"] = metric
        rewritten_backends.append(rewritten)
        backend_index_names[backend_family] = index_name

    prepared_config = CampaignConfig(
        dataset=replace(source_config.dataset, name=f"{source_config.dataset.name}__{suffix}"),
        embedding=source_config.embedding,
        chunking=source_config.chunking,
        schema={"documents_table": documents_table, "passages_table": passages_table},
        backends=rewritten_backends,
        regression_gate=source_config.regression_gate,
    )

    embedder = build_embedder(prepared_config.embedding.model, prepared_config.embedding.normalized)
    with connect_fn(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_turboquant")
            cursor.execute(f"DROP TABLE IF EXISTS {passages_table} CASCADE")
            cursor.execute(f"DROP TABLE IF EXISTS {documents_table} CASCADE")
        manifest = run_hf_ingestion(connection, prepared_config, embedder=embedder)

    return {
        "passage_table": PassageTable(
            table_name=passages_table,
            id_column=passage_table.id_column,
            text_column=passage_table.text_column,
            embedding_column=passage_table.embedding_column,
            query_vector_cast=query_vector_cast,
        ),
        "index_names": {
            "pg_turboquant": backend_index_names["pg_turboquant"],
            "pgvector_hnsw": backend_index_names["pgvector_hnsw"],
            "pgvector_ivfflat": backend_index_names["pgvector_ivfflat"],
        },
        "manifest": manifest,
    }


def _serialize_dataset_source_layout(layout: dict[str, object]) -> dict[str, object]:
    passage_table = layout["passage_table"]
    if isinstance(passage_table, PassageTable):
        serialized_table = asdict(passage_table)
    else:
        serialized_table = dict(passage_table)
    return {
        "passage_table": serialized_table,
        "index_names": dict(layout["index_names"]),
        "manifest": dict(layout["manifest"]),
    }


def _prepare_backend_isolated_layout(
    *,
    dsn: str,
    connect_fn: Callable[[str], Any],
    source_table: PassageTable,
    output_dir: Path,
    turboquant_index_name: str,
    hnsw_index_name: str,
    ivfflat_index_name: str,
) -> dict[str, dict[str, str]]:
    suffix = output_dir.name.replace("-", "_")
    layout = {
        "pg_turboquant": {
            "table_name": _isolated_relation_name(source_table.table_name, "pg_turboquant", suffix),
            "index_name": _isolated_relation_name(turboquant_index_name, "pg_turboquant_idx", suffix),
            "source_index_name": turboquant_index_name,
        },
        "pgvector_hnsw": {
            "table_name": _isolated_relation_name(source_table.table_name, "pgvector_hnsw", suffix),
            "index_name": _isolated_relation_name(hnsw_index_name, "pgvector_hnsw_idx", suffix),
            "source_index_name": hnsw_index_name,
        },
        "pgvector_ivfflat": {
            "table_name": _isolated_relation_name(source_table.table_name, "pgvector_ivfflat", suffix),
            "index_name": _isolated_relation_name(ivfflat_index_name, "pgvector_ivfflat_idx", suffix),
            "source_index_name": ivfflat_index_name,
        },
    }
    with connect_fn(dsn) as connection:
        with connection.cursor() as cursor:
            for backend_family in BACKEND_FAMILIES:
                entry = layout[backend_family]
                cursor.execute(f"DROP TABLE IF EXISTS {entry['table_name']} CASCADE")
                cursor.execute(f"CREATE TABLE {entry['table_name']} AS TABLE {source_table.table_name} WITH NO DATA")
                cursor.execute(f"INSERT INTO {entry['table_name']} SELECT * FROM {source_table.table_name}")
                cursor.execute(
                    "SELECT indexdef FROM pg_indexes WHERE indexname = %s",
                    (entry["source_index_name"],),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(f"missing source index definition for backend isolation: {entry['source_index_name']}")
                cursor.execute(
                    _rewrite_indexdef_for_clone(
                        row[0],
                        new_table_name=entry["table_name"],
                        new_index_name=entry["index_name"],
                    )
                )
        if hasattr(connection, "commit"):
            connection.commit()
    return {backend: {"table_name": values["table_name"], "index_name": values["index_name"]} for backend, values in layout.items()}


def _isolated_relation_name(base_name: str, backend_family: str, suffix: str) -> str:
    stem = base_name.split(".")[-1]
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{stem}__{suffix}__{backend_family}").strip("_")
    if len(sanitized) <= 63:
        return sanitized
    digest = hashlib.sha1(sanitized.encode("utf-8")).hexdigest()[:8]
    head = sanitized[: 63 - len(digest) - 2].rstrip("_")
    return f"{head}__{digest}"


def _rewrite_indexdef_for_clone(indexdef: str, *, new_table_name: str, new_index_name: str) -> str:
    match = re.match(r"^CREATE INDEX\s+\S+\s+ON\s+\S+\s+", indexdef)
    if match is None:
        raise ValueError(f"unsupported index definition for clone rewrite: {indexdef}")
    return f"CREATE INDEX {new_index_name} ON {new_table_name} " + indexdef[match.end() :]


def _run_retrieval_scenario(
    *,
    dsn: str,
    connect_fn: Callable[[str], Any],
    passage_table: PassageTable,
    dataset_config: dict[str, Any],
    dataset_samples: Sequence[QuerySample],
    method_id: str,
    metric: str,
    query_encoder: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    dataset_top_k: int,
    rerank_top_k: int,
    eval_ks: Sequence[int],
    retrieval_root: Path,
    clock_fn: Callable[[], float],
    turboquant_index_name: str,
    hnsw_index_name: str,
    ivfflat_index_name: str,
    turboquant_probes: int,
    turboquant_oversampling: int,
    turboquant_max_visited_codes: int,
    turboquant_max_visited_pages: int,
    hnsw_ef_search: int,
    ivfflat_probes: int,
) -> dict[str, object]:
    backend = _make_backend(
        method_id=method_id,
        metric=metric,
        turboquant_index_name=turboquant_index_name,
        hnsw_index_name=hnsw_index_name,
        ivfflat_index_name=ivfflat_index_name,
        dataset_top_k=dataset_top_k,
        rerank_top_k=rerank_top_k,
    )
    approx_backend = (
        _make_backend(
            method_id=method_id.replace("_rerank", "_approx"),
            metric=metric,
            turboquant_index_name=turboquant_index_name,
            hnsw_index_name=hnsw_index_name,
            ivfflat_index_name=ivfflat_index_name,
            dataset_top_k=rerank_top_k,
            rerank_top_k=rerank_top_k,
        )
        if method_id.endswith("_rerank")
        else None
    )

    ann = _ann_settings_for_method(
        method_id=method_id,
        turboquant_probes=turboquant_probes,
        turboquant_oversampling=turboquant_oversampling,
        turboquant_max_visited_codes=turboquant_max_visited_codes,
        turboquant_max_visited_pages=turboquant_max_visited_pages,
        hnsw_ef_search=hnsw_ef_search,
        ivfflat_probes=ivfflat_probes,
    )

    adapter = PostgresRetrieverAdapter(
        dsn=dsn,
        table=passage_table,
        backend=backend,
        connect_fn=connect_fn,
    )
    approx_adapter = (
        PostgresRetrieverAdapter(
            dsn=dsn,
            table=passage_table,
            backend=approx_backend,
            connect_fn=connect_fn,
        )
        if approx_backend is not None
        else None
    )

    run_id = f"{dataset_config['dataset_id']}:{method_id}:retrieval_only"
    raw_queries: list[dict[str, object]] = []
    final_evaluations: list[QueryEvaluation] = []
    pre_rerank_evaluations: list[QueryEvaluation] = []
    post_rerank_evaluations: list[QueryEvaluation] = []
    operational_metrics: list[QueryOperationalMetrics] = []

    for sample in dataset_samples:
        query_vector = list(query_encoder([sample.question])[0])

        if approx_adapter is not None:
            started_at = clock_fn()
            approx_rows, approx_scan_stats = approx_adapter.retrieve_with_metadata(
                RetrievalRequest(
                    query_vector=query_vector,
                    top_k=rerank_top_k,
                    metric=metric,
                    ann=ann,
                )
            )
            approx_latency_ms = (clock_fn() - started_at) * 1000.0
            pre_rerank_evaluations.append(
                QueryEvaluation(
                    query_id=sample.query_id,
                    retrieved_ids=_canonicalize_ids(dataset_config, [row["id"] for row in approx_rows]),
                    relevant_ids=sample.relevant_ids,
                    evidence_ids=sample.evidence_ids,
                    latency_ms=approx_latency_ms,
                )
            )
        else:
            approx_rows = None
            approx_latency_ms = None
            approx_scan_stats = None

        started_at = clock_fn()
        final_rows, final_scan_stats = adapter.retrieve_with_metadata(
            RetrievalRequest(
                query_vector=query_vector,
                top_k=dataset_top_k,
                metric=metric,
                ann=ann,
            )
        )
        retrieval_latency_ms = (clock_fn() - started_at) * 1000.0
        final_ids = _canonicalize_ids(dataset_config, [row["id"] for row in final_rows])
        final_eval = QueryEvaluation(
            query_id=sample.query_id,
            retrieved_ids=final_ids,
            relevant_ids=sample.relevant_ids,
            evidence_ids=sample.evidence_ids,
            latency_ms=retrieval_latency_ms,
        )
        final_evaluations.append(final_eval)
        if approx_adapter is not None:
            post_rerank_evaluations.append(final_eval)

        operational_metrics.append(
            QueryOperationalMetrics(
                retrieval_latency_ms=approx_latency_ms if approx_latency_ms is not None else retrieval_latency_ms,
                rerank_latency_ms=retrieval_latency_ms if approx_latency_ms is not None else None,
                scan_stats=approx_scan_stats if approx_scan_stats is not None else final_scan_stats,
            )
        )
        raw_queries.append(
            {
                "query_id": sample.query_id,
                "question": sample.question,
                "answers": sample.answers,
                "relevant_ids": sample.relevant_ids,
                "evidence_ids": sample.evidence_ids,
                "retrieved_rows": final_rows,
            }
        )

    footprint_bytes = _fetch_relation_size_bytes(
        dsn=dsn,
        connect_fn=connect_fn,
        relation_name=_relation_name_for_method(
            method_id=method_id,
            turboquant_index_name=turboquant_index_name,
            hnsw_index_name=hnsw_index_name,
            ivfflat_index_name=ivfflat_index_name,
        ),
    )

    run_metadata = {
        "run_id": run_id,
        "dataset_id": dataset_config["dataset_id"],
        "method_id": method_id,
        "result_kind": "retrieval_only",
        "backend": method_id.rsplit("_", 1)[0],
        "rerank_enabled": method_id.endswith("_rerank"),
        "footprint_bytes": footprint_bytes,
    }
    if hasattr(backend, "serialize_run_metadata"):
        run_metadata.update(backend.serialize_run_metadata(adapter.build_plan(_request_stub(metric, dataset_top_k, ann))))
    if method_id.startswith("pg_turboquant_"):
        run_metadata["index_metadata"] = _fetch_turboquant_index_metadata(
            dsn=dsn,
            connect_fn=connect_fn,
            index_name=turboquant_index_name,
        )

    if approx_adapter is None:
        metrics = compute_retrieval_metrics(final_evaluations, ks=eval_ks)
        artifacts = export_retrieval_run(
            output_dir=retrieval_root,
            run_metadata=run_metadata,
            metrics=metrics,
            operational_metrics=operational_metrics,
        )
    else:
        run_metadata["candidate_pool_size"] = rerank_top_k
        artifacts = export_two_stage_retrieval_run(
            output_dir=retrieval_root,
            run_metadata=run_metadata,
            pre_rerank_queries=pre_rerank_evaluations,
            post_rerank_queries=post_rerank_evaluations,
            ks=eval_ks,
            operational_metrics=operational_metrics,
        )
        metrics = compute_retrieval_metrics(post_rerank_evaluations, ks=eval_ks)

    (retrieval_root / "retrieval-query-results.json").write_text(
        json.dumps(raw_queries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result_payload = json.loads((retrieval_root / artifacts["json"]).read_text(encoding="utf-8"))
    return {
        "run_metadata": run_metadata,
        "metrics": metrics,
        "operational_summary": result_payload.get("operational_summary", {}),
        "artifacts": artifacts,
        "query_results": raw_queries,
    }


def _run_end_to_end_scenario(
    *,
    dataset_config: dict[str, Any],
    dataset_samples: Sequence[QuerySample],
    method_id: str,
    generation_top_k: int,
    retrieval_result: dict[str, object],
    end_to_end_root: Path,
    generator_runner: Callable[[QuerySample, list[dict[str, str]], dict[str, Any]], dict[str, str]],
    answer_metrics_fn: Callable[[Sequence[str], Sequence[Sequence[str]], Sequence[str]], dict[str, float]],
    clock_fn: Callable[[], float],
) -> dict[str, object]:
    generation_results: list[dict[str, object]] = []
    predictions: list[str] = []
    references: list[list[str]] = []
    questions: list[str] = []

    raw_queries = list(retrieval_result["query_results"])
    for sample, query_result in zip(dataset_samples, raw_queries):
        contexts = [
            {"id": str(row["id"]), "text": str(row["text"])}
            for row in query_result["retrieved_rows"][:generation_top_k]
        ]
        started_at = clock_fn()
        generation = generator_runner(sample, contexts, dataset_config)
        generator_latency_ms = (clock_fn() - started_at) * 1000.0

        generation_results.append(
            {
                "query_id": sample.query_id,
                "prompt": str(generation["prompt"]),
                "contexts": contexts,
                "answer": str(generation["answer"]),
                "reference_answer": sample.answers[0] if sample.answers else "",
                "operational_metrics": QueryOperationalMetrics(
                    retrieval_latency_ms=float(
                        query_result["retrieved_rows"] and 0.0 or 0.0
                    ),
                    generator_latency_ms=generator_latency_ms,
                ).to_dict(),
            }
        )
        predictions.append(str(generation["answer"]))
        references.append(list(sample.answers))
        questions.append(sample.question)

    answer_metrics = answer_metrics_fn(predictions, references, questions)
    run_metadata = {
        "run_id": f"{dataset_config['dataset_id']}:{method_id}:end_to_end",
        "dataset_id": dataset_config["dataset_id"],
        "method_id": method_id,
        "result_kind": "end_to_end",
        "generator_id": dataset_config.get("generator_id", "bergen_generator"),
    }
    artifacts = export_end_to_end_run(
        output_dir=end_to_end_root,
        run_metadata=run_metadata,
        retrieval_summary={
            "result_kind": "retrieval_only",
            **dict(retrieval_result["metrics"]),
        },
        generation_results=generation_results,
        answer_metrics=answer_metrics,
    )
    (end_to_end_root / "end-to-end-query-results.json").write_text(
        json.dumps(generation_results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "run_metadata": run_metadata,
        "retrieval_summary": dict(retrieval_result["metrics"]),
        "answer_metrics": answer_metrics,
        "operational_summary": json.loads(
            (end_to_end_root / artifacts["json"]).read_text(encoding="utf-8")
        )["operational_summary"],
        "artifacts": artifacts,
    }


def _request_stub(metric: str, top_k: int, ann: dict[str, Any]) -> RetrievalRequest:
    return RetrievalRequest(query_vector=[0.0], top_k=top_k, metric=metric, ann=ann)


def _relation_name_for_method(
    *,
    method_id: str,
    turboquant_index_name: str,
    hnsw_index_name: str,
    ivfflat_index_name: str,
) -> str:
    if method_id.startswith("pg_turboquant_"):
        return turboquant_index_name
    if method_id.startswith("pgvector_hnsw_"):
        return hnsw_index_name
    return ivfflat_index_name


def _fetch_relation_size_bytes(
    *,
    dsn: str,
    connect_fn: Callable[[str], Any],
    relation_name: str,
) -> int | None:
    try:
        with connect_fn(dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_relation_size(%s::regclass)", (relation_name,))
                row = cursor.fetchone()
                return int(row[0]) if row else None
    except Exception:
        return None


def _fetch_turboquant_index_metadata(
    *,
    dsn: str,
    connect_fn: Callable[[str], Any],
    index_name: str,
) -> dict[str, Any] | None:
    try:
        with connect_fn(dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT tq_index_metadata(%s::regclass)", (index_name,))
                row = cursor.fetchone()
                if not row or row[0] is None:
                    return None
                value = row[0]
                if isinstance(value, dict):
                    return value
                return json.loads(value)
    except Exception:
        return None


def _metric_ks(top_k: int) -> tuple[int, ...]:
    return tuple(sorted({1, 5, 10, int(top_k)}))


def _ann_settings_for_method(
    *,
    method_id: str,
    turboquant_probes: int,
    turboquant_oversampling: int,
    turboquant_max_visited_codes: int,
    turboquant_max_visited_pages: int,
    hnsw_ef_search: int,
    ivfflat_probes: int,
) -> dict[str, int]:
    if method_id.startswith("pg_turboquant_"):
        return {
            "probes": turboquant_probes,
            "oversampling": turboquant_oversampling,
            "max_visited_codes": turboquant_max_visited_codes,
            "max_visited_pages": turboquant_max_visited_pages,
        }
    if method_id.startswith("pgvector_hnsw_"):
        return {"ef_search": hnsw_ef_search}
    return {"probes": ivfflat_probes}


def _make_backend(
    *,
    method_id: str,
    metric: str,
    turboquant_index_name: str,
    hnsw_index_name: str,
    ivfflat_index_name: str,
    dataset_top_k: int,
    rerank_top_k: int,
) -> object:
    if method_id == "pg_turboquant_approx":
        return PgTurboquantBackend(
            index_name=turboquant_index_name,
            metric=metric,
            normalized=metric != "inner_product",
            mode=TURBOQUANT_MODE_APPROX,
        )
    if method_id == "pg_turboquant_rerank":
        return PgTurboquantBackend(
            index_name=turboquant_index_name,
            metric=metric,
            normalized=metric != "inner_product",
            mode=TURBOQUANT_MODE_APPROX_RERANK,
            rerank_k=rerank_top_k,
        )
    if method_id == "pgvector_hnsw_approx":
        return PgvectorHnswBackend(
            index_name=hnsw_index_name,
            metric=metric,
            mode=PGVECTOR_MODE_APPROX,
        )
    if method_id == "pgvector_hnsw_rerank":
        return PgvectorHnswBackend(
            index_name=hnsw_index_name,
            metric=metric,
            mode=PGVECTOR_MODE_APPROX_RERANK,
            rerank_k=rerank_top_k,
        )
    if method_id == "pgvector_ivfflat_approx":
        return PgvectorIvfflatBackend(
            index_name=ivfflat_index_name,
            metric=metric,
            mode=PGVECTOR_MODE_APPROX,
        )
    if method_id == "pgvector_ivfflat_rerank":
        return PgvectorIvfflatBackend(
            index_name=ivfflat_index_name,
            metric=metric,
            mode=PGVECTOR_MODE_APPROX_RERANK,
            rerank_k=rerank_top_k,
        )
    raise ValueError(f"unsupported comparative method: {method_id}")


def _canonicalize_ids(dataset_config: dict[str, Any], ids: Sequence[str]) -> list[str]:
    stable_fields = list(dataset_config.get("stable_passage_id_fields") or [])
    if len(stable_fields) > 1:
        return [str(doc_id).split(":", 1)[0] for doc_id in ids]
    return [str(doc_id) for doc_id in ids]


def _default_connect_fn() -> Callable[[str], Any]:
    try:
        import psycopg  # type: ignore

        return psycopg.connect
    except ImportError:
        try:
            import psycopg2  # type: ignore

            return psycopg2.connect
        except ImportError as exc:
            raise RuntimeError("psycopg or psycopg2 is required for live campaign runs") from exc


def _build_default_dataset_loader() -> Callable[[str, dict[str, Any], int | None], list[QuerySample]]:
    def load_dataset(dataset_id: str, dataset_config: dict[str, Any], query_limit: int | None) -> list[QuerySample]:
        try:
            import datasets  # type: ignore
        except ImportError as exc:
            raise RuntimeError("huggingface datasets is required for BERGEN live runs") from exc

        split = _hf_split_from_config(dataset_config)
        if dataset_id == "kilt_nq":
            dataset = datasets.load_dataset("kilt_tasks", "nq")[split]
            samples = [
                QuerySample(
                    query_id=str(item["id"]),
                    question=str(item["input"]),
                    answers=[entry["answer"] for entry in item["output"] if entry.get("answer")],
                    relevant_ids=_flatten_kilt_wiki_ids(item["output"]),
                    evidence_ids=_flatten_kilt_wiki_ids(item["output"]),
                )
                for item in dataset
            ]
        elif dataset_id == "kilt_hotpotqa":
            dataset = datasets.load_dataset("kilt_tasks", "hotpotqa")[split]
            samples = [
                QuerySample(
                    query_id=str(item["id"]),
                    question=str(item["input"]),
                    answers=[entry["answer"] for entry in item["output"] if entry.get("answer")],
                    relevant_ids=_flatten_kilt_wiki_ids(item["output"]),
                    evidence_ids=_flatten_kilt_wiki_ids(item["output"]),
                )
                for item in dataset
            ]
        elif dataset_id == "kilt_triviaqa":
            support_only = datasets.load_dataset("kilt_tasks", "triviaqa_support_only")[split]
            trivia_qa = datasets.load_dataset("trivia_qa", "unfiltered.nocontext")[split]
            triviaqa_map = {item["question_id"]: item for item in trivia_qa}
            samples = []
            for item in support_only:
                trivia = triviaqa_map.get(item["id"])
                if trivia is None:
                    continue
                answers = [entry["answer"] for entry in item["output"] if entry.get("answer")]
                if not answers and trivia.get("answer", {}).get("value"):
                    answers = [str(trivia["answer"]["value"])]
                samples.append(
                    QuerySample(
                        query_id=str(item["id"]),
                        question=str(trivia["question"]),
                        answers=answers,
                        relevant_ids=_flatten_kilt_wiki_ids(item["output"]),
                        evidence_ids=_flatten_kilt_wiki_ids(item["output"]),
                    )
                )
        elif dataset_id == "popqa":
            dataset = datasets.load_dataset("akariasai/PopQA")[split]
            samples = [
                QuerySample(
                    query_id=f"{split}{index}",
                    question=str(item["question"]),
                    answers=[str(answer) for answer in ast.literal_eval(item["possible_answers"])],
                    relevant_ids=[str(item["subj_id"])],
                    evidence_ids=[],
                )
                for index, item in enumerate(dataset)
            ]
        else:
            raise ValueError(f"unsupported BERGEN live dataset: {dataset_id}")

        if query_limit is not None:
            return samples[:query_limit]
        return samples

    return load_dataset


def _build_default_query_encoder(
    *,
    bergen_root: str | Path | None,
    retriever_name: str,
) -> Callable[[Sequence[str]], Sequence[Sequence[float]]]:
    bergen_root = _resolve_bergen_root(bergen_root)
    _ensure_bergen_import_path(bergen_root)

    try:
        import torch  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("BERGEN live query encoding requires torch and pyyaml") from exc

    retriever_config_path = bergen_root / "config" / "retriever" / f"{retriever_name}.yaml"
    if not retriever_config_path.exists():
        raise FileNotFoundError(f"BERGEN retriever config not found: {retriever_config_path}")

    retriever_config = yaml.safe_load(retriever_config_path.read_text(encoding="utf-8"))
    init_args = dict(retriever_config["init_args"])
    pooler = _instantiate_from_target(init_args.pop("pooler"))
    similarity = _instantiate_from_target(init_args.pop("similarity"))
    init_args["pooler"] = pooler
    init_args["similarity"] = similarity
    model = _instantiate_from_target(init_args.pop("_target_"), **init_args)

    def encode(texts: Sequence[str]) -> list[list[float]]:
        encoded: list[list[float]] = []
        for text in texts:
            batch = model.collate_fn([{"generated_query": text}], query_or_doc="query")
            with torch.no_grad():
                embedding = model("query", batch)["embedding"]
            encoded.extend(embedding.detach().cpu().float().tolist())
        return encoded

    return encode


def _build_default_generator_runner(
    *,
    bergen_root: str | Path | None,
    generator_name: str,
    prompt_name: str,
) -> Callable[[QuerySample, list[dict[str, str]], dict[str, Any]], dict[str, str]]:
    if generator_name == "oracle_answer":
        def run_generator(
            sample: QuerySample,
            contexts: list[dict[str, str]],
            _dataset_config: dict[str, Any],
        ) -> dict[str, str]:
            prompt_lines = [
                f"Prompt: {prompt_name}",
                f"Question: {sample.question}",
            ]
            if contexts:
                prompt_lines.append("Contexts:")
                prompt_lines.extend(f"- {context['id']}: {context['text']}" for context in contexts)
            return {
                "prompt": "\n".join(prompt_lines),
                "answer": sample.answers[0] if sample.answers else "",
            }

        return run_generator

    bergen_root = _resolve_bergen_root(bergen_root)
    _ensure_bergen_import_path(bergen_root)

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("BERGEN live generation requires pyyaml") from exc

    generator_config_path = bergen_root / "config" / "generator" / f"{generator_name}.yaml"
    if not generator_config_path.exists():
        raise FileNotFoundError(f"BERGEN generator config not found: {generator_config_path}")
    generator_config = yaml.safe_load(generator_config_path.read_text(encoding="utf-8"))
    init_args = dict(generator_config["init_args"])

    prompt = None
    prompt_config_path = bergen_root / "config" / "prompt" / f"{prompt_name}.yaml"
    if prompt_config_path.exists():
        prompt = SimpleNamespace(
            **yaml.safe_load(prompt_config_path.read_text(encoding="utf-8"))
        )

    target = init_args.pop("_target_")
    if prompt is not None and "prompt" not in init_args:
        init_args["prompt"] = prompt
    generator = _instantiate_from_target(target, **init_args)

    def run_generator(
        sample: QuerySample,
        contexts: list[dict[str, str]],
        _dataset_config: dict[str, Any],
    ) -> dict[str, str]:
        dataset = [
            {
                "q_id": sample.query_id,
                "query": sample.question,
                "label": list(sample.answers),
                "ranking_label": list(sample.relevant_ids),
                "doc": [context["text"] for context in contexts],
            }
        ]
        _, _, instructions, responses, _, _ = generator.eval(dataset)
        return {
            "prompt": str(instructions[0]),
            "answer": str(responses[0]),
        }

    return run_generator


def _build_answer_metrics_computer(
    bergen_root: str | Path | None,
) -> Callable[[Sequence[str], Sequence[Sequence[str]], Sequence[str]], dict[str, float]]:
    try:
        bergen_root = _resolve_bergen_root(bergen_root)
        _ensure_bergen_import_path(bergen_root)
        metrics_module = importlib.import_module("modules.metrics")
        rag_metrics = getattr(metrics_module, "RAGMetrics")

        def compute(
            predictions: Sequence[str],
            references: Sequence[Sequence[str]],
            questions: Sequence[str],
        ) -> dict[str, float]:
            payload = rag_metrics.compute(list(predictions), list(references), list(questions))
            return {
                "answer_exact_match": float(payload.get("EM", 0.0)),
                "answer_f1": float(payload.get("F1", 0.0)),
            }

        return compute
    except Exception:
        return _fallback_answer_metrics


def _fallback_answer_metrics(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    _questions: Sequence[str],
) -> dict[str, float]:
    def normalize(text: str) -> list[str]:
        return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split()

    exact_matches = []
    f1_scores = []
    for prediction, answers in zip(predictions, references):
        normalized_prediction = " ".join(normalize(prediction))
        exact_matches.append(
            1.0 if any(normalized_prediction == " ".join(normalize(answer)) for answer in answers) else 0.0
        )
        prediction_tokens = normalize(prediction)
        best_f1 = 0.0
        for answer in answers:
            answer_tokens = normalize(answer)
            common = set(prediction_tokens) & set(answer_tokens)
            if not common:
                continue
            precision = len(common) / len(prediction_tokens) if prediction_tokens else 0.0
            recall = len(common) / len(answer_tokens) if answer_tokens else 0.0
            if precision + recall:
                best_f1 = max(best_f1, (2 * precision * recall) / (precision + recall))
        f1_scores.append(best_f1)
    return {
        "answer_exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        "answer_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }


def _hf_split_from_config(dataset_config: dict[str, Any]) -> str:
    split = str(dataset_config["source"]["split"])
    return {"dev": "validation"}.get(split, split)


def _flatten_kilt_wiki_ids(outputs: Sequence[dict[str, Any]]) -> list[str]:
    flattened: list[str] = []
    for item in outputs:
        if not item.get("answer"):
            continue
        for provenance in item.get("provenance", []):
            wiki_id = provenance.get("wikipedia_id")
            if wiki_id is not None:
                flattened.append(str(wiki_id))
    seen: set[str] = set()
    deduped: list[str] = []
    for wiki_id in flattened:
        if wiki_id in seen:
            continue
        seen.add(wiki_id)
        deduped.append(wiki_id)
    return deduped


def _resolve_bergen_root(bergen_root: str | Path | None) -> Path:
    if bergen_root is not None:
        return Path(bergen_root).resolve()
    env_value = os.environ.get("BERGEN_ROOT")
    if env_value:
        return Path(env_value).resolve()
    return (Path(__file__).resolve().parent / "vendor" / "bergen").resolve()


def _ensure_bergen_import_path(bergen_root: Path) -> None:
    bergen_root_str = str(bergen_root)
    if bergen_root_str not in sys.path:
        sys.path.insert(0, bergen_root_str)


def _instantiate_from_target(target_or_config: object, **kwargs: Any) -> object:
    if isinstance(target_or_config, dict):
        config = dict(target_or_config)
        target = config.pop("_target_")
        config.update(kwargs)
        return _instantiate_from_target(target, **config)

    target = str(target_or_config)
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def _redact_dsn(dsn: str) -> str:
    if "@" not in dsn:
        return dsn
    prefix, suffix = dsn.rsplit("@", 1)
    if "://" in prefix:
        scheme, credentials = prefix.split("://", 1)
        if ":" in credentials:
            username, _password = credentials.split(":", 1)
            return f"{scheme}://{username}:***@{suffix}"
    return f"***@{suffix}"
