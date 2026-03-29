from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .bergen_adapter import PostgresRetrieverAdapter, RetrievalRequest
from .retrieval_eval import QueryEvaluation, compute_retrieval_metrics


REGRESSION_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "regression"
VALID_HARNESSES = frozenset({"beir", "lotte"})


@dataclass(frozen=True)
class BeirFixtureQuery:
    query_id: str
    query_text: str
    query_vector: list[float]
    relevant_ids: list[str]


def load_regression_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"harness", "dataset_id", "top_k", "metric"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"regression config missing required keys: {', '.join(missing)}")
    if payload["harness"] not in VALID_HARNESSES:
        raise ValueError(f"unsupported regression harness: {payload['harness']}")
    if int(payload["top_k"]) <= 0:
        raise ValueError("top_k must be positive")
    return payload


def load_regression_fixture(path: str | Path) -> list[BeirFixtureQuery]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        BeirFixtureQuery(
            query_id=str(item["query_id"]),
            query_text=str(item["query_text"]),
            query_vector=[float(value) for value in item["query_vector"]],
            relevant_ids=[str(doc_id) for doc_id in item["relevant_ids"]],
        )
        for item in payload["queries"]
    ]


def resolve_regression_harness(config: dict[str, Any]) -> dict[str, Any]:
    harness = config["harness"]
    if harness == "beir":
        return {
            "harness": "beir",
            "runner": "beir_smoke",
            "available": True,
            "dataset_id": config["dataset_id"],
            "fixture_path": config.get("fixture_path"),
        }
    return {
        "harness": "lotte",
        "runner": "lotte_adapter",
        "available": False,
        "dataset_id": config["dataset_id"],
        "fixture_path": config.get("fixture_path"),
    }


def run_beir_smoke_evaluation(
    *,
    adapter: PostgresRetrieverAdapter,
    fixture: Sequence[BeirFixtureQuery],
    top_k: int,
    metric: str,
    ann: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluations: list[QueryEvaluation] = []
    for query in fixture:
        rows = adapter.retrieve(
            RetrievalRequest(
                query_vector=query.query_vector,
                top_k=top_k,
                metric=metric,
                ann=ann or {},
            )
        )
        evaluations.append(
            QueryEvaluation(
                query_id=query.query_id,
                retrieved_ids=[row["id"] for row in rows],
                relevant_ids=query.relevant_ids,
                evidence_ids=[],
                latency_ms=0.0,
            )
        )

    metrics = compute_retrieval_metrics(evaluations, ks=(1, top_k))
    return {
        "run_metadata": {
            "harness": "beir",
            "result_kind": "retrieval_only",
            "query_count": len(fixture),
            "top_k": top_k,
            "metric": metric,
        },
        "metrics": metrics,
    }
