from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .bergen_adapter.adapter import vector_literal
from .operational_metrics import QueryOperationalMetrics, summarize_query_operational_metrics
from .retrieval_eval import QueryEvaluation, compute_retrieval_metrics


METRIC_OPERATORS = {
    "cosine": "<=>",
    "inner_product": "<#>",
    "l2": "<->",
}


@dataclass(frozen=True)
class ExactRerankPlan:
    sql: str
    params: tuple[object, ...]
    sql_template_hash: str


def build_exact_rerank_sql(
    *,
    table_name: str,
    id_column: str,
    text_column: str,
    embedding_column: str,
    metric: str,
    query_vector: Sequence[float],
    candidate_ids: Sequence[str],
    final_k: int,
) -> ExactRerankPlan:
    operator = METRIC_OPERATORS[metric]
    sql = (
        "WITH candidate_pool AS ("
        f"SELECT unnest(%s::text[]) AS {id_column}"
        "), query_vector AS ("
        "SELECT %s::vector AS embedding"
        ") "
        f"SELECT base.{id_column} AS id, "
        f"base.{embedding_column} {operator} query_vector.embedding AS exact_score, "
        f"base.{text_column} AS text "
        f"FROM {table_name} AS base "
        f"JOIN candidate_pool USING ({id_column}) "
        "CROSS JOIN query_vector "
        "ORDER BY exact_score ASC "
        "LIMIT %s"
    )
    params = (list(candidate_ids), vector_literal(query_vector), final_k)
    return ExactRerankPlan(
        sql=sql,
        params=params,
        sql_template_hash=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
    )


def rerank_results(
    approx_results: Sequence[dict[str, object]],
    exact_scores: dict[str, float],
    *,
    final_k: int,
) -> list[dict[str, object]]:
    reranked = []
    for item in approx_results:
        doc_id = str(item["id"])
        if doc_id not in exact_scores:
            continue
        reranked.append(
            {
                "id": doc_id,
                "score": exact_scores[doc_id],
                "text": item["text"],
            }
        )
    reranked.sort(key=lambda item: (float(item["score"]), str(item["id"])))
    return reranked[:final_k]


def export_two_stage_retrieval_run(
    *,
    output_dir: str | Path,
    run_metadata: dict[str, object],
    pre_rerank_queries: Sequence[QueryEvaluation],
    post_rerank_queries: Sequence[QueryEvaluation],
    ks: Sequence[int],
    operational_metrics: Sequence[QueryOperationalMetrics] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_metadata": run_metadata,
        "metrics": {
            "pre_rerank": compute_retrieval_metrics(pre_rerank_queries, ks),
            "post_rerank": compute_retrieval_metrics(post_rerank_queries, ks),
        },
        "operational_summary": summarize_query_operational_metrics(
            list(operational_metrics or [])
        ),
    }

    json_name = "two-stage-retrieval-results.json"
    csv_name = "two-stage-retrieval-results.csv"
    markdown_name = "two-stage-retrieval-results.md"

    (output_dir / json_name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_lines = ["stage,metric,value"]
    for stage in ("pre_rerank", "post_rerank"):
        for metric, value in payload["metrics"][stage].items():
            csv_lines.append(f"{stage},{metric},{value}")
    (output_dir / csv_name).write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    markdown_lines = [
        "# Two-Stage Retrieval Results",
        "",
        f"- run_id: {run_metadata.get('run_id', '')}",
        f"- result_kind: {run_metadata.get('result_kind', '')}",
        f"- candidate_pool_size: {run_metadata.get('candidate_pool_size', '')}",
        f"- rerank_enabled: {run_metadata.get('rerank_enabled', '')}",
        "",
        "| Stage | Metric | Value |",
        "|---|---|---:|",
    ]
    for stage in ("pre_rerank", "post_rerank"):
        for metric, value in payload["metrics"][stage].items():
            markdown_lines.append(f"| {stage} | {metric} | {value:.6f} |")
    operational_summary = payload["operational_summary"]
    if operational_summary:
        markdown_lines.extend(
            [
                "",
                "## Operational Summary",
                "",
                "| Metric | P50 | P95 | P99 |",
                "|---|---:|---:|---:|",
            ]
        )
        for stage, distribution in operational_summary.get("latency_ms", {}).items():
            markdown_lines.append(
                f"| {stage}_latency_ms | {distribution['p50']:.6f} | "
                f"{distribution['p95']:.6f} | {distribution['p99']:.6f} |"
            )
    (output_dir / markdown_name).write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return {"json": json_name, "csv": csv_name, "markdown": markdown_name}
