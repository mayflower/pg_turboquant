from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .operational_metrics import QueryOperationalMetrics, summarize_query_operational_metrics


@dataclass(frozen=True)
class QueryEvaluation:
    query_id: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    evidence_ids: list[str]
    latency_ms: float


def compute_retrieval_metrics(
    queries: Sequence[QueryEvaluation],
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    if not queries:
        raise ValueError("at least one query is required")

    ks = tuple(sorted(set(int(k) for k in ks)))
    metrics: dict[str, float] = {}

    latencies = [query.latency_ms for query in queries]
    total_latency_seconds = sum(latencies) / 1000.0

    for k in ks:
        recall_sum = 0.0
        reciprocal_rank_sum = 0.0
        ndcg_sum = 0.0
        hit_count = 0
        evidence_hits = 0

        for query in queries:
            top_k = query.retrieved_ids[:k]
            relevant = set(query.relevant_ids)
            evidence = set(query.evidence_ids)

            recall_sum += (
                len(relevant.intersection(top_k)) / len(relevant) if relevant else 0.0
            )
            reciprocal_rank_sum += reciprocal_rank(top_k, relevant)
            ndcg_sum += ndcg_at_k(top_k, relevant, k)
            if relevant.intersection(top_k):
                hit_count += 1
            if evidence and evidence.intersection(top_k):
                evidence_hits += 1

        metrics[f"recall@{k}"] = recall_sum / len(queries)
        metrics[f"mrr@{k}"] = reciprocal_rank_sum / len(queries)
        metrics[f"ndcg@{k}"] = ndcg_sum / len(queries)
        metrics[f"hit_rate@{k}"] = hit_count / len(queries)
        metrics[f"evidence_coverage@{k}"] = evidence_hits / len(queries)

    metrics["latency_p50_ms"] = percentile(latencies, 50)
    metrics["latency_p95_ms"] = percentile(latencies, 95)
    metrics["latency_p99_ms"] = percentile(latencies, 99)
    metrics["throughput_qps"] = len(queries) / total_latency_seconds if total_latency_seconds else 0.0
    return metrics


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for index, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for index, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg


def percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * weight


def export_retrieval_run(
    *,
    output_dir: str | Path,
    run_metadata: dict[str, object],
    metrics: dict[str, float],
    operational_metrics: Sequence[QueryOperationalMetrics] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_metadata.get("result_kind") != "retrieval_only":
        raise ValueError("retrieval exports must be marked as result_kind='retrieval_only'")

    json_name = "retrieval-results.json"
    csv_name = "retrieval-results.csv"
    markdown_name = "retrieval-results.md"

    payload = {
        "run_metadata": run_metadata,
        "metrics": metrics,
        "operational_summary": summarize_query_operational_metrics(list(operational_metrics or [])),
    }

    (output_dir / json_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with (output_dir / csv_name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in ordered_metric_items(metrics):
            writer.writerow({"metric": key, "value": value})

    markdown_lines = [
        "# Retrieval Results",
        "",
        f"- run_id: {run_metadata.get('run_id', '')}",
        f"- result_kind: {run_metadata.get('result_kind', '')}",
        f"- backend: {run_metadata.get('backend', '')}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in ordered_metric_items(metrics):
        markdown_lines.append(f"| {key} | {value:.6f} |")

    operational_summary = payload["operational_summary"]
    if operational_summary:
        markdown_lines.extend(
            [
                "",
                "## Operational Summary",
                "",
            ]
        )
        latency_summary = operational_summary.get("latency_ms", {})
        if latency_summary:
            markdown_lines.extend(
                [
                    "| Metric | Avg | P50 | P95 | P99 |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for stage, distribution in latency_summary.items():
                markdown_lines.append(
                    f"| {stage}_latency_ms | {distribution['avg']:.6f} | {distribution['p50']:.6f} | "
                    f"{distribution['p95']:.6f} | {distribution['p99']:.6f} |"
                )

        scan_stats = operational_summary.get("scan_stats", {})
        if scan_stats:
            markdown_lines.extend(
                [
                    "",
                    "| Metric | Value |",
                    "|---|---:|",
                ]
            )
            for key, summary in scan_stats.items():
                if "uniform" in summary:
                    markdown_lines.append(f"| scan_{key} | {summary['uniform'] or ''} |")
                else:
                    markdown_lines.append(f"| avg_{key} | {summary['avg']:.6f} |")
    (output_dir / markdown_name).write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return {"json": json_name, "csv": csv_name, "markdown": markdown_name}


def ordered_metric_items(metrics: dict[str, float]) -> list[tuple[str, float]]:
    def metric_key(item: tuple[str, float]) -> tuple[int, int, str]:
        name = item[0]
        for prefix, bucket in (
            ("recall@", 0),
            ("mrr@", 1),
            ("ndcg@", 2),
            ("hit_rate@", 3),
            ("evidence_coverage@", 4),
        ):
            if name.startswith(prefix):
                return (bucket, int(name.split("@", 1)[1]), name)
        if name == "latency_p50_ms":
            return (5, 50, name)
        if name == "latency_p95_ms":
            return (5, 95, name)
        if name == "latency_p99_ms":
            return (5, 99, name)
        if name == "throughput_qps":
            return (6, 0, name)
        return (7, 0, name)

    return sorted(metrics.items(), key=metric_key)
