from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .outcome_report import write_outcome_html


COMPARATIVE_METHOD_VARIANTS = [
    "pg_turboquant_approx",
    "pg_turboquant_rerank",
    "pgvector_hnsw_approx",
    "pgvector_hnsw_rerank",
    "pgvector_ivfflat_approx",
    "pgvector_ivfflat_rerank",
]


def build_comparative_campaign_plan(
    *,
    dataset_ids: Sequence[str],
    generator_id: str,
) -> dict[str, object]:
    datasets = list(dataset_ids)
    method_variants = [_method_variant(method_id) for method_id in COMPARATIVE_METHOD_VARIANTS]
    retrieval_scenarios = [
        {
            "scenario_id": f"{dataset_id}:{variant['method_id']}:retrieval_only",
            "dataset_id": dataset_id,
            "result_kind": "retrieval_only",
            **variant,
        }
        for dataset_id in datasets
        for variant in method_variants
    ]
    end_to_end_scenarios = [
        {
            "scenario_id": f"{dataset_id}:{variant['method_id']}:end_to_end",
            "dataset_id": dataset_id,
            "result_kind": "end_to_end",
            "generator_id": generator_id,
            **variant,
        }
        for dataset_id in datasets
        for variant in method_variants
    ]
    return {
        "campaign_kind": "comparative_rag",
        "datasets": datasets,
        "generator_id": generator_id,
        "method_variants": method_variants,
        "retrieval_scenarios": retrieval_scenarios,
        "end_to_end_scenarios": end_to_end_scenarios,
    }


def run_comparative_campaign(
    *,
    output_dir: str | Path,
    plan: dict[str, object],
    retrieval_runner: Callable[[dict[str, object]], dict[str, object]],
    end_to_end_runner: Callable[[dict[str, object], dict[str, object]], dict[str, object]],
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_results: dict[tuple[str, str], dict[str, object]] = {}
    retrieval_rows: list[dict[str, object]] = []
    for scenario in plan["retrieval_scenarios"]:
        result = retrieval_runner(scenario)
        key = (scenario["dataset_id"], scenario["method_id"])
        retrieval_results[key] = result
        retrieval_rows.append(_retrieval_row(scenario, result))

    end_to_end_rows: list[dict[str, object]] = []
    for scenario in plan["end_to_end_scenarios"]:
        key = (scenario["dataset_id"], scenario["method_id"])
        retrieval_result = retrieval_results[key]
        result = end_to_end_runner(scenario, retrieval_result)
        end_to_end_rows.append(_end_to_end_row(scenario, result))

    report = _build_report(plan, retrieval_rows, end_to_end_rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "tables": {
            "retrieval_only": retrieval_rows,
            "end_to_end": end_to_end_rows,
        },
        "report": report,
    }

    campaign_json = "rag-campaign.json"
    retrieval_csv = "retrieval-comparison.csv"
    end_to_end_csv = "end-to-end-comparison.csv"
    report_markdown = "rag-campaign-report.md"
    report_html = "outcome.html"

    (output_dir / campaign_json).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(output_dir / retrieval_csv, retrieval_rows)
    _write_csv(output_dir / end_to_end_csv, end_to_end_rows)
    (output_dir / report_markdown).write_text(
        _render_markdown_report(report, retrieval_rows, end_to_end_rows),
        encoding="utf-8",
    )
    write_outcome_html(
        output_dir / report_html,
        [payload],
        source_labels=[output_dir.name],
    )

    return {
        "campaign_json": campaign_json,
        "retrieval_csv": retrieval_csv,
        "end_to_end_csv": end_to_end_csv,
        "report_markdown": report_markdown,
        "report_html": report_html,
    }


def _method_variant(method_id: str) -> dict[str, object]:
    backend, mode = method_id.rsplit("_", 1)
    return {
        "method_id": method_id,
        "backend_family": backend,
        "rerank_enabled": mode == "rerank",
    }


def _retrieval_row(scenario: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    metrics = result.get("metrics", {})
    run_metadata = result.get("run_metadata", {})
    row = {
        "dataset_id": scenario["dataset_id"],
        "method_id": scenario["method_id"],
        "backend_family": scenario["backend_family"],
        "rerank_enabled": scenario["rerank_enabled"],
        "recall@10": metrics.get("recall@10"),
        "latency_p95_ms": metrics.get("latency_p95_ms"),
        "footprint_bytes": run_metadata.get("footprint_bytes"),
    }
    return row


def _end_to_end_row(scenario: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    answer_metrics = result.get("answer_metrics", {})
    operational = result.get("operational_summary", {}).get("latency_ms", {}).get("total", {})
    row = {
        "dataset_id": scenario["dataset_id"],
        "method_id": scenario["method_id"],
        "backend_family": scenario["backend_family"],
        "rerank_enabled": scenario["rerank_enabled"],
        "answer_exact_match": answer_metrics.get("answer_exact_match"),
        "answer_f1": answer_metrics.get("answer_f1"),
        "total_latency_p95_ms": operational.get("p95"),
    }
    return row


def _build_report(
    plan: dict[str, object],
    retrieval_rows: Sequence[dict[str, object]],
    end_to_end_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "summary": {
            "dataset_count": len(plan["datasets"]),
            "retrieval_scenario_count": len(retrieval_rows),
            "end_to_end_scenario_count": len(end_to_end_rows),
            "methods": list(COMPARATIVE_METHOD_VARIANTS),
            "datasets": list(plan["datasets"]),
        },
        "retrieval_findings": _find_best_rows(
            retrieval_rows,
            metric_name="recall@10",
            higher_is_better=True,
        ),
        "answer_findings": _find_best_rows(
            end_to_end_rows,
            metric_name="answer_exact_match",
            higher_is_better=True,
        ),
        "latency_findings": _find_best_rows(
            end_to_end_rows,
            metric_name="total_latency_p95_ms",
            higher_is_better=False,
        ),
        "footprint_findings": _find_best_rows(
            retrieval_rows,
            metric_name="footprint_bytes",
            higher_is_better=False,
        ),
        "metric_validity_caveats": [
            "Retrieval quality, answer quality, latency, and footprint are reported separately and should not be collapsed into one score.",
            "Answer metrics depend on a fixed-generator contract and should be compared only within the same generator configuration.",
            "Footprint fields are only as precise as the saved run metadata for each scenario.",
        ],
    }


def _find_best_rows(
    rows: Sequence[dict[str, object]],
    *,
    metric_name: str,
    higher_is_better: bool,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset_id"]), []).append(row)

    findings: list[dict[str, object]] = []
    for dataset_id, dataset_rows in grouped.items():
        comparable = [row for row in dataset_rows if row.get(metric_name) is not None]
        if not comparable:
            continue
        best = sorted(
            comparable,
            key=lambda row: (
                -float(row[metric_name]) if higher_is_better else float(row[metric_name]),
                str(row["method_id"]),
            ),
        )[0]
        findings.append(
            {
                "dataset_id": dataset_id,
                "method_id": best["method_id"],
                "metric": metric_name,
                "value": best[metric_name],
            }
        )
    return findings


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown_report(
    report: dict[str, object],
    retrieval_rows: Sequence[dict[str, object]],
    end_to_end_rows: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# Comparative RAG Campaign",
        "",
        "## Retrieval-Only Comparison",
        "",
        "| Dataset | Method | Recall@10 | P95 Latency (ms) | Footprint (bytes) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in retrieval_rows:
        lines.append(
            f"| {row['dataset_id']} | {row['method_id']} | "
            f"{_fmt(row['recall@10'])} | {_fmt(row['latency_p95_ms'])} | {_fmt(row['footprint_bytes'])} |"
        )

    lines.extend(
        [
            "",
            "## End-to-End Comparison",
            "",
            "| Dataset | Method | Answer EM | Answer F1 | Total P95 Latency (ms) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in end_to_end_rows:
        lines.append(
            f"| {row['dataset_id']} | {row['method_id']} | "
            f"{_fmt(row['answer_exact_match'])} | {_fmt(row['answer_f1'])} | {_fmt(row['total_latency_p95_ms'])} |"
        )

    lines.extend(["", "## Narrative Findings", ""])
    for section_name, label in (
        ("retrieval_findings", "Retrieval quality"),
        ("answer_findings", "Answer quality"),
        ("latency_findings", "Latency"),
        ("footprint_findings", "Footprint"),
    ):
        lines.append(f"### {label}")
        for finding in report[section_name]:
            lines.append(
                f"- {finding['dataset_id']}: {finding['method_id']} leads on "
                f"{finding['metric']} ({_fmt(finding['value'])})."
            )
        lines.append("")

    lines.extend(["## Metric Validity Caveats", ""])
    for caveat in report["metric_validity_caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
