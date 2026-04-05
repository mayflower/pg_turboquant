from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .outcome_report import write_outcome_html
from .regression_gate import evaluate_hotpot_regression_gate


DEFAULT_RAG_SYSTEM_VARIANTS = [
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
    system_variants = [_system_variant(system_id) for system_id in DEFAULT_RAG_SYSTEM_VARIANTS]
    retrieval_scenarios = [
        {
            "scenario_id": f"{dataset_id}:{variant['system_id']}:retrieval_only",
            "dataset_id": dataset_id,
            "result_kind": "retrieval_only",
            **variant,
        }
        for dataset_id in datasets
        for variant in system_variants
    ]
    end_to_end_scenarios = [
        {
            "scenario_id": f"{dataset_id}:{variant['system_id']}:end_to_end",
            "dataset_id": dataset_id,
            "result_kind": "end_to_end",
            "generator_id": generator_id,
            **variant,
        }
        for dataset_id in datasets
        for variant in system_variants
    ]
    return {
        "campaign_kind": "rag_benchmark",
        "datasets": datasets,
        "generator_id": generator_id,
        "system_variants": system_variants,
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
    diagnostics_rows: list[dict[str, object]] = []
    for scenario in plan["retrieval_scenarios"]:
        result = retrieval_runner(scenario)
        key = (str(scenario["dataset_id"]), str(scenario["system_id"]))
        retrieval_results[key] = result
        retrieval_rows.append(_retrieval_row(scenario, result))
        diagnostics_rows.append(_diagnostics_row(scenario, result))

    end_to_end_rows: list[dict[str, object]] = []
    for scenario in plan["end_to_end_scenarios"]:
        key = (str(scenario["dataset_id"]), str(scenario["system_id"]))
        retrieval_result = retrieval_results[key]
        result = end_to_end_runner(scenario, retrieval_result)
        end_to_end_rows.append(_end_to_end_row(scenario, result))

    report = _build_report(plan, retrieval_rows, end_to_end_rows, diagnostics_rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "tables": {
            "retrieval_benchmark": retrieval_rows,
            "end_to_end_benchmark": end_to_end_rows,
            "retrieval_diagnostics": diagnostics_rows,
        },
        "report": report,
    }

    campaign_json = "rag-benchmark.json"
    retrieval_csv = "retrieval-benchmark.csv"
    end_to_end_csv = "end-to-end-benchmark.csv"
    retrieval_diagnostics_csv = "retrieval-diagnostics.csv"
    report_markdown = "rag-benchmark-report.md"
    report_html = "outcome.html"

    (output_dir / campaign_json).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(output_dir / retrieval_csv, retrieval_rows)
    _write_csv(output_dir / end_to_end_csv, end_to_end_rows)
    _write_csv(output_dir / retrieval_diagnostics_csv, diagnostics_rows)
    (output_dir / report_markdown).write_text(
        _render_markdown_report(report, retrieval_rows, end_to_end_rows, diagnostics_rows),
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
        "retrieval_diagnostics_csv": retrieval_diagnostics_csv,
        "report_markdown": report_markdown,
        "report_html": report_html,
    }


def _system_variant(system_id: str) -> dict[str, object]:
    backend, mode = system_id.rsplit("_", 1)
    retrieval_mode = "rerank" if mode == "rerank" else "approx"
    return {
        "system_id": system_id,
        "method_id": system_id,
        "system_label": f"{backend} ({retrieval_mode})",
        "retriever_backend": backend,
        "retrieval_mode": retrieval_mode,
        "rerank_enabled": mode == "rerank",
    }


def _retrieval_row(scenario: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    metrics = result.get("metrics", {})
    run_metadata = result.get("run_metadata", {})
    return {
        "dataset_id": scenario["dataset_id"],
        "system_id": scenario["system_id"],
        "method_id": scenario["method_id"],
        "system_label": scenario["system_label"],
        "retriever_backend": scenario["retriever_backend"],
        "retrieval_mode": scenario["retrieval_mode"],
        "rerank_enabled": scenario["rerank_enabled"],
        "retrieval_execution_mode": run_metadata.get("retrieval_execution_mode"),
        "context_fetch_mode": run_metadata.get("context_fetch_mode"),
        "recall@10": metrics.get("recall@10"),
        "mrr@10": metrics.get("mrr@10"),
        "ndcg@10": metrics.get("ndcg@10"),
        "hit_rate@10": metrics.get("hit_rate@10"),
        "evidence_coverage@10": metrics.get("evidence_coverage@10"),
        "latency_p50_ms": metrics.get("latency_p50_ms"),
        "latency_p95_ms": metrics.get("latency_p95_ms"),
        "footprint_bytes": run_metadata.get("footprint_bytes"),
    }


def _diagnostics_row(scenario: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    run_metadata = result.get("run_metadata", {})
    scan_stats = result.get("operational_summary", {}).get("scan_stats", {})
    index_metadata = run_metadata.get("index_metadata", {}) or {}
    router = index_metadata.get("router", {}) if isinstance(index_metadata, dict) else {}
    list_distribution = index_metadata.get("list_distribution", {}) if isinstance(index_metadata, dict) else {}
    delta_health = index_metadata.get("delta_health", {}) if isinstance(index_metadata, dict) else {}
    maintenance = index_metadata.get("maintenance", {}) if isinstance(index_metadata, dict) else {}
    footprint_bytes = run_metadata.get("footprint_bytes")
    live_count = index_metadata.get("live_count") if isinstance(index_metadata, dict) else None
    avg_selected_list_count = _avg_or_p50(scan_stats.get("selected_list_count"))
    avg_visited_page_count = _avg_or_p50(scan_stats.get("visited_page_count"))
    return {
        "dataset_id": scenario["dataset_id"],
        "system_id": scenario["system_id"],
        "method_id": scenario["method_id"],
        "system_label": scenario["system_label"],
        "retriever_backend": scenario["retriever_backend"],
        "retrieval_mode": scenario["retrieval_mode"],
        "rerank_enabled": scenario["rerank_enabled"],
        "retrieval_execution_mode": run_metadata.get("retrieval_execution_mode"),
        "context_fetch_mode": run_metadata.get("context_fetch_mode"),
        "score_mode": _uniform_value(scan_stats.get("score_mode")),
        "avg_selected_list_count": avg_selected_list_count,
        "avg_selected_live_count": _avg_or_p50(scan_stats.get("selected_live_count")),
        "avg_visited_page_count": avg_visited_page_count,
        "avg_visited_code_count": _avg_or_p50(scan_stats.get("visited_code_count")),
        "avg_effective_probe_count": _avg_or_p50(scan_stats.get("effective_probe_count")),
        "avg_page_prune_count": _avg_or_p50(scan_stats.get("page_prune_count")),
        "visited_code_fraction": _scan_work_fraction(_avg_or_p50(scan_stats.get("visited_code_count")), live_count),
        "visited_page_fraction": _scan_work_fraction(
            avg_visited_page_count,
            _estimated_index_pages(footprint_bytes),
        ),
        "page_prune_count_p50": scan_stats.get("page_prune_count", {}).get("p50"),
        "early_stop_count_p50": scan_stats.get("early_stop_count", {}).get("p50"),
        "visited_code_count_p50": scan_stats.get("visited_code_count", {}).get("p50"),
        "router_restarts": router.get("restart_count"),
        "router_balance_penalty": router.get("balance_penalty"),
        "max_list_size": list_distribution.get("max_list_size"),
        "list_coeff_var": list_distribution.get("coeff_var"),
        "delta_head_block": index_metadata.get("delta_head_block"),
        "delta_tail_block": index_metadata.get("delta_tail_block"),
        "delta_live_count": index_metadata.get("delta_live_count"),
        "delta_batch_page_count": index_metadata.get("delta_batch_page_count"),
        "delta_page_depth": delta_health.get("delta_page_depth"),
        "delta_merge_recommended": delta_health.get("merge_recommended"),
        "exact_key_head_block": index_metadata.get("exact_key_head_block"),
        "exact_key_tail_block": index_metadata.get("exact_key_tail_block"),
        "exact_key_page_count": index_metadata.get("exact_key_page_count"),
        "compaction_recommended": maintenance.get("compaction_recommended"),
        "maintenance_action_recommended": index_metadata.get("maintenance_action_recommended"),
    }


def _end_to_end_row(scenario: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    answer_metrics = result.get("answer_metrics", {})
    operational = result.get("operational_summary", {}).get("latency_ms", {}).get("total", {})
    return {
        "dataset_id": scenario["dataset_id"],
        "system_id": scenario["system_id"],
        "method_id": scenario["method_id"],
        "system_label": scenario["system_label"],
        "retriever_backend": scenario["retriever_backend"],
        "retrieval_mode": scenario["retrieval_mode"],
        "rerank_enabled": scenario["rerank_enabled"],
        "generator_id": result.get("run_metadata", {}).get("generator_id", scenario.get("generator_id")),
        "answer_exact_match": answer_metrics.get("answer_exact_match"),
        "answer_f1": answer_metrics.get("answer_f1"),
        "total_latency_p50_ms": operational.get("p50"),
        "total_latency_p95_ms": operational.get("p95"),
        "total_latency_p99_ms": operational.get("p99"),
    }


def _build_report(
    plan: dict[str, object],
    retrieval_rows: Sequence[dict[str, object]],
    end_to_end_rows: Sequence[dict[str, object]],
    diagnostics_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    report = {
        "summary": {
            "dataset_count": len(plan["datasets"]),
            "retrieval_scenario_count": len(retrieval_rows),
            "end_to_end_scenario_count": len(end_to_end_rows),
            "systems": list(DEFAULT_RAG_SYSTEM_VARIANTS),
            "datasets": list(plan["datasets"]),
            "generator_id": plan["generator_id"],
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
            "Retriever diagnostics are operational evidence, not quality metrics.",
        ],
    }
    regression_gate = _build_regression_gate(plan, retrieval_rows, diagnostics_rows)
    if regression_gate is not None:
        report["regression_gate"] = regression_gate
    return report


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
                str(row["system_id"]),
            ),
        )[0]
        findings.append(
            {
                "dataset_id": dataset_id,
                "system_id": best["system_id"],
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
    diagnostics_rows: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# RAG Benchmark",
        "",
        "## Retrieval Benchmark",
        "",
        "| Dataset | System | Backend | Retrieval Mode | Recall@10 | MRR@10 | NDCG@10 | Hit Rate@10 | Evidence Coverage@10 | P50 Latency (ms) | P95 Latency (ms) | Footprint (bytes) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in retrieval_rows:
        lines.append(
            f"| {row['dataset_id']} | {row['system_label']} | {row['retriever_backend']} | "
            f"{row['retrieval_mode']} | {_fmt(row['recall@10'])} | {_fmt(row['mrr@10'])} | "
            f"{_fmt(row['ndcg@10'])} | {_fmt(row['hit_rate@10'])} | "
            f"{_fmt(row['evidence_coverage@10'])} | {_fmt(row['latency_p50_ms'])} | "
            f"{_fmt(row['latency_p95_ms'])} | {_fmt(row['footprint_bytes'])} |"
        )

    lines.extend(
        [
            "",
            "## End-to-End Benchmark",
            "",
            "| Dataset | System | Backend | Retrieval Mode | Generator | Answer EM | Answer F1 | Total P50 Latency (ms) | Total P95 Latency (ms) |",
            "|---|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in end_to_end_rows:
        lines.append(
            f"| {row['dataset_id']} | {row['system_label']} | {row['retriever_backend']} | "
            f"{row['retrieval_mode']} | {row['generator_id']} | {_fmt(row['answer_exact_match'])} | "
            f"{_fmt(row['answer_f1'])} | {_fmt(row['total_latency_p50_ms'])} | {_fmt(row['total_latency_p95_ms'])} |"
        )

    lines.extend(
        [
            "",
            "## Retriever Diagnostics",
            "",
            "| Dataset | System | Score Mode | avg_selected_list_count | avg_selected_live_count | avg_visited_page_count | avg_visited_code_count | avg_effective_probe_count | avg_page_prune_count | visited_code_fraction | visited_page_fraction | router_restarts | router_balance_penalty | max_list_size | list_coeff_var |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics_rows:
        lines.append(
            f"| {row['dataset_id']} | {row['system_label']} | {_fmt(row['score_mode'])} | "
            f"{_fmt(row['avg_selected_list_count'])} | {_fmt(row['avg_selected_live_count'])} | "
            f"{_fmt(row['avg_visited_page_count'])} | {_fmt(row['avg_visited_code_count'])} | "
            f"{_fmt(row['avg_effective_probe_count'])} | {_fmt(row['avg_page_prune_count'])} | "
            f"{_fmt(row['visited_code_fraction'])} | {_fmt(row['visited_page_fraction'])} | "
            f"{_fmt(row['router_restarts'])} | {_fmt(row['router_balance_penalty'])} | "
            f"{_fmt(row['max_list_size'])} | {_fmt(row['list_coeff_var'])} |"
        )

    if "regression_gate" in report:
        gate = report["regression_gate"]
        lines.extend(
            [
                "",
                "## Regression Gate",
                "",
                f"- Dataset: {gate['dataset_id']}",
                f"- Method: {gate['method_id']}",
                f"- Passed: {_fmt(gate['passed'])}",
                "",
                "| Check | Passed | Observed | Threshold |",
                "|---|---|---:|---:|",
            ]
        )
        for name, check in gate["checks"].items():
            threshold = check.get("minimum", check.get("maximum", check.get("expected", "")))
            lines.append(
                f"| {name} | {_fmt(check['passed'])} | {_fmt(check.get('observed'))} | {_fmt(threshold)} |"
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
                f"- {finding['dataset_id']}: {finding['system_id']} leads on "
                f"{finding['metric']} ({_fmt(finding['value'])})."
            )
        lines.append("")

    lines.extend(["## Metric Validity Caveats", ""])
    for caveat in report["metric_validity_caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def _build_regression_gate(
    plan: dict[str, object],
    retrieval_rows: Sequence[dict[str, object]],
    diagnostics_rows: Sequence[dict[str, object]],
) -> dict[str, object] | None:
    raw_configs = plan.get("regression_gate")
    if not isinstance(raw_configs, dict) or not raw_configs:
        return None

    configs = [config for config in raw_configs.values() if isinstance(config, dict)]
    if not configs:
        return None
    if len(configs) == 1:
        return evaluate_hotpot_regression_gate(
            retrieval_rows=_merge_rows_for_regression_gate(retrieval_rows, diagnostics_rows),
            gate_config=configs[0],
        )

    evaluations = [
        evaluate_hotpot_regression_gate(
            retrieval_rows=_merge_rows_for_regression_gate(retrieval_rows, diagnostics_rows),
            gate_config=config,
        )
        for config in configs
    ]
    return {
        "passed": all(item["passed"] for item in evaluations),
        "evaluations": evaluations,
    }


def _merge_rows_for_regression_gate(
    retrieval_rows: Sequence[dict[str, object]],
    diagnostics_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    diagnostics_by_key = {
        (str(row["dataset_id"]), str(row["method_id"])): row
        for row in diagnostics_rows
    }
    merged: list[dict[str, object]] = []
    for retrieval_row in retrieval_rows:
        key = (str(retrieval_row["dataset_id"]), str(retrieval_row["method_id"]))
        combined = dict(retrieval_row)
        combined.update(diagnostics_by_key.get(key, {}))
        merged.append(combined)
    return merged


def _avg_or_p50(distribution: object) -> float | None:
    if not isinstance(distribution, dict):
        return None
    value = distribution.get("avg", distribution.get("p50"))
    return None if value is None else float(value)


def _uniform_value(distribution: object) -> object:
    if not isinstance(distribution, dict):
        return None
    return distribution.get("uniform")


def _scan_work_fraction(numerator: float | None, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        return None
    return round(numerator / denominator_value, 6)


def _estimated_index_pages(footprint_bytes: object) -> int | None:
    if footprint_bytes is None:
        return None
    footprint = int(footprint_bytes)
    if footprint <= 0:
        return None
    return max(1, math.ceil(footprint / 8192))


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
