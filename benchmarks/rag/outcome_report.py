from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


TURBOQUANT_METHOD_ID = "pg_turboquant_approx"
BASELINE_METHOD_IDS = ("pgvector_hnsw_approx", "pgvector_ivfflat_approx")


def load_campaign_payload(path: str | Path) -> dict[str, object]:
    payload_path = Path(path)
    return json.loads(payload_path.read_text(encoding="utf-8"))


def build_outcome_summary(campaign_payloads: Sequence[dict[str, object]]) -> dict[str, object]:
    comparisons: list[dict[str, object]] = []
    datasets: list[str] = []
    method_rows: list[dict[str, object]] = []

    for payload in campaign_payloads:
        retrieval_rows = list(payload.get("tables", {}).get("retrieval_only", []))
        rows_by_dataset: dict[str, dict[str, dict[str, object]]] = {}
        for row in retrieval_rows:
            dataset_id = str(row["dataset_id"])
            datasets.append(dataset_id)
            method_rows.append(_method_row(row))
            rows_by_dataset.setdefault(dataset_id, {})[str(row["method_id"])] = row

        for dataset_id, dataset_methods in rows_by_dataset.items():
            turboquant_row = dataset_methods.get(TURBOQUANT_METHOD_ID)
            if turboquant_row is None:
                continue
            for baseline_method_id in BASELINE_METHOD_IDS:
                baseline_row = dataset_methods.get(baseline_method_id)
                if baseline_row is None:
                    continue
                comparisons.append(_comparison_row(dataset_id, turboquant_row, baseline_row))

    unique_datasets = sorted(set(datasets))
    footprint_pass_count = sum(1 for item in comparisons if item["smaller_footprint"])
    latency_pass_count = sum(1 for item in comparisons if item["lower_p95"])
    overall_pass_count = sum(1 for item in comparisons if item["smaller_and_faster"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(unique_datasets),
        "datasets": unique_datasets,
        "comparison_count": len(comparisons),
        "smaller_footprint_count": footprint_pass_count,
        "lower_p95_count": latency_pass_count,
        "smaller_and_faster_count": overall_pass_count,
        "method_rows": method_rows,
        "comparisons": comparisons,
        "measured_scope": {
            "scope": "approximate retrieval only",
            "turboquant_method_id": TURBOQUANT_METHOD_ID,
            "baseline_method_ids": list(BASELINE_METHOD_IDS),
            "comparison_basis": "This report lists measured retrieval results for pg_turboquant against approximate pgvector baselines under the included source campaigns.",
            "fairness_note": "It reports the current measurement contract only and does not infer unmeasured prior benchmark inflation.",
        },
    }


def write_outcome_html(
    output_path: str | Path,
    campaign_payloads: Sequence[dict[str, object]],
    *,
    source_labels: Sequence[str] | None = None,
) -> dict[str, str]:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_outcome_summary(campaign_payloads)
    rendered = _render_html(summary, list(source_labels or ()))
    target_path.write_text(rendered, encoding="utf-8")
    return {"output_html": target_path.name}


def _comparison_row(
    dataset_id: str,
    turboquant_row: dict[str, object],
    baseline_row: dict[str, object],
) -> dict[str, object]:
    tq_footprint = _as_float(turboquant_row.get("footprint_bytes"))
    baseline_footprint = _as_float(baseline_row.get("footprint_bytes"))
    tq_latency = _as_float(turboquant_row.get("latency_p95_ms"))
    baseline_latency = _as_float(baseline_row.get("latency_p95_ms"))
    tq_recall = _as_float(turboquant_row.get("recall@10"))
    baseline_recall = _as_float(baseline_row.get("recall@10"))

    smaller_footprint = (
        tq_footprint is not None and baseline_footprint is not None and tq_footprint < baseline_footprint
    )
    lower_p95 = (
        tq_latency is not None and baseline_latency is not None and tq_latency <= baseline_latency
    )
    footprint_ratio = (
        baseline_footprint / tq_footprint
        if tq_footprint not in (None, 0.0) and baseline_footprint is not None
        else None
    )
    latency_ratio = (
        baseline_latency / tq_latency if tq_latency not in (None, 0.0) and baseline_latency is not None else None
    )
    return {
        "dataset_id": dataset_id,
        "turboquant_method_id": str(turboquant_row["method_id"]),
        "baseline_method_id": str(baseline_row["method_id"]),
        "turboquant_recall@10": tq_recall,
        "baseline_recall@10": baseline_recall,
        "recall_delta": None if tq_recall is None or baseline_recall is None else tq_recall - baseline_recall,
        "turboquant_latency_p95_ms": tq_latency,
        "baseline_latency_p95_ms": baseline_latency,
        "latency_ratio_vs_turboquant": latency_ratio,
        "lower_p95": lower_p95,
        "turboquant_footprint_bytes": tq_footprint,
        "baseline_footprint_bytes": baseline_footprint,
        "footprint_ratio_vs_turboquant": footprint_ratio,
        "smaller_footprint": smaller_footprint,
        "smaller_and_faster": smaller_footprint and lower_p95,
    }


def _render_html(summary: dict[str, object], source_labels: Sequence[str]) -> str:
    source_items = "".join(f"<li>{html.escape(label)}</li>" for label in source_labels)
    source_section = (
        f"<section><h2>Source Campaigns</h2><ul>{source_items}</ul></section>"
        if source_items
        else ""
    )
    method_rows = "".join(_render_method_row(row) for row in summary["method_rows"])
    comparison_rows = "".join(_render_comparison_row(row) for row in summary["comparisons"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TurboQuant Outcome</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #14231a;
      --muted: #5a6b5f;
      --line: #d7cfbf;
      --pass: #1f7a4d;
      --fail: #9f2d2d;
      --accent: #c55a11;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background: radial-gradient(circle at top, #fff7e9 0%, var(--bg) 55%);
      color: var(--ink);
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 40px 24px 80px;
    }}
    h1, h2 {{
      margin-bottom: 12px;
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.45;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(197, 90, 17, 0.12), rgba(20, 35, 26, 0.04));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 28px;
      box-shadow: 0 14px 40px rgba(20, 35, 26, 0.08);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-top: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: rgba(20, 35, 26, 0.04);
    }}
    .pass {{
      color: var(--pass);
      font-weight: 700;
    }}
    .fail {{
      color: var(--fail);
      font-weight: 700;
    }}
    .mono {{
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: 0.95em;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>TurboQuant Outcome</h1>
      <p>This report aggregates measured retrieval-only results for the included campaigns. It lists footprint, p50/p95 latency, scan-work fields, and measured pg_turboquant versus pgvector comparisons for the exact dataset/method matrix below.</p>
      <div class="grid">
        <div class="card"><strong>Datasets</strong><br>{summary["dataset_count"]}</div>
        <div class="card"><strong>Comparisons</strong><br>{summary["comparison_count"]}</div>
        <div class="card"><strong>Smaller Footprint</strong><br>{summary["smaller_footprint_count"]}/{summary["comparison_count"]}</div>
        <div class="card"><strong>Lower p95</strong><br>{summary["lower_p95_count"]}/{summary["comparison_count"]}</div>
        <div class="card"><strong>Smaller And Faster</strong><br>{summary["smaller_and_faster_count"]}/{summary["comparison_count"]}</div>
      </div>
    </section>
    <section>
      <h2>Measured Comparison Scope</h2>
      <p>{html.escape(summary["measured_scope"]["comparison_basis"])}</p>
      <p>{html.escape(summary["measured_scope"]["fairness_note"])}</p>
      <p class="mono">TurboQuant method: {html.escape(summary["measured_scope"]["turboquant_method_id"])}</p>
    </section>
    {source_section}
    <section>
      <h2>Method Metrics</h2>
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Method</th>
            <th>Recall@10</th>
            <th>p50</th>
            <th>p95</th>
            <th>Footprint</th>
            <th>visited_code_count</th>
            <th>visited_page_count</th>
            <th>selected_live_count</th>
            <th>selected_page_count</th>
            <th>score_kernel</th>
          </tr>
        </thead>
        <tbody>
          {method_rows}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Dataset Comparison</h2>
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Baseline</th>
            <th>Recall Delta</th>
            <th>Footprint</th>
            <th>Memory Ratio</th>
            <th>Latency</th>
            <th>Speed Ratio</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {comparison_rows}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def _render_comparison_row(row: dict[str, object]) -> str:
    status_parts = []
    if row["smaller_footprint"]:
        status_parts.append("smaller")
    else:
        status_parts.append("larger")
    if row["lower_p95"]:
        status_parts.append("faster")
    else:
        status_parts.append("slower")
    status_label = " / ".join(status_parts)
    status_class = "pass" if row["smaller_and_faster"] else "fail"
    return (
        "<tr>"
        f"<td>{html.escape(str(row['dataset_id']))}</td>"
        f"<td class=\"mono\">{html.escape(str(row['baseline_method_id']))}</td>"
        f"<td>{_fmt(row['recall_delta'])}</td>"
        f"<td>{_fmt(row['turboquant_footprint_bytes'])} vs {_fmt(row['baseline_footprint_bytes'])}</td>"
        f"<td>{_fmt_ratio(row['footprint_ratio_vs_turboquant'])}</td>"
        f"<td>{_fmt(row['turboquant_latency_p95_ms'])} ms vs {_fmt(row['baseline_latency_p95_ms'])} ms</td>"
        f"<td>{_fmt_ratio(row['latency_ratio_vs_turboquant'])}</td>"
        f"<td class=\"{status_class}\">{status_label}</td>"
        "</tr>"
    )


def _method_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "dataset_id": str(row["dataset_id"]),
        "method_id": str(row["method_id"]),
        "backend_family": str(row["backend_family"]),
        "recall@10": _as_float(row.get("recall@10")),
        "latency_p50_ms": _as_float(row.get("latency_p50_ms")),
        "latency_p95_ms": _as_float(row.get("latency_p95_ms")),
        "footprint_bytes": _as_float(row.get("footprint_bytes")),
        "visited_code_count": _as_float(row.get("avg_visited_code_count")),
        "visited_page_count": _as_float(row.get("avg_visited_page_count")),
        "selected_live_count": _as_float(row.get("avg_selected_live_count")),
        "selected_page_count": _as_float(row.get("avg_selected_page_count")),
        "score_kernel": str(row.get("score_kernel", "none")),
    }


def _render_method_row(row: dict[str, object]) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(str(row['dataset_id']))}</td>"
        f"<td class=\"mono\">{html.escape(str(row['method_id']))}</td>"
        f"<td>{_fmt(row['recall@10'])}</td>"
        f"<td>{_fmt(row['latency_p50_ms'])}</td>"
        f"<td>{_fmt(row['latency_p95_ms'])}</td>"
        f"<td>{_fmt(row['footprint_bytes'])}</td>"
        f"<td>{_fmt(row['visited_code_count'])}</td>"
        f"<td>{_fmt(row['visited_page_count'])}</td>"
        f"<td>{_fmt(row['selected_live_count'])}</td>"
        f"<td>{_fmt(row['selected_page_count'])}</td>"
        f"<td>{html.escape(str(row['score_kernel']))}</td>"
        "</tr>"
    )


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    numeric = float(value)
    if numeric.is_integer() and abs(numeric) >= 1000:
        return f"{int(numeric):,}"
    if numeric.is_integer():
        return f"{int(numeric)}"
    return f"{numeric:.4f}"


def _fmt_ratio(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}x"
