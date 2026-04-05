from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def load_campaign_payload(path: str | Path) -> dict[str, object]:
    payload_path = Path(path)
    return json.loads(payload_path.read_text(encoding="utf-8"))


def build_outcome_summary(campaign_payloads: Sequence[dict[str, object]]) -> dict[str, object]:
    retrieval_rows: list[dict[str, object]] = []
    diagnostics_by_key: dict[tuple[str, str], dict[str, object]] = {}

    for payload in campaign_payloads:
        for row in payload.get("tables", {}).get("retrieval_benchmark", []):
            retrieval_rows.append(dict(row))
        for row in payload.get("tables", {}).get("retrieval_diagnostics", []):
            diagnostics_by_key[(str(row["dataset_id"]), str(row["system_id"]))] = dict(row)

    datasets = sorted({str(row["dataset_id"]) for row in retrieval_rows})
    systems = sorted({str(row["system_id"]) for row in retrieval_rows})
    system_rows = [_system_row(row, diagnostics_by_key.get((str(row["dataset_id"]), str(row["system_id"])))) for row in retrieval_rows]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(datasets),
        "datasets": datasets,
        "system_count": len(systems),
        "systems": systems,
        "system_rows": system_rows,
        "retrieval_leaders": _leader_rows(retrieval_rows, metric_name="recall@10", higher_is_better=True),
        "latency_leaders": _leader_rows(retrieval_rows, metric_name="latency_p95_ms", higher_is_better=False),
        "footprint_leaders": _leader_rows(retrieval_rows, metric_name="footprint_bytes", higher_is_better=False),
        "measured_scope": {
            "scope": "retrieval and end-to-end rag benchmark results",
            "comparison_basis": "This report summarizes measured benchmark rows for each retrieval system under the included campaign payloads.",
            "fairness_note": "Quality, latency, and footprint remain separate dimensions and should be compared within the same dataset and generator contract.",
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


def _leader_rows(
    retrieval_rows: Sequence[dict[str, object]],
    *,
    metric_name: str,
    higher_is_better: bool,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in retrieval_rows:
        grouped.setdefault(str(row["dataset_id"]), []).append(row)

    leaders: list[dict[str, object]] = []
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
        leaders.append(
            {
                "dataset_id": dataset_id,
                "system_id": str(best["system_id"]),
                "system_label": str(best.get("system_label", best["system_id"])),
                "metric": metric_name,
                "value": best[metric_name],
            }
        )
    return leaders


def _system_row(row: dict[str, object], diagnostics: dict[str, object] | None) -> dict[str, object]:
    return {
        "dataset_id": str(row["dataset_id"]),
        "system_id": str(row["system_id"]),
        "system_label": str(row.get("system_label", row["system_id"])),
        "retriever_backend": str(row.get("retriever_backend", "")),
        "retrieval_mode": str(row.get("retrieval_mode", "")),
        "recall@10": _as_float(row.get("recall@10")),
        "mrr@10": _as_float(row.get("mrr@10")),
        "ndcg@10": _as_float(row.get("ndcg@10")),
        "hit_rate@10": _as_float(row.get("hit_rate@10")),
        "evidence_coverage@10": _as_float(row.get("evidence_coverage@10")),
        "latency_p50_ms": _as_float(row.get("latency_p50_ms")),
        "latency_p95_ms": _as_float(row.get("latency_p95_ms")),
        "footprint_bytes": _as_float(row.get("footprint_bytes")),
        "diagnostics": {
            "score_mode": None if diagnostics is None else diagnostics.get("score_mode"),
            "avg_selected_list_count": None if diagnostics is None else _as_float(diagnostics.get("avg_selected_list_count")),
            "avg_selected_live_count": None if diagnostics is None else _as_float(diagnostics.get("avg_selected_live_count")),
            "avg_visited_page_count": None if diagnostics is None else _as_float(diagnostics.get("avg_visited_page_count")),
            "avg_visited_code_count": None if diagnostics is None else _as_float(diagnostics.get("avg_visited_code_count")),
        },
    }


def _render_html(summary: dict[str, object], source_labels: Sequence[str]) -> str:
    source_items = "".join(f"<li>{html.escape(label)}</li>" for label in source_labels)
    source_section = (
        f"<section><h2>Source Campaigns</h2><ul>{source_items}</ul></section>"
        if source_items
        else ""
    )
    system_rows = "".join(_render_system_row(row) for row in summary["system_rows"])
    leader_rows = "".join(_render_leader_row(row) for row in _flatten_leaders(summary))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Benchmark Outcome</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #14231a;
      --muted: #5a6b5f;
      --line: #d7cfbf;
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
    .mono {{
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: 0.95em;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>RAG Benchmark Outcome</h1>
      <p>This report summarizes benchmarked retrieval systems across the included campaigns. It keeps retrieval quality, latency, footprint, and retriever diagnostics visible as separate dimensions.</p>
      <div class="grid">
        <div class="card"><strong>Datasets</strong><br>{summary["dataset_count"]}</div>
        <div class="card"><strong>Systems</strong><br>{summary["system_count"]}</div>
        <div class="card"><strong>System Rows</strong><br>{len(summary["system_rows"])}</div>
      </div>
    </section>
    <section>
      <h2>Measured Comparison Scope</h2>
      <p>{html.escape(summary["measured_scope"]["comparison_basis"])}</p>
      <p>{html.escape(summary["measured_scope"]["fairness_note"])}</p>
    </section>
    {source_section}
    <section>
      <h2>Retrieval Systems</h2>
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>System</th>
            <th>Backend</th>
            <th>Mode</th>
            <th>Recall@10</th>
            <th>MRR@10</th>
            <th>NDCG@10</th>
            <th>Hit Rate@10</th>
            <th>evidence_coverage@10</th>
            <th>p50</th>
            <th>p95</th>
            <th>Footprint</th>
            <th>score_mode</th>
          </tr>
        </thead>
        <tbody>
          {system_rows}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Dataset Leaders</h2>
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Metric</th>
            <th>Leader</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {leader_rows}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def _flatten_leaders(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in ("retrieval_leaders", "latency_leaders", "footprint_leaders"):
        rows.extend(summary[key])
    return rows


def _render_system_row(row: dict[str, object]) -> str:
    diagnostics = row["diagnostics"]
    return (
        "<tr>"
        f"<td>{html.escape(str(row['dataset_id']))}</td>"
        f"<td class=\"mono\">{html.escape(str(row['system_label']))}</td>"
        f"<td>{html.escape(str(row['retriever_backend']))}</td>"
        f"<td>{html.escape(str(row['retrieval_mode']))}</td>"
        f"<td>{_fmt(row['recall@10'])}</td>"
        f"<td>{_fmt(row['mrr@10'])}</td>"
        f"<td>{_fmt(row['ndcg@10'])}</td>"
        f"<td>{_fmt(row['hit_rate@10'])}</td>"
        f"<td>{_fmt(row['evidence_coverage@10'])}</td>"
        f"<td>{_fmt(row['latency_p50_ms'])}</td>"
        f"<td>{_fmt(row['latency_p95_ms'])}</td>"
        f"<td>{_fmt(row['footprint_bytes'])}</td>"
        f"<td>{html.escape(str(diagnostics['score_mode']))}</td>"
        "</tr>"
    )


def _render_leader_row(row: dict[str, object]) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(str(row['dataset_id']))}</td>"
        f"<td>{html.escape(str(row['metric']))}</td>"
        f"<td class=\"mono\">{html.escape(str(row['system_label']))}</td>"
        f"<td>{_fmt(row['value'])}</td>"
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
