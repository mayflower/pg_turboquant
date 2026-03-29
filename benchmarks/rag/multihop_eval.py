from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class MultihopQueryEvaluation:
    query_id: str
    retrieved_ids: list[str]
    supporting_ids: list[str]


def compute_multihop_support_metrics(
    queries: Sequence[MultihopQueryEvaluation],
    ks: Sequence[int] = (10,),
) -> tuple[dict[str, float], list[dict[str, object]]]:
    if not queries:
        raise ValueError("at least one query is required")

    ordered_ks = tuple(sorted(set(int(k) for k in ks)))
    metrics: dict[str, float] = {}
    diagnostics: list[dict[str, object]] = []

    for query in queries:
        diagnostic_entry = {
            "query_id": query.query_id,
            "supporting_ids": list(query.supporting_ids),
            "per_k": {},
        }
        supporting_set = set(query.supporting_ids)
        for k in ordered_ks:
            top_k = query.retrieved_ids[:k]
            retrieved_supporting_ids = [
                doc_id for doc_id in query.supporting_ids if doc_id in set(top_k)
            ]
            missing_supporting_ids = [
                doc_id for doc_id in query.supporting_ids if doc_id not in set(top_k)
            ]
            diagnostic_entry["per_k"][str(k)] = {
                "retrieved_supporting_ids": retrieved_supporting_ids,
                "missing_supporting_ids": missing_supporting_ids,
                "all_supporting_retrieved": len(retrieved_supporting_ids) == len(supporting_set),
            }
        diagnostics.append(diagnostic_entry)

    for k in ordered_ks:
        full_support_hits = sum(
            1
            for diagnostic in diagnostics
            if diagnostic["per_k"][str(k)]["all_supporting_retrieved"]
        )
        metrics[f"multihop_support_coverage@{k}"] = full_support_hits / len(queries)

    return metrics, diagnostics


def export_multihop_diagnostics(
    *,
    output_dir: str | Path,
    run_metadata: dict[str, object],
    overlay_metrics: dict[str, float],
    diagnostics: Sequence[dict[str, object]],
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_metadata": run_metadata,
        "overlay_metrics": dict(overlay_metrics),
        "diagnostics": list(diagnostics),
    }

    json_name = "multihop-diagnostics.json"
    (output_dir / json_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {"json": json_name}
