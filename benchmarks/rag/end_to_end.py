from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .operational_metrics import (
    ApproximateQueryCost,
    QueryOperationalMetrics,
    estimate_context_token_count,
    estimate_token_count,
    summarize_query_operational_metrics,
)


@dataclass(frozen=True)
class FixedGeneratorConfig:
    generator_id: str
    system_prompt: str
    max_contexts: int


@dataclass(frozen=True)
class RetrievalCacheEntry:
    query_id: str
    question: str
    retrieved_contexts: list[dict[str, str]]
    answer_reference: str
    retrieval_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    approximate_query_costs: tuple[ApproximateQueryCost, ...] = ()


def build_prompt(*, system_prompt: str, question: str, contexts: Sequence[dict[str, str]]) -> str:
    lines = [system_prompt, "", f"Question: {question}", "", "Contexts:"]
    for context in contexts:
        lines.append(f"[{context['id']}] {context['text']}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def run_fixed_generator_stage(
    *,
    cache_entries: Sequence[RetrievalCacheEntry],
    config: FixedGeneratorConfig,
    generator_fn: Callable[[str, RetrievalCacheEntry], str],
    clock_fn: Callable[[], float] = time.perf_counter,
) -> list[dict[str, object]]:
    results = []
    for entry in cache_entries:
        contexts = entry.retrieved_contexts[: config.max_contexts]
        prompt = build_prompt(
            system_prompt=config.system_prompt,
            question=entry.question,
            contexts=contexts,
        )
        started_at = clock_fn()
        answer = generator_fn(prompt, entry)
        finished_at = clock_fn()

        operational_metrics = QueryOperationalMetrics(
            retrieval_latency_ms=entry.retrieval_latency_ms,
            rerank_latency_ms=entry.rerank_latency_ms,
            generator_latency_ms=(finished_at - started_at) * 1000.0,
            retrieved_context_tokens=estimate_context_token_count(contexts),
            prompt_tokens=estimate_token_count(prompt),
            prompt_context_count=len(contexts),
            approximate_query_costs=entry.approximate_query_costs,
        )

        results.append(
            {
                "query_id": entry.query_id,
                "prompt": prompt,
                "contexts": contexts,
                "answer": answer,
                "reference_answer": entry.answer_reference,
                "operational_metrics": operational_metrics.to_dict(),
            }
        )
    return results


def export_end_to_end_run(
    *,
    output_dir: str | Path,
    run_metadata: dict[str, object],
    retrieval_summary: dict[str, object],
    generation_results: Sequence[dict[str, object]],
    answer_metrics: dict[str, float],
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_metadata.get("result_kind") != "end_to_end":
        raise ValueError("end-to-end export requires result_kind='end_to_end'")

    payload = {
        "run_metadata": run_metadata,
        "retrieval_summary": retrieval_summary,
        "generation_results": list(generation_results),
        "answer_metrics": dict(answer_metrics),
        "operational_summary": summarize_query_operational_metrics(
            [
                item["operational_metrics"]
                for item in generation_results
                if isinstance(item, dict) and "operational_metrics" in item
            ]
        ),
    }

    json_name = "end-to-end-results.json"
    markdown_name = "end-to-end-results.md"

    (output_dir / json_name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    markdown_lines = [
        "# End-to-End Results",
        "",
        f"- run_id: {run_metadata.get('run_id', '')}",
        f"- result_kind: {run_metadata.get('result_kind', '')}",
        f"- generator_id: {run_metadata.get('generator_id', '')}",
        "",
        "## Answer Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in sorted(answer_metrics.items()):
        markdown_lines.append(f"| {key} | {value:.6f} |")
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
        for name, distribution in operational_summary.get("budgets", {}).items():
            markdown_lines.append(
                f"| {name} | {distribution['p50']:.6f} | "
                f"{distribution['p95']:.6f} | {distribution['p99']:.6f} |"
            )
    (output_dir / markdown_name).write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return {"json": json_name, "markdown": markdown_name}
