from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ApproximateQueryCost:
    name: str
    unit: str
    value: float

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "unit": self.unit, "value": float(self.value)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ApproximateQueryCost":
        return cls(
            name=str(payload["name"]),
            unit=str(payload["unit"]),
            value=float(payload["value"]),
        )


@dataclass(frozen=True)
class QueryOperationalMetrics:
    retrieval_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    generator_latency_ms: float | None = None
    retrieved_context_tokens: int | None = None
    prompt_tokens: int | None = None
    prompt_context_count: int | None = None
    approximate_query_costs: tuple[ApproximateQueryCost, ...] = ()
    scan_stats: Mapping[str, Any] | None = None

    def total_latency_ms(self) -> float | None:
        stages = [
            latency
            for latency in (
                self.retrieval_latency_ms,
                self.rerank_latency_ms,
                self.generator_latency_ms,
            )
            if latency is not None
        ]
        if not stages:
            return None
        return sum(stages)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}

        latency_ms: dict[str, float] = {}
        if self.retrieval_latency_ms is not None:
            latency_ms["retrieval"] = float(self.retrieval_latency_ms)
        if self.rerank_latency_ms is not None:
            latency_ms["rerank"] = float(self.rerank_latency_ms)
        if self.generator_latency_ms is not None:
            latency_ms["generator"] = float(self.generator_latency_ms)

        total_latency_ms = self.total_latency_ms()
        if total_latency_ms is not None:
            latency_ms["total"] = float(total_latency_ms)
        if latency_ms:
            payload["latency_ms"] = latency_ms

        budgets: dict[str, int] = {}
        if self.retrieved_context_tokens is not None:
            budgets["retrieved_context_tokens"] = int(self.retrieved_context_tokens)
        if self.prompt_tokens is not None:
            budgets["prompt_tokens"] = int(self.prompt_tokens)
        if self.prompt_context_count is not None:
            budgets["prompt_context_count"] = int(self.prompt_context_count)
        if budgets:
            payload["budgets"] = budgets

        if self.approximate_query_costs:
            payload["approximate_query_costs"] = [
                cost.to_dict() for cost in self.approximate_query_costs
            ]

        if self.scan_stats is not None:
            payload["scan_stats"] = dict(self.scan_stats)

        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "QueryOperationalMetrics":
        latency_ms = payload.get("latency_ms", {})
        budgets = payload.get("budgets", {})
        approximate_query_costs = tuple(
            ApproximateQueryCost.from_dict(cost)
            for cost in payload.get("approximate_query_costs", [])
        )
        return cls(
            retrieval_latency_ms=_maybe_float(latency_ms, "retrieval"),
            rerank_latency_ms=_maybe_float(latency_ms, "rerank"),
            generator_latency_ms=_maybe_float(latency_ms, "generator"),
            retrieved_context_tokens=_maybe_int(budgets, "retrieved_context_tokens"),
            prompt_tokens=_maybe_int(budgets, "prompt_tokens"),
            prompt_context_count=_maybe_int(budgets, "prompt_context_count"),
            approximate_query_costs=approximate_query_costs,
            scan_stats=dict(payload["scan_stats"]) if isinstance(payload.get("scan_stats"), Mapping) else None,
        )


def estimate_token_count(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return len(normalized.split())


def estimate_context_token_count(contexts: Sequence[Mapping[str, object]]) -> int:
    return sum(estimate_token_count(str(context.get("text", ""))) for context in contexts)


def summarize_query_operational_metrics(
    metrics: Sequence[QueryOperationalMetrics | Mapping[str, object]],
) -> dict[str, object]:
    if not metrics:
        return {}

    normalized = [
        item
        if isinstance(item, QueryOperationalMetrics)
        else QueryOperationalMetrics.from_dict(item)
        for item in metrics
    ]

    summary: dict[str, object] = {}

    latency_summary = _summarize_stage_latencies(normalized)
    if latency_summary:
        summary["latency_ms"] = latency_summary

    budget_summary = _summarize_budgets(normalized)
    if budget_summary:
        summary["budgets"] = budget_summary

    cost_summary = _summarize_costs(normalized)
    if cost_summary:
        summary["approximate_query_costs"] = cost_summary

    scan_stats_summary = _summarize_scan_stats(normalized)
    if scan_stats_summary:
        summary["scan_stats"] = scan_stats_summary

    return summary


def _summarize_stage_latencies(
    metrics: Sequence[QueryOperationalMetrics],
) -> dict[str, dict[str, object]]:
    latency_summary: dict[str, dict[str, object]] = {}
    for stage, values in (
        ("retrieval", [m.retrieval_latency_ms for m in metrics if m.retrieval_latency_ms is not None]),
        ("rerank", [m.rerank_latency_ms for m in metrics if m.rerank_latency_ms is not None]),
        ("generator", [m.generator_latency_ms for m in metrics if m.generator_latency_ms is not None]),
        ("total", [m.total_latency_ms() for m in metrics if m.total_latency_ms() is not None]),
    ):
        if values:
            latency_summary[stage] = _distribution([float(value) for value in values if value is not None])
    return latency_summary


def _summarize_budgets(
    metrics: Sequence[QueryOperationalMetrics],
) -> dict[str, dict[str, object]]:
    budget_summary: dict[str, dict[str, object]] = {}
    for name, values in (
        (
            "retrieved_context_tokens",
            [m.retrieved_context_tokens for m in metrics if m.retrieved_context_tokens is not None],
        ),
        ("prompt_tokens", [m.prompt_tokens for m in metrics if m.prompt_tokens is not None]),
        (
            "prompt_context_count",
            [m.prompt_context_count for m in metrics if m.prompt_context_count is not None],
        ),
    ):
        if values:
            budget_summary[name] = _distribution([float(value) for value in values])
    return budget_summary


def _summarize_costs(
    metrics: Sequence[QueryOperationalMetrics],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for item in metrics:
        for cost in item.approximate_query_costs:
            existing = grouped.get(cost.name)
            if existing is None:
                existing = {"unit": cost.unit, "values": []}
                grouped[cost.name] = existing
            elif existing["unit"] != cost.unit:
                raise ValueError(
                    f"approximate query cost '{cost.name}' changed unit from "
                    f"{existing['unit']} to {cost.unit}"
                )
            existing["values"].append(float(cost.value))

    summary: dict[str, dict[str, object]] = {}
    for name, payload in grouped.items():
        distribution = _distribution(payload["values"])
        distribution["unit"] = payload["unit"]
        summary[name] = distribution
    return summary


def _distribution(values: Sequence[float]) -> dict[str, object]:
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _summarize_scan_stats(
    metrics: Sequence[QueryOperationalMetrics],
) -> dict[str, object]:
    grouped_numeric: dict[str, list[float]] = {}
    grouped_text: dict[str, list[str]] = {}

    for item in metrics:
        if not isinstance(item.scan_stats, Mapping):
            continue
        for key, value in item.scan_stats.items():
            if isinstance(value, (int, float)):
                grouped_numeric.setdefault(key, []).append(float(value))
            elif isinstance(value, str):
                grouped_text.setdefault(key, []).append(value)

    summary: dict[str, object] = {}
    for key, values in grouped_numeric.items():
        summary[key] = _distribution(values)
    for key, values in grouped_text.items():
        distinct = sorted(set(values))
        summary[key] = {
            "count": len(values),
            "uniform": distinct[0] if len(distinct) == 1 else None,
            "values": distinct,
        }
    return summary


def _percentile(values: Sequence[float], pct: float) -> float:
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


def _maybe_float(payload: object, key: str) -> float | None:
    if not isinstance(payload, Mapping) or key not in payload:
        return None
    return float(payload[key])


def _maybe_int(payload: object, key: str) -> int | None:
    if not isinstance(payload, Mapping) or key not in payload:
        return None
    return int(payload[key])
