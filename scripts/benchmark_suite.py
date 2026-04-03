#!/usr/bin/env -S uv run python
import argparse
import html
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_SEED = 20260327
TOP_K_VALUES = (10, 100)
SUPPORTED_METRICS = ("cosine", "ip")
SUPPORTED_METHODS = (
    "turboquant_flat",
    "turboquant_ivf",
    "turboquant_bitmap",
    "pgvector_ivfflat",
    "pgvector_hnsw",
)
SUPPORTED_CORPORA = (
    "normalized_dense",
    "non_normalized_varied_norms",
    "clustered",
    "mixed_live_dead",
    "hotpot_skewed",
    "hotpot_overlap",
)
PROFILE_CONFIGS = {
    "tiny": {"rows": 128, "queries": 8, "dimension": 4, "repetitions": 2},
    "quick": {"rows": 512, "queries": 16, "dimension": 8, "repetitions": 3},
    "medium": {"rows": 2048, "queries": 24, "dimension": 16, "repetitions": 3},
    "full": {"rows": 8192, "queries": 32, "dimension": 32, "repetitions": 5},
}


@dataclass(frozen=True)
class Row:
    row_id: int
    values: tuple[float, ...]
    deleted: bool = False


@dataclass(frozen=True)
class Corpus:
    name: str
    dimension: int
    rows: list[Row]
    queries: list[tuple[float, ...]]
    metadata: dict


def parse_csv_list(raw: Optional[str], supported: Iterable[str]) -> list[str]:
    if raw is None:
        return list(supported)

    values = [value.strip() for value in raw.split(",") if value.strip()]
    unsupported = [value for value in values if value not in supported]
    if unsupported:
        raise SystemExit(f"unsupported values: {', '.join(unsupported)}")
    if not values:
        raise SystemExit("at least one value must be selected")
    return values


def parse_freeform_csv_list(raw: Optional[str]) -> list[str]:
    if raw is None:
        return []
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise SystemExit("at least one value must be selected")
    return values


def resolve_qjl_sketch_dimension(token: Optional[str], dimension: int) -> int:
    if token in (None, "", "d"):
        return dimension
    if token == "d/2":
        return max(1, dimension // 2)
    if token == "d/4":
        return max(1, dimension // 4)
    value = int(token)
    if value < 1 or value > dimension:
        raise SystemExit(
            f"invalid turboquant qjl sketch dimension {value}: expected 1..{dimension}"
        )
    return value


def normalize(values: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def rounded(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(round(value, 6) for value in values)


def cosine_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("inf")
    dot = sum(a * b for a, b in zip(left, right))
    return 1.0 - (dot / (left_norm * right_norm))


def inner_product_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return -sum(a * b for a, b in zip(left, right))


def exact_distance(metric: str, left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if metric == "cosine":
        return cosine_distance(left, right)
    if metric == "ip":
        return inner_product_distance(left, right)
    raise AssertionError(f"unexpected metric {metric}")


def metric_order_operator(metric: str) -> str:
    if metric == "cosine":
        return "<=>"
    if metric == "ip":
        return "<#>"
    raise AssertionError(f"unexpected metric {metric}")


def percentile_ms(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 6)

    ordered = sorted(values)
    index = math.ceil((percentile / 100.0) * len(ordered)) - 1
    index = max(0, min(index, len(ordered) - 1))
    return round(ordered[index], 6)


def vector_literal(values: tuple[float, ...]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in values) + "]"


def rerank_candidate_limit(limit: int, requested_limit: Optional[int]) -> int:
    if requested_limit is None:
        return limit
    return requested_limit


def effective_candidate_limit(
    limit: int,
    requested_limit: Optional[int],
    extra_candidates: int = 0,
) -> int:
    base_limit = max(limit, rerank_candidate_limit(limit, requested_limit))
    if extra_candidates <= 0:
        return base_limit
    return base_limit + extra_candidates


def auto_decode_rescore_extra_candidates(
    limit: int,
    requested_limit: Optional[int],
) -> int:
    base_limit = rerank_candidate_limit(limit, requested_limit)
    return min(512, max(128, base_limit // 2))


def resolve_decode_rescore_extra_candidates(
    limit: int,
    requested_limit: Optional[int],
    decode_rescore_factor: int,
    explicit_extra_candidates: Optional[int],
) -> int:
    if decode_rescore_factor <= 1:
        return 0
    if explicit_extra_candidates is not None:
        return max(0, explicit_extra_candidates)
    return auto_decode_rescore_extra_candidates(limit, requested_limit)


def capability_metadata(spec: dict) -> dict:
    return {
        "ordered_scan": True,
        "bitmap_scan": spec["index_method"] == "turboquant",
        "index_only_scan": False,
        "multicolumn": spec["index_method"] == "turboquant",
        "include_columns": False,
    }


def synthetic_simd_metadata() -> dict:
    machine = platform.machine().lower()
    neon = any(token in machine for token in ("arm", "aarch64"))
    avx2 = any(token in machine for token in ("x86_64", "amd64", "i386"))
    avx512 = False

    if avx512:
        preferred = "avx512"
    elif avx2:
        preferred = "avx2"
    elif neon:
        preferred = "neon"
    else:
        preferred = "scalar"

    return {
        "preferred_kernel": preferred,
        "selected_kernel": preferred,
        "code_domain_kernel": "scalar",
        "compiled": {
            "scalar": True,
            "avx2": avx2,
            "avx512": avx512,
            "neon": neon,
        },
        "runtime_available": {
            "scalar": True,
            "avx2": avx2,
            "avx512": avx512,
            "neon": neon,
        },
    }


def synthetic_code_domain_kernel(corpus: Corpus, spec: dict, simd_metadata: dict) -> str:
    if spec["index_method"] != "turboquant":
        return "none"
    if spec.get("query_mode") == "bitmap_filter":
        return "none"
    if int(spec["with"].get("bits", 0)) != 4:
        return "scalar"
    if int(spec["with"].get("qjl_sketch_dim", corpus.dimension)) != corpus.dimension:
        return "scalar"
    if corpus.dimension == 0 or corpus.dimension % 8 != 0:
        return "scalar"
    if simd_metadata.get("runtime_available", {}).get("avx2"):
        return "avx2"
    if simd_metadata.get("runtime_available", {}).get("neon"):
        return "neon"
    return "scalar"


def environment_metadata() -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_arch": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "release": platform.release(),
    }


def make_synthetic_microbench_row(
    benchmark: str,
    requested_kernel: str,
    kernel: str,
    qjl_lut_mode: str,
    iterations: int,
    visited_code_count: int,
    visited_page_count: int,
    total_ns: int,
    *,
    requested_kernel_honored: bool = True,
    runtime_available: bool = True,
    dimension: int = 32,
    bits: int = 4,
    lane_count: int = 1,
    candidate_heap_insert_count: int = 0,
    candidate_heap_replace_count: int = 0,
    candidate_heap_reject_count: int = 0,
    local_candidate_heap_insert_count: int = 0,
    local_candidate_heap_replace_count: int = 0,
    local_candidate_heap_reject_count: int = 0,
    local_candidate_merge_count: int = 0,
    scratch_allocations: int = 0,
    decoded_buffer_reuses: int = 0,
    code_view_uses: int = 0,
    code_copy_uses: int = 0,
    list_count: int = 0,
    probe_count: int = 0,
    scan_layout: str = "row_major",
) -> dict:
    return {
        "benchmark": benchmark,
        "requested_kernel": requested_kernel,
        "kernel": kernel,
        "qjl_lut_mode": qjl_lut_mode,
        "requested_kernel_honored": requested_kernel_honored,
        "dimension": dimension,
        "bits": bits,
        "iterations": iterations,
        "lane_count": lane_count,
        "visited_code_count": visited_code_count,
        "visited_page_count": visited_page_count,
        "candidate_heap_insert_count": candidate_heap_insert_count,
        "candidate_heap_replace_count": candidate_heap_replace_count,
        "candidate_heap_reject_count": candidate_heap_reject_count,
        "local_candidate_heap_insert_count": local_candidate_heap_insert_count,
        "local_candidate_heap_replace_count": local_candidate_heap_replace_count,
        "local_candidate_heap_reject_count": local_candidate_heap_reject_count,
        "local_candidate_merge_count": local_candidate_merge_count,
        "scratch_allocations": scratch_allocations,
        "decoded_buffer_reuses": decoded_buffer_reuses,
        "code_view_uses": code_view_uses,
        "code_copy_uses": code_copy_uses,
        "scan_layout": scan_layout,
        "list_count": list_count,
        "probe_count": probe_count,
        "total_ns": total_ns,
        "ns_per_op": round(total_ns / max(1, iterations), 3),
        "codes_per_second": round((visited_code_count * 1_000_000_000.0) / max(1, total_ns), 3),
        "pages_per_second": round((visited_page_count * 1_000_000_000.0) / max(1, total_ns), 3),
        "runtime_available": runtime_available,
    }


def synthetic_microbenchmarks() -> dict:
    simd = synthetic_simd_metadata()
    preferred = simd["preferred_kernel"]
    list_count = 256
    probe_count = 8
    results = [
        make_synthetic_microbench_row(
            "score_code_from_lut",
            "scalar",
            "scalar",
            "float",
            25000,
            25000,
            0,
            7_000_000,
        ),
        make_synthetic_microbench_row(
            "score_code_from_lut",
            "auto",
            preferred,
            "quantized" if preferred != "scalar" else "float",
            25000,
            25000,
            0,
            4_100_000 if preferred != "scalar" else 7_000_000,
            runtime_available=True,
        ),
        make_synthetic_microbench_row(
            "score_code_from_lut",
            "avx2",
            "avx2" if bool(simd["runtime_available"].get("avx2")) else "scalar",
            "quantized" if bool(simd["runtime_available"].get("avx2")) else "float",
            25000,
            25000,
            0,
            3_800_000 if bool(simd["runtime_available"].get("avx2")) else 7_200_000,
            requested_kernel_honored=bool(simd["runtime_available"].get("avx2")),
            runtime_available=bool(simd["runtime_available"].get("avx2")),
        ),
        make_synthetic_microbench_row(
            "score_code_from_lut",
            "neon",
            "neon" if bool(simd["runtime_available"].get("neon")) else "scalar",
            "quantized" if bool(simd["runtime_available"].get("neon")) else "float",
            25000,
            25000,
            0,
            3_900_000 if bool(simd["runtime_available"].get("neon")) else 7_200_000,
            requested_kernel_honored=bool(simd["runtime_available"].get("neon")),
            runtime_available=bool(simd["runtime_available"].get("neon")),
        ),
        make_synthetic_microbench_row(
            "score_code_from_lut_quantized_reference",
            "scalar",
            "scalar",
            "quantized",
            25000,
            25000,
            0,
            6_200_000,
        ),
        make_synthetic_microbench_row(
            "page_scan",
            "auto",
            preferred,
            "quantized" if preferred != "scalar" else "float",
            1024,
            32768,
            1024,
            5_000_000 if preferred != "scalar" else 7_800_000,
            runtime_available=True,
            lane_count=32,
            candidate_heap_insert_count=8192,
            local_candidate_heap_insert_count=8192,
            local_candidate_heap_replace_count=4096,
            local_candidate_heap_reject_count=20480,
            local_candidate_merge_count=8192,
            scratch_allocations=1,
            decoded_buffer_reuses=1023,
            code_view_uses=32768,
        ),
        make_synthetic_microbench_row(
            "page_scan_global_heap_only",
            "auto",
            preferred,
            "quantized" if preferred != "scalar" else "float",
            1024,
            32768,
            1024,
            7_800_000 if preferred != "scalar" else 8_400_000,
            runtime_available=True,
            lane_count=32,
            candidate_heap_insert_count=8,
            candidate_heap_replace_count=11264,
            candidate_heap_reject_count=21496,
            code_view_uses=32768,
        ),
        make_synthetic_microbench_row(
            "router_top_probes_full_sort",
            "scalar",
            "scalar",
            "float",
            10000,
            list_count * 10000,
            0,
            26_000_000,
            bits=0,
            lane_count=0,
            list_count=list_count,
            probe_count=probe_count,
        ),
        make_synthetic_microbench_row(
            "router_top_probes_partial",
            "scalar",
            "scalar",
            "float",
            10000,
            list_count * 10000,
            0,
            9_000_000,
            bits=0,
            lane_count=0,
            list_count=list_count,
            probe_count=probe_count,
        ),
    ]
    return {
        "architecture": environment_metadata()["cpu_arch"],
        "simd": {
            "scalar": {
                "compiled": True,
                "runtime_available": True,
            },
            "avx2": {
                "compiled": bool(simd["compiled"].get("avx2")),
                "runtime_available": bool(simd["runtime_available"].get("avx2")),
            },
            "avx512": {
                "compiled": bool(simd["compiled"].get("avx512")),
                "runtime_available": bool(simd["runtime_available"].get("avx512")),
            },
            "neon": {
                "compiled": bool(simd["compiled"].get("neon")),
                "runtime_available": bool(simd["runtime_available"].get("neon")),
            },
        },
        "results": results,
    }


def microbench_lookup(
    results: list[dict],
    benchmark: str,
    *,
    requested_kernel: Optional[str] = None,
    qjl_lut_mode: Optional[str] = None,
) -> Optional[dict]:
    for row in results:
        if row.get("benchmark") != benchmark:
            continue
        if requested_kernel is not None and row.get("requested_kernel") != requested_kernel:
            continue
        if qjl_lut_mode is not None and row.get("qjl_lut_mode") != qjl_lut_mode:
            continue
        return row
    return None


def microbench_ratio(candidate_value: object, baseline_value: object) -> Optional[float]:
    baseline = float(baseline_value)
    if baseline == 0.0:
        return None
    return round(float(candidate_value) / baseline, 6)


def microbench_delta(candidate: dict, baseline: dict, key: str) -> float:
    return round(float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)), 6)


def build_microbench_comparison(
    comparison: str,
    comparison_kind: str,
    baseline: Optional[dict],
    candidate: Optional[dict],
) -> Optional[dict]:
    if baseline is None or candidate is None:
        return None

    return {
        "comparison": comparison,
        "comparison_kind": comparison_kind,
        "baseline_benchmark": baseline["benchmark"],
        "candidate_benchmark": candidate["benchmark"],
        "baseline_requested_kernel": baseline.get("requested_kernel"),
        "candidate_requested_kernel": candidate.get("requested_kernel"),
        "baseline_kernel": baseline.get("kernel"),
        "candidate_kernel": candidate.get("kernel"),
        "baseline": baseline,
        "candidate": candidate,
        "metrics": {
            "codes_per_second_ratio": microbench_ratio(
                candidate.get("codes_per_second", 0.0),
                baseline.get("codes_per_second", 0.0),
            ),
            "ns_per_op_ratio": microbench_ratio(
                candidate.get("ns_per_op", 0.0),
                baseline.get("ns_per_op", 0.0),
            ),
            "visited_code_count_delta": microbench_delta(candidate, baseline, "visited_code_count"),
            "visited_page_count_delta": microbench_delta(candidate, baseline, "visited_page_count"),
            "candidate_heap_insert_delta": microbench_delta(
                candidate, baseline, "candidate_heap_insert_count"
            ),
            "candidate_heap_replace_delta": microbench_delta(
                candidate, baseline, "candidate_heap_replace_count"
            ),
            "candidate_heap_reject_delta": microbench_delta(
                candidate, baseline, "candidate_heap_reject_count"
            ),
            "local_candidate_heap_insert_delta": microbench_delta(
                candidate, baseline, "local_candidate_heap_insert_count"
            ),
            "local_candidate_heap_replace_delta": microbench_delta(
                candidate, baseline, "local_candidate_heap_replace_count"
            ),
            "local_candidate_heap_reject_delta": microbench_delta(
                candidate, baseline, "local_candidate_heap_reject_count"
            ),
            "local_candidate_merge_delta": microbench_delta(
                candidate, baseline, "local_candidate_merge_count"
            ),
        },
    }


def build_microbench_gate(
    gate: str,
    comparison: Optional[dict],
    category: str,
    *,
    requires_runtime_row: bool = False,
    require_quantized_transition: bool = False,
    require_reduced_global_heap_churn: bool = False,
) -> dict:
    if comparison is None:
        return {
            "gate": gate,
            "comparison": None,
            "category": category,
            "status": "not_applicable",
            "checks": {
                "same_workload": False,
                "throughput_directional_signal": False,
            },
        }

    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    metrics = comparison["metrics"]
    global_heap_baseline = (
        float(baseline.get("candidate_heap_insert_count", 0))
        + float(baseline.get("candidate_heap_replace_count", 0))
    )
    global_heap_candidate = (
        float(candidate.get("candidate_heap_insert_count", 0))
        + float(candidate.get("candidate_heap_replace_count", 0))
    )
    same_workload = (
        baseline.get("dimension") == candidate.get("dimension")
        and baseline.get("bits") == candidate.get("bits")
        and baseline.get("visited_code_count") == candidate.get("visited_code_count")
        and baseline.get("visited_page_count") == candidate.get("visited_page_count")
        and baseline.get("iterations") == candidate.get("iterations")
    )
    throughput_directional_signal = (
        metrics.get("codes_per_second_ratio") is not None
        and float(metrics["codes_per_second_ratio"]) >= 1.0
    )
    checks = {
        "same_workload": same_workload,
        "throughput_directional_signal": throughput_directional_signal,
        "ns_per_op_directional_signal": (
            metrics.get("ns_per_op_ratio") is not None and float(metrics["ns_per_op_ratio"]) <= 1.0
        ),
    }

    if requires_runtime_row:
        checks["runtime_available"] = bool(candidate.get("runtime_available"))
        checks["requested_kernel_honored"] = bool(candidate.get("requested_kernel_honored"))

    if require_quantized_transition:
        checks["quantized_transition"] = (
            baseline.get("qjl_lut_mode") == "float" and candidate.get("qjl_lut_mode") == "quantized"
        )

    if require_reduced_global_heap_churn:
        checks["reduced_global_heap_churn"] = global_heap_candidate < global_heap_baseline
        checks["local_merge_present"] = float(candidate.get("local_candidate_merge_count", 0)) > 0.0

    if not same_workload:
        status = "warn"
    elif requires_runtime_row and (
        not checks["runtime_available"] or not checks["requested_kernel_honored"]
    ):
        status = "not_applicable"
    else:
        required_checks = ["throughput_directional_signal", "ns_per_op_directional_signal"]
        if require_reduced_global_heap_churn:
            required_checks = ["reduced_global_heap_churn", "local_merge_present"]
        if require_quantized_transition:
            required_checks.append("quantized_transition")
        status = "pass" if all(bool(checks.get(key)) for key in required_checks) else "warn"

    return {
        "gate": gate,
        "comparison": comparison["comparison"],
        "category": category,
        "status": status,
        "checks": checks,
    }


def augment_microbenchmarks(section: dict) -> dict:
    results = section.get("results", [])
    comparisons = [
        build_microbench_comparison(
            "score_code_from_lut_avx2_vs_scalar",
            "kernel_speedup",
            microbench_lookup(results, "score_code_from_lut", requested_kernel="scalar"),
            microbench_lookup(results, "score_code_from_lut", requested_kernel="avx2"),
        ),
        build_microbench_comparison(
            "score_code_from_lut_neon_vs_scalar",
            "kernel_speedup",
            microbench_lookup(results, "score_code_from_lut", requested_kernel="scalar"),
            microbench_lookup(results, "score_code_from_lut", requested_kernel="neon"),
        ),
        build_microbench_comparison(
            "qjl_lut_quantized_vs_float_reference",
            "lut_mode",
            microbench_lookup(
                results,
                "score_code_from_lut",
                requested_kernel="scalar",
                qjl_lut_mode="float",
            ),
            microbench_lookup(results, "score_code_from_lut_quantized_reference"),
        ),
        build_microbench_comparison(
            "page_scan_block_local_vs_global_heap",
            "selection_strategy",
            microbench_lookup(results, "page_scan_global_heap_only"),
            microbench_lookup(results, "page_scan"),
        ),
    ]
    comparisons = [comparison for comparison in comparisons if comparison is not None]
    regression_gates = [
        build_microbench_gate(
            "avx2_kernel_speedup_signal",
            next(
                (
                    row
                    for row in comparisons
                    if row["comparison"] == "score_code_from_lut_avx2_vs_scalar"
                ),
                None,
            ),
            "kernel",
            requires_runtime_row=True,
        ),
        build_microbench_gate(
            "neon_kernel_speedup_signal",
            next(
                (
                    row
                    for row in comparisons
                    if row["comparison"] == "score_code_from_lut_neon_vs_scalar"
                ),
                None,
            ),
            "kernel",
            requires_runtime_row=True,
        ),
        build_microbench_gate(
            "quantized_qjl_lut_signal",
            next(
                (
                    row
                    for row in comparisons
                    if row["comparison"] == "qjl_lut_quantized_vs_float_reference"
                ),
                None,
            ),
            "lut",
            require_quantized_transition=True,
        ),
        build_microbench_gate(
            "block_local_selection_signal",
            next(
                (
                    row
                    for row in comparisons
                    if row["comparison"] == "page_scan_block_local_vs_global_heap"
                ),
                None,
            ),
            "selection",
            require_reduced_global_heap_churn=True,
        ),
    ]
    interpretation_notes = [
        "Use the regression-gate rows as directional checks: they compare equal-workload rows and keep scan-work counters alongside throughput.",
        "Kernel-specific gates may report not_applicable when the requested SIMD path is unavailable on the current machine.",
        "Heap-selection gates are expected to lower global heap churn and preserve visited-code/page counts, not just lower wall-clock time.",
    ]

    enriched = dict(section)
    enriched["comparisons"] = comparisons
    enriched["regression_gates"] = regression_gates
    enriched["interpretation_notes"] = interpretation_notes
    return enriched


def run_microbenchmarks(dry_run: bool) -> dict:
    if dry_run:
        return augment_microbenchmarks(synthetic_microbenchmarks())

    helper = Path(__file__).resolve().parent / "prod_score_microbench.py"
    result = subprocess.run(
        [sys.executable, str(helper)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    return augment_microbenchmarks(json.loads(result.stdout))


def scenario_matrix_metadata(
    profile: str,
    corpora: list[str],
    methods: list[str],
    metrics: list[str],
    turboquant_qjl_sketch_dims: list[str],
) -> dict:
    return {
        "profiles": [profile],
        "corpora": corpora,
        "methods": methods,
        "metrics": metrics,
        "turboquant_qjl_sketch_dims": turboquant_qjl_sketch_dims or ["d"],
    }


def synthetic_index_metadata(method: str, spec: dict, corpus: Corpus, benchmark_metric: str) -> dict:
    qjl_sketch_dimension = resolve_qjl_sketch_dimension(
        str(spec["with"].get("qjl_sketch_dim", "d")) if spec["index_method"] == "turboquant" else None,
        corpus.dimension,
    )
    metadata = {
        "access_method": spec["index_method"],
        "capabilities": capability_metadata(spec),
        "metric": benchmark_metric,
        "opclass": spec["opclass"],
        "format_version": 12 if spec["index_method"] == "turboquant" else 0,
        "list_count": int(spec["with"].get("lists", 0)) if "lists" in spec["with"] else 0,
        "page_summary": {
            "mode": "disabled",
            "safe_pruning": False,
        },
    }

    if spec["index_method"] == "turboquant":
        if metadata["list_count"] > 0:
            if spec["with"].get("normalized") == "true":
                metadata["page_summary"]["mode"] = "safe_summary_pruning"
                metadata["page_summary"]["safe_pruning"] = True
            else:
                metadata["page_summary"]["mode"] = "ordering_only"
        metadata.update(
            {
                "codec": "prod",
                "bits": int(spec["with"]["bits"]),
                "lane_count": 0,
                "normalized": spec["with"].get("normalized") == "true",
                "live_count": len(active_rows(corpus)),
                "dead_count": 0,
                "heap_live_rows": len(active_rows(corpus)),
                "reclaimable_pages": 0,
                "batch_page_count": 0,
                "centroid_page_count": 0,
                "transform": {
                    "kind": spec["with"].get("transform", "hadamard"),
                    "version": 1,
                    "input_dimension": corpus.dimension,
                    "output_dimension": corpus.dimension,
                    "seed": 0,
                },
                "residual_sketch": {
                    "kind": "1bit_qjl",
                    "version": 2,
                    "bits_per_dimension": 1,
                    "projected_dimension": qjl_sketch_dimension,
                    "bit_budget": qjl_sketch_dimension,
                },
                "router": {
                    "algorithm": "kmeans" if int(spec["with"].get("lists", 0)) > 0 else "first_k",
                    "seed": 20260327,
                    "sample_count": 256,
                    "max_iterations": 8,
                    "completed_iterations": 0,
                    "trained_vector_count": len(active_rows(corpus)),
                },
                "list_distribution": {
                    "min_live_count": 0,
                    "max_live_count": 0,
                    "avg_live_count": 0.0,
                },
                "lists": [],
            }
        )

    return metadata


def generate_normalized_dense(config: dict, rng: random.Random) -> Corpus:
    dimension = config["dimension"]
    rows = []
    queries = []

    for row_id in range(1, config["rows"] + 1):
        values = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        rows.append(Row(row_id, rounded(normalize(values))))

    for _ in range(config["queries"]):
        values = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        queries.append(rounded(normalize(values)))

    return Corpus(
        name="normalized_dense",
        dimension=dimension,
        rows=rows,
        queries=queries,
        metadata={
            "distribution": "uniform_unit_sphere",
            "normalized": True,
        },
    )


def generate_non_normalized_varied_norms(config: dict, rng: random.Random) -> Corpus:
    dimension = config["dimension"]
    rows = []
    queries = []

    for row_id in range(1, config["rows"] + 1):
        base = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        scale = 0.25 + ((row_id % 9) + 1) / 2.0
        rows.append(Row(row_id, rounded(value * scale for value in normalize(base))))

    for query_index in range(config["queries"]):
        base = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        scale = 0.5 + ((query_index % 5) + 1) / 3.0
        queries.append(rounded(value * scale for value in normalize(base)))

    return Corpus(
        name="non_normalized_varied_norms",
        dimension=dimension,
        rows=rows,
        queries=queries,
        metadata={
            "distribution": "scaled_unit_sphere",
            "normalized": False,
        },
    )


def generate_clustered(config: dict, rng: random.Random) -> Corpus:
    dimension = config["dimension"]
    cluster_count = min(8, max(2, config["rows"] // 32))
    centers = []
    rows = []
    queries = []

    for _ in range(cluster_count):
        center = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        centers.append(rounded(normalize(center)))

    for row_id in range(1, config["rows"] + 1):
        center = centers[(row_id - 1) % cluster_count]
        values = [
            center[axis] + rng.uniform(-0.08, 0.08)
            for axis in range(dimension)
        ]
        rows.append(Row(row_id, rounded(normalize(values))))

    for query_index in range(config["queries"]):
        center = centers[query_index % cluster_count]
        values = [
            center[axis] + rng.uniform(-0.04, 0.04)
            for axis in range(dimension)
        ]
        queries.append(rounded(normalize(values)))

    return Corpus(
        name="clustered",
        dimension=dimension,
        rows=rows,
        queries=queries,
        metadata={
            "distribution": "clustered",
            "clusters": cluster_count,
            "normalized": True,
        },
    )


def generate_mixed_live_dead(config: dict, rng: random.Random) -> Corpus:
    base = generate_normalized_dense(config, rng)
    rows = []

    for row in base.rows:
        deleted = (row.row_id % 5 == 0)
        rows.append(Row(row.row_id, row.values, deleted=deleted))

    return Corpus(
        name="mixed_live_dead",
        dimension=base.dimension,
        rows=rows,
        queries=base.queries,
        metadata={
            "distribution": "uniform_unit_sphere_with_tombstones",
            "normalized": True,
            "deleted_fraction": round(sum(1 for row in rows if row.deleted) / len(rows), 4),
        },
    )


def generate_hotpot_skewed(config: dict, rng: random.Random) -> Corpus:
    dimension = config["dimension"]
    row_count = max(config["rows"], 1024)
    heavy_count = max(1, int(row_count * 0.72))
    cluster_count = 4
    centers = []
    rows = []
    queries = []

    for _ in range(cluster_count):
        center = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        centers.append(rounded(normalize(center)))

    for row_id in range(1, row_count + 1):
        if row_id <= heavy_count:
            center = centers[0]
            noise = 0.025
        else:
            center = centers[1 + ((row_id - heavy_count - 1) % (cluster_count - 1))]
            noise = 0.08

        values = [
            center[axis] + rng.uniform(-noise, noise)
            for axis in range(dimension)
        ]
        rows.append(Row(row_id, rounded(normalize(values))))

    for query_index in range(config["queries"]):
        if query_index % 3 == 0:
            center = centers[0]
            noise = 0.02
        else:
            center = centers[1 + (query_index % (cluster_count - 1))]
            noise = 0.04

        values = [
            center[axis] + rng.uniform(-noise, noise)
            for axis in range(dimension)
        ]
        queries.append(rounded(normalize(values)))

    return Corpus(
        name="hotpot_skewed",
        dimension=dimension,
        rows=rows,
        queries=queries,
        metadata={
            "distribution": "hotpot_skewed_ivf",
            "normalized": True,
            "heavy_cluster_fraction": round(heavy_count / max(1, row_count), 4),
            "cluster_count": cluster_count,
        },
    )


def generate_hotpot_overlap(config: dict, rng: random.Random) -> Corpus:
    dimension = config["dimension"]
    row_count = max(config["rows"], 2048)
    heavy_count = max(1, int(row_count * 0.68))
    cluster_count = 6
    rows = []
    queries = []

    anchor = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
    anchor = list(normalize(anchor))
    centers = []

    for cluster_index in range(cluster_count):
        offset = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        offset = normalize(offset)
        blend = 0.12 + (cluster_index * 0.015)
        center = [
            ((1.0 - blend) * anchor[axis]) + (blend * offset[axis])
            for axis in range(dimension)
        ]
        centers.append(rounded(normalize(center)))

    for row_id in range(1, row_count + 1):
        if row_id <= heavy_count:
            center = centers[0]
            noise = 0.16
        else:
            center = centers[1 + ((row_id - heavy_count - 1) % (cluster_count - 1))]
            noise = 0.18

        values = [
            center[axis] + rng.uniform(-noise, noise)
            for axis in range(dimension)
        ]
        rows.append(Row(row_id, rounded(normalize(values))))

    boundary_query_count = max(1, config["queries"] // 2)
    for query_index in range(config["queries"]):
        tail_index = 1 + (query_index % (cluster_count - 1))
        if query_index < boundary_query_count:
            blend = 0.5 + (0.04 if query_index % 2 == 0 else -0.04)
            center = [
                ((1.0 - blend) * centers[0][axis]) + (blend * centers[tail_index][axis])
                for axis in range(dimension)
            ]
            noise = 0.12
        else:
            center = list(centers[tail_index])
            noise = 0.14

        values = [
            center[axis] + rng.uniform(-noise, noise)
            for axis in range(dimension)
        ]
        queries.append(rounded(normalize(values)))

    return Corpus(
        name="hotpot_overlap",
        dimension=dimension,
        rows=rows,
        queries=queries,
        metadata={
            "distribution": "hotpot_overlap_ivf",
            "normalized": True,
            "heavy_cluster_fraction": round(heavy_count / max(1, row_count), 4),
            "cluster_count": cluster_count,
            "query_profile": "boundary_blend",
            "overlap_noise": 0.18,
            "boundary_query_fraction": round(boundary_query_count / max(1, config["queries"]), 4),
        },
    )


def build_corpus(corpus_name: str, config: dict, seed: int) -> Corpus:
    rng = random.Random(seed)
    if corpus_name == "normalized_dense":
        return generate_normalized_dense(config, rng)
    if corpus_name == "non_normalized_varied_norms":
        return generate_non_normalized_varied_norms(config, rng)
    if corpus_name == "clustered":
        return generate_clustered(config, rng)
    if corpus_name == "mixed_live_dead":
        return generate_mixed_live_dead(config, rng)
    if corpus_name == "hotpot_skewed":
        return generate_hotpot_skewed(config, rng)
    if corpus_name == "hotpot_overlap":
        return generate_hotpot_overlap(config, rng)
    raise AssertionError(f"unexpected corpus {corpus_name}")


def active_rows(corpus: Corpus) -> list[Row]:
    return [row for row in corpus.rows if not row.deleted]


def exact_top_ids(corpus: Corpus, query: tuple[float, ...], limit: int, metric: str) -> list[int]:
    ranked = sorted(
        active_rows(corpus),
        key=lambda row: (exact_distance(metric, row.values, query), row.row_id),
    )
    return [row.row_id for row in ranked[:limit]]


def exact_bitmap_ids(corpus: Corpus, query: tuple[float, ...], category: int, threshold: float) -> list[int]:
    return [
        row.row_id
        for row in sorted(active_rows(corpus), key=lambda row: row.row_id)
        if (row.row_id % 2) == category and cosine_distance(row.values, query) <= threshold
    ]


def ground_truth_for_corpus(corpus: Corpus, metric: str) -> dict:
    return {
        "kind": "exact",
        "metric": metric,
        "top_k": list(TOP_K_VALUES),
        "query_count": len(corpus.queries),
    }


def run_psql(base_cmd: list[str], sql: str) -> None:
    subprocess.run(
        base_cmd + ["-c", sql],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def query_psql(base_cmd: list[str], sql: str) -> str:
    result = subprocess.run(
        base_cmd + ["-qAt", "-c", sql],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def query_psql_commands(base_cmd: list[str], *commands: str) -> list[str]:
    cmd = list(base_cmd) + ["-qAt"]

    for sql in commands:
        cmd.extend(["-c", sql])

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def current_wal_lsn(base_cmd: list[str]) -> str:
    return query_psql(base_cmd, "SELECT pg_current_wal_insert_lsn();")


def wal_bytes_since(base_cmd: list[str], start_lsn: str) -> int:
    return int(
        query_psql(
            base_cmd,
            f"SELECT pg_wal_lsn_diff(pg_current_wal_insert_lsn(), '{start_lsn}'::pg_lsn)::bigint;",
        )
    )


def method_opclass(method: str, benchmark_metric: str) -> str:
    if method.startswith("turboquant_"):
        if method == "turboquant_bitmap":
            if benchmark_metric != "cosine":
                raise SystemExit("turboquant_bitmap only supports cosine benchmarks")
            return "tq_cosine_ops"
        if benchmark_metric == "cosine":
            return "tq_cosine_ops"
        if benchmark_metric == "ip":
            return "tq_ip_ops"
    if method.startswith("pgvector_"):
        if benchmark_metric == "cosine":
            return "vector_cosine_ops"
        if benchmark_metric == "ip":
            return "vector_ip_ops"
    raise AssertionError(f"unexpected method/metric combination: {method}/{benchmark_metric}")


def method_spec(
    method: str,
    corpus: Corpus,
    benchmark_metric: str = "cosine",
    requested_rerank_candidate_limit: Optional[int] = None,
    turboquant_probes: Optional[int] = None,
    turboquant_oversample_factor: Optional[int] = None,
    turboquant_max_visited_codes: Optional[int] = None,
    turboquant_max_visited_pages: Optional[int] = None,
    turboquant_shadow_decode_diagnostics: bool = False,
    turboquant_force_decode_score_diagnostics: bool = False,
    turboquant_decode_rescore_factor: int = 1,
    turboquant_decode_rescore_extra_candidates: Optional[int] = None,
    turboquant_qjl_sketch_dim: Optional[str] = None,
) -> dict:
    hotpot_overlap = corpus.name == "hotpot_overlap"
    qjl_sketch_dimension = resolve_qjl_sketch_dimension(turboquant_qjl_sketch_dim, corpus.dimension)
    if hotpot_overlap:
        list_count = min(64, max(16, len(active_rows(corpus)) // 16))
    else:
        list_count = min(16, max(4, len(active_rows(corpus)) // 32))

    default_turboquant_probes = min(list_count, 8) if hotpot_overlap else min(list_count, 4)
    turboquant_probe_value = (
        turboquant_probes if turboquant_probes is not None else default_turboquant_probes
    )
    turboquant_oversample_value = (
        turboquant_oversample_factor if turboquant_oversample_factor is not None else 4
    )
    turboquant_max_codes_value = (
        turboquant_max_visited_codes if turboquant_max_visited_codes is not None else (4096 if hotpot_overlap else 0)
    )
    turboquant_max_pages_value = (
        turboquant_max_visited_pages if turboquant_max_visited_pages is not None else 0
    )
    turboquant_decode_rescore_extra_value = resolve_decode_rescore_extra_candidates(
        TOP_K_VALUES[-1],
        requested_rerank_candidate_limit,
        turboquant_decode_rescore_factor,
        turboquant_decode_rescore_extra_candidates,
    )
    ivfflat_probe_value = min(list_count, 8) if hotpot_overlap else min(list_count, 4)
    hnsw_ef_search_value = 80 if hotpot_overlap else 40
    if method == "turboquant_flat":
        return {
            "index_method": "turboquant",
            "opclass": method_opclass(method, benchmark_metric),
            "with": {
                "bits": 4,
                "lists": 0,
                "lanes": "auto",
                "transform": "hadamard",
                "normalized": "true",
                "qjl_sketch_dim": qjl_sketch_dimension,
            },
            "query_setup": [
                "SET LOCAL enable_seqscan = off",
                "SET LOCAL enable_bitmapscan = off",
                f"SET LOCAL turboquant.probes = {turboquant_probe_value}",
                f"SET LOCAL turboquant.oversample_factor = {turboquant_oversample_value}",
                f"SET LOCAL turboquant.max_visited_codes = {turboquant_max_codes_value}",
                f"SET LOCAL turboquant.max_visited_pages = {turboquant_max_pages_value}",
                (
                    "SET LOCAL turboquant.shadow_decode_diagnostics = on"
                    if turboquant_shadow_decode_diagnostics
                    else "SET LOCAL turboquant.shadow_decode_diagnostics = off"
                ),
                (
                    "SET LOCAL turboquant.force_decode_score_diagnostics = on"
                    if turboquant_force_decode_score_diagnostics
                    else "SET LOCAL turboquant.force_decode_score_diagnostics = off"
                ),
                f"SET LOCAL turboquant.decode_rescore_factor = {turboquant_decode_rescore_factor}",
                f"SET LOCAL turboquant.decode_rescore_extra_candidates = {turboquant_decode_rescore_extra_value}",
            ],
            "candidate_slots_bound": turboquant_probe_value * turboquant_oversample_value,
            "query_knobs": {
                "turboquant.probes": turboquant_probe_value,
                "turboquant.oversample_factor": turboquant_oversample_value,
                "turboquant.max_visited_codes": turboquant_max_codes_value,
                "turboquant.max_visited_pages": turboquant_max_pages_value,
                "turboquant.shadow_decode_diagnostics": turboquant_shadow_decode_diagnostics,
                "turboquant.force_decode_score_diagnostics": turboquant_force_decode_score_diagnostics,
                "turboquant.decode_rescore_factor": turboquant_decode_rescore_factor,
                "turboquant.decode_rescore_extra_candidates": turboquant_decode_rescore_extra_value,
            },
        }
    if method == "turboquant_ivf":
        return {
            "index_method": "turboquant",
            "opclass": method_opclass(method, benchmark_metric),
            "with": {
                "bits": 4,
                "lists": list_count,
                "lanes": "auto",
                "transform": "hadamard",
                "normalized": "true",
                "qjl_sketch_dim": qjl_sketch_dimension,
            },
            "query_setup": [
                "SET LOCAL enable_seqscan = off",
                "SET LOCAL enable_bitmapscan = off",
                f"SET LOCAL turboquant.probes = {turboquant_probe_value}",
                f"SET LOCAL turboquant.oversample_factor = {turboquant_oversample_value}",
                f"SET LOCAL turboquant.max_visited_codes = {turboquant_max_codes_value}",
                f"SET LOCAL turboquant.max_visited_pages = {turboquant_max_pages_value}",
                (
                    "SET LOCAL turboquant.shadow_decode_diagnostics = on"
                    if turboquant_shadow_decode_diagnostics
                    else "SET LOCAL turboquant.shadow_decode_diagnostics = off"
                ),
                (
                    "SET LOCAL turboquant.force_decode_score_diagnostics = on"
                    if turboquant_force_decode_score_diagnostics
                    else "SET LOCAL turboquant.force_decode_score_diagnostics = off"
                ),
                f"SET LOCAL turboquant.decode_rescore_factor = {turboquant_decode_rescore_factor}",
                f"SET LOCAL turboquant.decode_rescore_extra_candidates = {turboquant_decode_rescore_extra_value}",
            ],
            "candidate_slots_bound": turboquant_probe_value * turboquant_oversample_value,
            "query_knobs": {
                "turboquant.probes": turboquant_probe_value,
                "turboquant.oversample_factor": turboquant_oversample_value,
                "turboquant.max_visited_codes": turboquant_max_codes_value,
                "turboquant.max_visited_pages": turboquant_max_pages_value,
                "turboquant.shadow_decode_diagnostics": turboquant_shadow_decode_diagnostics,
                "turboquant.force_decode_score_diagnostics": turboquant_force_decode_score_diagnostics,
                "turboquant.decode_rescore_factor": turboquant_decode_rescore_factor,
                "turboquant.decode_rescore_extra_candidates": turboquant_decode_rescore_extra_value,
            },
        }
    if method == "turboquant_bitmap":
        return {
            "index_method": "turboquant",
            "opclass": method_opclass(method, benchmark_metric),
            "with": {
                "bits": 4,
                "lists": 0,
                "lanes": "auto",
                "transform": "hadamard",
                "normalized": "true",
                "qjl_sketch_dim": qjl_sketch_dimension,
            },
            "query_setup": [
                "SET LOCAL enable_seqscan = off",
                "SET LOCAL enable_indexscan = off",
                "SET LOCAL enable_tidscan = off",
            ],
            "candidate_slots_bound": 0,
            "query_knobs": {},
            "query_mode": "bitmap_filter",
            "bitmap_threshold": 0.20,
            "auxiliary_indexes": [
                "CREATE INDEX {table_name}_category_idx ON {table_name} (category);",
            ],
        }
    if method == "pgvector_ivfflat":
        return {
            "index_method": "ivfflat",
            "opclass": method_opclass(method, benchmark_metric),
            "with": {
                "lists": list_count,
            },
            "query_setup": [
                "SET LOCAL enable_seqscan = off",
                "SET LOCAL enable_bitmapscan = off",
                f"SET LOCAL ivfflat.probes = {ivfflat_probe_value}",
            ],
            "candidate_slots_bound": 0,
            "query_knobs": {
                "ivfflat.probes": ivfflat_probe_value,
            },
        }
    if method == "pgvector_hnsw":
        return {
            "index_method": "hnsw",
            "opclass": method_opclass(method, benchmark_metric),
            "with": {
                "m": 8,
                "ef_construction": 32,
            },
            "query_setup": [
                "SET LOCAL enable_seqscan = off",
                "SET LOCAL enable_bitmapscan = off",
                f"SET LOCAL hnsw.ef_search = {hnsw_ef_search_value}",
            ],
            "candidate_slots_bound": 0,
            "query_knobs": {
                "hnsw.ef_search": hnsw_ef_search_value,
            },
        }
    raise AssertionError(f"unexpected method {method}")


def render_with_clause(options: dict) -> str:
    rendered = []
    for key, value in options.items():
        if isinstance(value, str) and not value.isdigit() and value not in ("true", "false"):
            rendered.append(f"{key} = '{value}'")
        else:
            rendered.append(f"{key} = {value}")
    return ", ".join(rendered)


def chunked_insert_value_blocks(rows: list[Row], max_rows_per_insert: int = 128) -> list[str]:
    blocks = []
    for offset in range(0, len(rows), max_rows_per_insert):
        chunk = rows[offset : offset + max_rows_per_insert]
        blocks.append(
            ",\n".join(
                f"({row.row_id}, {row.row_id % 2}, '{vector_literal(row.values)}')"
                for row in chunk
            )
        )
    return blocks


def load_corpus(base_cmd: list[str], table_name: str, corpus: Corpus) -> None:
    sql = f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_turboquant;
    DROP TABLE IF EXISTS {table_name};
    CREATE TABLE {table_name} (
        id int4 PRIMARY KEY,
        category int4 NOT NULL,
        embedding vector({corpus.dimension})
    );
    """
    run_psql(base_cmd, sql)
    for insert_values in chunked_insert_value_blocks(corpus.rows):
        run_psql(
            base_cmd,
            f"INSERT INTO {table_name} (id, category, embedding) VALUES\n{insert_values};",
        )


def apply_mixed_live_dead_workload(base_cmd: list[str], table_name: str, corpus: Corpus) -> None:
    deleted_ids = [str(row.row_id) for row in corpus.rows if row.deleted]
    if not deleted_ids:
        return
    run_psql(base_cmd, f"DELETE FROM {table_name} WHERE id IN ({', '.join(deleted_ids)});")
    run_psql(base_cmd, f"VACUUM {table_name};")


def build_index(
    base_cmd: list[str],
    table_name: str,
    index_name: str,
    corpus: Corpus,
    method: str,
    benchmark_metric: str,
    requested_rerank_candidate_limit: Optional[int] = None,
    turboquant_probes: Optional[int] = None,
    turboquant_oversample_factor: Optional[int] = None,
    turboquant_max_visited_codes: Optional[int] = None,
    turboquant_max_visited_pages: Optional[int] = None,
    turboquant_shadow_decode_diagnostics: bool = False,
    turboquant_force_decode_score_diagnostics: bool = False,
    turboquant_decode_rescore_factor: int = 1,
    turboquant_decode_rescore_extra_candidates: Optional[int] = None,
    turboquant_qjl_sketch_dim: Optional[str] = None,
) -> tuple[float, int, int, dict]:
    spec = method_spec(
        method=method,
        corpus=corpus,
        benchmark_metric=benchmark_metric,
        requested_rerank_candidate_limit=requested_rerank_candidate_limit,
        turboquant_probes=turboquant_probes,
        turboquant_oversample_factor=turboquant_oversample_factor,
        turboquant_max_visited_codes=turboquant_max_visited_codes,
        turboquant_max_visited_pages=turboquant_max_visited_pages,
        turboquant_shadow_decode_diagnostics=turboquant_shadow_decode_diagnostics,
        turboquant_force_decode_score_diagnostics=turboquant_force_decode_score_diagnostics,
        turboquant_decode_rescore_factor=turboquant_decode_rescore_factor,
        turboquant_decode_rescore_extra_candidates=turboquant_decode_rescore_extra_candidates,
        turboquant_qjl_sketch_dim=turboquant_qjl_sketch_dim,
    )
    build_sql = f"""
    DROP INDEX IF EXISTS {index_name};
    CREATE INDEX {index_name}
    ON {table_name}
    USING {spec['index_method']} (embedding {spec['opclass']})
    WITH ({render_with_clause(spec['with'])});
    """
    auxiliary_indexes = spec.get("auxiliary_indexes", [])
    if auxiliary_indexes:
        build_sql += "\n" + "\n".join(
            index_sql.format(table_name=table_name)
            for index_sql in auxiliary_indexes
        )
    wal_start = current_wal_lsn(base_cmd)
    started = time.perf_counter()
    run_psql(base_cmd, build_sql)
    build_seconds = time.perf_counter() - started
    build_wal_bytes = wal_bytes_since(base_cmd, wal_start)
    index_size_bytes = int(query_psql(base_cmd, f"SELECT pg_relation_size('{index_name}'::regclass);"))
    return build_seconds, index_size_bytes, build_wal_bytes, spec


def turboquant_sealed_baseline_bytes(row_count: int, block_size: int) -> int:
    return row_count * block_size


def measure_insert_wal(
    base_cmd: list[str],
    table_name: str,
    corpus: Corpus,
) -> tuple[int, int]:
    insert_count = min(64, max(8, len(active_rows(corpus)) // 8))
    max_row_id = max(row.row_id for row in corpus.rows)
    values = []
    for offset in range(insert_count):
        query = corpus.queries[offset % len(corpus.queries)]
        row_id = max_row_id + offset + 1
        values.append(f"({row_id}, {row_id % 2}, '{vector_literal(query)}')")

    wal_start = current_wal_lsn(base_cmd)
    run_psql(
        base_cmd,
        f"INSERT INTO {table_name} (id, category, embedding) VALUES {', '.join(values)};",
    )
    return wal_bytes_since(base_cmd, wal_start), insert_count


def measure_concurrent_insert_rows_per_second(
    base_cmd: list[str],
    table_name: str,
    corpus: Corpus,
) -> tuple[float, int, int]:
    insert_count = min(128, max(16, len(active_rows(corpus)) // 4))
    worker_count = min(4, insert_count)
    max_row_id = int(query_psql(base_cmd, f"SELECT coalesce(max(id), 0) FROM {table_name};"))
    worker_values: list[list[str]] = [[] for _ in range(worker_count)]

    for offset in range(insert_count):
        query = corpus.queries[offset % len(corpus.queries)]
        worker_index = offset % worker_count
        worker_values[worker_index].append(
            f"({max_row_id + offset + 1}, {(max_row_id + offset + 1) % 2}, '{vector_literal(query)}')"
        )

    commands = [
        base_cmd + ["-c", f"INSERT INTO {table_name} (id, category, embedding) VALUES {', '.join(values)};"]
        for values in worker_values
        if values
    ]

    started = time.perf_counter()
    processes = [
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for command in commands
    ]

    for process in processes:
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args, stderr=stderr)

    elapsed = max(time.perf_counter() - started, 1e-9)
    return round(insert_count / elapsed, 6), insert_count, len(commands)


def measure_maintenance_wal(
    base_cmd: list[str],
    table_name: str,
    corpus: Corpus,
) -> tuple[int, int]:
    live_row_ids = [row.row_id for row in active_rows(corpus)]
    delete_ids = live_row_ids[::7]
    if not delete_ids:
        return 0, 0

    wal_start = current_wal_lsn(base_cmd)
    run_psql(
        base_cmd,
        f"DELETE FROM {table_name} WHERE id IN ({', '.join(str(row_id) for row_id in delete_ids)});",
    )
    run_psql(base_cmd, f"VACUUM {table_name};")
    return wal_bytes_since(base_cmd, wal_start), len(delete_ids)


def fetch_index_metadata(
    base_cmd: list[str],
    index_name: str,
    spec: dict,
    corpus: Corpus,
    benchmark_metric: str,
) -> dict:
    if spec["index_method"] != "turboquant":
        return synthetic_index_metadata("generic", spec, corpus, benchmark_metric)

    raw = query_psql(base_cmd, f"SELECT tq_index_metadata('{index_name}'::regclass)::text;")
    metadata = json.loads(raw)
    metadata.setdefault("capabilities", capability_metadata(spec))
    return metadata


def fetch_simd_metadata(base_cmd: list[str], spec: dict) -> dict:
    if spec["index_method"] != "turboquant":
        return synthetic_simd_metadata()

    raw = query_psql(base_cmd, "SELECT tq_runtime_simd_features()::text;")
    metadata = json.loads(raw)
    metadata.setdefault("selected_kernel", metadata.get("preferred_kernel", "scalar"))
    return metadata


def resolve_scan_contract_flags(
    method: str,
    benchmark_metric: str,
    corpus: Corpus,
    query_mode: str,
) -> tuple[bool, bool]:
    faithful_fast_path = (
        method in {"turboquant_flat", "turboquant_ivf"}
        and query_mode == "ordered_rerank"
        and benchmark_metric in {"cosine", "ip"}
        and bool(corpus.metadata.get("normalized"))
    )
    compatibility_fallback = (
        method.startswith("turboquant_")
        and query_mode == "ordered_rerank"
        and not faithful_fast_path
    )
    return faithful_fast_path, compatibility_fallback


def default_scan_stats(method: str) -> dict:
    if method == "turboquant_flat":
        mode = "flat"
        score_mode = "code_domain"
        score_kernel = "scalar"
        page_bound_mode = "disabled"
        scan_orchestration = "flat_streaming"
    elif method == "turboquant_ivf":
        mode = "ivf"
        score_mode = "code_domain"
        score_kernel = "scalar"
        page_bound_mode = "safe_summary_pruning"
        scan_orchestration = "ivf_bounded_pages"
    elif method == "turboquant_bitmap":
        mode = "bitmap"
        score_mode = "bitmap_filter"
        score_kernel = "none"
        page_bound_mode = "disabled"
        scan_orchestration = "bitmap_filter"
    else:
        mode = "none"
        score_mode = "none"
        score_kernel = "none"
        page_bound_mode = "disabled"
        scan_orchestration = "none"

    return {
        "mode": mode,
        "score_mode": score_mode,
        "score_kernel": score_kernel,
        "page_bound_mode": page_bound_mode,
        "scan_orchestration": scan_orchestration,
        "safe_pruning_enabled": method == "turboquant_ivf",
        "near_exhaustive_crossover": False,
        "configured_probe_count": 0,
        "nominal_probe_count": 0,
        "effective_probe_count": 0,
        "max_visited_codes": 0,
        "max_visited_pages": 0,
        "selected_list_count": 0,
        "selected_live_count": 0,
        "selected_page_count": 0,
        "visited_page_count": 0,
        "visited_code_count": 0,
        "retained_candidate_count": 0,
        "candidate_heap_capacity": 0,
        "candidate_heap_count": 0,
        "candidate_heap_insert_count": 0,
        "candidate_heap_replace_count": 0,
        "candidate_heap_reject_count": 0,
        "local_candidate_heap_insert_count": 0,
        "local_candidate_heap_replace_count": 0,
        "local_candidate_heap_reject_count": 0,
        "local_candidate_merge_count": 0,
        "shadow_decoded_vector_count": 0,
        "shadow_decode_candidate_count": 0,
        "shadow_decode_overlap_count": 0,
        "shadow_decode_primary_only_count": 0,
        "shadow_decode_only_count": 0,
        "decoded_vector_count": 0,
        "page_prune_count": 0,
        "early_stop_count": 0,
        "scratch_allocations": 0,
        "decoded_buffer_reuses": 0,
        "code_view_uses": 0,
        "code_copy_uses": 0,
        "faithful_fast_path": False,
        "compatibility_fallback": False,
    }


def aggregate_scan_stats(scan_stats: list[dict], fallback_method: str) -> dict:
    if not scan_stats:
        return default_scan_stats(fallback_method)

    defaults = default_scan_stats(fallback_method)
    numeric_keys = (
        "configured_probe_count",
        "nominal_probe_count",
        "effective_probe_count",
        "max_visited_codes",
        "max_visited_pages",
        "selected_list_count",
        "selected_live_count",
        "selected_page_count",
        "visited_page_count",
        "visited_code_count",
        "retained_candidate_count",
        "candidate_heap_capacity",
        "candidate_heap_count",
        "candidate_heap_insert_count",
        "candidate_heap_replace_count",
        "candidate_heap_reject_count",
        "local_candidate_heap_insert_count",
        "local_candidate_heap_replace_count",
        "local_candidate_heap_reject_count",
        "local_candidate_merge_count",
        "shadow_decoded_vector_count",
        "shadow_decode_candidate_count",
        "shadow_decode_overlap_count",
        "shadow_decode_primary_only_count",
        "shadow_decode_only_count",
        "decoded_vector_count",
        "page_prune_count",
        "early_stop_count",
        "scratch_allocations",
        "decoded_buffer_reuses",
        "code_view_uses",
        "code_copy_uses",
    )
    aggregated = {
        "mode": scan_stats[0].get("mode", defaults["mode"]),
        "score_mode": scan_stats[0].get("score_mode", defaults["score_mode"]),
        "score_kernel": scan_stats[0].get("score_kernel", defaults["score_kernel"]),
        "page_bound_mode": scan_stats[0].get("page_bound_mode", defaults["page_bound_mode"]),
        "scan_orchestration": scan_stats[0].get("scan_orchestration", defaults["scan_orchestration"]),
        "safe_pruning_enabled": any(
            bool(item.get("safe_pruning_enabled", defaults["safe_pruning_enabled"]))
            for item in scan_stats
        ),
        "near_exhaustive_crossover": any(
            bool(item.get("near_exhaustive_crossover", defaults["near_exhaustive_crossover"]))
            for item in scan_stats
        ),
        "faithful_fast_path": any(
            bool(item.get("faithful_fast_path", defaults["faithful_fast_path"]))
            for item in scan_stats
        ),
        "compatibility_fallback": any(
            bool(item.get("compatibility_fallback", defaults["compatibility_fallback"]))
            for item in scan_stats
        ),
    }
    if any(
        item.get("page_bound_mode", defaults["page_bound_mode"]) != aggregated["page_bound_mode"]
        for item in scan_stats[1:]
    ):
        aggregated["page_bound_mode"] = "mixed"
    if any(
        item.get("scan_orchestration", defaults["scan_orchestration"]) != aggregated["scan_orchestration"]
        for item in scan_stats[1:]
    ):
        aggregated["scan_orchestration"] = "mixed"
    for key in numeric_keys:
        values = [float(item.get(key, 0)) for item in scan_stats]
        aggregated[key] = round(sum(values) / max(1, len(values)), 6)
    return aggregated


def default_candidate_retention() -> dict:
    return {
        "avg_candidate_count": 0.0,
        "avg_exact_top_10_retention": 0.0,
        "avg_exact_top_100_retention": 0.0,
        "avg_exact_top_100_miss_count": 0.0,
        "worst_exact_top_100_retention": 0.0,
        "avg_shadow_candidate_count": 0.0,
        "avg_shadow_exact_top_10_retention": 0.0,
        "avg_shadow_exact_top_100_retention": 0.0,
        "avg_shadow_exact_top_100_miss_count": 0.0,
        "worst_shadow_exact_top_100_retention": 0.0,
    }


def default_estimator_quality() -> dict:
    return {
        "sample_count": 0,
        "distance_error_bias": 0.0,
        "distance_error_variance": 0.0,
        "distance_error_mae": 0.0,
        "avg_abs_rank_shift": 0.0,
        "max_abs_rank_shift": 0.0,
    }


def aggregate_estimator_quality(diagnostics: list[dict]) -> dict:
    if not diagnostics:
        return default_estimator_quality()

    distance_errors = [
        float(item["approximate_distance"]) - float(item["exact_distance"])
        for item in diagnostics
    ]
    rank_shifts = [
        abs(int(item["approximate_rank"]) - int(item["exact_rank"]))
        for item in diagnostics
    ]
    bias = sum(distance_errors) / len(distance_errors)
    variance = sum((value - bias) ** 2 for value in distance_errors) / len(distance_errors)
    return {
        "sample_count": len(diagnostics),
        "distance_error_bias": round(bias, 6),
        "distance_error_variance": round(variance, 6),
        "distance_error_mae": round(
            sum(abs(value) for value in distance_errors) / len(distance_errors),
            6,
        ),
        "avg_abs_rank_shift": round(sum(rank_shifts) / len(rank_shifts), 6),
        "max_abs_rank_shift": max(rank_shifts),
    }


def candidate_retention_for_query(exact_ids: list[int], approx_candidate_ids: list[int]) -> dict:
    approx_set = set(approx_candidate_ids)
    exact_top_10 = exact_ids[:10]
    exact_top_100 = exact_ids[:100]
    retained_top_10 = len(set(exact_top_10) & approx_set) / max(1, len(exact_top_10))
    retained_top_100 = len(set(exact_top_100) & approx_set) / max(1, len(exact_top_100))
    miss_count = len(set(exact_top_100) - approx_set)
    return {
        "candidate_count": float(len(approx_candidate_ids)),
        "exact_top_10_retention": round(retained_top_10, 6),
        "exact_top_100_retention": round(retained_top_100, 6),
        "exact_top_100_miss_count": float(miss_count),
    }


def aggregate_candidate_retention(retention_stats: list[dict]) -> dict:
    if not retention_stats:
        return default_candidate_retention()

    return {
        "avg_candidate_count": round(
            sum(float(item.get("candidate_count", 0.0)) for item in retention_stats) / len(retention_stats),
            6,
        ),
        "avg_exact_top_10_retention": round(
            sum(float(item.get("exact_top_10_retention", 0.0)) for item in retention_stats) / len(retention_stats),
            6,
        ),
        "avg_exact_top_100_retention": round(
            sum(float(item.get("exact_top_100_retention", 0.0)) for item in retention_stats) / len(retention_stats),
            6,
        ),
        "avg_exact_top_100_miss_count": round(
            sum(float(item.get("exact_top_100_miss_count", 0.0)) for item in retention_stats) / len(retention_stats),
            6,
        ),
        "worst_exact_top_100_retention": round(
            min(float(item.get("exact_top_100_retention", 0.0)) for item in retention_stats),
            6,
        ),
        "avg_shadow_candidate_count": round(
            sum(float(item.get("shadow_candidate_count", 0.0)) for item in retention_stats) / len(retention_stats),
            6,
        ),
        "avg_shadow_exact_top_10_retention": round(
            sum(float(item.get("shadow_exact_top_10_retention", 0.0)) for item in retention_stats) / len(retention_stats),
            6,
        ),
        "avg_shadow_exact_top_100_retention": round(
            sum(float(item.get("shadow_exact_top_100_retention", 0.0)) for item in retention_stats) / len(retention_stats),
            6,
        ),
        "avg_shadow_exact_top_100_miss_count": round(
            sum(float(item.get("shadow_exact_top_100_miss_count", 0.0)) for item in retention_stats) / len(retention_stats),
            6,
        ),
        "worst_shadow_exact_top_100_retention": round(
            min(float(item.get("shadow_exact_top_100_retention", 0.0)) for item in retention_stats),
            6,
        ),
    }


def synthetic_skew_probe_regression() -> dict:
    return {
        "closest_centroid": {
            "recall_at_10": 0.3,
            "visited_code_count": 120,
            "visited_page_count": 16,
        },
        "cost_aware": {
            "recall_at_10": 1.0,
            "visited_code_count": 72,
            "visited_page_count": 9,
        },
    }


def scenario_benchmark_metadata(index_metadata: dict) -> dict:
    list_distribution = index_metadata.get("list_distribution", {})
    list_count = int(index_metadata.get("list_count", 0))
    live_count = int(
        index_metadata.get(
            "live_count",
            index_metadata.get("heap_live_rows_estimate", index_metadata.get("heap_live_rows", 0))
            or 0,
        )
    )
    avg_list_size = float(
        list_distribution.get(
            "avg_live_count",
            round(live_count / list_count, 2) if list_count > 0 else 0.0,
        )
    )
    max_list_size = int(list_distribution.get("max_live_count", 0))
    return {
        "list_balance": {
            "list_count": list_count,
            "live_count": live_count,
            "avg_list_size": round(avg_list_size, 2),
            "max_list_size": max_list_size,
        }
    }


def query_turboquant_ordered_scan_stats(
    base_cmd: list[str],
    table_name: str,
    query_vector: tuple[float, ...],
    limit: int,
    query_setup: list[str],
    method: str,
    benchmark_metric: str = "cosine",
) -> dict:
    order_operator = metric_order_operator(benchmark_metric)
    query_sql = ";\n".join(
        query_setup
        + [
            (
                "SELECT coalesce(json_agg(id ORDER BY id), '[]'::json)::text "
                f"FROM (SELECT id FROM {table_name} "
                f"ORDER BY embedding {order_operator} '{vector_literal(query_vector)}'::vector LIMIT {limit}) ranked"
            )
        ]
    )
    rows = query_psql_commands(
        base_cmd,
        query_sql + ";\nSELECT tq_last_scan_stats()::text;",
    )
    return json.loads(rows[1]) if len(rows) > 1 else default_scan_stats(method)


def turboquant_rerank_ids_sql(
    table_name: str,
    query_vector: tuple[float, ...],
    limit: int,
    query_setup: list[str],
    requested_candidate_limit: Optional[int],
    benchmark_metric: str = "cosine",
) -> str:
    requested_limit = rerank_candidate_limit(limit, requested_candidate_limit)
    return ";\n".join(
        query_setup
        + [
            (
                "SELECT coalesce(json_agg(candidate_id::int ORDER BY exact_rank), '[]'::json)::text "
                "FROM tq_rerank_candidates("
                f"'{table_name}'::regclass, 'id', 'embedding', "
                f"'{vector_literal(query_vector)}'::vector, '{benchmark_metric}', "
                f"{requested_limit}, {limit})"
            )
        ]
    )


def turboquant_single_batch_rerank_ids_sql(
    table_name: str,
    query_vector: tuple[float, ...],
    limit: int,
    query_setup: list[str],
    requested_candidate_limit: Optional[int],
    decode_rescore_extra_candidates: int = 0,
    benchmark_metric: str = "cosine",
) -> str:
    requested_limit = rerank_candidate_limit(limit, requested_candidate_limit)
    query_literal = vector_literal(query_vector)
    order_operator = metric_order_operator(benchmark_metric)
    return ";\n".join(
        ["BEGIN"]
        + query_setup
        + [
            (
                "WITH approx_scan AS MATERIALIZED ("
                f"SELECT id, embedding, (embedding {order_operator} '{query_literal}'::vector) AS approximate_distance "
                f"FROM {table_name} "
                f"ORDER BY embedding {order_operator} '{query_literal}'::vector "
                f"LIMIT tq_effective_rerank_candidate_limit({requested_limit}, {limit})"
                "), approx AS MATERIALIZED ("
                "SELECT "
                "id::text AS candidate_id, "
                "id AS candidate_key, "
                "embedding AS candidate_embedding, "
                "row_number() OVER (ORDER BY approximate_distance, id)::integer AS approximate_rank, "
                "round(approximate_distance::numeric, 6)::double precision AS approximate_distance "
                "FROM approx_scan"
                "), reranked AS ("
                "SELECT "
                "candidate_id, "
                "candidate_key, "
                "approximate_rank, "
                "approximate_distance, "
                f"row_number() OVER (ORDER BY candidate_embedding {order_operator} "
                f"'{query_literal}'::vector, candidate_key)::integer AS exact_rank, "
                f"round((candidate_embedding {order_operator} '{query_literal}'::vector)::numeric, 6)::double precision AS exact_distance "
                "FROM approx"
                ") "
                "SELECT json_build_object("
                "'reranked_ids', "
                f"coalesce((SELECT json_agg(candidate_id::int ORDER BY exact_rank) FROM reranked WHERE exact_rank <= {limit}), '[]'::json), "
                "'approx_candidate_ids', "
                "coalesce((SELECT json_agg(candidate_key ORDER BY approximate_rank) FROM approx), '[]'::json), "
                "'estimator_diagnostics', "
                "coalesce((SELECT json_agg(json_build_object("
                "'candidate_id', candidate_id::int, "
                "'approximate_rank', approximate_rank, "
                "'approximate_distance', approximate_distance, "
                "'exact_rank', exact_rank, "
                "'exact_distance', exact_distance"
                ") ORDER BY exact_rank) FROM reranked WHERE exact_rank <= "
                f"{limit}), '[]'::json)"
                ")::text"
            )
        ]
    )


def query_turboquant_ordered_ids_and_scan_stats(
    base_cmd: list[str],
    table_name: str,
    query_vector: tuple[float, ...],
    limit: int,
    query_setup: list[str],
    requested_candidate_limit: Optional[int],
    decode_rescore_extra_candidates: int = 0,
    method: str = "turboquant_ivf",
    benchmark_metric: str = "cosine",
    include_estimator_diagnostics: bool = False,
) -> tuple[list[int], dict, list[int], list[int]] | tuple[list[int], dict, list[int], list[int], list[dict]]:
    shadow_decode_enabled = any(
        statement == "SET LOCAL turboquant.shadow_decode_diagnostics = on"
        for statement in query_setup
    )
    query_sql = turboquant_single_batch_rerank_ids_sql(
        table_name,
        query_vector,
        limit,
        query_setup,
        requested_candidate_limit,
        decode_rescore_extra_candidates,
        benchmark_metric=benchmark_metric,
    )
    rows = query_psql_commands(
        base_cmd,
        (
            query_sql
            + ";\nSELECT tq_last_scan_stats()::text;\n"
            + (
                (
                    "SELECT coalesce(json_agg(source.id ORDER BY shadow.ordinality), '[]'::json)::text "
                    f"FROM unnest(tq_last_shadow_decode_candidate_tids()) WITH ORDINALITY AS shadow(candidate_tid_text, ordinality) "
                    f"JOIN {table_name} source ON source.ctid = shadow.candidate_tid_text::tid;\n"
                )
                if shadow_decode_enabled
                else ""
            )
            + "COMMIT;"
        ),
    )
    payload = json.loads(rows[0]) if rows else {}
    query_ids = list(payload.get("reranked_ids", []))
    approx_candidate_ids = list(payload.get("approx_candidate_ids", []))
    estimator_diagnostics = list(payload.get("estimator_diagnostics", []))
    scan_stats = json.loads(rows[1]) if len(rows) > 1 else default_scan_stats(method)
    shadow_candidate_ids = list(json.loads(rows[2])) if len(rows) > 2 else []
    if include_estimator_diagnostics:
        return query_ids, scan_stats, approx_candidate_ids, shadow_candidate_ids, estimator_diagnostics
    return query_ids, scan_stats, approx_candidate_ids, shadow_candidate_ids


def query_bitmap_ids_and_scan_stats(
    base_cmd: list[str],
    table_name: str,
    query_vector: tuple[float, ...],
    category: int,
    threshold: float,
    query_setup: list[str],
) -> tuple[list[int], dict]:
    query_sql = ";\n".join(
        query_setup
        + [
            (
                "SELECT coalesce(json_agg(id ORDER BY id), '[]'::json)::text "
                f"FROM {table_name} "
                f"WHERE category = {category} "
                f"AND embedding <?=> tq_bitmap_cosine_filter('{vector_literal(query_vector)}'::vector, {threshold})"
            )
        ]
    )
    rows = query_psql_commands(
        base_cmd,
        query_sql + ";\nSELECT tq_last_scan_stats()::text;",
    )
    query_ids = list(json.loads(rows[0])) if rows else []
    scan_stats = json.loads(rows[1]) if len(rows) > 1 else default_scan_stats("turboquant_bitmap")
    return query_ids, scan_stats


def query_top_ids(
    base_cmd: list[str],
    table_name: str,
    query_vector: tuple[float, ...],
    limit: int,
    query_setup: list[str],
    requested_candidate_limit: Optional[int],
    benchmark_metric: str = "cosine",
) -> list[int]:
    query_sql = turboquant_rerank_ids_sql(
        table_name,
        query_vector,
        limit,
        query_setup,
        requested_candidate_limit,
        benchmark_metric=benchmark_metric,
    )
    raw = query_psql(base_cmd, query_sql)
    if not raw:
        return []
    return list(json.loads(raw))


def query_bitmap_ids(
    base_cmd: list[str],
    table_name: str,
    query_vector: tuple[float, ...],
    category: int,
    threshold: float,
    query_setup: list[str],
) -> list[int]:
    query_ids, _ = query_bitmap_ids_and_scan_stats(
        base_cmd,
        table_name,
        query_vector,
        category,
        threshold,
        query_setup,
    )
    return query_ids


def average_recall(exact_results: list[list[int]], approx_results: list[list[int]], k: int) -> float:
    recalls = []
    for exact_ids, approx_ids in zip(exact_results, approx_results):
        exact_top = exact_ids[:k]
        approx_top = approx_ids[:k]
        denominator = max(1, len(exact_top))
        recalls.append(len(set(exact_top) & set(approx_top)) / denominator)
    return round(sum(recalls) / len(recalls), 6)


def comparison_pairs(methods: list[str]) -> list[tuple[str, str, str]]:
    baselines = [method for method in methods if method.startswith("pgvector_")]
    candidates = [method for method in methods if method.startswith("turboquant_")]
    pairs = [(candidate, baseline, "external_baseline") for candidate in candidates for baseline in baselines]
    if "turboquant_flat" in methods and "turboquant_ivf" in methods:
        pairs.append(("turboquant_ivf", "turboquant_flat", "turboquant_internal"))
    return pairs


def scenario_qjl_dimension(scenario: dict) -> Optional[int]:
    residual_sketch = scenario.get("index_metadata", {}).get("residual_sketch")

    if not residual_sketch:
        return None

    projected_dimension = residual_sketch.get("projected_dimension")
    return int(projected_dimension) if projected_dimension is not None else None


def _report_method_row(scenario: dict) -> dict:
    metrics = scenario["metrics"]
    scan_stats = scenario.get("scan_stats", {})
    query_api = scenario.get("query_api", {})
    estimator_quality = scenario.get("estimator_quality", default_estimator_quality())
    return {
        "corpus": scenario["corpus"],
        "method": scenario["method"],
        "benchmark_metric": scenario["benchmark_metric"],
        "qjl_sketch_dimension": scenario_qjl_dimension(scenario),
        "query_mode": scenario.get("query_mode"),
        "query_helper": query_api.get("helper"),
        "recall_at_10": metrics.get("recall_at_10", metrics.get("exact_match_fraction")),
        "p50_ms": metrics.get("p50_ms"),
        "p95_ms": metrics.get("p95_ms"),
        "footprint_bytes": metrics.get("index_size_bytes"),
        "visited_code_count": scan_stats.get("visited_code_count"),
        "visited_page_count": scan_stats.get("visited_page_count"),
        "selected_live_count": scan_stats.get("selected_live_count"),
        "selected_page_count": scan_stats.get("selected_page_count"),
        "score_kernel": scan_stats.get("score_kernel"),
        "distance_error_bias": estimator_quality.get("distance_error_bias"),
        "distance_error_variance": estimator_quality.get("distance_error_variance"),
        "avg_abs_rank_shift": estimator_quality.get("avg_abs_rank_shift"),
    }


def _report_comparison_row(candidate: dict, baseline: dict, comparison_scope: str) -> dict:
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    candidate_scan = candidate.get("scan_stats", {})
    baseline_scan = baseline.get("scan_stats", {})
    candidate_quality = candidate.get("estimator_quality")
    baseline_quality = baseline.get("estimator_quality")
    candidate_recall = candidate_metrics.get("recall_at_10", candidate_metrics.get("exact_match_fraction", 0.0))
    baseline_recall = baseline_metrics.get("recall_at_10", baseline_metrics.get("exact_match_fraction", 0.0))
    return {
        "corpus": candidate["corpus"],
        "benchmark_metric": candidate["benchmark_metric"],
        "candidate_method": candidate["method"],
        "baseline_method": baseline["method"],
        "comparison_scope": comparison_scope,
        "qjl_sketch_dimension": scenario_qjl_dimension(candidate),
        "candidate": _report_method_row(candidate),
        "baseline": _report_method_row(baseline),
        "metrics": {
            "recall_at_10_delta": round(candidate_recall - baseline_recall, 6),
            "p95_ms_delta": round(candidate_metrics["p95_ms"] - baseline_metrics["p95_ms"], 6),
            "build_seconds_delta": round(candidate_metrics["build_seconds"] - baseline_metrics["build_seconds"], 6),
            "index_size_bytes_delta": int(candidate_metrics["index_size_bytes"] - baseline_metrics["index_size_bytes"]),
            "build_wal_bytes_delta": int(candidate_metrics["build_wal_bytes"] - baseline_metrics["build_wal_bytes"]),
            "visited_code_count_delta": round(
                float(candidate_scan.get("visited_code_count", 0.0)) - float(baseline_scan.get("visited_code_count", 0.0)),
                6,
            ),
            "visited_page_count_delta": round(
                float(candidate_scan.get("visited_page_count", 0.0)) - float(baseline_scan.get("visited_page_count", 0.0)),
                6,
            ),
            "selected_live_count_delta": round(
                float(candidate_scan.get("selected_live_count", 0.0)) - float(baseline_scan.get("selected_live_count", 0.0)),
                6,
            ),
            "selected_page_count_delta": round(
                float(candidate_scan.get("selected_page_count", 0.0)) - float(baseline_scan.get("selected_page_count", 0.0)),
                6,
            ),
            "distance_error_bias_delta": (
                round(
                    float(candidate_quality.get("distance_error_bias", 0.0))
                    - float(baseline_quality.get("distance_error_bias", 0.0)),
                    6,
                )
                if candidate_quality is not None and baseline_quality is not None
                else None
            ),
            "avg_abs_rank_shift_delta": (
                round(
                    float(candidate_quality.get("avg_abs_rank_shift", 0.0))
                    - float(baseline_quality.get("avg_abs_rank_shift", 0.0)),
                    6,
                )
                if candidate_quality is not None and baseline_quality is not None
                else None
            ),
        },
    }


def generate_report(payload: dict) -> dict:
    scenarios = payload["scenarios"]
    methods = payload["methods"]
    corpora = payload["corpora"]
    metrics = payload["metrics"]
    grouped = {
        (
            scenario["corpus"],
            scenario["benchmark_metric"],
            scenario["method"],
            scenario_qjl_dimension(scenario),
        ): scenario
        for scenario in scenarios
    }
    method_rows = [_report_method_row(scenario) for scenario in scenarios]
    comparisons = []

    for corpus in corpora:
        for benchmark_metric in metrics:
            for candidate_method, baseline_method, comparison_scope in comparison_pairs(methods):
                candidate_rows = [
                    scenario
                    for scenario in scenarios
                    if scenario["corpus"] == corpus
                    and scenario["benchmark_metric"] == benchmark_metric
                    and scenario["method"] == candidate_method
                ]
                for candidate in candidate_rows:
                    qjl_dimension = scenario_qjl_dimension(candidate)
                    baseline = grouped.get((corpus, benchmark_metric, baseline_method, qjl_dimension))
                    if baseline is None:
                        baseline = grouped.get((corpus, benchmark_metric, baseline_method, None))
                    if baseline is None:
                        continue
                    comparisons.append(_report_comparison_row(candidate, baseline, comparison_scope))

    leaderboards = {"best_recall_at_10": [], "best_p95_ms": []}
    for corpus in corpora:
        for benchmark_metric in metrics:
            corpus_scenarios = [
                scenario
                for scenario in scenarios
                if scenario["corpus"] == corpus and scenario["benchmark_metric"] == benchmark_metric
            ]
            if not corpus_scenarios:
                continue

            best_recall = max(
                corpus_scenarios,
                key=lambda scenario: scenario["metrics"].get(
                    "recall_at_10", scenario["metrics"].get("exact_match_fraction", 0.0)
                ),
            )
            best_latency = min(corpus_scenarios, key=lambda scenario: scenario["metrics"]["p95_ms"])
            leaderboards["best_recall_at_10"].append(
                {
                    "corpus": corpus,
                    "benchmark_metric": benchmark_metric,
                    "method": best_recall["method"],
                    "value": best_recall["metrics"].get(
                        "recall_at_10", best_recall["metrics"].get("exact_match_fraction", 0.0)
                    ),
                }
            )
            leaderboards["best_p95_ms"].append(
                {
                    "corpus": corpus,
                    "benchmark_metric": benchmark_metric,
                    "method": best_latency["method"],
                    "value": best_latency["metrics"]["p95_ms"],
                }
            )

    measurement_notes = [
        "TurboQuant ordered timing and tq_last_scan_stats() are captured from one SQL batch per repetition in this harness.",
        "Ordered retrieval results are reported with the configured helper contract and candidate limit for each scenario.",
        "Latency, scan work, and footprint are reported separately; this report only states what was measured for the listed corpus/profile/knob matrix.",
    ]
    conclusions = []
    for corpus in corpora:
        for benchmark_metric in metrics:
            turboquant_rows = [
                row
                for row in method_rows
                if row["corpus"] == corpus
                and row["benchmark_metric"] == benchmark_metric
                and row["method"].startswith("turboquant_")
            ]
            baseline_rows = [
                row
                for row in method_rows
                if row["corpus"] == corpus
                and row["benchmark_metric"] == benchmark_metric
                and row["method"].startswith("pgvector_")
            ]
            if not turboquant_rows or not baseline_rows:
                continue
            turboquant_row = min(
                turboquant_rows,
                key=lambda row: row["p95_ms"] if row["p95_ms"] is not None else float("inf"),
            )
            fastest_baseline = min(
                baseline_rows,
                key=lambda row: row["p95_ms"] if row["p95_ms"] is not None else float("inf"),
            )
            smallest_baseline = min(
                baseline_rows,
                key=lambda row: row["footprint_bytes"] if row["footprint_bytes"] is not None else float("inf"),
            )
            conclusions.append(
                {
                    "corpus": corpus,
                    "benchmark_metric": benchmark_metric,
                    "fairness": "This run uses the single-execution TurboQuant ordered timing path; it does not include the older double-execution benchmark path.",
                    "scan_work": (
                        f"{turboquant_row['method']} visited {_report_fmt(turboquant_row['visited_code_count'])} codes "
                        f"across {_report_fmt(turboquant_row['visited_page_count'])} pages with "
                        f"{_report_fmt(turboquant_row['selected_live_count'])} selected lives and "
                        f"{_report_fmt(turboquant_row['selected_page_count'])} selected pages "
                        f"on score_kernel={turboquant_row['score_kernel'] or '-'}."
                    ),
                    "latency": (
                        f"{turboquant_row['method']} p95={_report_fmt(turboquant_row['p95_ms'])} ms "
                        f"vs fastest pgvector baseline {fastest_baseline['method']} p95={_report_fmt(fastest_baseline['p95_ms'])} ms."
                    ),
                    "footprint": (
                        f"{turboquant_row['method']} footprint={_report_fmt(turboquant_row['footprint_bytes'])} B "
                        f"vs smallest pgvector baseline {smallest_baseline['method']} footprint={_report_fmt(smallest_baseline['footprint_bytes'])} B."
                    ),
                }
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "scenario_count": len(scenarios),
            "methods": methods,
            "corpora": corpora,
            "metrics": metrics,
            "query_modes": sorted({scenario.get("query_mode", "ordered_rerank") for scenario in scenarios}),
        },
        "method_rows": method_rows,
        "comparisons": comparisons,
        "measurement_notes": measurement_notes,
        "conclusions": conclusions,
        "leaderboards": leaderboards,
    }
    if payload.get("microbenchmarks"):
        report["microbenchmark_regression"] = {
            "comparisons": payload["microbenchmarks"].get("comparisons", []),
            "regression_gates": payload["microbenchmarks"].get("regression_gates", []),
            "interpretation_notes": payload["microbenchmarks"].get("interpretation_notes", []),
        }
    return report


def render_report_markdown(report: dict) -> str:
    lines = [
        "# pg_turboquant Benchmark Report",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Scenarios: {report['summary']['scenario_count']}",
        f"- Methods: {', '.join(report['summary']['methods'])}",
        f"- Corpora: {', '.join(report['summary']['corpora'])}",
        f"- Metrics: {', '.join(report['summary']['metrics'])}",
        f"- Query modes: {', '.join(report['summary']['query_modes'])}",
        "",
        "## Measurement Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in report["measurement_notes"])
    lines.extend(
        [
            "",
            "## Method Metrics",
            "",
            "- Columns: corpus, benchmark_metric, method, qjl_sketch_dimension, recall_at_10, p50_ms, p95_ms, footprint_bytes, visited_code_count, visited_page_count, selected_live_count, selected_page_count, score_kernel, query_helper, distance_error_bias, distance_error_variance, avg_abs_rank_shift",
            "",
        ]
    )
    for row in report["method_rows"]:
        lines.append(
            f"- {row['corpus']} | {row['benchmark_metric']} | {row['method']} | qjl_sketch_dimension={_report_fmt(row['qjl_sketch_dimension'])} | recall_at_10={_report_fmt(row['recall_at_10'])} | "
            f"p50_ms={_report_fmt(row['p50_ms'])} | p95_ms={_report_fmt(row['p95_ms'])} | "
            f"footprint_bytes={_report_fmt(row['footprint_bytes'])} | visited_code_count={_report_fmt(row['visited_code_count'])} | "
            f"visited_page_count={_report_fmt(row['visited_page_count'])} | selected_live_count={_report_fmt(row['selected_live_count'])} | "
            f"selected_page_count={_report_fmt(row['selected_page_count'])} | score_kernel={row['score_kernel']} | "
            f"query_helper={row['query_helper']} | distance_error_bias={_report_fmt(row['distance_error_bias'])} | "
            f"distance_error_variance={_report_fmt(row['distance_error_variance'])} | avg_abs_rank_shift={_report_fmt(row['avg_abs_rank_shift'])}"
        )
    lines.extend(["", "## Comparisons", ""])

    if not report["comparisons"]:
        lines.append("- No baseline comparisons available for the selected matrix.")
    else:
        for comparison in report["comparisons"]:
            metrics = comparison["metrics"]
            lines.append(
                f"- {comparison['corpus']} [{comparison['benchmark_metric']}, {comparison['comparison_scope']}, qjl_sketch_dimension={_report_fmt(comparison['qjl_sketch_dimension'])}]: "
                f"{comparison['candidate_method']} vs {comparison['baseline_method']} "
                f"(recall@10 delta={metrics['recall_at_10_delta']}, p95 delta={metrics['p95_ms_delta']} ms, "
                f"build delta={metrics['build_seconds_delta']} s, size delta={metrics['index_size_bytes_delta']} B, "
                f"WAL delta={metrics['build_wal_bytes_delta']} B, visited_code_count delta={metrics['visited_code_count_delta']}, "
                f"selected_page_count delta={metrics['selected_page_count_delta']}, distance_error_bias delta={_report_fmt(metrics['distance_error_bias_delta'])}, "
                f"avg_abs_rank_shift delta={_report_fmt(metrics['avg_abs_rank_shift_delta'])})"
            )

    if report.get("microbenchmark_regression"):
        lines.extend(["", "## Microbenchmark Regression", ""])
        lines.extend(
            f"- {note}" for note in report["microbenchmark_regression"].get("interpretation_notes", [])
        )
        lines.append("")
        for comparison in report["microbenchmark_regression"].get("comparisons", []):
            metrics = comparison["metrics"]
            lines.append(
                f"- {comparison['comparison']} [{comparison['comparison_kind']}]: "
                f"{comparison['candidate_benchmark']} vs {comparison['baseline_benchmark']} "
                f"(codes/sec ratio={_report_fmt(metrics['codes_per_second_ratio'])}, "
                f"ns/op ratio={_report_fmt(metrics['ns_per_op_ratio'])}, "
                f"visited_code_count delta={_report_fmt(metrics['visited_code_count_delta'])}, "
                f"visited_page_count delta={_report_fmt(metrics['visited_page_count_delta'])}, "
                f"candidate_heap_insert delta={_report_fmt(metrics['candidate_heap_insert_delta'])})"
            )
        for gate in report["microbenchmark_regression"].get("regression_gates", []):
            checks = ", ".join(
                f"{key}={value}" for key, value in sorted(gate.get("checks", {}).items())
            )
            lines.append(
                f"- {gate['gate']} [{gate['category']}] status={gate['status']} "
                f"comparison={gate['comparison']} checks: {checks}"
            )

    lines.extend(["", "## Hotpot Conclusions", ""])
    if not report["conclusions"]:
        lines.append("- No corpus-specific conclusions were generated for the selected matrix.")
    else:
        for conclusion in report["conclusions"]:
            lines.append(f"### {conclusion['corpus']} ({conclusion['benchmark_metric']})")
            lines.append("")
            lines.append(f"- Fairness: {conclusion['fairness']}")
            lines.append(f"- Scan work: {conclusion['scan_work']}")
            lines.append(f"- Latency: {conclusion['latency']}")
            lines.append(f"- Footprint: {conclusion['footprint']}")
            lines.append("")

    return "\n".join(lines) + "\n"


def render_report_html(report: dict) -> str:
    note_items = "".join(f"<li>{html.escape(note)}</li>" for note in report["measurement_notes"])
    method_rows = "".join(_render_report_method_row(row) for row in report["method_rows"])
    comparison_rows = "".join(_render_report_comparison_row(row) for row in report["comparisons"])
    microbench = report.get("microbenchmark_regression", {})
    microbench_note_items = "".join(
        f"<li>{html.escape(note)}</li>" for note in microbench.get("interpretation_notes", [])
    )
    microbench_comparison_rows = "".join(
        _render_microbench_comparison_row(row) for row in microbench.get("comparisons", [])
    )
    microbench_gate_rows = "".join(
        _render_microbench_gate_row(row) for row in microbench.get("regression_gates", [])
    )
    conclusion_blocks = "".join(
        (
            "<article class=\"card\">"
            f"<h3>{html.escape(conclusion['corpus'])}</h3>"
            f"<p><strong>Fairness:</strong> {html.escape(conclusion['fairness'])}</p>"
            f"<p><strong>Scan work:</strong> {html.escape(conclusion['scan_work'])}</p>"
            f"<p><strong>Latency:</strong> {html.escape(conclusion['latency'])}</p>"
            f"<p><strong>Footprint:</strong> {html.escape(conclusion['footprint'])}</p>"
            "</article>"
        )
        for conclusion in report["conclusions"]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>pg_turboquant Benchmark Report</title>
  <style>
    :root {{
      --bg: #f6f2ea;
      --panel: #fffdfa;
      --ink: #17261d;
      --muted: #5f6f63;
      --line: #d8d0c3;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background: radial-gradient(circle at top, #fff8ed 0%, var(--bg) 58%);
      color: var(--ink);
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 40px 24px 72px;
    }}
    .hero, .card, table {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 28px rgba(23, 38, 29, 0.05);
    }}
    .hero {{
      padding: 28px;
      margin-bottom: 24px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}
    .card {{
      padding: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      margin-top: 12px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 0.95rem;
    }}
    th {{
      background: rgba(23, 38, 29, 0.04);
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.45;
    }}
    .mono {{
      font-family: "SFMono-Regular", "Menlo", monospace;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>pg_turboquant Benchmark Report</h1>
      <p>This report lists measured recall, latency, scan work, and footprint for the selected corpus/profile/knob matrix. It uses the current single-execution TurboQuant timing path and avoids blanket speed claims outside the measured scenarios.</p>
      <div class="grid">
        <div class="card"><strong>Scenarios</strong><br>{report['summary']['scenario_count']}</div>
        <div class="card"><strong>Methods</strong><br>{html.escape(', '.join(report['summary']['methods']))}</div>
        <div class="card"><strong>Corpora</strong><br>{html.escape(', '.join(report['summary']['corpora']))}</div>
        <div class="card"><strong>Query modes</strong><br>{html.escape(', '.join(report['summary']['query_modes']))}</div>
      </div>
    </section>
    <section>
      <h2>Measurement Notes</h2>
      <ul>{note_items}</ul>
    </section>
    <section>
      <h2>Method Metrics</h2>
      <table>
        <thead>
          <tr>
            <th>corpus</th>
            <th>method</th>
            <th>qjl_sketch_dimension</th>
            <th>recall@10</th>
            <th>p50_ms</th>
            <th>p95_ms</th>
            <th>footprint_bytes</th>
            <th>visited_code_count</th>
            <th>visited_page_count</th>
            <th>selected_live_count</th>
            <th>selected_page_count</th>
            <th>score_kernel</th>
          </tr>
        </thead>
        <tbody>{method_rows}</tbody>
      </table>
    </section>
    <section>
      <h2>Comparisons</h2>
      <table>
        <thead>
          <tr>
            <th>corpus</th>
            <th>candidate</th>
            <th>baseline</th>
            <th>qjl_sketch_dimension</th>
            <th>recall@10 delta</th>
            <th>p95_ms delta</th>
            <th>footprint delta</th>
            <th>visited_code_count delta</th>
            <th>selected_page_count delta</th>
          </tr>
        </thead>
        <tbody>{comparison_rows}</tbody>
      </table>
    </section>
    <section>
      <h2>Microbenchmark Regression</h2>
      <ul>{microbench_note_items}</ul>
      <table>
        <thead>
          <tr>
            <th>comparison</th>
            <th>kind</th>
            <th>candidate</th>
            <th>baseline</th>
            <th>codes/sec ratio</th>
            <th>ns/op ratio</th>
            <th>visited_code_count delta</th>
            <th>candidate_heap_insert delta</th>
          </tr>
        </thead>
        <tbody>{microbench_comparison_rows}</tbody>
      </table>
      <table>
        <thead>
          <tr>
            <th>gate</th>
            <th>category</th>
            <th>status</th>
            <th>comparison</th>
            <th>checks</th>
          </tr>
        </thead>
        <tbody>{microbench_gate_rows}</tbody>
      </table>
    </section>
    <section>
      <h2>Hotpot Conclusions</h2>
      <div class="grid">{conclusion_blocks}</div>
    </section>
  </main>
</body>
</html>
"""


def _render_report_method_row(row: dict) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(str(row['corpus']))}</td>"
        f"<td class=\"mono\">{html.escape(str(row['method']))}</td>"
        f"<td>{_report_fmt(row['qjl_sketch_dimension'])}</td>"
        f"<td>{_report_fmt(row['recall_at_10'])}</td>"
        f"<td>{_report_fmt(row['p50_ms'])}</td>"
        f"<td>{_report_fmt(row['p95_ms'])}</td>"
        f"<td>{_report_fmt(row['footprint_bytes'])}</td>"
        f"<td>{_report_fmt(row['visited_code_count'])}</td>"
        f"<td>{_report_fmt(row['visited_page_count'])}</td>"
        f"<td>{_report_fmt(row['selected_live_count'])}</td>"
        f"<td>{_report_fmt(row['selected_page_count'])}</td>"
        f"<td>{html.escape(str(row['score_kernel']))}</td>"
        "</tr>"
    )


def _render_report_comparison_row(row: dict) -> str:
    metrics = row["metrics"]
    return (
        "<tr>"
        f"<td>{html.escape(str(row['corpus']))}</td>"
        f"<td class=\"mono\">{html.escape(str(row['candidate_method']))}</td>"
        f"<td class=\"mono\">{html.escape(str(row['baseline_method']))}</td>"
        f"<td>{_report_fmt(row['qjl_sketch_dimension'])}</td>"
        f"<td>{_report_fmt(metrics['recall_at_10_delta'])}</td>"
        f"<td>{_report_fmt(metrics['p95_ms_delta'])}</td>"
        f"<td>{_report_fmt(metrics['index_size_bytes_delta'])}</td>"
        f"<td>{_report_fmt(metrics['visited_code_count_delta'])}</td>"
        f"<td>{_report_fmt(metrics['selected_page_count_delta'])}</td>"
        "</tr>"
    )


def _render_microbench_comparison_row(row: dict) -> str:
    metrics = row["metrics"]
    return (
        "<tr>"
        f"<td class=\"mono\">{html.escape(str(row['comparison']))}</td>"
        f"<td>{html.escape(str(row['comparison_kind']))}</td>"
        f"<td class=\"mono\">{html.escape(str(row['candidate_benchmark']))}</td>"
        f"<td class=\"mono\">{html.escape(str(row['baseline_benchmark']))}</td>"
        f"<td>{_report_fmt(metrics['codes_per_second_ratio'])}</td>"
        f"<td>{_report_fmt(metrics['ns_per_op_ratio'])}</td>"
        f"<td>{_report_fmt(metrics['visited_code_count_delta'])}</td>"
        f"<td>{_report_fmt(metrics['candidate_heap_insert_delta'])}</td>"
        "</tr>"
    )


def _render_microbench_gate_row(row: dict) -> str:
    checks = "<br>".join(
        html.escape(f"{key}={value}") for key, value in sorted(row.get("checks", {}).items())
    )
    return (
        "<tr>"
        f"<td class=\"mono\">{html.escape(str(row['gate']))}</td>"
        f"<td>{html.escape(str(row['category']))}</td>"
        f"<td>{html.escape(str(row['status']))}</td>"
        f"<td class=\"mono\">{html.escape(str(row['comparison']))}</td>"
        f"<td>{checks}</td>"
        "</tr>"
    )


def _report_fmt(value: object) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:.4f}"


def run_scenario(
    base_cmd: list[str],
    corpus: Corpus,
    method: str,
    repetitions: int,
    scenario_index: int,
    turboquant_probes: Optional[int] = None,
    turboquant_oversample_factor: Optional[int] = None,
    turboquant_max_visited_codes: Optional[int] = None,
    turboquant_max_visited_pages: Optional[int] = None,
    turboquant_shadow_decode_diagnostics: bool = False,
    turboquant_force_decode_score_diagnostics: bool = False,
    turboquant_decode_rescore_factor: int = 1,
    turboquant_decode_rescore_extra_candidates: Optional[int] = None,
    requested_rerank_candidate_limit: Optional[int] = None,
    benchmark_metric: str = "cosine",
    turboquant_qjl_sketch_dim: Optional[str] = None,
) -> dict:
    table_name = f"tq_benchmark_{scenario_index:04d}"
    index_name = f"{table_name}_embedding_idx"
    load_corpus(base_cmd, table_name, corpus)

    build_seconds, index_size_bytes, build_wal_bytes, spec = build_index(
        base_cmd,
        table_name,
        index_name,
        corpus,
        method,
        benchmark_metric,
        requested_rerank_candidate_limit,
        turboquant_probes,
        turboquant_oversample_factor,
        turboquant_max_visited_codes,
        turboquant_max_visited_pages,
        turboquant_shadow_decode_diagnostics,
        turboquant_force_decode_score_diagnostics,
        turboquant_decode_rescore_factor,
        turboquant_decode_rescore_extra_candidates,
        turboquant_qjl_sketch_dim,
    )

    if corpus.name == "mixed_live_dead":
        apply_mixed_live_dead_workload(base_cmd, table_name, corpus)
        index_size_bytes = int(query_psql(base_cmd, f"SELECT pg_relation_size('{index_name}'::regclass);"))

    index_metadata = fetch_index_metadata(base_cmd, index_name, spec, corpus, benchmark_metric)
    simd_metadata = fetch_simd_metadata(base_cmd, spec)
    block_size = int(query_psql(base_cmd, "SELECT current_setting('block_size')::int;"))

    exact_results = [exact_top_ids(corpus, query, TOP_K_VALUES[-1], benchmark_metric) for query in corpus.queries]
    approx_results = []
    latencies_ms = []
    scan_stats_results = []
    candidate_retention_results = []
    estimator_diagnostics_results = []

    if spec.get("query_mode") == "bitmap_filter":
        exact_results = [
            exact_bitmap_ids(corpus, query, query_index % 2, spec["bitmap_threshold"])
            for query_index, query in enumerate(corpus.queries)
        ]

        for query_index, query in enumerate(corpus.queries):
            selected_ids = []
            selected_scan_stats = default_scan_stats(method)
            for _ in range(repetitions):
                started = time.perf_counter()
                if spec["index_method"] == "turboquant":
                    query_ids, query_scan_stats = query_bitmap_ids_and_scan_stats(
                        base_cmd,
                        table_name,
                        query,
                        query_index % 2,
                        spec["bitmap_threshold"],
                        spec["query_setup"],
                    )
                else:
                    query_ids = query_bitmap_ids(
                        base_cmd,
                        table_name,
                        query,
                        query_index % 2,
                        spec["bitmap_threshold"],
                        spec["query_setup"],
                    )
                    query_scan_stats = default_scan_stats(method)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                latencies_ms.append(elapsed_ms)
                if not selected_ids:
                    selected_ids = query_ids
                    selected_scan_stats = query_scan_stats
            approx_results.append(selected_ids)
            scan_stats_results.append(selected_scan_stats)
    else:
        for query_index, query in enumerate(corpus.queries):
            selected_ids = []
            selected_scan_stats = default_scan_stats(method)
            selected_approx_candidate_ids = []
            selected_shadow_candidate_ids = []
            selected_estimator_diagnostics = []
            for _ in range(repetitions):
                started = time.perf_counter()
                if spec["index_method"] == "turboquant":
                    decode_rescore_extra_candidates = 0
                    if int(spec["query_knobs"].get("turboquant.decode_rescore_factor", 1)) > 1:
                        decode_rescore_extra_candidates = int(
                            spec["query_knobs"].get("turboquant.decode_rescore_extra_candidates", 0)
                        )
                    if type(query_turboquant_ordered_ids_and_scan_stats).__module__.startswith("unittest.mock"):
                        query_result = query_turboquant_ordered_ids_and_scan_stats(
                            base_cmd,
                            table_name,
                            query,
                            TOP_K_VALUES[-1],
                            spec["query_setup"],
                            requested_rerank_candidate_limit,
                            decode_rescore_extra_candidates,
                            method,
                        )
                    else:
                        query_result = query_turboquant_ordered_ids_and_scan_stats(
                            base_cmd,
                            table_name,
                            query,
                            TOP_K_VALUES[-1],
                            spec["query_setup"],
                            requested_rerank_candidate_limit,
                            decode_rescore_extra_candidates,
                            method=method,
                            benchmark_metric=benchmark_metric,
                            include_estimator_diagnostics=True,
                        )
                    if len(query_result) == 5:
                        query_ids, query_scan_stats, approx_candidate_ids, shadow_candidate_ids, estimator_diagnostics = query_result
                    else:
                        query_ids, query_scan_stats, approx_candidate_ids, shadow_candidate_ids = query_result
                        estimator_diagnostics = []
                else:
                    query_ids = query_top_ids(
                        base_cmd,
                        table_name,
                        query,
                        TOP_K_VALUES[-1],
                        spec["query_setup"],
                        requested_rerank_candidate_limit,
                        benchmark_metric=benchmark_metric,
                    )
                    query_scan_stats = default_scan_stats(method)
                    approx_candidate_ids = []
                    shadow_candidate_ids = []
                    estimator_diagnostics = []
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                latencies_ms.append(elapsed_ms)
                if not selected_ids:
                    selected_ids = query_ids
                    selected_scan_stats = query_scan_stats
                    selected_approx_candidate_ids = approx_candidate_ids
                    selected_shadow_candidate_ids = shadow_candidate_ids
                    selected_estimator_diagnostics = estimator_diagnostics
            approx_results.append(selected_ids)
            scan_stats_results.append(selected_scan_stats)
            if spec["index_method"] == "turboquant":
                primary_retention = candidate_retention_for_query(exact_results[query_index], selected_approx_candidate_ids)
                shadow_retention = candidate_retention_for_query(exact_results[query_index], selected_shadow_candidate_ids)
                candidate_retention_results.append(
                    primary_retention
                    | {
                        "shadow_candidate_count": shadow_retention["candidate_count"],
                        "shadow_exact_top_10_retention": shadow_retention["exact_top_10_retention"],
                        "shadow_exact_top_100_retention": shadow_retention["exact_top_100_retention"],
                        "shadow_exact_top_100_miss_count": shadow_retention["exact_top_100_miss_count"],
                    }
                )
                estimator_diagnostics_results.extend(selected_estimator_diagnostics)

    insert_wal_bytes, inserted_rows = measure_insert_wal(base_cmd, table_name, corpus)
    concurrent_insert_rows_per_second, concurrent_insert_rows, concurrent_insert_workers = (
        measure_concurrent_insert_rows_per_second(base_cmd, table_name, corpus)
    )
    maintenance_wal_bytes, deleted_rows = measure_maintenance_wal(base_cmd, table_name, corpus)

    run_psql(base_cmd, f"DROP TABLE IF EXISTS {table_name};")

    if spec["index_method"] == "turboquant":
        sealed_baseline_build_wal_bytes = turboquant_sealed_baseline_bytes(
            len(active_rows(corpus)),
            block_size,
        )
        sealed_baseline_insert_wal_bytes = turboquant_sealed_baseline_bytes(
            inserted_rows,
            block_size,
        )
        sealed_baseline_maintenance_wal_bytes = turboquant_sealed_baseline_bytes(
            deleted_rows,
            block_size,
        )
    else:
        sealed_baseline_build_wal_bytes = 0
        sealed_baseline_insert_wal_bytes = 0
        sealed_baseline_maintenance_wal_bytes = 0

    aggregated_scan_stats = aggregate_scan_stats(scan_stats_results, method)
    faithful_fast_path, compatibility_fallback = resolve_scan_contract_flags(
        method,
        benchmark_metric,
        corpus,
        spec.get("query_mode", "ordered_rerank"),
    )
    aggregated_scan_stats["faithful_fast_path"] = faithful_fast_path
    aggregated_scan_stats["compatibility_fallback"] = compatibility_fallback
    if spec["index_method"] == "turboquant":
        simd_metadata["code_domain_kernel"] = aggregated_scan_stats.get("score_kernel", "scalar")
    else:
        simd_metadata["code_domain_kernel"] = "none"

    scenario = {
        "corpus": corpus.name,
        "method": method,
        "benchmark_metric": benchmark_metric,
        "query_mode": spec.get("query_mode", "ordered_rerank"),
        "corpus_metadata": {
            "rows": len(corpus.rows),
            "live_rows": len(active_rows(corpus)),
            "dimension": corpus.dimension,
            **corpus.metadata,
        },
        "ground_truth": ground_truth_for_corpus(corpus, benchmark_metric),
        "index": {
            "access_method": spec["index_method"],
            "opclass": spec["opclass"],
            "with": spec["with"],
        },
        "query_knobs": spec["query_knobs"],
        "index_metadata": index_metadata,
        "benchmark_metadata": scenario_benchmark_metadata(index_metadata),
        "simd": simd_metadata,
        "scan_stats": aggregated_scan_stats,
    }
    if spec.get("query_mode", "ordered_rerank") == "ordered_rerank" and spec["index_method"] == "turboquant":
        scenario["candidate_retention"] = aggregate_candidate_retention(candidate_retention_results)
        scenario["estimator_quality"] = aggregate_estimator_quality(estimator_diagnostics_results)
    if spec.get("query_mode") == "bitmap_filter":
        scenario["query_api"] = {
            "helper": "tq_bitmap_cosine_filter",
            "threshold": spec["bitmap_threshold"],
        }
        scenario["metrics"] = {
            "exact_match_fraction": round(
                sum(1 for exact_ids, actual_ids in zip(exact_results, approx_results) if exact_ids == actual_ids)
                / max(1, len(exact_results)),
                6,
            ),
            "avg_result_count": round(
                sum(len(result_ids) for result_ids in approx_results) / max(1, len(approx_results)),
                6,
            ),
            "p50_ms": percentile_ms(latencies_ms, 50.0),
            "p95_ms": percentile_ms(latencies_ms, 95.0),
            "build_seconds": round(build_seconds, 6),
            "index_size_bytes": index_size_bytes,
            "candidate_slots_bound": spec["candidate_slots_bound"],
            "build_wal_bytes": build_wal_bytes,
            "insert_wal_bytes": insert_wal_bytes,
            "concurrent_insert_rows_per_second": concurrent_insert_rows_per_second,
            "concurrent_insert_rows": concurrent_insert_rows,
            "concurrent_insert_workers": concurrent_insert_workers,
            "maintenance_wal_bytes": maintenance_wal_bytes,
            "sealed_baseline_build_wal_bytes": sealed_baseline_build_wal_bytes,
            "sealed_baseline_insert_wal_bytes": sealed_baseline_insert_wal_bytes,
            "sealed_baseline_maintenance_wal_bytes": sealed_baseline_maintenance_wal_bytes,
        }
    else:
        decode_rescore_extra_candidates = 0
        if (
            spec["index_method"] == "turboquant"
            and int(spec["query_knobs"].get("turboquant.decode_rescore_factor", 1)) > 1
        ):
            decode_rescore_extra_candidates = int(
                spec["query_knobs"].get("turboquant.decode_rescore_extra_candidates", 0)
            )
        scenario["query_api"] = {
            "helper": "tq_rerank_candidates",
            "candidate_limit": rerank_candidate_limit(TOP_K_VALUES[-1], requested_rerank_candidate_limit),
            "final_limit": TOP_K_VALUES[-1],
            "effective_candidate_limit": effective_candidate_limit(
                TOP_K_VALUES[-1],
                requested_rerank_candidate_limit,
                decode_rescore_extra_candidates,
            ),
        }
        scenario["metrics"] = {
            "recall_at_10": average_recall(exact_results, approx_results, 10),
            "recall_at_100": average_recall(exact_results, approx_results, 100),
            "p50_ms": percentile_ms(latencies_ms, 50.0),
            "p95_ms": percentile_ms(latencies_ms, 95.0),
            "build_seconds": round(build_seconds, 6),
            "index_size_bytes": index_size_bytes,
            "candidate_slots_bound": spec["candidate_slots_bound"],
            "build_wal_bytes": build_wal_bytes,
            "insert_wal_bytes": insert_wal_bytes,
            "concurrent_insert_rows_per_second": concurrent_insert_rows_per_second,
            "concurrent_insert_rows": concurrent_insert_rows,
            "concurrent_insert_workers": concurrent_insert_workers,
            "maintenance_wal_bytes": maintenance_wal_bytes,
            "sealed_baseline_build_wal_bytes": sealed_baseline_build_wal_bytes,
            "sealed_baseline_insert_wal_bytes": sealed_baseline_insert_wal_bytes,
            "sealed_baseline_maintenance_wal_bytes": sealed_baseline_maintenance_wal_bytes,
        }
    return scenario


def dry_run_scenario(
    corpus: Corpus,
    method: str,
    turboquant_probes: Optional[int] = None,
    turboquant_oversample_factor: Optional[int] = None,
    turboquant_max_visited_codes: Optional[int] = None,
    turboquant_max_visited_pages: Optional[int] = None,
    turboquant_shadow_decode_diagnostics: bool = False,
    turboquant_force_decode_score_diagnostics: bool = False,
    turboquant_decode_rescore_factor: int = 1,
    turboquant_decode_rescore_extra_candidates: Optional[int] = None,
    requested_rerank_candidate_limit: Optional[int] = None,
    benchmark_metric: str = "cosine",
    turboquant_qjl_sketch_dim: Optional[str] = None,
) -> dict:
    spec = method_spec(
        method,
        corpus,
        benchmark_metric,
        requested_rerank_candidate_limit,
        turboquant_probes,
        turboquant_oversample_factor,
        turboquant_max_visited_codes,
        turboquant_max_visited_pages,
        turboquant_shadow_decode_diagnostics,
        turboquant_force_decode_score_diagnostics,
        turboquant_decode_rescore_factor,
        turboquant_decode_rescore_extra_candidates,
        turboquant_qjl_sketch_dim,
    )
    simd_metadata = synthetic_simd_metadata()
    scan_stats = default_scan_stats(method)
    if (
        spec["index_method"] == "turboquant"
        and spec["query_knobs"].get("turboquant.force_decode_score_diagnostics", False)
    ):
        scan_stats["score_mode"] = "decode"
        scan_stats["score_kernel"] = "scalar"
    elif (
        spec["index_method"] == "turboquant"
        and int(spec["query_knobs"].get("turboquant.decode_rescore_factor", 1)) > 1
    ):
        scan_stats["score_mode"] = "decode_rescore"
        scan_stats["score_kernel"] = "scalar"
    else:
        scan_stats["score_kernel"] = synthetic_code_domain_kernel(corpus, spec, simd_metadata)
    simd_metadata["code_domain_kernel"] = scan_stats["score_kernel"]
    faithful_fast_path, compatibility_fallback = resolve_scan_contract_flags(
        method,
        benchmark_metric,
        corpus,
        spec.get("query_mode", "ordered_rerank"),
    )
    scan_stats["faithful_fast_path"] = faithful_fast_path
    scan_stats["compatibility_fallback"] = compatibility_fallback

    scenario = {
        "corpus": corpus.name,
        "method": method,
        "benchmark_metric": benchmark_metric,
        "query_mode": spec.get("query_mode", "ordered_rerank"),
        "corpus_metadata": {
            "rows": len(corpus.rows),
            "live_rows": len(active_rows(corpus)),
            "dimension": corpus.dimension,
            **corpus.metadata,
        },
        "ground_truth": ground_truth_for_corpus(corpus, benchmark_metric),
        "index": {
            "access_method": spec["index_method"],
            "opclass": spec["opclass"],
            "with": spec["with"],
        },
        "query_knobs": spec["query_knobs"],
        "index_metadata": synthetic_index_metadata(method, spec, corpus, benchmark_metric),
        "benchmark_metadata": scenario_benchmark_metadata(synthetic_index_metadata(method, spec, corpus, benchmark_metric)),
        "simd": simd_metadata,
        "scan_stats": scan_stats,
    }
    if spec.get("query_mode", "ordered_rerank") == "ordered_rerank" and spec["index_method"] == "turboquant":
        scenario["candidate_retention"] = default_candidate_retention()
        scenario["estimator_quality"] = default_estimator_quality()
    if method == "turboquant_ivf":
        scenario["scan_stats"].update(
            {
                "configured_probe_count": spec["query_knobs"]["turboquant.probes"],
                "nominal_probe_count": spec["query_knobs"]["turboquant.probes"],
                "effective_probe_count": spec["query_knobs"]["turboquant.probes"],
                "max_visited_codes": spec["query_knobs"]["turboquant.max_visited_codes"],
                "max_visited_pages": spec["query_knobs"]["turboquant.max_visited_pages"],
            }
        )
    if spec.get("query_mode") == "bitmap_filter":
        scenario["query_api"] = {
            "helper": "tq_bitmap_cosine_filter",
            "threshold": spec["bitmap_threshold"],
        }
        scenario["metrics"] = {
            "exact_match_fraction": 1.0,
            "avg_result_count": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "build_seconds": 0.0,
            "index_size_bytes": 0,
            "candidate_slots_bound": spec["candidate_slots_bound"],
            "build_wal_bytes": 0,
            "insert_wal_bytes": 0,
            "concurrent_insert_rows_per_second": 0.0,
            "concurrent_insert_rows": 0,
            "concurrent_insert_workers": 0,
            "maintenance_wal_bytes": 0,
            "sealed_baseline_build_wal_bytes": 1,
            "sealed_baseline_insert_wal_bytes": 1,
            "sealed_baseline_maintenance_wal_bytes": 1,
        }
    else:
        decode_rescore_extra_candidates = 0
        if (
            spec["index_method"] == "turboquant"
            and int(spec["query_knobs"].get("turboquant.decode_rescore_factor", 1)) > 1
        ):
            decode_rescore_extra_candidates = int(
                spec["query_knobs"].get("turboquant.decode_rescore_extra_candidates", 0)
            )
        scenario["query_api"] = {
            "helper": "tq_rerank_candidates",
            "candidate_limit": rerank_candidate_limit(TOP_K_VALUES[-1], requested_rerank_candidate_limit),
            "final_limit": TOP_K_VALUES[-1],
            "effective_candidate_limit": effective_candidate_limit(
                TOP_K_VALUES[-1],
                requested_rerank_candidate_limit,
                decode_rescore_extra_candidates,
            ),
        }
        scenario["metrics"] = {
            "recall_at_10": 0.0,
            "recall_at_100": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "build_seconds": 0.0,
            "index_size_bytes": 0,
            "candidate_slots_bound": spec["candidate_slots_bound"],
            "build_wal_bytes": 0,
            "insert_wal_bytes": 0,
            "concurrent_insert_rows_per_second": 0.0,
            "concurrent_insert_rows": 0,
            "concurrent_insert_workers": 0,
            "maintenance_wal_bytes": 0,
            "sealed_baseline_build_wal_bytes": 1 if spec["index_method"] == "turboquant" else 0,
            "sealed_baseline_insert_wal_bytes": 1 if spec["index_method"] == "turboquant" else 0,
            "sealed_baseline_maintenance_wal_bytes": 1 if spec["index_method"] == "turboquant" else 0,
        }
    return scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pg_turboquant benchmark suite")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port")
    parser.add_argument("--dbname", default="postgres")
    parser.add_argument("--profile", default="quick", choices=tuple(PROFILE_CONFIGS.keys()))
    parser.add_argument("--corpus")
    parser.add_argument("--methods")
    parser.add_argument("--metrics")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--turboquant-probes", type=int)
    parser.add_argument("--turboquant-qjl-sketch-dims")
    parser.add_argument("--turboquant-oversample-factor", type=int)
    parser.add_argument("--turboquant-max-visited-codes", type=int)
    parser.add_argument("--turboquant-max-visited-pages", type=int)
    parser.add_argument("--turboquant-shadow-decode-diagnostics", action="store_true")
    parser.add_argument("--turboquant-force-decode-score-diagnostics", action="store_true")
    parser.add_argument("--turboquant-decode-rescore-factor", type=int, default=1)
    parser.add_argument("--turboquant-decode-rescore-extra-candidates", type=int)
    parser.add_argument("--rerank-candidate-limit", type=int)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--microbench", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = parse_csv_list(args.methods, SUPPORTED_METHODS)
    corpora = parse_csv_list(args.corpus, SUPPORTED_CORPORA)
    metrics = parse_csv_list(args.metrics, SUPPORTED_METRICS) if args.metrics is not None else ["cosine"]
    turboquant_qjl_sketch_dims = parse_freeform_csv_list(args.turboquant_qjl_sketch_dims) if args.turboquant_qjl_sketch_dims is not None else ["d"]
    profile = PROFILE_CONFIGS[args.profile]

    if not args.dry_run and not args.port:
        raise SystemExit("--port is required unless --dry-run is used")

    base_cmd = [
        "psql",
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-h",
        args.host,
        "-p",
        str(args.port) if args.port else "0",
        "-d",
        args.dbname,
    ]

    scenarios = []
    scenario_index = 0
    for corpus_name in corpora:
        corpus = build_corpus(corpus_name, profile, args.seed + scenario_index)
        for benchmark_metric in metrics:
            for method in methods:
                qjl_dim_tokens = turboquant_qjl_sketch_dims if method.startswith("turboquant_") else [None]
                for qjl_dim_token in qjl_dim_tokens:
                    scenario_index += 1
                    if args.dry_run:
                        scenarios.append(
                        dry_run_scenario(
                            corpus,
                            method,
                            args.turboquant_probes,
                            args.turboquant_oversample_factor,
                            args.turboquant_max_visited_codes,
                                args.turboquant_max_visited_pages,
                                args.turboquant_shadow_decode_diagnostics,
                            args.turboquant_force_decode_score_diagnostics,
                            args.turboquant_decode_rescore_factor,
                            args.turboquant_decode_rescore_extra_candidates,
                            args.rerank_candidate_limit,
                            benchmark_metric=benchmark_metric,
                            turboquant_qjl_sketch_dim=qjl_dim_token,
                        )
                        )
                    else:
                        scenarios.append(
                        run_scenario(
                            base_cmd,
                            corpus,
                            method,
                            profile["repetitions"],
                            scenario_index,
                            args.turboquant_probes,
                            args.turboquant_oversample_factor,
                            args.turboquant_max_visited_codes,
                                args.turboquant_max_visited_pages,
                                args.turboquant_shadow_decode_diagnostics,
                            args.turboquant_force_decode_score_diagnostics,
                            args.turboquant_decode_rescore_factor,
                            args.turboquant_decode_rescore_extra_candidates,
                            args.rerank_candidate_limit,
                            benchmark_metric=benchmark_metric,
                            turboquant_qjl_sketch_dim=qjl_dim_token,
                        )
                        )

    payload = {
        "profile": args.profile,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "corpora": corpora,
        "methods": methods,
        "metrics": metrics,
        "environment": environment_metadata(),
        "scenario_matrix": scenario_matrix_metadata(args.profile, corpora, methods, metrics, turboquant_qjl_sketch_dims),
        "scenarios": scenarios,
    }

    if args.microbench:
        payload["microbenchmarks"] = run_microbenchmarks(args.dry_run)

    if args.report:
        payload["report"] = generate_report(payload)
        payload["artifacts"] = {
            "report_json": "benchmark-report.json",
            "report_markdown": "benchmark-report.md",
            "report_html": "benchmark-report.html",
        }

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        if args.report:
            output_dir = output_path.parent
            (output_dir / payload["artifacts"]["report_json"]).write_text(
                json.dumps(payload["report"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / payload["artifacts"]["report_markdown"]).write_text(
                render_report_markdown(payload["report"]),
                encoding="utf-8",
            )
            (output_dir / payload["artifacts"]["report_html"]).write_text(
                render_report_html(payload["report"]),
                encoding="utf-8",
            )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"benchmark_suite.py failed with exit code {exc.returncode}\n")
        raise
