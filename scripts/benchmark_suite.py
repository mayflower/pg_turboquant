#!/usr/bin/env -S uv run python
import argparse
import json
import math
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


def effective_candidate_limit(limit: int, requested_limit: Optional[int]) -> int:
    return max(limit, rerank_candidate_limit(limit, requested_limit))


def capability_metadata(spec: dict) -> dict:
    return {
        "ordered_scan": True,
        "bitmap_scan": spec["index_method"] == "turboquant",
        "index_only_scan": False,
        "multicolumn": False,
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


def environment_metadata() -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_arch": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "release": platform.release(),
    }


def scenario_matrix_metadata(profile: str, corpora: list[str], methods: list[str]) -> dict:
    return {
        "profiles": [profile],
        "corpora": corpora,
        "methods": methods,
    }


def synthetic_index_metadata(method: str, spec: dict, corpus: Corpus) -> dict:
    metric = "cosine"
    metadata = {
        "access_method": spec["index_method"],
        "capabilities": capability_metadata(spec),
        "metric": metric,
        "opclass": spec["opclass"],
        "format_version": 4 if spec["index_method"] == "turboquant" else 0,
        "list_count": int(spec["with"].get("lists", 0)) if "lists" in spec["with"] else 0,
    }

    if spec["index_method"] == "turboquant":
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
    raise AssertionError(f"unexpected corpus {corpus_name}")


def active_rows(corpus: Corpus) -> list[Row]:
    return [row for row in corpus.rows if not row.deleted]


def exact_top_ids(corpus: Corpus, query: tuple[float, ...], limit: int) -> list[int]:
    ranked = sorted(
        active_rows(corpus),
        key=lambda row: (cosine_distance(row.values, query), row.row_id),
    )
    return [row.row_id for row in ranked[:limit]]


def exact_bitmap_ids(corpus: Corpus, query: tuple[float, ...], category: int, threshold: float) -> list[int]:
    return [
        row.row_id
        for row in sorted(active_rows(corpus), key=lambda row: row.row_id)
        if (row.row_id % 2) == category and cosine_distance(row.values, query) <= threshold
    ]


def ground_truth_for_corpus(corpus: Corpus) -> dict:
    return {
        "kind": "exact",
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


def current_wal_lsn(base_cmd: list[str]) -> str:
    return query_psql(base_cmd, "SELECT pg_current_wal_insert_lsn();")


def wal_bytes_since(base_cmd: list[str], start_lsn: str) -> int:
    return int(
        query_psql(
            base_cmd,
            f"SELECT pg_wal_lsn_diff(pg_current_wal_insert_lsn(), '{start_lsn}'::pg_lsn)::bigint;",
        )
    )


def method_spec(
    method: str,
    corpus: Corpus,
    turboquant_probes: Optional[int] = None,
    turboquant_oversample_factor: Optional[int] = None,
) -> dict:
    list_count = min(16, max(4, len(active_rows(corpus)) // 32))
    turboquant_probe_value = turboquant_probes if turboquant_probes is not None else min(list_count, 4)
    turboquant_oversample_value = (
        turboquant_oversample_factor if turboquant_oversample_factor is not None else 4
    )
    if method == "turboquant_flat":
        return {
            "index_method": "turboquant",
            "opclass": "tq_cosine_ops",
            "with": {
                "bits": 4,
                "lists": 0,
                "lanes": "auto",
                "transform": "hadamard",
                "normalized": "true",
            },
            "query_setup": [
                "SET LOCAL enable_seqscan = off",
                "SET LOCAL enable_bitmapscan = off",
                f"SET LOCAL turboquant.probes = {turboquant_probe_value}",
                f"SET LOCAL turboquant.oversample_factor = {turboquant_oversample_value}",
            ],
            "candidate_slots_bound": turboquant_probe_value * turboquant_oversample_value,
            "query_knobs": {
                "turboquant.probes": turboquant_probe_value,
                "turboquant.oversample_factor": turboquant_oversample_value,
            },
        }
    if method == "turboquant_ivf":
        return {
            "index_method": "turboquant",
            "opclass": "tq_cosine_ops",
            "with": {
                "bits": 4,
                "lists": list_count,
                "lanes": "auto",
                "transform": "hadamard",
                "normalized": "true",
            },
            "query_setup": [
                "SET LOCAL enable_seqscan = off",
                "SET LOCAL enable_bitmapscan = off",
                f"SET LOCAL turboquant.probes = {turboquant_probe_value}",
                f"SET LOCAL turboquant.oversample_factor = {turboquant_oversample_value}",
            ],
            "candidate_slots_bound": turboquant_probe_value * turboquant_oversample_value,
            "query_knobs": {
                "turboquant.probes": turboquant_probe_value,
                "turboquant.oversample_factor": turboquant_oversample_value,
            },
        }
    if method == "turboquant_bitmap":
        return {
            "index_method": "turboquant",
            "opclass": "tq_cosine_ops",
            "with": {
                "bits": 4,
                "lists": 0,
                "lanes": "auto",
                "transform": "hadamard",
                "normalized": "true",
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
            "opclass": "vector_cosine_ops",
            "with": {
                "lists": list_count,
            },
            "query_setup": [
                "SET LOCAL enable_seqscan = off",
                "SET LOCAL enable_bitmapscan = off",
                f"SET LOCAL ivfflat.probes = {min(list_count, 4)}",
            ],
            "candidate_slots_bound": 0,
            "query_knobs": {
                "ivfflat.probes": min(list_count, 4),
            },
        }
    if method == "pgvector_hnsw":
        return {
            "index_method": "hnsw",
            "opclass": "vector_cosine_ops",
            "with": {
                "m": 8,
                "ef_construction": 32,
            },
            "query_setup": [
                "SET LOCAL enable_seqscan = off",
                "SET LOCAL enable_bitmapscan = off",
                "SET LOCAL hnsw.ef_search = 40",
            ],
            "candidate_slots_bound": 0,
            "query_knobs": {
                "hnsw.ef_search": 40,
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


def load_corpus(base_cmd: list[str], table_name: str, corpus: Corpus) -> None:
    insert_values = ",\n".join(
        f"({row.row_id}, {row.row_id % 2}, '{vector_literal(row.values)}')"
        for row in corpus.rows
    )
    sql = f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_turboquant;
    DROP TABLE IF EXISTS {table_name};
    CREATE TABLE {table_name} (
        id int4 PRIMARY KEY,
        category int4 NOT NULL,
        embedding vector({corpus.dimension})
    );
    INSERT INTO {table_name} (id, category, embedding) VALUES
    {insert_values};
    """
    run_psql(base_cmd, sql)


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
    turboquant_probes: Optional[int],
    turboquant_oversample_factor: Optional[int],
) -> tuple[float, int, int, dict]:
    spec = method_spec(method, corpus, turboquant_probes, turboquant_oversample_factor)
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


def fetch_index_metadata(base_cmd: list[str], index_name: str, spec: dict, corpus: Corpus) -> dict:
    if spec["index_method"] != "turboquant":
        return synthetic_index_metadata("generic", spec, corpus)

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


def query_top_ids(
    base_cmd: list[str],
    table_name: str,
    query_vector: tuple[float, ...],
    limit: int,
    query_setup: list[str],
    requested_candidate_limit: Optional[int],
) -> list[int]:
    helper_candidate_limit = effective_candidate_limit(limit, requested_candidate_limit)
    query_sql = ";\n".join(
        query_setup
        + [
            (
                "SELECT coalesce(json_agg(candidate_id::int ORDER BY exact_rank), '[]'::json)::text "
                "FROM tq_rerank_candidates("
                f"'{table_name}'::regclass, 'id', 'embedding', "
                f"'{vector_literal(query_vector)}'::vector, 'cosine', "
                f"{helper_candidate_limit}, {limit})"
            )
        ]
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
    raw = query_psql(base_cmd, query_sql)
    if not raw:
        return []
    return list(json.loads(raw))


def average_recall(exact_results: list[list[int]], approx_results: list[list[int]], k: int) -> float:
    recalls = []
    for exact_ids, approx_ids in zip(exact_results, approx_results):
        exact_top = exact_ids[:k]
        approx_top = approx_ids[:k]
        denominator = max(1, len(exact_top))
        recalls.append(len(set(exact_top) & set(approx_top)) / denominator)
    return round(sum(recalls) / len(recalls), 6)


def comparison_pairs(methods: list[str]) -> list[tuple[str, str]]:
    baselines = [method for method in methods if method.startswith("pgvector_")]
    candidates = [method for method in methods if method.startswith("turboquant_")]
    return [(candidate, baseline) for candidate in candidates for baseline in baselines]


def generate_report(payload: dict) -> dict:
    scenarios = payload["scenarios"]
    methods = payload["methods"]
    corpora = payload["corpora"]
    grouped = {(scenario["corpus"], scenario["method"]): scenario for scenario in scenarios}
    comparisons = []

    for corpus in corpora:
        for candidate_method, baseline_method in comparison_pairs(methods):
            candidate = grouped.get((corpus, candidate_method))
            baseline = grouped.get((corpus, baseline_method))
            if candidate is None or baseline is None:
                continue

            candidate_metrics = candidate["metrics"]
            baseline_metrics = baseline["metrics"]
            candidate_recall = candidate_metrics.get(
                "recall_at_10", candidate_metrics.get("exact_match_fraction", 0.0)
            )
            baseline_recall = baseline_metrics.get(
                "recall_at_10", baseline_metrics.get("exact_match_fraction", 0.0)
            )
            comparisons.append(
                {
                    "corpus": corpus,
                    "candidate_method": candidate_method,
                    "baseline_method": baseline_method,
                    "metrics": {
                        "recall_at_10_delta": round(candidate_recall - baseline_recall, 6),
                        "p95_ms_delta": round(candidate_metrics["p95_ms"] - baseline_metrics["p95_ms"], 6),
                        "build_seconds_delta": round(
                            candidate_metrics["build_seconds"] - baseline_metrics["build_seconds"], 6
                        ),
                        "index_size_bytes_delta": int(
                            candidate_metrics["index_size_bytes"] - baseline_metrics["index_size_bytes"]
                        ),
                        "build_wal_bytes_delta": int(
                            candidate_metrics["build_wal_bytes"] - baseline_metrics["build_wal_bytes"]
                        ),
                    },
                }
            )

    leaderboards = {"best_recall_at_10": [], "best_p95_ms": []}
    for corpus in corpora:
        corpus_scenarios = [scenario for scenario in scenarios if scenario["corpus"] == corpus]
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
                "method": best_recall["method"],
                "value": best_recall["metrics"].get(
                    "recall_at_10", best_recall["metrics"].get("exact_match_fraction", 0.0)
                ),
            }
        )
        leaderboards["best_p95_ms"].append(
            {
                "corpus": corpus,
                "method": best_latency["method"],
                "value": best_latency["metrics"]["p95_ms"],
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "scenario_count": len(scenarios),
            "methods": methods,
            "corpora": corpora,
        },
        "comparisons": comparisons,
        "leaderboards": leaderboards,
    }


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
        "",
        "## Comparisons",
        "",
    ]

    if not report["comparisons"]:
        lines.append("- No baseline comparisons available for the selected matrix.")
    else:
        for comparison in report["comparisons"]:
            metrics = comparison["metrics"]
            lines.append(
                f"- {comparison['corpus']}: {comparison['candidate_method']} vs {comparison['baseline_method']} "
                f"(recall@10 delta={metrics['recall_at_10_delta']}, p95 delta={metrics['p95_ms_delta']} ms, "
                f"build delta={metrics['build_seconds_delta']} s, size delta={metrics['index_size_bytes_delta']} B, "
                f"WAL delta={metrics['build_wal_bytes_delta']} B)"
            )

    return "\n".join(lines) + "\n"


def run_scenario(
    base_cmd: list[str],
    corpus: Corpus,
    method: str,
    repetitions: int,
    scenario_index: int,
    turboquant_probes: Optional[int],
    turboquant_oversample_factor: Optional[int],
    requested_rerank_candidate_limit: Optional[int],
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
        turboquant_probes,
        turboquant_oversample_factor,
    )

    if corpus.name == "mixed_live_dead":
        apply_mixed_live_dead_workload(base_cmd, table_name, corpus)
        index_size_bytes = int(query_psql(base_cmd, f"SELECT pg_relation_size('{index_name}'::regclass);"))

    index_metadata = fetch_index_metadata(base_cmd, index_name, spec, corpus)
    simd_metadata = fetch_simd_metadata(base_cmd, spec)
    block_size = int(query_psql(base_cmd, "SELECT current_setting('block_size')::int;"))

    exact_results = [exact_top_ids(corpus, query, TOP_K_VALUES[-1]) for query in corpus.queries]
    approx_results = []
    latencies_ms = []

    if spec.get("query_mode") == "bitmap_filter":
        exact_results = [
            exact_bitmap_ids(corpus, query, query_index % 2, spec["bitmap_threshold"])
            for query_index, query in enumerate(corpus.queries)
        ]

        for query_index, query in enumerate(corpus.queries):
            best_ids = []
            best_latency_ms = None
            for _ in range(repetitions):
                started = time.perf_counter()
                query_ids = query_bitmap_ids(
                    base_cmd,
                    table_name,
                    query,
                    query_index % 2,
                    spec["bitmap_threshold"],
                    spec["query_setup"],
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if best_latency_ms is None or elapsed_ms < best_latency_ms:
                    best_latency_ms = elapsed_ms
                    best_ids = query_ids
            approx_results.append(best_ids)
            latencies_ms.append(best_latency_ms or 0.0)
    else:
        for query in corpus.queries:
            best_ids = []
            best_latency_ms = None
            for _ in range(repetitions):
                started = time.perf_counter()
                query_ids = query_top_ids(
                    base_cmd,
                    table_name,
                    query,
                    TOP_K_VALUES[-1],
                    spec["query_setup"],
                    requested_rerank_candidate_limit,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if best_latency_ms is None or elapsed_ms < best_latency_ms:
                    best_latency_ms = elapsed_ms
                    best_ids = query_ids
            approx_results.append(best_ids)
            latencies_ms.append(best_latency_ms or 0.0)

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

    scenario = {
        "corpus": corpus.name,
        "method": method,
        "query_mode": spec.get("query_mode", "ordered_rerank"),
        "corpus_metadata": {
            "rows": len(corpus.rows),
            "live_rows": len(active_rows(corpus)),
            "dimension": corpus.dimension,
            **corpus.metadata,
        },
        "ground_truth": ground_truth_for_corpus(corpus),
        "index": {
            "access_method": spec["index_method"],
            "opclass": spec["opclass"],
            "with": spec["with"],
        },
        "query_knobs": spec["query_knobs"],
        "index_metadata": index_metadata,
        "simd": simd_metadata,
    }
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
        scenario["query_api"] = {
            "helper": "tq_rerank_candidates",
            "candidate_limit": rerank_candidate_limit(TOP_K_VALUES[-1], requested_rerank_candidate_limit),
            "final_limit": TOP_K_VALUES[-1],
            "effective_candidate_limit": effective_candidate_limit(TOP_K_VALUES[-1], requested_rerank_candidate_limit),
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
    turboquant_probes: Optional[int],
    turboquant_oversample_factor: Optional[int],
    requested_rerank_candidate_limit: Optional[int],
) -> dict:
    spec = method_spec(method, corpus, turboquant_probes, turboquant_oversample_factor)
    scenario = {
        "corpus": corpus.name,
        "method": method,
        "query_mode": spec.get("query_mode", "ordered_rerank"),
        "corpus_metadata": {
            "rows": len(corpus.rows),
            "live_rows": len(active_rows(corpus)),
            "dimension": corpus.dimension,
            **corpus.metadata,
        },
        "ground_truth": ground_truth_for_corpus(corpus),
        "index": {
            "access_method": spec["index_method"],
            "opclass": spec["opclass"],
            "with": spec["with"],
        },
        "query_knobs": spec["query_knobs"],
        "index_metadata": synthetic_index_metadata(method, spec, corpus),
        "simd": synthetic_simd_metadata(),
    }
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
        scenario["query_api"] = {
            "helper": "tq_rerank_candidates",
            "candidate_limit": rerank_candidate_limit(TOP_K_VALUES[-1], requested_rerank_candidate_limit),
            "final_limit": TOP_K_VALUES[-1],
            "effective_candidate_limit": effective_candidate_limit(TOP_K_VALUES[-1], requested_rerank_candidate_limit),
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
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--turboquant-probes", type=int)
    parser.add_argument("--turboquant-oversample-factor", type=int)
    parser.add_argument("--rerank-candidate-limit", type=int)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = parse_csv_list(args.methods, SUPPORTED_METHODS)
    corpora = parse_csv_list(args.corpus, SUPPORTED_CORPORA)
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
        for method in methods:
            scenario_index += 1
            if args.dry_run:
                scenarios.append(
                    dry_run_scenario(
                        corpus,
                        method,
                        args.turboquant_probes,
                        args.turboquant_oversample_factor,
                        args.rerank_candidate_limit,
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
                        args.rerank_candidate_limit,
                    )
                )

    payload = {
        "profile": args.profile,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "corpora": corpora,
        "methods": methods,
        "environment": environment_metadata(),
        "scenario_matrix": scenario_matrix_metadata(args.profile, corpora, methods),
        "scenarios": scenarios,
    }

    if args.report:
        payload["report"] = generate_report(payload)
        payload["artifacts"] = {
            "report_json": "benchmark-report.json",
            "report_markdown": "benchmark-report.md",
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


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"benchmark_suite.py failed with exit code {exc.returncode}\n")
        raise
