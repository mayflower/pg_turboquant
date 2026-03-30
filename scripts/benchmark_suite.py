#!/usr/bin/env -S uv run python
import argparse
import html
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


def method_spec(
    method: str,
    corpus: Corpus,
    turboquant_probes: Optional[int] = None,
    turboquant_oversample_factor: Optional[int] = None,
    turboquant_max_visited_codes: Optional[int] = None,
    turboquant_max_visited_pages: Optional[int] = None,
) -> dict:
    hotpot_overlap = corpus.name == "hotpot_overlap"
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
    ivfflat_probe_value = min(list_count, 8) if hotpot_overlap else min(list_count, 4)
    hnsw_ef_search_value = 80 if hotpot_overlap else 40
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
                f"SET LOCAL turboquant.max_visited_codes = {turboquant_max_codes_value}",
                f"SET LOCAL turboquant.max_visited_pages = {turboquant_max_pages_value}",
            ],
            "candidate_slots_bound": turboquant_probe_value * turboquant_oversample_value,
            "query_knobs": {
                "turboquant.probes": turboquant_probe_value,
                "turboquant.oversample_factor": turboquant_oversample_value,
                "turboquant.max_visited_codes": turboquant_max_codes_value,
                "turboquant.max_visited_pages": turboquant_max_pages_value,
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
                f"SET LOCAL turboquant.max_visited_codes = {turboquant_max_codes_value}",
                f"SET LOCAL turboquant.max_visited_pages = {turboquant_max_pages_value}",
            ],
            "candidate_slots_bound": turboquant_probe_value * turboquant_oversample_value,
            "query_knobs": {
                "turboquant.probes": turboquant_probe_value,
                "turboquant.oversample_factor": turboquant_oversample_value,
                "turboquant.max_visited_codes": turboquant_max_codes_value,
                "turboquant.max_visited_pages": turboquant_max_pages_value,
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
            "opclass": "vector_cosine_ops",
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
    turboquant_probes: Optional[int],
    turboquant_oversample_factor: Optional[int],
    turboquant_max_visited_codes: Optional[int],
    turboquant_max_visited_pages: Optional[int],
) -> tuple[float, int, int, dict]:
    spec = method_spec(
        method,
        corpus,
        turboquant_probes,
        turboquant_oversample_factor,
        turboquant_max_visited_codes,
        turboquant_max_visited_pages,
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


def default_scan_stats(method: str) -> dict:
    if method == "turboquant_flat":
        mode = "flat"
        score_mode = "code_domain"
        score_kernel = "scalar"
    elif method == "turboquant_ivf":
        mode = "ivf"
        score_mode = "code_domain"
        score_kernel = "scalar"
    elif method == "turboquant_bitmap":
        mode = "bitmap"
        score_mode = "bitmap_filter"
        score_kernel = "none"
    else:
        mode = "none"
        score_mode = "none"
        score_kernel = "none"

    return {
        "mode": mode,
        "score_mode": score_mode,
        "score_kernel": score_kernel,
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
        "decoded_vector_count": 0,
        "page_prune_count": 0,
        "early_stop_count": 0,
    }


def aggregate_scan_stats(scan_stats: list[dict], fallback_method: str) -> dict:
    if not scan_stats:
        return default_scan_stats(fallback_method)

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
        "decoded_vector_count",
        "page_prune_count",
        "early_stop_count",
    )
    aggregated = {
        "mode": scan_stats[0].get("mode", default_scan_stats(fallback_method)["mode"]),
        "score_mode": scan_stats[0].get("score_mode", default_scan_stats(fallback_method)["score_mode"]),
        "score_kernel": scan_stats[0].get("score_kernel", default_scan_stats(fallback_method)["score_kernel"]),
    }
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
    live_count = int(index_metadata.get("live_count", index_metadata.get("heap_live_rows", 0) or 0))
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
) -> dict:
    query_sql = ";\n".join(
        query_setup
        + [
            (
                "SELECT coalesce(json_agg(id ORDER BY id), '[]'::json)::text "
                f"FROM (SELECT id FROM {table_name} "
                f"ORDER BY embedding <=> '{vector_literal(query_vector)}'::vector LIMIT {limit}) ranked"
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
) -> str:
    helper_candidate_limit = effective_candidate_limit(limit, requested_candidate_limit)
    return ";\n".join(
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


def turboquant_single_batch_rerank_ids_sql(
    table_name: str,
    query_vector: tuple[float, ...],
    limit: int,
    query_setup: list[str],
    requested_candidate_limit: Optional[int],
) -> str:
    helper_candidate_limit = effective_candidate_limit(limit, requested_candidate_limit)
    query_literal = vector_literal(query_vector)
    session_setup = [statement for statement in query_setup if not statement.startswith("SET LOCAL turboquant.")]
    return ";\n".join(
        ["BEGIN"]
        + session_setup
        + [
            (
                "DO $$ "
                "DECLARE resolved record; "
                "BEGIN "
                "SELECT probes, oversample_factor, max_visited_codes, max_visited_pages "
                "INTO resolved "
                f"FROM tq_resolve_query_knobs({helper_candidate_limit}, {limit}, NULL, NULL); "
                "PERFORM set_config('turboquant.probes', resolved.probes::text, true); "
                "PERFORM set_config('turboquant.oversample_factor', resolved.oversample_factor::text, true); "
                "PERFORM set_config('turboquant.max_visited_codes', resolved.max_visited_codes::text, true); "
                "PERFORM set_config('turboquant.max_visited_pages', resolved.max_visited_pages::text, true); "
                "END $$ LANGUAGE plpgsql"
            ),
            (
                "WITH approx_scan AS MATERIALIZED ("
                f"SELECT id, embedding, (embedding <=> '{query_literal}'::vector) AS approximate_distance "
                f"FROM {table_name} "
                f"ORDER BY embedding <=> '{query_literal}'::vector "
                f"LIMIT {helper_candidate_limit}"
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
                "row_number() OVER (ORDER BY candidate_embedding <=> "
                f"'{query_literal}'::vector, candidate_key)::integer AS exact_rank "
                "FROM approx"
                ") "
                "SELECT json_build_object("
                "'reranked_ids', "
                f"coalesce((SELECT json_agg(candidate_id::int ORDER BY exact_rank) FROM reranked WHERE exact_rank <= {limit}), '[]'::json), "
                "'approx_candidate_ids', "
                "coalesce((SELECT json_agg(candidate_key ORDER BY approximate_rank) FROM approx), '[]'::json)"
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
    method: str = "turboquant_ivf",
) -> tuple[list[int], dict, list[int]]:
    query_sql = turboquant_single_batch_rerank_ids_sql(
        table_name,
        query_vector,
        limit,
        query_setup,
        requested_candidate_limit,
    )
    rows = query_psql_commands(
        base_cmd,
        query_sql + ";\nSELECT tq_last_scan_stats()::text;\nCOMMIT;",
    )
    payload = json.loads(rows[0]) if rows else {}
    query_ids = list(payload.get("reranked_ids", []))
    approx_candidate_ids = list(payload.get("approx_candidate_ids", []))
    scan_stats = json.loads(rows[1]) if len(rows) > 1 else default_scan_stats(method)
    return query_ids, scan_stats, approx_candidate_ids


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
) -> list[int]:
    query_sql = turboquant_rerank_ids_sql(
        table_name,
        query_vector,
        limit,
        query_setup,
        requested_candidate_limit,
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


def comparison_pairs(methods: list[str]) -> list[tuple[str, str]]:
    baselines = [method for method in methods if method.startswith("pgvector_")]
    candidates = [method for method in methods if method.startswith("turboquant_")]
    return [(candidate, baseline) for candidate in candidates for baseline in baselines]


def _report_method_row(scenario: dict) -> dict:
    metrics = scenario["metrics"]
    scan_stats = scenario.get("scan_stats", {})
    query_api = scenario.get("query_api", {})
    return {
        "corpus": scenario["corpus"],
        "method": scenario["method"],
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
    }


def _report_comparison_row(candidate: dict, baseline: dict) -> dict:
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    candidate_scan = candidate.get("scan_stats", {})
    baseline_scan = baseline.get("scan_stats", {})
    candidate_recall = candidate_metrics.get("recall_at_10", candidate_metrics.get("exact_match_fraction", 0.0))
    baseline_recall = baseline_metrics.get("recall_at_10", baseline_metrics.get("exact_match_fraction", 0.0))
    return {
        "corpus": candidate["corpus"],
        "candidate_method": candidate["method"],
        "baseline_method": baseline["method"],
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
        },
    }


def generate_report(payload: dict) -> dict:
    scenarios = payload["scenarios"]
    methods = payload["methods"]
    corpora = payload["corpora"]
    grouped = {(scenario["corpus"], scenario["method"]): scenario for scenario in scenarios}
    method_rows = [_report_method_row(scenario) for scenario in scenarios]
    comparisons = []

    for corpus in corpora:
        for candidate_method, baseline_method in comparison_pairs(methods):
            candidate = grouped.get((corpus, candidate_method))
            baseline = grouped.get((corpus, baseline_method))
            if candidate is None or baseline is None:
                continue
            comparisons.append(_report_comparison_row(candidate, baseline))

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

    measurement_notes = [
        "TurboQuant ordered timing and tq_last_scan_stats() are captured from one SQL batch per repetition in this harness.",
        "Ordered retrieval results are reported with the configured helper contract and candidate limit for each scenario.",
        "Latency, scan work, and footprint are reported separately; this report only states what was measured for the listed corpus/profile/knob matrix.",
    ]
    conclusions = []
    for corpus in corpora:
        turboquant_rows = [row for row in method_rows if row["corpus"] == corpus and row["method"].startswith("turboquant_")]
        baseline_rows = [row for row in method_rows if row["corpus"] == corpus and row["method"].startswith("pgvector_")]
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

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "scenario_count": len(scenarios),
            "methods": methods,
            "corpora": corpora,
            "query_modes": sorted({scenario.get("query_mode", "ordered_rerank") for scenario in scenarios}),
        },
        "method_rows": method_rows,
        "comparisons": comparisons,
        "measurement_notes": measurement_notes,
        "conclusions": conclusions,
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
            "- Columns: corpus, method, recall_at_10, p50_ms, p95_ms, footprint_bytes, visited_code_count, visited_page_count, selected_live_count, selected_page_count, score_kernel, query_helper",
            "",
        ]
    )
    for row in report["method_rows"]:
        lines.append(
            f"- {row['corpus']} | {row['method']} | recall_at_10={_report_fmt(row['recall_at_10'])} | "
            f"p50_ms={_report_fmt(row['p50_ms'])} | p95_ms={_report_fmt(row['p95_ms'])} | "
            f"footprint_bytes={_report_fmt(row['footprint_bytes'])} | visited_code_count={_report_fmt(row['visited_code_count'])} | "
            f"visited_page_count={_report_fmt(row['visited_page_count'])} | selected_live_count={_report_fmt(row['selected_live_count'])} | "
            f"selected_page_count={_report_fmt(row['selected_page_count'])} | score_kernel={row['score_kernel']} | "
            f"query_helper={row['query_helper']}"
        )
    lines.extend(["", "## Comparisons", ""])

    if not report["comparisons"]:
        lines.append("- No baseline comparisons available for the selected matrix.")
    else:
        for comparison in report["comparisons"]:
            metrics = comparison["metrics"]
            lines.append(
                f"- {comparison['corpus']}: {comparison['candidate_method']} vs {comparison['baseline_method']} "
                f"(recall@10 delta={metrics['recall_at_10_delta']}, p95 delta={metrics['p95_ms_delta']} ms, "
                f"build delta={metrics['build_seconds_delta']} s, size delta={metrics['index_size_bytes_delta']} B, "
                f"WAL delta={metrics['build_wal_bytes_delta']} B, visited_code_count delta={metrics['visited_code_count_delta']}, "
                f"selected_page_count delta={metrics['selected_page_count_delta']})"
            )

    lines.extend(["", "## Hotpot Conclusions", ""])
    if not report["conclusions"]:
        lines.append("- No corpus-specific conclusions were generated for the selected matrix.")
    else:
        for conclusion in report["conclusions"]:
            lines.append(f"### {conclusion['corpus']}")
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
        f"<td>{_report_fmt(metrics['recall_at_10_delta'])}</td>"
        f"<td>{_report_fmt(metrics['p95_ms_delta'])}</td>"
        f"<td>{_report_fmt(metrics['index_size_bytes_delta'])}</td>"
        f"<td>{_report_fmt(metrics['visited_code_count_delta'])}</td>"
        f"<td>{_report_fmt(metrics['selected_page_count_delta'])}</td>"
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
    turboquant_probes: Optional[int],
    turboquant_oversample_factor: Optional[int],
    turboquant_max_visited_codes: Optional[int],
    turboquant_max_visited_pages: Optional[int],
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
        turboquant_max_visited_codes,
        turboquant_max_visited_pages,
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
    scan_stats_results = []
    candidate_retention_results = []

    if spec.get("query_mode") == "bitmap_filter":
        exact_results = [
            exact_bitmap_ids(corpus, query, query_index % 2, spec["bitmap_threshold"])
            for query_index, query in enumerate(corpus.queries)
        ]

        for query_index, query in enumerate(corpus.queries):
            best_ids = []
            best_latency_ms = None
            best_scan_stats = default_scan_stats(method)
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
                if best_latency_ms is None or elapsed_ms < best_latency_ms:
                    best_latency_ms = elapsed_ms
                    best_ids = query_ids
                    best_scan_stats = query_scan_stats
            approx_results.append(best_ids)
            latencies_ms.append(best_latency_ms or 0.0)
            scan_stats_results.append(best_scan_stats)
    else:
        for query_index, query in enumerate(corpus.queries):
            best_ids = []
            best_latency_ms = None
            best_scan_stats = default_scan_stats(method)
            best_approx_candidate_ids = []
            for _ in range(repetitions):
                started = time.perf_counter()
                if spec["index_method"] == "turboquant":
                    query_ids, query_scan_stats, approx_candidate_ids = query_turboquant_ordered_ids_and_scan_stats(
                        base_cmd,
                        table_name,
                        query,
                        TOP_K_VALUES[-1],
                        spec["query_setup"],
                        requested_rerank_candidate_limit,
                        method=method,
                    )
                else:
                    query_ids = query_top_ids(
                        base_cmd,
                        table_name,
                        query,
                        TOP_K_VALUES[-1],
                        spec["query_setup"],
                        requested_rerank_candidate_limit,
                    )
                    query_scan_stats = default_scan_stats(method)
                    approx_candidate_ids = []
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if best_latency_ms is None or elapsed_ms < best_latency_ms:
                    best_latency_ms = elapsed_ms
                    best_ids = query_ids
                    best_scan_stats = query_scan_stats
                    best_approx_candidate_ids = approx_candidate_ids
            approx_results.append(best_ids)
            latencies_ms.append(best_latency_ms or 0.0)
            scan_stats_results.append(best_scan_stats)
            if spec["index_method"] == "turboquant":
                candidate_retention_results.append(
                    candidate_retention_for_query(exact_results[query_index], best_approx_candidate_ids)
                )

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
    if spec["index_method"] == "turboquant":
        simd_metadata["code_domain_kernel"] = aggregated_scan_stats.get("score_kernel", "scalar")
    else:
        simd_metadata["code_domain_kernel"] = "none"

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
        "benchmark_metadata": scenario_benchmark_metadata(index_metadata),
        "simd": simd_metadata,
        "scan_stats": aggregated_scan_stats,
    }
    if spec.get("query_mode", "ordered_rerank") == "ordered_rerank" and spec["index_method"] == "turboquant":
        scenario["candidate_retention"] = aggregate_candidate_retention(candidate_retention_results)
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
    turboquant_max_visited_codes: Optional[int],
    turboquant_max_visited_pages: Optional[int],
    requested_rerank_candidate_limit: Optional[int],
) -> dict:
    spec = method_spec(
        method,
        corpus,
        turboquant_probes,
        turboquant_oversample_factor,
        turboquant_max_visited_codes,
        turboquant_max_visited_pages,
    )
    simd_metadata = synthetic_simd_metadata()
    scan_stats = default_scan_stats(method)
    scan_stats["score_kernel"] = synthetic_code_domain_kernel(corpus, spec, simd_metadata)
    simd_metadata["code_domain_kernel"] = scan_stats["score_kernel"]

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
        "benchmark_metadata": scenario_benchmark_metadata(synthetic_index_metadata(method, spec, corpus)),
        "simd": simd_metadata,
        "scan_stats": scan_stats,
    }
    if spec.get("query_mode", "ordered_rerank") == "ordered_rerank" and spec["index_method"] == "turboquant":
        scenario["candidate_retention"] = default_candidate_retention()
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
    parser.add_argument("--turboquant-max-visited-codes", type=int)
    parser.add_argument("--turboquant-max-visited-pages", type=int)
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
                        args.turboquant_max_visited_codes,
                        args.turboquant_max_visited_pages,
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
                        args.turboquant_max_visited_codes,
                        args.turboquant_max_visited_pages,
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
