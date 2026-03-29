from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATASET_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "datasets"

REQUIRED_TOP_LEVEL_KEYS = {
    "dataset_id",
    "display_name",
    "stress_focus",
    "source",
    "retrieval_profile",
    "evidence",
    "answer_metrics",
    "stable_passage_id_fields",
    "capabilities",
}


def dataset_config_paths() -> list[Path]:
    return sorted(DATASET_CONFIG_DIR.glob("*.json"))


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ValueError(f"dataset config missing required keys: {', '.join(missing)}")

    retrieval = payload["retrieval_profile"]
    evidence = payload["evidence"]
    source = payload["source"]
    if retrieval["top_k_default"] <= 0:
        raise ValueError("retrieval_profile.top_k_default must be positive")
    if "kind" not in source or "split" not in source:
        raise ValueError("source must define kind and split")
    if "enabled" not in evidence:
        raise ValueError("evidence must define enabled")
    if not payload["stable_passage_id_fields"]:
        raise ValueError("stable_passage_id_fields must not be empty")
    return payload


def resolve_benchmark_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": config["dataset_id"],
        "display_name": config["display_name"],
        "source": {
            "kind": config["source"]["kind"],
            "dataset": config["source"]["dataset"],
            "split": config["source"]["split"],
        },
        "retrieval_top_k": config["retrieval_profile"]["top_k_default"],
        "rerank_top_k": config["retrieval_profile"].get("rerank_top_k_default"),
        "evidence_enabled": bool(config["evidence"]["enabled"]),
        "answer_metrics": list(config["answer_metrics"]),
        "stable_passage_id_fields": list(config["stable_passage_id_fields"]),
        "stress_focus": config["stress_focus"],
    }


def load_primary_dataset_pack() -> dict[str, dict[str, Any]]:
    return {
        config["dataset_id"]: config
        for config in (load_dataset_config(path) for path in dataset_config_paths())
    }
