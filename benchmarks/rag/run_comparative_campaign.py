#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.rag.campaign_report import (
        build_comparative_campaign_plan,
        run_comparative_campaign,
    )
else:
    from .campaign_report import build_comparative_campaign_plan, run_comparative_campaign


def load_campaign_fixture_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"campaign_kind", "datasets", "generator_id", "execution"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"comparative campaign config missing required keys: {', '.join(missing)}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fixture-backed RAG benchmark from local fixtures")
    parser.add_argument("--config", required=True, help="Path to comparative campaign JSON config")
    parser.add_argument("--output-dir", required=True, help="Directory for generated campaign artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Only print the resolved campaign plan")
    args = parser.parse_args()

    config = load_campaign_fixture_config(args.config)
    plan = build_comparative_campaign_plan(
        dataset_ids=config["datasets"],
        generator_id=config["generator_id"],
    )

    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    execution = dict(config["execution"])
    if execution.get("mode") != "fixture_payload":
        raise ValueError(f"unsupported comparative campaign mode: {execution.get('mode')}")

    fixture_path = Path(execution["fixture_path"])
    if not fixture_path.is_absolute():
        fixture_path = Path(args.config).resolve().parent / fixture_path
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    def retrieval_runner(scenario: dict[str, object]) -> dict[str, object]:
        key = f"{scenario['dataset_id']}:{scenario['system_id']}"
        payload = fixture["retrieval"][key]
        return {
            "run_metadata": {
                "dataset_id": scenario["dataset_id"],
                "method_id": scenario["system_id"],
                "result_kind": "retrieval_only",
                "footprint_bytes": payload["footprint_bytes"],
            },
            "metrics": dict(payload["metrics"]),
        }

    def end_to_end_runner(
        scenario: dict[str, object],
        retrieval_result: dict[str, object],
    ) -> dict[str, object]:
        key = f"{scenario['dataset_id']}:{scenario['system_id']}"
        payload = fixture["end_to_end"][key]
        return {
            "run_metadata": {
                "dataset_id": scenario["dataset_id"],
                "method_id": scenario["system_id"],
                "result_kind": "end_to_end",
                "generator_id": config["generator_id"],
            },
            "retrieval_summary": dict(retrieval_result["metrics"]),
            "answer_metrics": dict(payload["answer_metrics"]),
            "operational_summary": {
                "latency_ms": {
                    "total": {
                        "p50": payload["total_latency_p50_ms"],
                        "p95": payload["total_latency_p95_ms"],
                        "p99": payload["total_latency_p99_ms"],
                    }
                }
            },
        }

    artifacts = run_comparative_campaign(
        output_dir=Path(args.output_dir),
        plan=plan,
        retrieval_runner=retrieval_runner,
        end_to_end_runner=end_to_end_runner,
    )
    print(
        json.dumps(
            {
                "campaign_kind": plan["campaign_kind"],
                "datasets": plan["datasets"],
                "generator_id": plan["generator_id"],
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
