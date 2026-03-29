#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.rag.outcome_report import load_campaign_payload, write_outcome_html
else:
    from .outcome_report import load_campaign_payload, write_outcome_html


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an aggregated HTML outcome report from RAG campaign JSON files")
    parser.add_argument(
        "--campaign-json",
        action="append",
        dest="campaign_jsons",
        required=True,
        help="Path to a rag-campaign.json artifact. Repeat for multiple campaign runs.",
    )
    parser.add_argument("--output", required=True, help="Target HTML path")
    args = parser.parse_args()

    campaign_paths = [Path(item).resolve() for item in args.campaign_jsons]
    payloads = [load_campaign_payload(path) for path in campaign_paths]
    labels = [str(path.parent.name) for path in campaign_paths]
    artifact = write_outcome_html(args.output, payloads, source_labels=labels)
    print(
        json.dumps(
            {
                "campaign_jsons": [str(path) for path in campaign_paths],
                "output_html": str(Path(args.output).resolve()),
                "artifact": artifact,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
