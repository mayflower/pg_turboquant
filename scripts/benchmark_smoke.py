#!/usr/bin/env -S uv run python
import argparse
import subprocess
import sys
from pathlib import Path


def profile_for_rows(rows: int) -> str:
    if rows <= 128:
        return "tiny"
    if rows <= 512:
        return "quick"
    if rows <= 2048:
        return "medium"
    return "full"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper around scripts/benchmark_suite.py",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True)
    parser.add_argument("--dbname", default="postgres")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.rows <= 0:
        raise SystemExit("--rows must be positive")

    suite_path = Path(__file__).with_name("benchmark_suite.py")
    cmd = [
        "uv",
        "run",
        "python",
        str(suite_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dbname",
        args.dbname,
        "--profile",
        profile_for_rows(args.rows),
        "--corpus",
        "normalized_dense",
        "--methods",
        "turboquant_flat",
    ]
    if args.output:
        cmd.extend(["--output", args.output])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"benchmark_smoke.py failed with exit code {exc.returncode}\n")
        raise
