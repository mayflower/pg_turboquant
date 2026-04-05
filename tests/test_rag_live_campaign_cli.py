import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks.rag.runtime_env import maybe_reexec_into_bergen_venv

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "rag" / "run_live_campaign.py"


class RagLiveCampaignCliContractTest(unittest.TestCase):
    def test_runtime_bootstrap_reexecs_live_run_into_bergen_venv_when_modules_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "run_live_campaign.py"
            script_path.write_text("", encoding="utf-8")
            venv_python = script_path.parent / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text("", encoding="utf-8")

            exec_calls = []
            did_reexec = maybe_reexec_into_bergen_venv(
                script_path=script_path,
                argv=[str(script_path), "--output-dir", "/tmp/out"],
                required_modules=("psycopg", "yaml"),
                find_spec=lambda name: None,
                execvpe=lambda executable, argv, env: exec_calls.append((executable, argv, env)),
                current_executable="/usr/bin/python3",
                environ={"PATH": "/usr/bin"},
            )

            self.assertTrue(did_reexec)
            expected_python = str(venv_python.resolve())
            self.assertEqual(exec_calls[0][0], expected_python)
            self.assertEqual(exec_calls[0][1], [expected_python, str(script_path), "--output-dir", "/tmp/out"])

    def test_runtime_bootstrap_keeps_dry_run_in_current_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "run_live_campaign.py"
            script_path.write_text("", encoding="utf-8")
            venv_python = script_path.parent / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text("", encoding="utf-8")

            execvpe = mock.Mock()
            did_reexec = maybe_reexec_into_bergen_venv(
                script_path=script_path,
                argv=[str(script_path), "--dry-run", "--output-dir", "/tmp/out"],
                required_modules=("psycopg", "yaml"),
                find_spec=lambda name: None,
                execvpe=execvpe,
                current_executable="/usr/bin/python3",
                environ={"PATH": "/usr/bin"},
            )

            self.assertFalse(did_reexec)
            execvpe.assert_not_called()

    def test_dry_run_prints_resolved_plan_without_live_dependencies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--output-dir",
                    tmpdir,
                    "--dry-run",
                    "--datasets",
                    "kilt_nq",
                    "kilt_hotpotqa",
                    "popqa",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(result.stdout)
            self.assertEqual(payload["plan"]["campaign_kind"], "rag_benchmark")
            self.assertEqual(payload["plan"]["datasets"], ["kilt_nq", "kilt_hotpotqa", "popqa"])
            self.assertEqual(len(payload["plan"]["system_variants"]), 6)
            self.assertEqual(payload["generator_name"], "oracle_answer")
            self.assertEqual(payload["retriever_name"], "bge-small-en-v1.5")


if __name__ == "__main__":
    unittest.main()
