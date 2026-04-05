import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "rag" / "run_comparative_campaign.py"
CONFIG = ROOT / "benchmarks" / "rag" / "configs" / "comparative" / "toy_campaign.json"


class RagComparativeCampaignCliContractTest(unittest.TestCase):
    def test_fixture_backed_cli_emits_campaign_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--config",
                    str(CONFIG),
                    "--output-dir",
                    tmpdir,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(result.stdout)
            self.assertEqual(payload["campaign_kind"], "rag_benchmark")
            self.assertIn("campaign_json", payload["artifacts"])
            self.assertIn("report_markdown", payload["artifacts"])
            self.assertIn("report_html", payload["artifacts"])

            campaign_json = Path(tmpdir) / payload["artifacts"]["campaign_json"]
            report_markdown = Path(tmpdir) / payload["artifacts"]["report_markdown"]
            report_html = Path(tmpdir) / payload["artifacts"]["report_html"]

            self.assertTrue(campaign_json.exists())
            self.assertTrue(report_markdown.exists())
            self.assertTrue(report_html.exists())

            report_payload = json.loads(campaign_json.read_text(encoding="utf-8"))
            self.assertIn("tables", report_payload)
            self.assertIn("report", report_payload)
            self.assertGreater(len(report_payload["tables"]["retrieval_benchmark"]), 0)
            self.assertGreater(len(report_payload["tables"]["end_to_end_benchmark"]), 0)
            self.assertGreater(len(report_payload["tables"]["retrieval_diagnostics"]), 0)
            self.assertIn("RAG Benchmark Outcome", report_html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
