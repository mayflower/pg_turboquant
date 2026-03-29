import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.rag.outcome_report import build_outcome_summary, write_outcome_html


class RagOutcomeReportContractTest(unittest.TestCase):
    def make_campaign_payload(self, dataset_id: str, tq_latency: float, hnsw_latency: float, ivf_latency: float):
        return {
            "generated_at": "2026-03-29T12:00:00+00:00",
            "plan": {"datasets": [dataset_id], "campaign_kind": "comparative_rag"},
            "tables": {
                "retrieval_only": [
                    {
                        "dataset_id": dataset_id,
                        "method_id": "pg_turboquant_approx",
                        "backend_family": "pg_turboquant",
                        "rerank_enabled": False,
                        "recall@10": 0.95,
                        "latency_p95_ms": tq_latency,
                        "footprint_bytes": 1000,
                    },
                    {
                        "dataset_id": dataset_id,
                        "method_id": "pgvector_hnsw_approx",
                        "backend_family": "pgvector_hnsw",
                        "rerank_enabled": False,
                        "recall@10": 0.95,
                        "latency_p95_ms": hnsw_latency,
                        "footprint_bytes": 4000,
                    },
                    {
                        "dataset_id": dataset_id,
                        "method_id": "pgvector_ivfflat_approx",
                        "backend_family": "pgvector_ivfflat",
                        "rerank_enabled": False,
                        "recall@10": 0.95,
                        "latency_p95_ms": ivf_latency,
                        "footprint_bytes": 3000,
                    },
                ],
                "end_to_end": [],
            },
            "report": {},
        }

    def test_outcome_summary_compares_turboquant_against_approx_pgvector_baselines(self):
        payloads = [
            self.make_campaign_payload("popqa", tq_latency=20.0, hnsw_latency=50.0, ivf_latency=40.0),
            self.make_campaign_payload("kilt_nq", tq_latency=10.0, hnsw_latency=25.0, ivf_latency=30.0),
        ]

        summary = build_outcome_summary(payloads)

        self.assertEqual(summary["dataset_count"], 2)
        self.assertEqual(summary["comparison_count"], 4)
        self.assertEqual(summary["footprint_expectation_pass_count"], 4)
        self.assertEqual(summary["latency_expectation_pass_count"], 4)
        self.assertEqual(summary["overall_expectation_pass_count"], 4)
        self.assertEqual(summary["comparisons"][0]["baseline_method_id"], "pgvector_hnsw_approx")
        self.assertAlmostEqual(summary["comparisons"][0]["footprint_ratio_vs_turboquant"], 4.0)
        self.assertAlmostEqual(summary["comparisons"][0]["latency_ratio_vs_turboquant"], 2.5)

    def test_write_outcome_html_emits_expectation_and_dataset_comparison_sections(self):
        payloads = [self.make_campaign_payload("popqa", tq_latency=20.0, hnsw_latency=50.0, ivf_latency=40.0)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "outcome.html"
            artifact = write_outcome_html(output_path, payloads, source_labels=["popqa-small-live"])

            self.assertEqual(artifact["output_html"], "outcome.html")
            html = output_path.read_text(encoding="utf-8")

        self.assertIn("TurboQuant Outcome", html)
        self.assertIn("Expected TurboQuant Profile", html)
        self.assertIn("popqa-small-live", html)
        self.assertIn("pgvector_hnsw_approx", html)
        self.assertIn("4.00x", html)
        self.assertIn("2.50x", html)
        self.assertIn("PASS", html)


if __name__ == "__main__":
    unittest.main()
