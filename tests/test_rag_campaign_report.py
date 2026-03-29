import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.rag.campaign_report import (
    COMPARATIVE_METHOD_VARIANTS,
    build_comparative_campaign_plan,
    run_comparative_campaign,
)


class RagCampaignReportContractTest(unittest.TestCase):
    def test_campaign_plan_covers_required_methods_and_datasets(self):
        plan = build_comparative_campaign_plan(
            dataset_ids=["kilt_nq", "kilt_hotpotqa", "popqa"],
            generator_id="fixed-debug-generator",
        )

        self.assertEqual(plan["campaign_kind"], "comparative_rag")
        self.assertEqual(plan["datasets"], ["kilt_nq", "kilt_hotpotqa", "popqa"])
        self.assertEqual(plan["generator_id"], "fixed-debug-generator")
        self.assertEqual(
            [variant["method_id"] for variant in plan["method_variants"]],
            COMPARATIVE_METHOD_VARIANTS,
        )
        self.assertEqual(
            len(plan["retrieval_scenarios"]),
            len(COMPARATIVE_METHOD_VARIANTS) * 3,
        )
        self.assertEqual(
            len(plan["end_to_end_scenarios"]),
            len(COMPARATIVE_METHOD_VARIANTS) * 3,
        )

    def test_tiny_campaign_smoke_emits_expected_artifacts_and_report_schema(self):
        plan = build_comparative_campaign_plan(
            dataset_ids=["kilt_nq"],
            generator_id="fixed-debug-generator",
        )

        def fake_retrieval_runner(scenario):
            rank = COMPARATIVE_METHOD_VARIANTS.index(scenario["method_id"]) + 1
            return {
                "run_metadata": {
                    "dataset_id": scenario["dataset_id"],
                    "method_id": scenario["method_id"],
                    "result_kind": "retrieval_only",
                    "footprint_bytes": 1000 * rank,
                },
                "metrics": {
                    "recall@10": 0.9 - (rank * 0.01),
                    "latency_p95_ms": 12.0 + rank,
                    "latency_p50_ms": 10.0 + rank,
                },
            }

        def fake_end_to_end_runner(scenario, retrieval_result):
            rank = COMPARATIVE_METHOD_VARIANTS.index(scenario["method_id"]) + 1
            return {
                "run_metadata": {
                    "dataset_id": scenario["dataset_id"],
                    "method_id": scenario["method_id"],
                    "result_kind": "end_to_end",
                    "generator_id": "fixed-debug-generator",
                },
                "retrieval_summary": retrieval_result["metrics"],
                "answer_metrics": {
                    "answer_exact_match": 0.7 - (rank * 0.01),
                    "answer_f1": 0.8 - (rank * 0.01),
                },
                "operational_summary": {
                    "latency_ms": {
                        "total": {
                            "p50": 20.0 + rank,
                            "p95": 25.0 + rank,
                            "p99": 30.0 + rank,
                        }
                    }
                },
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = run_comparative_campaign(
                output_dir=Path(tmpdir),
                plan=plan,
                retrieval_runner=fake_retrieval_runner,
                end_to_end_runner=fake_end_to_end_runner,
            )

            payload = json.loads((Path(tmpdir) / artifacts["campaign_json"]).read_text(encoding="utf-8"))
            self.assertIn("plan", payload)
            self.assertIn("retrieval_only", payload["tables"])
            self.assertIn("end_to_end", payload["tables"])
            self.assertIn("report", payload)
            self.assertIn("summary", payload["report"])
            self.assertIn("retrieval_findings", payload["report"])
            self.assertIn("answer_findings", payload["report"])
            self.assertIn("latency_findings", payload["report"])
            self.assertIn("footprint_findings", payload["report"])
            self.assertIn("metric_validity_caveats", payload["report"])
            self.assertIn("report_markdown", artifacts)
            self.assertIn("report_html", artifacts)
            self.assertIn("retrieval_csv", artifacts)
            self.assertIn("end_to_end_csv", artifacts)

            retrieval_rows = payload["tables"]["retrieval_only"]
            end_to_end_rows = payload["tables"]["end_to_end"]
            self.assertEqual(len(retrieval_rows), len(COMPARATIVE_METHOD_VARIANTS))
            self.assertEqual(len(end_to_end_rows), len(COMPARATIVE_METHOD_VARIANTS))
            self.assertEqual(retrieval_rows[0]["dataset_id"], "kilt_nq")
            self.assertIn("recall@10", retrieval_rows[0])
            self.assertIn("answer_exact_match", end_to_end_rows[0])
            self.assertIn("total_latency_p95_ms", end_to_end_rows[0])

            markdown = (Path(tmpdir) / artifacts["report_markdown"]).read_text(encoding="utf-8")
            report_html = (Path(tmpdir) / artifacts["report_html"]).read_text(encoding="utf-8")
            self.assertIn("Retrieval-Only Comparison", markdown)
            self.assertIn("End-to-End Comparison", markdown)
            self.assertIn("Metric Validity Caveats", markdown)
            self.assertIn("TurboQuant Outcome", report_html)
            self.assertIn("Expected TurboQuant Profile", report_html)
            self.assertIn("pgvector_hnsw_approx", report_html)


if __name__ == "__main__":
    unittest.main()
