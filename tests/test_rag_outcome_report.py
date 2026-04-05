import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.rag.outcome_report import build_outcome_summary, write_outcome_html


class RagOutcomeReportContractTest(unittest.TestCase):
    def make_campaign_payload(self, dataset_id: str, tq_latency: float, hnsw_latency: float, ivf_latency: float):
        return {
            "generated_at": "2026-03-29T12:00:00+00:00",
            "plan": {"datasets": [dataset_id], "campaign_kind": "rag_benchmark"},
            "tables": {
                "retrieval_benchmark": [
                    {
                        "dataset_id": dataset_id,
                        "system_id": "pg_turboquant_approx",
                        "system_label": "pg_turboquant (approx)",
                        "method_id": "pg_turboquant_approx",
                        "retriever_backend": "pg_turboquant",
                        "retrieval_mode": "approx",
                        "rerank_enabled": False,
                        "recall@10": 0.95,
                        "mrr@10": 0.90,
                        "ndcg@10": 0.92,
                        "hit_rate@10": 0.99,
                        "evidence_coverage@10": 0.91,
                        "latency_p50_ms": tq_latency - 4.0,
                        "latency_p95_ms": tq_latency,
                        "footprint_bytes": 1000,
                    },
                    {
                        "dataset_id": dataset_id,
                        "system_id": "pgvector_hnsw_approx",
                        "system_label": "pgvector_hnsw (approx)",
                        "method_id": "pgvector_hnsw_approx",
                        "retriever_backend": "pgvector_hnsw",
                        "retrieval_mode": "approx",
                        "rerank_enabled": False,
                        "recall@10": 0.94,
                        "mrr@10": 0.88,
                        "ndcg@10": 0.90,
                        "hit_rate@10": 0.98,
                        "evidence_coverage@10": 0.89,
                        "latency_p50_ms": hnsw_latency - 5.0,
                        "latency_p95_ms": hnsw_latency,
                        "footprint_bytes": 4000,
                    },
                    {
                        "dataset_id": dataset_id,
                        "system_id": "pgvector_ivfflat_approx",
                        "system_label": "pgvector_ivfflat (approx)",
                        "method_id": "pgvector_ivfflat_approx",
                        "retriever_backend": "pgvector_ivfflat",
                        "retrieval_mode": "approx",
                        "rerank_enabled": False,
                        "recall@10": 0.93,
                        "mrr@10": 0.86,
                        "ndcg@10": 0.88,
                        "hit_rate@10": 0.97,
                        "evidence_coverage@10": 0.87,
                        "latency_p50_ms": ivf_latency - 3.0,
                        "latency_p95_ms": ivf_latency,
                        "footprint_bytes": 3000,
                    },
                ],
                "end_to_end_benchmark": [],
                "retrieval_diagnostics": [
                    {
                        "dataset_id": dataset_id,
                        "system_id": "pg_turboquant_approx",
                        "score_mode": "code_domain",
                        "avg_visited_code_count": 64.0,
                        "avg_visited_page_count": 8.0,
                        "avg_selected_live_count": 96.0,
                        "avg_selected_page_count": 12.0,
                    }
                ],
            },
            "report": {},
        }

    def test_outcome_summary_reports_generic_system_rows_and_dataset_leaders(self):
        payloads = [
            self.make_campaign_payload("popqa", tq_latency=20.0, hnsw_latency=50.0, ivf_latency=40.0),
            self.make_campaign_payload("kilt_nq", tq_latency=10.0, hnsw_latency=25.0, ivf_latency=30.0),
        ]

        summary = build_outcome_summary(payloads)

        self.assertEqual(summary["dataset_count"], 2)
        self.assertEqual(summary["system_count"], 3)
        self.assertEqual(len(summary["system_rows"]), 6)
        self.assertEqual(summary["retrieval_leaders"][0]["metric"], "recall@10")
        self.assertEqual(summary["retrieval_leaders"][0]["system_id"], "pg_turboquant_approx")
        self.assertEqual(summary["latency_leaders"][0]["system_id"], "pg_turboquant_approx")
        self.assertEqual(summary["footprint_leaders"][0]["system_id"], "pg_turboquant_approx")
        self.assertEqual(summary["system_rows"][0]["diagnostics"]["score_mode"], "code_domain")

    def test_write_outcome_html_emits_generic_benchmark_sections(self):
        payloads = [self.make_campaign_payload("popqa", tq_latency=20.0, hnsw_latency=50.0, ivf_latency=40.0)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "outcome.html"
            artifact = write_outcome_html(output_path, payloads, source_labels=["popqa-small-live"])

            self.assertEqual(artifact["output_html"], "outcome.html")
            html = output_path.read_text(encoding="utf-8")

        self.assertIn("RAG Benchmark Outcome", html)
        self.assertIn("Measured Comparison Scope", html)
        self.assertIn("Retrieval Systems", html)
        self.assertIn("Dataset Leaders", html)
        self.assertIn("popqa-small-live", html)
        self.assertIn("pgvector_hnsw (approx)", html)
        self.assertIn("evidence_coverage@10", html)
        self.assertIn("score_mode", html)
        self.assertIn("code_domain", html)
        self.assertNotIn("should be no slower", html)
        self.assertNotIn("TurboQuant method:", html)


if __name__ == "__main__":
    unittest.main()
