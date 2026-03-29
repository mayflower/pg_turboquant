import csv
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.rag.retrieval_eval import (
    QueryEvaluation,
    compute_retrieval_metrics,
    export_retrieval_run,
)


class RagRetrievalEvalContractTest(unittest.TestCase):
    def test_metric_computation_on_small_known_example(self):
        queries = [
            QueryEvaluation(
                query_id="q1",
                retrieved_ids=["d1", "d3", "d2"],
                relevant_ids=["d2", "d3"],
                evidence_ids=["d3"],
                latency_ms=10.0,
            ),
            QueryEvaluation(
                query_id="q2",
                retrieved_ids=["x1", "x2", "x3"],
                relevant_ids=["x9"],
                evidence_ids=[],
                latency_ms=20.0,
            ),
        ]

        metrics = compute_retrieval_metrics(queries, ks=(1, 3))

        self.assertAlmostEqual(metrics["recall@1"], 0.0)
        self.assertAlmostEqual(metrics["recall@3"], 0.5)
        self.assertAlmostEqual(metrics["mrr@1"], 0.0)
        self.assertAlmostEqual(metrics["mrr@3"], 0.25)
        self.assertAlmostEqual(metrics["ndcg@1"], 0.0)
        self.assertGreater(metrics["ndcg@3"], 0.3)
        self.assertAlmostEqual(metrics["hit_rate@1"], 0.0)
        self.assertAlmostEqual(metrics["hit_rate@3"], 0.5)
        self.assertAlmostEqual(metrics["evidence_coverage@3"], 0.5)
        self.assertEqual(metrics["latency_p50_ms"], 15.0)
        self.assertEqual(metrics["latency_p95_ms"], 19.5)
        self.assertEqual(metrics["latency_p99_ms"], 19.9)
        self.assertAlmostEqual(metrics["throughput_qps"], 66.6666666667, places=4)

    def test_json_csv_and_markdown_exports_are_valid(self):
        queries = [
            QueryEvaluation(
                query_id="q1",
                retrieved_ids=["d1", "d2"],
                relevant_ids=["d2"],
                evidence_ids=["d2"],
                latency_ms=12.0,
            )
        ]
        metrics = compute_retrieval_metrics(queries, ks=(1, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            artifacts = export_retrieval_run(
                output_dir=output_dir,
                run_metadata={
                    "run_id": "tiny-run",
                    "result_kind": "retrieval_only",
                    "backend": "pg_turboquant",
                },
                metrics=metrics,
            )

            json_payload = json.loads((output_dir / artifacts["json"]).read_text(encoding="utf-8"))
            self.assertEqual(json_payload["run_metadata"]["result_kind"], "retrieval_only")
            self.assertEqual(json_payload["metrics"]["recall@2"], 1.0)

            with (output_dir / artifacts["csv"]).open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["metric"], "recall@1")
            self.assertIn("value", rows[0])

            markdown = (output_dir / artifacts["markdown"]).read_text(encoding="utf-8")
            self.assertIn("| Metric | Value |", markdown)
            self.assertIn("retrieval_only", markdown)


if __name__ == "__main__":
    unittest.main()
