import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.rag.retrieval_eval import QueryEvaluation
from benchmarks.rag.operational_metrics import ApproximateQueryCost, QueryOperationalMetrics
from benchmarks.rag.rerank_eval import (
    ExactRerankPlan,
    build_exact_rerank_sql,
    export_two_stage_retrieval_run,
    rerank_results,
)


class RagExactRerankContractTest(unittest.TestCase):
    def test_rerank_query_generation_is_explicit_and_reproducible(self):
        plan = build_exact_rerank_sql(
            table_name="rag_passages",
            id_column="passage_id",
            text_column="passage_text",
            embedding_column="embedding",
            metric="cosine",
            query_vector=[1.0, 0.0, 0.0],
            candidate_ids=["p3", "p1", "p2"],
            final_k=2,
        )

        self.assertIsInstance(plan, ExactRerankPlan)
        self.assertIn("WITH candidate_pool AS", plan.sql)
        self.assertIn("unnest(%s::text[])", plan.sql)
        self.assertIn("ORDER BY exact_score ASC", plan.sql)
        self.assertEqual(plan.params[0], ["p3", "p1", "p2"])
        self.assertEqual(plan.params[-1], 2)
        self.assertEqual(len(plan.sql_template_hash), 64)

    def test_post_rerank_order_differs_on_crafted_example(self):
        approx_results = [
            {"id": "p3", "score": 0.01, "text": "wrong first"},
            {"id": "p1", "score": 0.02, "text": "best exact"},
            {"id": "p2", "score": 0.03, "text": "middle"},
        ]
        exact_scores = {"p1": 0.001, "p2": 0.010, "p3": 0.020}

        reranked = rerank_results(approx_results, exact_scores, final_k=2)

        self.assertEqual([item["id"] for item in reranked], ["p1", "p2"])
        self.assertNotEqual([item["id"] for item in reranked], [item["id"] for item in approx_results[:2]])

    def test_export_contains_pre_and_post_rerank_metrics(self):
        pre_queries = [
            QueryEvaluation(
                query_id="q1",
                retrieved_ids=["p3", "p1", "p2"],
                relevant_ids=["p1"],
                evidence_ids=[],
                latency_ms=8.0,
            )
        ]
        post_queries = [
            QueryEvaluation(
                query_id="q1",
                retrieved_ids=["p1", "p2"],
                relevant_ids=["p1"],
                evidence_ids=[],
                latency_ms=12.0,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = export_two_stage_retrieval_run(
                output_dir=Path(tmpdir),
                run_metadata={
                    "run_id": "rerank-run",
                    "result_kind": "retrieval_only",
                    "candidate_pool_size": 3,
                    "rerank_enabled": True,
                },
                pre_rerank_queries=pre_queries,
                post_rerank_queries=post_queries,
                operational_metrics=[
                    QueryOperationalMetrics(
                        retrieval_latency_ms=8.0,
                        rerank_latency_ms=4.0,
                        approximate_query_costs=(
                            ApproximateQueryCost(
                                name="candidate_pool_size",
                                unit="count",
                                value=3.0,
                            ),
                        ),
                    )
                ],
                ks=(1, 2),
            )

            payload = json.loads((Path(tmpdir) / artifacts["json"]).read_text(encoding="utf-8"))
            self.assertIn("pre_rerank", payload["metrics"])
            self.assertIn("post_rerank", payload["metrics"])
            self.assertEqual(payload["run_metadata"]["candidate_pool_size"], 3)
            self.assertTrue(payload["run_metadata"]["rerank_enabled"])
            self.assertEqual(payload["operational_summary"]["latency_ms"]["retrieval"]["p50"], 8.0)
            self.assertEqual(payload["operational_summary"]["latency_ms"]["rerank"]["p50"], 4.0)
            self.assertEqual(payload["operational_summary"]["latency_ms"]["total"]["p50"], 12.0)
            self.assertEqual(
                payload["operational_summary"]["approximate_query_costs"]["candidate_pool_size"]["unit"],
                "count",
            )


if __name__ == "__main__":
    unittest.main()
