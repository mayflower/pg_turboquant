import unittest

from benchmarks.rag.operational_metrics import (
    ApproximateQueryCost,
    QueryOperationalMetrics,
    estimate_token_count,
    summarize_query_operational_metrics,
)


class RagOperationalMetricsContractTest(unittest.TestCase):
    def test_token_budget_and_stage_latency_summary_are_deterministic(self):
        self.assertEqual(estimate_token_count("Alpha beta\ngamma"), 3)

        summary = summarize_query_operational_metrics(
            [
                QueryOperationalMetrics(
                    retrieval_latency_ms=10.0,
                    rerank_latency_ms=2.0,
                    generator_latency_ms=30.0,
                    retrieved_context_tokens=9,
                    prompt_tokens=14,
                    prompt_context_count=2,
                    approximate_query_costs=(
                        ApproximateQueryCost(
                            name="candidate_pool_size",
                            unit="count",
                            value=64.0,
                        ),
                    ),
                ),
                QueryOperationalMetrics(
                    retrieval_latency_ms=20.0,
                    rerank_latency_ms=4.0,
                    generator_latency_ms=50.0,
                    retrieved_context_tokens=5,
                    prompt_tokens=10,
                    prompt_context_count=1,
                    approximate_query_costs=(
                        ApproximateQueryCost(
                            name="candidate_pool_size",
                            unit="count",
                            value=32.0,
                        ),
                    ),
                ),
            ]
        )

        self.assertEqual(summary["latency_ms"]["retrieval"]["p50"], 15.0)
        self.assertEqual(summary["latency_ms"]["retrieval"]["p95"], 19.5)
        self.assertEqual(summary["latency_ms"]["retrieval"]["p99"], 19.9)
        self.assertEqual(summary["latency_ms"]["rerank"]["p50"], 3.0)
        self.assertEqual(summary["latency_ms"]["generator"]["p50"], 40.0)
        self.assertEqual(summary["latency_ms"]["total"]["p50"], 58.0)

        self.assertEqual(summary["budgets"]["retrieved_context_tokens"]["p50"], 7.0)
        self.assertEqual(summary["budgets"]["prompt_tokens"]["p50"], 12.0)
        self.assertEqual(summary["budgets"]["prompt_context_count"]["p50"], 1.5)

        self.assertEqual(
            summary["approximate_query_costs"]["candidate_pool_size"]["unit"],
            "count",
        )
        self.assertEqual(
            summary["approximate_query_costs"]["candidate_pool_size"]["p50"],
            48.0,
        )


if __name__ == "__main__":
    unittest.main()
