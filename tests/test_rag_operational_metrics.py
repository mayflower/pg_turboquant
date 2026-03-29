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
        self.assertEqual(summary["latency_ms"]["retrieval"]["avg"], 15.0)
        self.assertEqual(summary["latency_ms"]["retrieval"]["p95"], 19.5)
        self.assertEqual(summary["latency_ms"]["retrieval"]["p99"], 19.9)
        self.assertEqual(summary["latency_ms"]["rerank"]["p50"], 3.0)
        self.assertEqual(summary["latency_ms"]["generator"]["p50"], 40.0)
        self.assertEqual(summary["latency_ms"]["total"]["p50"], 58.0)

        self.assertEqual(summary["budgets"]["retrieved_context_tokens"]["p50"], 7.0)
        self.assertEqual(summary["budgets"]["retrieved_context_tokens"]["avg"], 7.0)
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

    def test_scan_stats_are_preserved_and_summarized(self):
        summary = summarize_query_operational_metrics(
            [
                QueryOperationalMetrics(
                    retrieval_latency_ms=10.0,
                    scan_stats={
                        "mode": "ivf",
                        "score_mode": "code_domain",
                        "nominal_probe_count": 4,
                        "effective_probe_count": 2,
                        "max_visited_codes": 256,
                        "max_visited_pages": 0,
                        "selected_list_count": 2,
                        "selected_live_count": 18,
                        "visited_page_count": 5,
                        "visited_code_count": 18,
                        "candidate_heap_count": 8,
                        "page_prune_count": 2,
                        "early_stop_count": 1,
                    },
                ),
                QueryOperationalMetrics(
                    retrieval_latency_ms=12.0,
                    scan_stats={
                        "mode": "ivf",
                        "score_mode": "code_domain",
                        "nominal_probe_count": 4,
                        "effective_probe_count": 3,
                        "max_visited_codes": 320,
                        "max_visited_pages": 0,
                        "selected_list_count": 3,
                        "selected_live_count": 30,
                        "visited_page_count": 7,
                        "visited_code_count": 30,
                        "candidate_heap_count": 8,
                        "page_prune_count": 4,
                        "early_stop_count": 2,
                    },
                ),
            ]
        )

        self.assertIn("scan_stats", summary)
        self.assertEqual(summary["scan_stats"]["mode"]["uniform"], "ivf")
        self.assertEqual(summary["scan_stats"]["score_mode"]["uniform"], "code_domain")
        self.assertEqual(summary["scan_stats"]["nominal_probe_count"]["p50"], 4.0)
        self.assertEqual(summary["scan_stats"]["effective_probe_count"]["p50"], 2.5)
        self.assertEqual(summary["scan_stats"]["effective_probe_count"]["avg"], 2.5)
        self.assertEqual(summary["scan_stats"]["max_visited_codes"]["p50"], 288.0)
        self.assertEqual(summary["scan_stats"]["selected_list_count"]["p50"], 2.5)
        self.assertEqual(summary["scan_stats"]["selected_list_count"]["avg"], 2.5)
        self.assertEqual(summary["scan_stats"]["selected_live_count"]["avg"], 24.0)
        self.assertEqual(summary["scan_stats"]["visited_page_count"]["avg"], 6.0)
        self.assertEqual(summary["scan_stats"]["visited_code_count"]["p50"], 24.0)
        self.assertEqual(summary["scan_stats"]["visited_code_count"]["avg"], 24.0)
        self.assertEqual(summary["scan_stats"]["page_prune_count"]["p50"], 3.0)
        self.assertEqual(summary["scan_stats"]["page_prune_count"]["avg"], 3.0)
        self.assertEqual(summary["scan_stats"]["early_stop_count"]["p50"], 1.5)


if __name__ == "__main__":
    unittest.main()
