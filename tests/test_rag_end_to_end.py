import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.rag.end_to_end import (
    FixedGeneratorConfig,
    RetrievalCacheEntry,
    build_prompt,
    export_end_to_end_run,
    run_fixed_generator_stage,
)
from benchmarks.rag.operational_metrics import ApproximateQueryCost


class RagEndToEndContractTest(unittest.TestCase):
    def test_retrieval_cache_can_be_consumed_by_generator_stage(self):
        cache = [
            RetrievalCacheEntry(
                query_id="q1",
                question="What is alpha?",
                retrieved_contexts=[
                    {"id": "p1", "text": "Alpha is the first letter."},
                    {"id": "p2", "text": "Beta follows alpha."},
                ],
                answer_reference="The first letter",
                retrieval_latency_ms=8.0,
                rerank_latency_ms=1.5,
                approximate_query_costs=(
                    ApproximateQueryCost(
                        name="candidate_pool_size",
                        unit="count",
                        value=24.0,
                    ),
                ),
            )
        ]

        config = FixedGeneratorConfig(
            generator_id="fixed-debug-generator",
            system_prompt="Answer using only the provided context.",
            max_contexts=2,
        )

        results = run_fixed_generator_stage(
            cache_entries=cache,
            config=config,
            generator_fn=lambda prompt, entry: f"generated::{entry.query_id}::{len(prompt)}",
            clock_fn=iter((0.0, 0.012)).__next__,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["query_id"], "q1")
        self.assertEqual(results[0]["answer"], f"generated::q1::{len(results[0]['prompt'])}")
        self.assertEqual(len(results[0]["contexts"]), 2)
        self.assertEqual(results[0]["operational_metrics"]["latency_ms"]["retrieval"], 8.0)
        self.assertEqual(results[0]["operational_metrics"]["latency_ms"]["rerank"], 1.5)
        self.assertEqual(results[0]["operational_metrics"]["latency_ms"]["generator"], 12.0)
        self.assertEqual(results[0]["operational_metrics"]["latency_ms"]["total"], 21.5)
        self.assertEqual(
            results[0]["operational_metrics"]["budgets"]["prompt_context_count"],
            2,
        )
        self.assertEqual(
            results[0]["operational_metrics"]["approximate_query_costs"][0]["name"],
            "candidate_pool_size",
        )

    def test_end_to_end_results_are_separated_from_retrieval_only_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = export_end_to_end_run(
                output_dir=Path(tmpdir),
                run_metadata={
                    "run_id": "e2e-run",
                    "result_kind": "end_to_end",
                    "generator_id": "fixed-debug-generator",
                },
                retrieval_summary={"result_kind": "retrieval_only", "recall@10": 0.8},
                generation_results=[
                    {
                        "query_id": "q1",
                        "prompt": "prompt text",
                        "contexts": [{"id": "p1", "text": "Alpha"}],
                        "answer": "Alpha",
                        "reference_answer": "Alpha",
                        "operational_metrics": {
                            "latency_ms": {
                                "retrieval": 9.0,
                                "rerank": 2.0,
                                "generator": 20.0,
                                "total": 31.0,
                            },
                            "budgets": {
                                "retrieved_context_tokens": 1,
                                "prompt_tokens": 2,
                                "prompt_context_count": 1,
                            },
                            "approximate_query_costs": [
                                {
                                    "name": "candidate_pool_size",
                                    "unit": "count",
                                    "value": 32.0,
                                }
                            ],
                        },
                    }
                ],
                answer_metrics={"answer_exact_match": 1.0},
            )

            payload = json.loads((Path(tmpdir) / artifacts["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["run_metadata"]["result_kind"], "end_to_end")
            self.assertEqual(payload["retrieval_summary"]["result_kind"], "retrieval_only")
            self.assertIn("generation_results", payload)
            self.assertIn("answer_metrics", payload)
            self.assertEqual(payload["operational_summary"]["latency_ms"]["retrieval"]["p50"], 9.0)
            self.assertEqual(payload["operational_summary"]["latency_ms"]["generator"]["p50"], 20.0)
            self.assertEqual(payload["operational_summary"]["latency_ms"]["total"]["p50"], 31.0)
            self.assertEqual(
                payload["operational_summary"]["approximate_query_costs"]["candidate_pool_size"]["unit"],
                "count",
            )

    def test_end_to_end_result_schema_golden(self):
        prompt = build_prompt(
            system_prompt="Answer using context.",
            question="Who is alpha?",
            contexts=[
                {"id": "p1", "text": "Alpha is a placeholder name."},
                {"id": "p2", "text": "Omega is the final letter."},
            ],
        )

        self.assertIn("Answer using context.", prompt)
        self.assertIn("Question: Who is alpha?", prompt)
        self.assertIn("[p1] Alpha is a placeholder name.", prompt)
        self.assertIn("[p2] Omega is the final letter.", prompt)


if __name__ == "__main__":
    unittest.main()
