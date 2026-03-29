import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.rag.multihop_eval import (
    MultihopQueryEvaluation,
    compute_multihop_support_metrics,
    export_multihop_diagnostics,
)


class RagMultihopEvalContractTest(unittest.TestCase):
    def test_tiny_synthetic_example_reports_multihop_support_coverage(self):
        queries = [
            MultihopQueryEvaluation(
                query_id="q1",
                retrieved_ids=["p_a", "p_noise", "p_b"],
                supporting_ids=["p_a", "p_b"],
            ),
            MultihopQueryEvaluation(
                query_id="q2",
                retrieved_ids=["x_a", "x_noise", "x_other"],
                supporting_ids=["x_a", "x_b"],
            ),
        ]

        metrics, diagnostics = compute_multihop_support_metrics(queries, ks=(2, 3))

        self.assertEqual(metrics["multihop_support_coverage@2"], 0.0)
        self.assertEqual(metrics["multihop_support_coverage@3"], 0.5)
        self.assertEqual(diagnostics[0]["per_k"]["2"]["missing_supporting_ids"], ["p_b"])
        self.assertEqual(diagnostics[0]["per_k"]["3"]["missing_supporting_ids"], [])
        self.assertEqual(diagnostics[1]["per_k"]["3"]["missing_supporting_ids"], ["x_b"])

    def test_diagnostic_export_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = export_multihop_diagnostics(
                output_dir=Path(tmpdir),
                run_metadata={
                    "run_id": "hotpotqa-run",
                    "dataset_id": "kilt_hotpotqa",
                    "result_kind": "retrieval_only",
                },
                overlay_metrics={"multihop_support_coverage@10": 0.75},
                diagnostics=[
                    {
                        "query_id": "q1",
                        "supporting_ids": ["p1", "p2"],
                        "per_k": {
                            "10": {
                                "retrieved_supporting_ids": ["p1"],
                                "missing_supporting_ids": ["p2"],
                                "all_supporting_retrieved": False,
                            }
                        },
                    }
                ],
            )

            payload = json.loads((Path(tmpdir) / artifacts["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["run_metadata"]["dataset_id"], "kilt_hotpotqa")
            self.assertEqual(payload["overlay_metrics"]["multihop_support_coverage@10"], 0.75)
            self.assertEqual(
                payload["diagnostics"][0]["per_k"]["10"]["missing_supporting_ids"],
                ["p2"],
            )


if __name__ == "__main__":
    unittest.main()
