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
            dataset_ids=["kilt_hotpotqa"],
            generator_id="fixed-debug-generator",
        )
        plan["regression_gate"] = {
            "kilt_hotpotqa": {
                "dataset_id": "kilt_hotpotqa",
                "method_id": "pg_turboquant_approx",
                "recall_at_10_floor": 0.90,
                "max_visited_code_fraction": 0.85,
                "max_visited_page_fraction": 0.60,
                "expected_score_mode": "code_domain",
                "max_effective_probe_count": 8,
            }
        }

        def fake_retrieval_runner(scenario):
            rank = COMPARATIVE_METHOD_VARIANTS.index(scenario["method_id"]) + 1
            is_turboquant = scenario["method_id"].startswith("pg_turboquant_")
            return {
                "run_metadata": {
                    "dataset_id": scenario["dataset_id"],
                    "method_id": scenario["method_id"],
                    "result_kind": "retrieval_only",
                    "footprint_bytes": 8192 * (10 + rank),
                    "index_metadata": {
                        "live_count": 200,
                        "router": {
                            "restart_count": 3,
                            "balance_penalty": round(0.05 * rank, 4),
                        },
                        "list_distribution": {
                            "max_list_size": 10 + rank,
                            "coeff_var": round(0.1 * rank, 4),
                        },
                    },
                },
                "metrics": {
                    "recall@10": 0.92 - (rank * 0.01),
                    "latency_p95_ms": 12.0 + rank,
                    "latency_p50_ms": 10.0 + rank,
                },
                "operational_summary": {
                    "scan_stats": {
                        "score_mode": {
                            "uniform": "code_domain" if is_turboquant else "none",
                            "values": ["code_domain" if is_turboquant else "none"],
                            "count": 1,
                        },
                        "selected_list_count": {
                            "avg": 2.0 + rank,
                            "p50": 2.0 + rank,
                            "p95": 3.0 + rank,
                        },
                        "selected_live_count": {
                            "avg": 40.0 + rank,
                            "p50": 40.0 + rank,
                            "p95": 44.0 + rank,
                        },
                        "visited_page_count": {
                            "avg": 1.0 + (rank / 10.0),
                            "p50": 1.0 + (rank / 10.0),
                            "p95": 2.0 + (rank / 10.0),
                        },
                        "page_prune_count": {
                            "avg": float(rank),
                            "p50": float(rank),
                            "p95": float(rank + 1),
                        },
                        "early_stop_count": {
                            "p50": float(rank) / 2.0,
                            "p95": float(rank),
                        },
                        "effective_probe_count": {
                            "avg": float(rank),
                            "p50": float(rank),
                            "p95": float(rank + 1),
                        },
                        "visited_code_count": {
                            "avg": float(10 + rank),
                            "p50": float(100 - rank),
                            "p95": float(120 - rank),
                        },
                    }
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
            self.assertEqual(retrieval_rows[0]["dataset_id"], "kilt_hotpotqa")
            self.assertIn("recall@10", retrieval_rows[0])
            self.assertIn("avg_selected_list_count", retrieval_rows[0])
            self.assertIn("avg_selected_live_count", retrieval_rows[0])
            self.assertIn("avg_visited_page_count", retrieval_rows[0])
            self.assertIn("avg_visited_code_count", retrieval_rows[0])
            self.assertIn("avg_effective_probe_count", retrieval_rows[0])
            self.assertIn("avg_page_prune_count", retrieval_rows[0])
            self.assertIn("visited_code_fraction", retrieval_rows[0])
            self.assertIn("visited_page_fraction", retrieval_rows[0])
            self.assertIn("score_mode", retrieval_rows[0])
            self.assertIn("router_restarts", retrieval_rows[0])
            self.assertIn("router_balance_penalty", retrieval_rows[0])
            self.assertIn("max_list_size", retrieval_rows[0])
            self.assertIn("list_coeff_var", retrieval_rows[0])
            self.assertIn("answer_exact_match", end_to_end_rows[0])
            self.assertIn("total_latency_p95_ms", end_to_end_rows[0])
            self.assertIn("regression_gate", payload["report"])
            self.assertTrue(payload["report"]["regression_gate"]["passed"])
            self.assertIn("kilt_hotpotqa", payload["report"]["regression_gate"]["dataset_id"])

            markdown = (Path(tmpdir) / artifacts["report_markdown"]).read_text(encoding="utf-8")
            report_html = (Path(tmpdir) / artifacts["report_html"]).read_text(encoding="utf-8")
            self.assertIn("Retrieval-Only Comparison", markdown)
            self.assertIn("End-to-End Comparison", markdown)
            self.assertIn("Metric Validity Caveats", markdown)
            self.assertIn("avg_selected_list_count", markdown)
            self.assertIn("avg_visited_page_count", markdown)
            self.assertIn("avg_effective_probe_count", markdown)
            self.assertIn("score_mode", markdown)
            self.assertIn("Regression Gate", markdown)
            self.assertIn("router_balance_penalty", markdown)
            self.assertIn("max_list_size", markdown)
            self.assertIn("TurboQuant Outcome", report_html)
            self.assertIn("Measured Comparison Scope", report_html)
            self.assertIn("Method Metrics", report_html)
            self.assertIn("selected_page_count", report_html)
            self.assertIn("pgvector_hnsw_approx", report_html)


if __name__ == "__main__":
    unittest.main()
