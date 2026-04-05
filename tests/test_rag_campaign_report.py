import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.rag.campaign_report import (
    DEFAULT_RAG_SYSTEM_VARIANTS,
    build_comparative_campaign_plan,
    run_comparative_campaign,
)


class RagCampaignReportContractTest(unittest.TestCase):
    def test_campaign_plan_covers_required_systems_and_datasets(self):
        plan = build_comparative_campaign_plan(
            dataset_ids=["kilt_nq", "kilt_hotpotqa", "popqa"],
            generator_id="fixed-debug-generator",
        )

        self.assertEqual(plan["campaign_kind"], "rag_benchmark")
        self.assertEqual(plan["datasets"], ["kilt_nq", "kilt_hotpotqa", "popqa"])
        self.assertEqual(plan["generator_id"], "fixed-debug-generator")
        self.assertEqual(
            [variant["system_id"] for variant in plan["system_variants"]],
            DEFAULT_RAG_SYSTEM_VARIANTS,
        )
        self.assertEqual(plan["system_variants"][0]["retriever_backend"], "pg_turboquant")
        self.assertEqual(plan["system_variants"][0]["retrieval_mode"], "approx")
        self.assertFalse(plan["system_variants"][0]["rerank_enabled"])
        self.assertEqual(
            len(plan["retrieval_scenarios"]),
            len(DEFAULT_RAG_SYSTEM_VARIANTS) * 3,
        )
        self.assertEqual(
            len(plan["end_to_end_scenarios"]),
            len(DEFAULT_RAG_SYSTEM_VARIANTS) * 3,
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
            rank = DEFAULT_RAG_SYSTEM_VARIANTS.index(scenario["system_id"]) + 1
            is_turboquant = scenario["system_id"].startswith("pg_turboquant_")
            return {
                "run_metadata": {
                    "dataset_id": scenario["dataset_id"],
                    "method_id": scenario["system_id"],
                    "result_kind": "retrieval_only",
                    "footprint_bytes": 8192 * (10 + rank),
                    "retrieval_execution_mode": "approx_exact_rerank" if scenario["rerank_enabled"] else "approx_stage1_only",
                    "context_fetch_mode": "post_limit_text_fetch",
                    "index_metadata": {
                        "live_count": 200,
                        "delta_live_count": rank,
                        "delta_batch_page_count": rank + 1,
                        "delta_head_block": 100 + rank,
                        "delta_tail_block": 110 + rank,
                        "exact_key_head_block": 200 + rank,
                        "exact_key_tail_block": 210 + rank,
                        "exact_key_page_count": rank + 2,
                        "maintenance_action_recommended": "merge_delta",
                        "delta_health": {
                            "merge_recommended": True,
                            "delta_page_depth": 2 + rank,
                        },
                        "maintenance": {
                            "compaction_recommended": rank % 2 == 0,
                        },
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
                    "mrr@10": 0.82 - (rank * 0.01),
                    "ndcg@10": 0.87 - (rank * 0.01),
                    "hit_rate@10": 0.95 - (rank * 0.01),
                    "evidence_coverage@10": 0.77 - (rank * 0.01),
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
            rank = DEFAULT_RAG_SYSTEM_VARIANTS.index(scenario["system_id"]) + 1
            return {
                "run_metadata": {
                    "dataset_id": scenario["dataset_id"],
                    "method_id": scenario["system_id"],
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
            self.assertIn("retrieval_benchmark", payload["tables"])
            self.assertIn("end_to_end_benchmark", payload["tables"])
            self.assertIn("retrieval_diagnostics", payload["tables"])
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
            self.assertIn("retrieval_diagnostics_csv", artifacts)

            retrieval_rows = payload["tables"]["retrieval_benchmark"]
            end_to_end_rows = payload["tables"]["end_to_end_benchmark"]
            diagnostics_rows = payload["tables"]["retrieval_diagnostics"]
            self.assertEqual(len(retrieval_rows), len(DEFAULT_RAG_SYSTEM_VARIANTS))
            self.assertEqual(len(end_to_end_rows), len(DEFAULT_RAG_SYSTEM_VARIANTS))
            self.assertEqual(len(diagnostics_rows), len(DEFAULT_RAG_SYSTEM_VARIANTS))
            self.assertEqual(retrieval_rows[0]["dataset_id"], "kilt_hotpotqa")
            self.assertEqual(retrieval_rows[0]["system_id"], "pg_turboquant_approx")
            self.assertIn("system_label", retrieval_rows[0])
            self.assertIn("retriever_backend", retrieval_rows[0])
            self.assertIn("retrieval_mode", retrieval_rows[0])
            self.assertIn("recall@10", retrieval_rows[0])
            self.assertIn("mrr@10", retrieval_rows[0])
            self.assertIn("ndcg@10", retrieval_rows[0])
            self.assertIn("hit_rate@10", retrieval_rows[0])
            self.assertIn("evidence_coverage@10", retrieval_rows[0])
            self.assertIn("latency_p50_ms", retrieval_rows[0])
            self.assertIn("retrieval_execution_mode", retrieval_rows[0])
            self.assertIn("context_fetch_mode", retrieval_rows[0])
            self.assertNotIn("avg_selected_list_count", retrieval_rows[0])
            self.assertIn("score_mode", diagnostics_rows[0])
            self.assertIn("avg_selected_list_count", diagnostics_rows[0])
            self.assertIn("avg_selected_live_count", diagnostics_rows[0])
            self.assertIn("avg_visited_page_count", diagnostics_rows[0])
            self.assertIn("avg_visited_code_count", diagnostics_rows[0])
            self.assertIn("avg_effective_probe_count", diagnostics_rows[0])
            self.assertIn("avg_page_prune_count", diagnostics_rows[0])
            self.assertIn("visited_code_fraction", diagnostics_rows[0])
            self.assertIn("visited_page_fraction", diagnostics_rows[0])
            self.assertIn("router_restarts", diagnostics_rows[0])
            self.assertIn("router_balance_penalty", diagnostics_rows[0])
            self.assertIn("max_list_size", diagnostics_rows[0])
            self.assertIn("list_coeff_var", diagnostics_rows[0])
            self.assertIn("delta_live_count", diagnostics_rows[0])
            self.assertIn("delta_batch_page_count", diagnostics_rows[0])
            self.assertIn("delta_page_depth", diagnostics_rows[0])
            self.assertIn("delta_merge_recommended", diagnostics_rows[0])
            self.assertIn("exact_key_page_count", diagnostics_rows[0])
            self.assertIn("maintenance_action_recommended", diagnostics_rows[0])
            self.assertIn("answer_exact_match", end_to_end_rows[0])
            self.assertIn("generator_id", end_to_end_rows[0])
            self.assertIn("total_latency_p50_ms", end_to_end_rows[0])
            self.assertIn("total_latency_p95_ms", end_to_end_rows[0])
            self.assertIn("regression_gate", payload["report"])
            self.assertTrue(payload["report"]["regression_gate"]["passed"])
            self.assertIn("kilt_hotpotqa", payload["report"]["regression_gate"]["dataset_id"])
            self.assertEqual(payload["report"]["summary"]["systems"], DEFAULT_RAG_SYSTEM_VARIANTS)

            markdown = (Path(tmpdir) / artifacts["report_markdown"]).read_text(encoding="utf-8")
            report_html = (Path(tmpdir) / artifacts["report_html"]).read_text(encoding="utf-8")
            self.assertIn("Retrieval Benchmark", markdown)
            self.assertIn("End-to-End Benchmark", markdown)
            self.assertIn("Retriever Diagnostics", markdown)
            self.assertIn("Metric Validity Caveats", markdown)
            self.assertIn("Evidence Coverage@10", markdown)
            self.assertIn("Regression Gate", markdown)
            self.assertIn("RAG Benchmark Outcome", report_html)
            self.assertIn("Measured Comparison Scope", report_html)
            self.assertIn("Retrieval Systems", report_html)
            self.assertIn("Dataset Leaders", report_html)
            self.assertIn("pgvector_hnsw (rerank)", report_html)


if __name__ == "__main__":
    unittest.main()
