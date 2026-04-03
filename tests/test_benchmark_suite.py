import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_suite.py"
SPEC = importlib.util.spec_from_file_location("benchmark_suite_module", SCRIPT)
BENCHMARK_SUITE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCHMARK_SUITE)


class BenchmarkSuiteContractTest(unittest.TestCase):
    def run_suite(self, *args):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "benchmark.json"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--dry-run",
                "--output",
                str(output),
                *args,
            ]
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            stdout_payload = json.loads(result.stdout)
            file_payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(stdout_payload, file_payload)
            return stdout_payload

    def test_matrix_output_schema_and_presets(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--methods",
            "turboquant_flat,turboquant_ivf,pgvector_ivfflat,pgvector_hnsw",
        )

        self.assertEqual(payload["profile"], "tiny")
        self.assertEqual(payload["metrics"], ["cosine"])
        self.assertEqual(
            payload["corpora"],
            [
                "normalized_dense",
                "non_normalized_varied_norms",
                "clustered",
                "mixed_live_dead",
                "hotpot_skewed",
                "hotpot_overlap",
            ],
        )
        self.assertEqual(
            payload["methods"],
            [
                "turboquant_flat",
                "turboquant_ivf",
                "pgvector_ivfflat",
                "pgvector_hnsw",
            ],
        )
        self.assertIn("scenarios", payload)
        self.assertIn("environment", payload)
        self.assertIn("scenario_matrix", payload)
        self.assertGreater(len(payload["scenarios"]), 0)

        scenario = payload["scenarios"][0]
        self.assertIn("corpus", scenario)
        self.assertIn("method", scenario)
        self.assertIn("benchmark_metric", scenario)
        self.assertIn("ground_truth", scenario)
        self.assertIn("metrics", scenario)
        self.assertIn("index", scenario)
        self.assertIn("query_knobs", scenario)
        self.assertIn("query_api", scenario)
        self.assertIn("index_metadata", scenario)
        self.assertIn("simd", scenario)
        self.assertIn("scan_stats", scenario)
        if scenario["query_mode"] == "ordered_rerank" and scenario["method"].startswith("turboquant_"):
            self.assertIn("candidate_retention", scenario)

        self.assertIn("recall_at_10", scenario["metrics"])
        self.assertIn("recall_at_100", scenario["metrics"])
        self.assertIn("p50_ms", scenario["metrics"])
        self.assertIn("p95_ms", scenario["metrics"])
        self.assertIn("build_seconds", scenario["metrics"])
        self.assertIn("index_size_bytes", scenario["metrics"])
        self.assertIn("candidate_slots_bound", scenario["metrics"])
        self.assertIn("build_wal_bytes", scenario["metrics"])
        self.assertIn("insert_wal_bytes", scenario["metrics"])
        self.assertIn("concurrent_insert_rows_per_second", scenario["metrics"])
        self.assertIn("concurrent_insert_rows", scenario["metrics"])
        self.assertIn("concurrent_insert_workers", scenario["metrics"])
        self.assertIn("maintenance_wal_bytes", scenario["metrics"])
        self.assertIn("sealed_baseline_build_wal_bytes", scenario["metrics"])
        self.assertIn("sealed_baseline_insert_wal_bytes", scenario["metrics"])
        self.assertIn("sealed_baseline_maintenance_wal_bytes", scenario["metrics"])
        self.assertIn("metric", scenario["ground_truth"])
        self.assertIn("turboquant.probes", scenario["query_knobs"])
        self.assertEqual(scenario["query_api"]["helper"], "tq_rerank_candidates")
        self.assertIn("candidate_limit", scenario["query_api"])
        self.assertIn("final_limit", scenario["query_api"])
        self.assertIn("format_version", scenario["index_metadata"])
        self.assertIn("metric", scenario["index_metadata"])
        self.assertIn("list_count", scenario["index_metadata"])
        self.assertIn("page_summary", scenario["index_metadata"])
        self.assertIn("capabilities", scenario["index_metadata"])
        self.assertIn("index_only_scan", scenario["index_metadata"]["capabilities"])
        self.assertIn("multicolumn", scenario["index_metadata"]["capabilities"])
        self.assertIn("include_columns", scenario["index_metadata"]["capabilities"])
        self.assertIn("bitmap_scan", scenario["index_metadata"]["capabilities"])
        self.assertIn("mode", scenario["index_metadata"]["page_summary"])
        self.assertIn("safe_pruning", scenario["index_metadata"]["page_summary"])
        if scenario["method"].startswith("turboquant_"):
            self.assertIn("residual_sketch", scenario["index_metadata"])
            self.assertIn("projected_dimension", scenario["index_metadata"]["residual_sketch"])
            self.assertIn("bit_budget", scenario["index_metadata"]["residual_sketch"])
        self.assertIn("preferred_kernel", scenario["simd"])
        self.assertIn("compiled", scenario["simd"])
        self.assertIn("runtime_available", scenario["simd"])
        self.assertIn("selected_kernel", scenario["simd"])
        self.assertIn("code_domain_kernel", scenario["simd"])
        self.assertIn("mode", scenario["scan_stats"])
        self.assertIn("score_mode", scenario["scan_stats"])
        self.assertIn("score_kernel", scenario["scan_stats"])
        self.assertIn("scan_orchestration", scenario["scan_stats"])
        self.assertIn("selected_list_count", scenario["scan_stats"])
        self.assertIn("selected_live_count", scenario["scan_stats"])
        self.assertIn("selected_page_count", scenario["scan_stats"])
        self.assertIn("visited_page_count", scenario["scan_stats"])
        self.assertIn("visited_code_count", scenario["scan_stats"])
        self.assertIn("nominal_probe_count", scenario["scan_stats"])
        self.assertIn("effective_probe_count", scenario["scan_stats"])
        self.assertIn("max_visited_codes", scenario["scan_stats"])
        self.assertIn("max_visited_pages", scenario["scan_stats"])
        self.assertIn("candidate_heap_count", scenario["scan_stats"])
        self.assertIn("candidate_heap_insert_count", scenario["scan_stats"])
        self.assertIn("candidate_heap_replace_count", scenario["scan_stats"])
        self.assertIn("candidate_heap_reject_count", scenario["scan_stats"])
        self.assertIn("local_candidate_heap_insert_count", scenario["scan_stats"])
        self.assertIn("local_candidate_heap_replace_count", scenario["scan_stats"])
        self.assertIn("local_candidate_heap_reject_count", scenario["scan_stats"])
        self.assertIn("local_candidate_merge_count", scenario["scan_stats"])
        self.assertIn("shadow_decoded_vector_count", scenario["scan_stats"])
        self.assertIn("shadow_decode_candidate_count", scenario["scan_stats"])
        self.assertIn("shadow_decode_overlap_count", scenario["scan_stats"])
        self.assertIn("shadow_decode_primary_only_count", scenario["scan_stats"])
        self.assertIn("shadow_decode_only_count", scenario["scan_stats"])
        self.assertIn("page_bound_mode", scenario["scan_stats"])
        self.assertIn("safe_pruning_enabled", scenario["scan_stats"])
        self.assertIn("faithful_fast_path", scenario["scan_stats"])
        self.assertIn("compatibility_fallback", scenario["scan_stats"])
        self.assertIn("near_exhaustive_crossover", scenario["scan_stats"])
        if scenario["query_mode"] == "ordered_rerank" and scenario["method"].startswith("turboquant_"):
            self.assertIn("avg_candidate_count", scenario["candidate_retention"])
            self.assertIn("avg_exact_top_10_retention", scenario["candidate_retention"])
            self.assertIn("avg_exact_top_100_retention", scenario["candidate_retention"])
            self.assertIn("avg_exact_top_100_miss_count", scenario["candidate_retention"])
            self.assertIn("worst_exact_top_100_retention", scenario["candidate_retention"])
            self.assertIn("avg_shadow_candidate_count", scenario["candidate_retention"])
            self.assertIn("avg_shadow_exact_top_10_retention", scenario["candidate_retention"])
            self.assertIn("avg_shadow_exact_top_100_retention", scenario["candidate_retention"])
            self.assertIn("avg_shadow_exact_top_100_miss_count", scenario["candidate_retention"])
            self.assertIn("worst_shadow_exact_top_100_retention", scenario["candidate_retention"])
            self.assertIn("estimator_quality", scenario)
            self.assertIn("sample_count", scenario["estimator_quality"])
            self.assertIn("distance_error_bias", scenario["estimator_quality"])
            self.assertIn("distance_error_variance", scenario["estimator_quality"])
            self.assertIn("distance_error_mae", scenario["estimator_quality"])
            self.assertIn("avg_abs_rank_shift", scenario["estimator_quality"])
            self.assertIn("max_abs_rank_shift", scenario["estimator_quality"])
        if scenario["method"] in {"turboquant_flat", "turboquant_ivf"}:
            self.assertEqual(scenario["scan_stats"]["score_mode"], "code_domain")
            self.assertEqual(scenario["scan_stats"]["decoded_vector_count"], 0)
            if scenario["method"] == "turboquant_ivf":
                self.assertEqual(
                    scenario["index_metadata"]["page_summary"]["mode"],
                    "safe_summary_pruning",
                )
                self.assertTrue(scenario["index_metadata"]["page_summary"]["safe_pruning"])
                self.assertEqual(scenario["scan_stats"]["page_bound_mode"], "safe_summary_pruning")
                self.assertTrue(scenario["scan_stats"]["safe_pruning_enabled"])
                self.assertIn(
                    scenario["scan_stats"]["scan_orchestration"],
                    ("ivf_bounded_pages", "ivf_near_exhaustive"),
                )
        self.assertIn("python_version", payload["environment"])
        self.assertIn("platform", payload["environment"])
        self.assertIn("cpu_arch", payload["environment"])
        self.assertIn("methods", payload["scenario_matrix"])
        self.assertIn("corpora", payload["scenario_matrix"])
        self.assertIn("profiles", payload["scenario_matrix"])
        self.assertIn("metrics", payload["scenario_matrix"])

    def test_report_artifact_generation_contract(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "normalized_dense,clustered",
            "--methods",
            "turboquant_flat,pgvector_ivfflat,pgvector_hnsw",
            "--report",
        )

        self.assertIn("report", payload)
        self.assertIn("artifacts", payload)
        self.assertIn("summary", payload["report"])
        self.assertIn("method_rows", payload["report"])
        self.assertIn("comparisons", payload["report"])
        self.assertIn("measurement_notes", payload["report"])
        self.assertIn("conclusions", payload["report"])
        self.assertIn("leaderboards", payload["report"])
        self.assertIn("generated_at", payload["report"])
        self.assertIn("report_json", payload["artifacts"])
        self.assertIn("report_markdown", payload["artifacts"])
        self.assertIn("report_html", payload["artifacts"])

        summary = payload["report"]["summary"]
        self.assertGreater(summary["scenario_count"], 0)
        self.assertIn("methods", summary)
        self.assertIn("corpora", summary)
        self.assertIn("metrics", summary)
        self.assertIn("query_modes", summary)

        method_row = payload["report"]["method_rows"][0]
        self.assertIn("corpus", method_row)
        self.assertIn("method", method_row)
        self.assertIn("benchmark_metric", method_row)
        self.assertIn("recall_at_10", method_row)
        self.assertIn("p50_ms", method_row)
        self.assertIn("p95_ms", method_row)
        self.assertIn("footprint_bytes", method_row)
        self.assertIn("visited_code_count", method_row)
        self.assertIn("visited_page_count", method_row)
        self.assertIn("selected_live_count", method_row)
        self.assertIn("selected_page_count", method_row)
        self.assertIn("score_kernel", method_row)
        self.assertIn("query_helper", method_row)
        self.assertIn("qjl_sketch_dimension", method_row)
        self.assertIn("distance_error_bias", method_row)
        self.assertIn("distance_error_variance", method_row)
        self.assertIn("avg_abs_rank_shift", method_row)

        comparison = payload["report"]["comparisons"][0]
        self.assertIn("corpus", comparison)
        self.assertIn("benchmark_metric", comparison)
        self.assertIn("baseline_method", comparison)
        self.assertIn("candidate_method", comparison)
        self.assertIn("comparison_scope", comparison)
        self.assertIn("metrics", comparison)
        self.assertIn("candidate", comparison)
        self.assertIn("baseline", comparison)
        self.assertIn("qjl_sketch_dimension", comparison)
        self.assertIn("recall_at_10_delta", comparison["metrics"])
        self.assertIn("p95_ms_delta", comparison["metrics"])
        self.assertIn("build_seconds_delta", comparison["metrics"])
        self.assertIn("index_size_bytes_delta", comparison["metrics"])
        self.assertIn("build_wal_bytes_delta", comparison["metrics"])
        self.assertIn("visited_code_count_delta", comparison["metrics"])
        self.assertIn("selected_page_count_delta", comparison["metrics"])
        self.assertIn("distance_error_bias_delta", comparison["metrics"])
        self.assertIn("avg_abs_rank_shift_delta", comparison["metrics"])

    def test_report_includes_internal_turboquant_ivf_vs_flat_comparison(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "normalized_dense",
            "--methods",
            "turboquant_flat,turboquant_ivf,pgvector_ivfflat",
            "--report",
        )

        self.assertTrue(
            any(
                comparison["comparison_scope"] == "turboquant_internal"
                and comparison["candidate_method"] == "turboquant_ivf"
                and comparison["baseline_method"] == "turboquant_flat"
                for comparison in payload["report"]["comparisons"]
            )
        )

    def test_ip_metric_contract_uses_ip_opclasses_and_ground_truth(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "normalized_dense",
            "--methods",
            "turboquant_flat,turboquant_ivf,pgvector_ivfflat",
            "--metrics",
            "ip",
        )

        self.assertEqual(payload["metrics"], ["ip"])
        for scenario in payload["scenarios"]:
            self.assertEqual(scenario["benchmark_metric"], "ip")
            self.assertEqual(scenario["ground_truth"]["metric"], "ip")
            if scenario["method"].startswith("turboquant_"):
                self.assertEqual(scenario["index"]["opclass"], "tq_ip_ops")
            elif scenario["method"].startswith("pgvector_"):
                self.assertEqual(scenario["index"]["opclass"], "vector_ip_ops")

    def test_qjl_sketch_budget_sweep_contract(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "normalized_dense",
            "--methods",
            "turboquant_flat,turboquant_ivf,pgvector_ivfflat",
            "--turboquant-qjl-sketch-dims",
            "d/4,d/2",
            "--report",
        )

        turboquant_dims = sorted(
            {
                scenario["index_metadata"]["residual_sketch"]["projected_dimension"]
                for scenario in payload["scenarios"]
                if scenario["method"].startswith("turboquant_")
            }
        )
        self.assertEqual(turboquant_dims, [1, 2])

        pgvector_scenarios = [
            scenario for scenario in payload["scenarios"] if scenario["method"].startswith("pgvector_")
        ]
        self.assertEqual(len(pgvector_scenarios), 1)

        self.assertTrue(
            any(
                comparison["comparison_scope"] == "turboquant_internal"
                and comparison["qjl_sketch_dimension"] in (1, 2)
                for comparison in payload["report"]["comparisons"]
            )
        )

    def test_report_markdown_is_factual_and_hotpot_ready(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "hotpot_skewed,hotpot_overlap",
            "--methods",
            "turboquant_ivf,pgvector_ivfflat,pgvector_hnsw",
            "--report",
        )

        markdown = BENCHMARK_SUITE.render_report_markdown(payload["report"])
        html = BENCHMARK_SUITE.render_report_html(payload["report"])

        self.assertIn("Measurement Notes", markdown)
        self.assertIn("Method Metrics", markdown)
        self.assertIn("Hotpot Conclusions", markdown)
        self.assertIn("selected_page_count", markdown)
        self.assertIn("score_kernel", markdown)
        self.assertNotIn("should be no slower", markdown)
        self.assertIn("Method Metrics", html)
        self.assertIn("Hotpot Conclusions", html)
        self.assertIn("hotpot_overlap", html)
        self.assertNotIn("should be no slower", html)

    def test_microbenchmark_section_is_machine_readable_when_requested(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "normalized_dense",
            "--methods",
            "turboquant_flat",
            "--microbench",
        )

        self.assertIn("microbenchmarks", payload)
        self.assertIn("results", payload["microbenchmarks"])
        self.assertGreater(len(payload["microbenchmarks"]["results"]), 0)

        first = payload["microbenchmarks"]["results"][0]
        self.assertIn("benchmark", first)
        self.assertIn("kernel", first)
        self.assertIn("iterations", first)
        self.assertIn("total_ns", first)
        self.assertIn("ns_per_op", first)
        self.assertIn("visited_code_count", first)
        self.assertIn("visited_page_count", first)
        self.assertIn("candidate_heap_insert_count", first)
        self.assertIn("candidate_heap_replace_count", first)
        self.assertIn("candidate_heap_reject_count", first)
        self.assertIn("local_candidate_heap_insert_count", first)
        self.assertIn("local_candidate_heap_replace_count", first)
        self.assertIn("local_candidate_heap_reject_count", first)
        self.assertIn("local_candidate_merge_count", first)
        self.assertIn("codes_per_second", first)
        self.assertIn("pages_per_second", first)
        self.assertIn("scratch_allocations", first)
        self.assertIn("decoded_buffer_reuses", first)
        self.assertIn("code_view_uses", first)
        self.assertIn("code_copy_uses", first)
        self.assertIn("list_count", first)
        self.assertIn("probe_count", first)
        self.assertIn("requested_kernel", first)
        self.assertIn("qjl_lut_mode", first)
        self.assertIn("scan_layout", first)
        self.assertIn("lookup_style", first)
        self.assertIn("block_width", first)
        self.assertIn("qjl_path", first)
        self.assertIn("gamma_path", first)
        self.assertIsInstance(first["lookup_style"], str)
        self.assertIsInstance(first["qjl_path"], str)
        self.assertIsInstance(first["gamma_path"], str)
        self.assertIsInstance(first["block_width"], int)
        self.assertIn("requested_kernel_honored", first)
        self.assertIn("dimension", first)
        self.assertIn("bits", first)
        self.assertIsInstance(first["kernel"], str)
        self.assertIsInstance(first["requested_kernel"], str)
        self.assertIn(first["qjl_lut_mode"], ("float", "quantized"))
        page_scan = next(
            row for row in payload["microbenchmarks"]["results"] if row["benchmark"] == "page_scan"
        )
        scalar_requested = next(
            row
            for row in payload["microbenchmarks"]["results"]
            if row["benchmark"] == "score_code_from_lut" and row["requested_kernel"] == "scalar"
        )
        avx2_requested = next(
            row
            for row in payload["microbenchmarks"]["results"]
            if row["benchmark"] == "score_code_from_lut" and row["requested_kernel"] == "avx2"
        )
        neon_requested = next(
            row
            for row in payload["microbenchmarks"]["results"]
            if row["benchmark"] == "score_code_from_lut" and row["requested_kernel"] == "neon"
        )
        router_full_sort = next(
            row
            for row in payload["microbenchmarks"]["results"]
            if row["benchmark"] == "router_top_probes_full_sort"
        )
        router_partial = next(
            row
            for row in payload["microbenchmarks"]["results"]
            if row["benchmark"] == "router_top_probes_partial"
        )
        self.assertGreater(page_scan["visited_page_count"], 0)
        self.assertGreater(page_scan["visited_code_count"], 0)
        self.assertGreater(page_scan["candidate_heap_insert_count"], 0)
        self.assertGreater(page_scan["local_candidate_heap_insert_count"], 0)
        self.assertGreater(page_scan["local_candidate_merge_count"], 0)
        self.assertLess(page_scan["candidate_heap_insert_count"], page_scan["visited_code_count"])
        self.assertEqual(
            page_scan["local_candidate_heap_insert_count"]
            + page_scan["local_candidate_heap_replace_count"]
            + page_scan["local_candidate_heap_reject_count"],
            page_scan["visited_code_count"],
        )
        self.assertGreaterEqual(page_scan["scratch_allocations"], 0)
        self.assertGreaterEqual(page_scan["decoded_buffer_reuses"], 0)
        self.assertEqual(page_scan["scan_layout"], "row_major")
        self.assertIn(page_scan["lookup_style"], ("scalar_loop", "float_gather"))
        self.assertEqual(page_scan["block_width"], 1)
        self.assertIn(page_scan["qjl_path"], ("float", "int16_quantized"))
        self.assertEqual(page_scan["gamma_path"], "float32_scalar")
        self.assertGreater(page_scan["code_view_uses"], 0)
        self.assertEqual(page_scan["code_copy_uses"], 0)
        self.assertEqual(scalar_requested["qjl_lut_mode"], "float")
        self.assertEqual(avx2_requested["requested_kernel"], "avx2")
        self.assertIn(avx2_requested["kernel"], ("scalar", "avx2"))
        self.assertIn(avx2_requested["qjl_lut_mode"], ("float", "quantized"))
        self.assertEqual(
            avx2_requested["requested_kernel_honored"],
            avx2_requested["kernel"] == "avx2",
        )
        self.assertEqual(neon_requested["requested_kernel"], "neon")
        self.assertIn(neon_requested["kernel"], ("scalar", "neon"))
        self.assertIn(neon_requested["qjl_lut_mode"], ("float", "quantized"))
        self.assertEqual(
            neon_requested["requested_kernel_honored"],
            neon_requested["kernel"] == "neon",
        )
        self.assertGreater(router_full_sort["iterations"], 0)
        self.assertGreater(router_partial["iterations"], 0)
        self.assertGreater(router_full_sort["visited_code_count"], 0)
        self.assertGreater(router_partial["visited_code_count"], 0)
        self.assertGreater(router_full_sort["list_count"], router_full_sort["probe_count"])
        self.assertEqual(router_partial["list_count"], router_full_sort["list_count"])
        self.assertEqual(router_partial["probe_count"], router_full_sort["probe_count"])
        self.assertEqual(router_full_sort["visited_page_count"], 0)
        self.assertEqual(router_partial["visited_page_count"], 0)
        if payload["microbenchmarks"]["simd"]["avx2"]["runtime_available"] or payload["microbenchmarks"]["simd"]["neon"]["runtime_available"]:
            self.assertTrue(
                any(
                    row["benchmark"] == "score_code_from_lut" and row["qjl_lut_mode"] == "quantized"
                    for row in payload["microbenchmarks"]["results"]
                )
        )
        self.assertNotIn("skipped", json.dumps(payload["microbenchmarks"]).lower())

        # Gate validation
        self.assertIn("gates", payload["microbenchmarks"])
        gates = payload["microbenchmarks"]["gates"]
        self.assertGreater(len(gates), 0)
        for gate in gates:
            self.assertIn("gate", gate)
            self.assertIn("passed", gate)
            self.assertIn("reason", gate)
            self.assertIsInstance(gate["passed"], bool)
            self.assertIsInstance(gate["reason"], str)
            self.assertTrue(gate["passed"], f"gate '{gate['gate']}' failed: {gate['reason']}")

        gate_names = {g["gate"] for g in gates}
        self.assertIn("lut16_dispatch_kernel_selection", gate_names)
        self.assertIn("page_scan_local_heap_reduces_global_inserts", gate_names)

    def test_microbenchmark_comparisons_and_regression_gates_are_reported(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "normalized_dense",
            "--methods",
            "turboquant_flat",
            "--microbench",
            "--report",
        )

        self.assertIn("comparisons", payload["microbenchmarks"])
        self.assertIn("regression_gates", payload["microbenchmarks"])
        self.assertIn("interpretation_notes", payload["microbenchmarks"])
        self.assertGreater(len(payload["microbenchmarks"]["comparisons"]), 0)
        self.assertGreater(len(payload["microbenchmarks"]["regression_gates"]), 0)
        self.assertGreater(len(payload["microbenchmarks"]["interpretation_notes"]), 0)

        comparison_names = {row["comparison"] for row in payload["microbenchmarks"]["comparisons"]}
        self.assertIn("score_code_from_lut_avx2_vs_scalar", comparison_names)
        self.assertIn("score_code_from_lut_neon_vs_scalar", comparison_names)
        self.assertIn("qjl_lut_quantized_vs_float_reference", comparison_names)
        self.assertIn("page_scan_block_local_vs_global_heap", comparison_names)

        first_comparison = payload["microbenchmarks"]["comparisons"][0]
        self.assertIn("comparison", first_comparison)
        self.assertIn("comparison_kind", first_comparison)
        self.assertIn("baseline_benchmark", first_comparison)
        self.assertIn("candidate_benchmark", first_comparison)
        self.assertIn("metrics", first_comparison)
        self.assertIn("codes_per_second_ratio", first_comparison["metrics"])
        self.assertIn("ns_per_op_ratio", first_comparison["metrics"])
        self.assertIn("visited_code_count_delta", first_comparison["metrics"])
        self.assertIn("visited_page_count_delta", first_comparison["metrics"])

        gate_names = {row["gate"] for row in payload["microbenchmarks"]["regression_gates"]}
        self.assertIn("avx2_kernel_speedup_signal", gate_names)
        self.assertIn("neon_kernel_speedup_signal", gate_names)
        self.assertIn("quantized_qjl_lut_signal", gate_names)
        self.assertIn("block_local_selection_signal", gate_names)

        first_gate = payload["microbenchmarks"]["regression_gates"][0]
        self.assertIn("gate", first_gate)
        self.assertIn("comparison", first_gate)
        self.assertIn("category", first_gate)
        self.assertIn("status", first_gate)
        self.assertIn("checks", first_gate)
        self.assertIn(first_gate["status"], ("pass", "warn", "not_applicable"))
        self.assertIn("same_workload", first_gate["checks"])
        self.assertIn("throughput_directional_signal", first_gate["checks"])

        self.assertIn("microbenchmark_regression", payload["report"])
        self.assertIn("comparisons", payload["report"]["microbenchmark_regression"])
        self.assertIn("regression_gates", payload["report"]["microbenchmark_regression"])

        markdown = BENCHMARK_SUITE.render_report_markdown(payload["report"])
        html = BENCHMARK_SUITE.render_report_html(payload["report"])
        self.assertIn("Microbenchmark Regression", markdown)
        self.assertIn("block_local_selection_signal", markdown)
        self.assertIn("Microbenchmark Regression", html)
        self.assertIn("block_local_selection_signal", html)

    def test_large_profile_matrix_contract(self):
        payload = self.run_suite(
            "--profile",
            "full",
            "--corpus",
            "normalized_dense,clustered,mixed_live_dead",
            "--methods",
            "turboquant_flat,turboquant_ivf,turboquant_bitmap,pgvector_ivfflat,pgvector_hnsw",
            "--report",
        )

        self.assertEqual(payload["profile"], "full")
        self.assertEqual(payload["scenario_matrix"]["profiles"], ["full"])
        self.assertEqual(len(payload["scenarios"]), 15)
        self.assertEqual(payload["report"]["summary"]["scenario_count"], 15)
        self.assertGreaterEqual(len(payload["report"]["comparisons"]), 3)

    def test_golden_acceptance_two_method_comparison(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "normalized_dense",
            "--methods",
            "turboquant_flat,pgvector_ivfflat",
        )

        self.assertEqual(len(payload["scenarios"]), 2)
        methods = [scenario["method"] for scenario in payload["scenarios"]]
        self.assertEqual(methods, ["turboquant_flat", "pgvector_ivfflat"])

        for scenario in payload["scenarios"]:
            self.assertEqual(scenario["corpus"], "normalized_dense")
            self.assertEqual(scenario["ground_truth"]["kind"], "exact")
            self.assertEqual(scenario["ground_truth"]["top_k"], [10, 100])
            self.assertIsInstance(scenario["metrics"]["index_size_bytes"], int)
            self.assertIsNotNone(scenario["metrics"]["candidate_slots_bound"])
            self.assertIn("build_wal_bytes", scenario["metrics"])
            self.assertIn("insert_wal_bytes", scenario["metrics"])
            self.assertIn("concurrent_insert_rows_per_second", scenario["metrics"])
            self.assertIn("maintenance_wal_bytes", scenario["metrics"])
            self.assertIn("query_knobs", scenario)
            self.assertEqual(scenario["query_api"]["helper"], "tq_rerank_candidates")
            self.assertIn("index_metadata", scenario)

        turboquant = payload["scenarios"][0]
        self.assertLess(
            turboquant["metrics"]["build_wal_bytes"],
            turboquant["metrics"]["sealed_baseline_build_wal_bytes"],
        )
        self.assertLess(
            turboquant["metrics"]["insert_wal_bytes"],
            turboquant["metrics"]["sealed_baseline_insert_wal_bytes"],
        )

    def test_rerank_candidate_limit_contract(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "normalized_dense",
            "--methods",
            "turboquant_flat",
            "--rerank-candidate-limit",
            "32",
        )

        scenario = payload["scenarios"][0]
        self.assertEqual(scenario["query_api"]["helper"], "tq_rerank_candidates")
        self.assertEqual(scenario["query_api"]["candidate_limit"], 32)
        self.assertEqual(scenario["query_api"]["final_limit"], 100)

    def test_decode_rescore_extra_candidates_expand_effective_limit_only(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "hotpot_skewed",
            "--methods",
            "turboquant_ivf",
            "--rerank-candidate-limit",
            "32",
            "--turboquant-decode-rescore-factor",
            "4",
            "--turboquant-decode-rescore-extra-candidates",
            "16",
        )

        scenario = payload["scenarios"][0]
        self.assertEqual(scenario["query_api"]["candidate_limit"], 32)
        self.assertEqual(scenario["query_api"]["effective_candidate_limit"], 116)
        self.assertEqual(scenario["query_knobs"]["turboquant.decode_rescore_extra_candidates"], 16)

    def test_decode_rescore_extra_candidates_auto_band_activates_by_default(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "hotpot_skewed",
            "--methods",
            "turboquant_ivf",
            "--rerank-candidate-limit",
            "1024",
            "--turboquant-decode-rescore-factor",
            "4",
        )

        scenario = payload["scenarios"][0]
        self.assertEqual(scenario["query_api"]["candidate_limit"], 1024)
        self.assertEqual(scenario["query_api"]["effective_candidate_limit"], 1536)
        self.assertEqual(scenario["query_knobs"]["turboquant.decode_rescore_extra_candidates"], 512)

    def test_decode_rescore_extra_candidates_explicit_zero_disables_auto_band(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "hotpot_skewed",
            "--methods",
            "turboquant_ivf",
            "--rerank-candidate-limit",
            "1024",
            "--turboquant-decode-rescore-factor",
            "4",
            "--turboquant-decode-rescore-extra-candidates",
            "0",
        )

        scenario = payload["scenarios"][0]
        self.assertEqual(scenario["query_api"]["effective_candidate_limit"], 1024)
        self.assertEqual(scenario["query_knobs"]["turboquant.decode_rescore_extra_candidates"], 0)

    def test_bitmap_filter_scenario_contract(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "normalized_dense",
            "--methods",
            "turboquant_bitmap",
        )

        scenario = payload["scenarios"][0]
        self.assertEqual(scenario["method"], "turboquant_bitmap")
        self.assertEqual(scenario["query_mode"], "bitmap_filter")
        self.assertEqual(scenario["query_api"]["helper"], "tq_bitmap_cosine_filter")
        self.assertIn("threshold", scenario["query_api"])
        self.assertIn("exact_match_fraction", scenario["metrics"])
        self.assertIn("avg_result_count", scenario["metrics"])
        self.assertTrue(scenario["index_metadata"]["capabilities"]["bitmap_scan"])
        self.assertTrue(scenario["index_metadata"]["capabilities"]["index_only_scan"])
        self.assertEqual(scenario["simd"]["selected_kernel"], scenario["simd"]["preferred_kernel"])
        self.assertEqual(scenario["scan_stats"]["score_kernel"], "none")
        self.assertEqual(scenario["simd"]["code_domain_kernel"], "none")

    def test_turboquant_capability_metadata_marks_multicolumn_support(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "normalized_dense",
            "--methods",
            "turboquant_flat",
        )

        scenario = payload["scenarios"][0]
        self.assertTrue(scenario["index_metadata"]["capabilities"]["multicolumn"])
        self.assertTrue(scenario["index_metadata"]["capabilities"]["include_columns"])

    def test_hotpot_skewed_profile_surfaces_scan_stats(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "hotpot_skewed",
            "--methods",
            "turboquant_ivf",
            "--turboquant-probes",
            "4",
            "--turboquant-max-visited-codes",
            "256",
        )

        scenario = payload["scenarios"][0]
        self.assertEqual(scenario["corpus"], "hotpot_skewed")
        self.assertEqual(scenario["method"], "turboquant_ivf")
        self.assertIn("scan_stats", scenario)
        self.assertEqual(scenario["query_knobs"]["turboquant.probes"], 4)
        self.assertEqual(scenario["query_knobs"]["turboquant.max_visited_codes"], 256)
        self.assertIn("selected_list_count", scenario["scan_stats"])
        self.assertIn("selected_live_count", scenario["scan_stats"])
        self.assertIn("selected_page_count", scenario["scan_stats"])
        self.assertIn("visited_page_count", scenario["scan_stats"])
        self.assertIn("visited_code_count", scenario["scan_stats"])
        self.assertIn("scan_orchestration", scenario["scan_stats"])
        self.assertIn("near_exhaustive_crossover", scenario["scan_stats"])
        self.assertIn("nominal_probe_count", scenario["scan_stats"])
        self.assertIn("effective_probe_count", scenario["scan_stats"])
        self.assertIn("max_visited_codes", scenario["scan_stats"])
        self.assertIn("max_visited_pages", scenario["scan_stats"])
        self.assertLessEqual(
            scenario["scan_stats"]["effective_probe_count"],
            scenario["scan_stats"]["nominal_probe_count"],
        )
        self.assertEqual(scenario["scan_stats"]["score_mode"], "code_domain")
        self.assertIn(scenario["scan_stats"]["score_kernel"], ("scalar", "avx2", "neon"))
        self.assertEqual(scenario["scan_stats"]["decoded_vector_count"], 0)
        self.assertEqual(scenario["simd"]["code_domain_kernel"], scenario["scan_stats"]["score_kernel"])

    def test_shadow_decode_diagnostics_flag_surfaces_in_query_setup(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "hotpot_skewed",
            "--methods",
            "turboquant_ivf",
            "--turboquant-shadow-decode-diagnostics",
        )

        scenario = payload["scenarios"][0]
        self.assertTrue(scenario["query_knobs"]["turboquant.shadow_decode_diagnostics"])
        self.assertEqual(scenario["scan_stats"]["shadow_decoded_vector_count"], 0)
        self.assertEqual(scenario["scan_stats"]["shadow_decode_candidate_count"], 0)
        self.assertEqual(scenario["scan_stats"]["shadow_decode_overlap_count"], 0)
        self.assertEqual(scenario["scan_stats"]["shadow_decode_primary_only_count"], 0)
        self.assertEqual(scenario["scan_stats"]["shadow_decode_only_count"], 0)

    def test_force_decode_score_diagnostics_flag_surfaces_in_query_setup(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "hotpot_skewed",
            "--methods",
            "turboquant_ivf",
            "--turboquant-force-decode-score-diagnostics",
        )

        scenario = payload["scenarios"][0]
        self.assertTrue(scenario["query_knobs"]["turboquant.force_decode_score_diagnostics"])

    def test_decode_rescore_factor_surfaces_in_query_setup(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "hotpot_skewed",
            "--methods",
            "turboquant_ivf",
            "--turboquant-decode-rescore-factor",
            "4",
        )

        scenario = payload["scenarios"][0]
        self.assertEqual(scenario["query_knobs"]["turboquant.decode_rescore_factor"], 4)
        self.assertEqual(scenario["scan_stats"]["score_mode"], "decode_rescore")

    def test_decode_rescore_extra_candidates_surfaces_in_query_knobs(self):
        payload = self.run_suite(
            "--profile",
            "tiny",
            "--corpus",
            "hotpot_skewed",
            "--methods",
            "turboquant_ivf",
            "--turboquant-decode-rescore-factor",
            "4",
            "--turboquant-decode-rescore-extra-candidates",
            "64",
        )

        scenario = payload["scenarios"][0]
        self.assertEqual(scenario["query_knobs"]["turboquant.decode_rescore_extra_candidates"], 64)
        self.assertEqual(
            scenario["query_api"]["effective_candidate_limit"],
            164,
        )

    def test_build_index_threads_qjl_and_rescore_knobs_without_positional_drift(self):
        corpus = BENCHMARK_SUITE.Corpus(
            name="normalized_dense",
            dimension=2,
            rows=[
                BENCHMARK_SUITE.Row(1, (1.0, 0.0)),
                BENCHMARK_SUITE.Row(2, (0.0, 1.0)),
            ],
            queries=[(1.0, 0.0)],
            metadata={"normalized": True},
        )

        with (
            mock.patch.object(
                BENCHMARK_SUITE,
                "method_spec",
                return_value={
                    "index_method": "turboquant",
                    "opclass": "tq_ip_ops",
                    "with": {"bits": 4, "lists": 0, "qjl_sketch_dim": 1},
                },
            ) as method_spec_mock,
            mock.patch.object(BENCHMARK_SUITE, "current_wal_lsn", return_value="0/0"),
            mock.patch.object(BENCHMARK_SUITE, "run_psql"),
            mock.patch.object(BENCHMARK_SUITE, "wal_bytes_since", return_value=0),
            mock.patch.object(BENCHMARK_SUITE, "query_psql", return_value="8192"),
        ):
            BENCHMARK_SUITE.build_index(
                ["psql"],
                "benchmark_items",
                "benchmark_items_embedding_idx",
                corpus,
                "turboquant_ivf",
                "ip",
                requested_rerank_candidate_limit=40,
                turboquant_probes=6,
                turboquant_oversample_factor=8,
                turboquant_max_visited_codes=1024,
                turboquant_max_visited_pages=32,
                turboquant_shadow_decode_diagnostics=True,
                turboquant_force_decode_score_diagnostics=False,
                turboquant_decode_rescore_factor=4,
                turboquant_decode_rescore_extra_candidates=64,
                turboquant_qjl_sketch_dim="half",
            )

        method_spec_mock.assert_called_once_with(
            method="turboquant_ivf",
            corpus=corpus,
            benchmark_metric="ip",
            requested_rerank_candidate_limit=40,
            turboquant_probes=6,
            turboquant_oversample_factor=8,
            turboquant_max_visited_codes=1024,
            turboquant_max_visited_pages=32,
            turboquant_shadow_decode_diagnostics=True,
            turboquant_force_decode_score_diagnostics=False,
            turboquant_decode_rescore_factor=4,
            turboquant_decode_rescore_extra_candidates=64,
            turboquant_qjl_sketch_dim="half",
        )

    def test_synthetic_skew_probe_regression_prefers_lower_work_without_recall_loss(self):
        regression = BENCHMARK_SUITE.synthetic_skew_probe_regression()

        self.assertIn("closest_centroid", regression)
        self.assertIn("cost_aware", regression)
        self.assertGreaterEqual(
            regression["cost_aware"]["recall_at_10"],
            regression["closest_centroid"]["recall_at_10"],
        )
        self.assertLess(
            regression["cost_aware"]["visited_code_count"],
            regression["closest_centroid"]["visited_code_count"],
        )
        self.assertLess(
            regression["cost_aware"]["visited_page_count"],
            regression["closest_centroid"]["visited_page_count"],
        )

    def test_hotpot_overlap_profile_surfaces_harder_boundary_metadata(self):
        payload = self.run_suite(
            "--profile",
            "medium",
            "--corpus",
            "hotpot_overlap",
            "--methods",
            "turboquant_ivf,pgvector_ivfflat,pgvector_hnsw",
        )

        self.assertEqual(len(payload["scenarios"]), 3)
        for scenario in payload["scenarios"]:
            self.assertEqual(scenario["corpus"], "hotpot_overlap")
            self.assertEqual(scenario["corpus_metadata"]["distribution"], "hotpot_overlap_ivf")
            self.assertTrue(scenario["corpus_metadata"]["normalized"])
            self.assertGreater(scenario["corpus_metadata"]["heavy_cluster_fraction"], 0.6)
            self.assertEqual(scenario["corpus_metadata"]["query_profile"], "boundary_blend")
            self.assertIn("cluster_count", scenario["corpus_metadata"])
            self.assertIn("overlap_noise", scenario["corpus_metadata"])
            if scenario["method"] == "turboquant_ivf":
                self.assertEqual(scenario["index"]["with"]["lists"], 64)
                self.assertEqual(scenario["query_knobs"]["turboquant.probes"], 8)
                self.assertEqual(scenario["query_knobs"]["turboquant.max_visited_codes"], 4096)
            elif scenario["method"] == "pgvector_ivfflat":
                self.assertEqual(scenario["index"]["with"]["lists"], 64)
                self.assertEqual(scenario["query_knobs"]["ivfflat.probes"], 8)
            elif scenario["method"] == "pgvector_hnsw":
                self.assertEqual(scenario["query_knobs"]["hnsw.ef_search"], 80)

    def test_insert_values_are_chunked_for_large_corpora(self):
        rows = [
            BENCHMARK_SUITE.Row(row_id, (0.1, 0.2, 0.3, 0.4))
            for row_id in range(1, 301)
        ]

        chunks = BENCHMARK_SUITE.chunked_insert_value_blocks(rows, max_rows_per_insert=128)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].count("\n"), 127)
        self.assertEqual(chunks[1].count("\n"), 127)
        self.assertIn("(1, 1, '[0.100000,0.200000,0.300000,0.400000]')", chunks[0])
        self.assertIn("(300, 0, '[0.100000,0.200000,0.300000,0.400000]')", chunks[-1])

    def test_query_turboquant_ordered_ids_and_scan_stats_uses_one_command_batch(self):
        command_batches = []

        def fake_query_psql_commands(base_cmd, *commands):
            command_batches.append((base_cmd, commands))
            return [
                '{"reranked_ids":[11,7,3],"approx_candidate_ids":[11,7,3,5]}',
                '{"mode":"ivf","score_mode":"code_domain","score_kernel":"scalar","visited_code_count":42}',
                '[11,5,9]',
            ]

        with mock.patch.object(
            BENCHMARK_SUITE,
            "query_psql_commands",
            side_effect=fake_query_psql_commands,
        ):
            query_ids, scan_stats, approx_candidate_ids, shadow_candidate_ids = BENCHMARK_SUITE.query_turboquant_ordered_ids_and_scan_stats(
                ["psql"],
                "benchmark_items",
                (0.25, 0.75),
                10,
                [
                    "SET LOCAL enable_seqscan = off",
                    "SET LOCAL turboquant.probes = 4",
                    "SET LOCAL turboquant.shadow_decode_diagnostics = on",
                    "SET LOCAL turboquant.force_decode_score_diagnostics = on",
                    "SET LOCAL turboquant.decode_rescore_factor = 4",
                    "SET LOCAL turboquant.decode_rescore_extra_candidates = 16",
                ],
                32,
                16,
            )

        self.assertEqual(query_ids, [11, 7, 3])
        self.assertEqual(approx_candidate_ids, [11, 7, 3, 5])
        self.assertEqual(shadow_candidate_ids, [11, 5, 9])
        self.assertEqual(scan_stats["visited_code_count"], 42)
        self.assertEqual(len(command_batches), 1)
        self.assertEqual(len(command_batches[0][1]), 1)
        sql_batch = command_batches[0][1][0]
        self.assertEqual(sql_batch.count("tq_rerank_candidates("), 0)
        self.assertEqual(sql_batch.count("ORDER BY embedding <=>"), 1)
        self.assertEqual(sql_batch.count("SELECT tq_last_scan_stats()::text"), 1)
        self.assertEqual(sql_batch.count("SELECT coalesce(json_agg(source.id ORDER BY shadow.ordinality), '[]'::json)::text"), 1)
        self.assertIn("BEGIN;", sql_batch)
        self.assertIn("COMMIT;", sql_batch)
        self.assertIn("SET LOCAL turboquant.probes = 4", sql_batch)
        self.assertIn("SET LOCAL turboquant.shadow_decode_diagnostics = on", sql_batch)
        self.assertIn("SET LOCAL turboquant.force_decode_score_diagnostics = on", sql_batch)
        self.assertIn("SET LOCAL turboquant.decode_rescore_factor = 4", sql_batch)
        self.assertIn("SET LOCAL turboquant.decode_rescore_extra_candidates = 16", sql_batch)
        self.assertNotIn("DO $$", sql_batch)
        self.assertNotIn("tq_resolve_query_knobs(", sql_batch)

    def test_turboquant_single_batch_sql_preserves_requested_turboquant_gucs(self):
        sql_batch = BENCHMARK_SUITE.turboquant_single_batch_rerank_ids_sql(
            "benchmark_items",
            (0.25, 0.75),
            10,
            [
                "SET LOCAL enable_seqscan = off",
                "SET LOCAL enable_bitmapscan = off",
                "SET LOCAL turboquant.probes = 4",
                "SET LOCAL turboquant.oversample_factor = 4",
                "SET LOCAL turboquant.max_visited_codes = 4096",
                "SET LOCAL turboquant.max_visited_pages = 0",
                "SET LOCAL turboquant.shadow_decode_diagnostics = on",
                "SET LOCAL turboquant.force_decode_score_diagnostics = on",
                "SET LOCAL turboquant.decode_rescore_factor = 4",
                "SET LOCAL turboquant.decode_rescore_extra_candidates = 16",
            ],
            None,
            16,
        )

        self.assertIn("SET LOCAL enable_seqscan = off", sql_batch)
        self.assertIn("SET LOCAL enable_bitmapscan = off", sql_batch)
        self.assertIn("BEGIN;", sql_batch)
        self.assertIn("tq_effective_rerank_candidate_limit(10, 10)", sql_batch)
        self.assertIn("SET LOCAL turboquant.probes = 4", sql_batch)
        self.assertIn("SET LOCAL turboquant.oversample_factor = 4", sql_batch)
        self.assertIn("SET LOCAL turboquant.max_visited_codes = 4096", sql_batch)
        self.assertIn("SET LOCAL turboquant.max_visited_pages = 0", sql_batch)
        self.assertIn("SET LOCAL turboquant.shadow_decode_diagnostics = on", sql_batch)
        self.assertIn("SET LOCAL turboquant.force_decode_score_diagnostics = on", sql_batch)
        self.assertIn("SET LOCAL turboquant.decode_rescore_factor = 4", sql_batch)
        self.assertIn("SET LOCAL turboquant.decode_rescore_extra_candidates = 16", sql_batch)
        self.assertNotIn("DO $$", sql_batch)
        self.assertNotIn("tq_resolve_query_knobs(", sql_batch)

    def test_run_scenario_turboquant_uses_new_helper_once_per_repetition(self):
        corpus = BENCHMARK_SUITE.Corpus(
            name="normalized_dense",
            dimension=2,
            rows=[
                BENCHMARK_SUITE.Row(1, (1.0, 0.0)),
                BENCHMARK_SUITE.Row(2, (0.0, 1.0)),
            ],
            queries=[(1.0, 0.0)],
            metadata={"normalized": True},
        )
        turbo_spec = BENCHMARK_SUITE.method_spec("turboquant_ivf", corpus)

        helper_calls = []

        def fake_helper(
            base_cmd,
            table_name,
            query_vector,
            limit,
            query_setup,
            requested_candidate_limit,
            decode_rescore_extra_candidates=0,
            method="turboquant_ivf",
        ):
            helper_calls.append(
                {
                    "table_name": table_name,
                    "query_vector": query_vector,
                    "limit": limit,
                    "query_setup": tuple(query_setup),
                    "requested_candidate_limit": requested_candidate_limit,
                    "decode_rescore_extra_candidates": decode_rescore_extra_candidates,
                    "method": method,
                }
            )
            return [1, 2], {
                "mode": "ivf",
                "score_mode": "code_domain",
                "score_kernel": "scalar",
                "visited_code_count": 8,
            }, [1], [2]

        with (
            mock.patch.object(BENCHMARK_SUITE, "load_corpus"),
            mock.patch.object(
                BENCHMARK_SUITE,
                "build_index",
                return_value=(0.01, 4096, 256, turbo_spec),
            ),
            mock.patch.object(BENCHMARK_SUITE, "fetch_index_metadata", return_value={"format_version": 6}),
            mock.patch.object(BENCHMARK_SUITE, "fetch_simd_metadata", return_value={"selected_kernel": "scalar"}),
            mock.patch.object(BENCHMARK_SUITE, "query_psql", return_value="8192"),
            mock.patch.object(BENCHMARK_SUITE, "measure_insert_wal", return_value=(0, 0)),
            mock.patch.object(
                BENCHMARK_SUITE,
                "measure_concurrent_insert_rows_per_second",
                return_value=(0.0, 0, 0),
            ),
            mock.patch.object(BENCHMARK_SUITE, "measure_maintenance_wal", return_value=(0, 0)),
            mock.patch.object(BENCHMARK_SUITE, "run_psql"),
            mock.patch.object(
                BENCHMARK_SUITE,
                "query_turboquant_ordered_ids_and_scan_stats",
                side_effect=fake_helper,
            ) as helper_mock,
            mock.patch.object(BENCHMARK_SUITE, "query_top_ids") as query_top_ids_mock,
            mock.patch.object(BENCHMARK_SUITE, "query_turboquant_ordered_scan_stats") as scan_stats_mock,
        ):
            scenario = BENCHMARK_SUITE.run_scenario(
                ["psql"],
                corpus,
                "turboquant_ivf",
                repetitions=3,
                scenario_index=1,
                turboquant_probes=None,
                turboquant_oversample_factor=None,
                turboquant_max_visited_codes=None,
                turboquant_max_visited_pages=None,
                turboquant_shadow_decode_diagnostics=False,
                turboquant_force_decode_score_diagnostics=False,
                turboquant_decode_rescore_factor=1,
                turboquant_decode_rescore_extra_candidates=0,
                requested_rerank_candidate_limit=32,
            )

        self.assertEqual(helper_mock.call_count, 3)
        self.assertEqual(len(helper_calls), 3)
        self.assertEqual(helper_calls[0]["requested_candidate_limit"], 32)
        self.assertEqual(helper_calls[0]["method"], "turboquant_ivf")
        query_top_ids_mock.assert_not_called()
        scan_stats_mock.assert_not_called()
        self.assertEqual(scenario["scan_stats"]["visited_code_count"], 8.0)
        self.assertEqual(scenario["simd"]["code_domain_kernel"], "scalar")
        self.assertEqual(scenario["candidate_retention"]["avg_candidate_count"], 1.0)
        self.assertEqual(scenario["candidate_retention"]["avg_exact_top_10_retention"], 0.5)
        self.assertEqual(scenario["candidate_retention"]["avg_exact_top_100_retention"], 0.5)
        self.assertEqual(scenario["candidate_retention"]["avg_exact_top_100_miss_count"], 1.0)
        self.assertEqual(scenario["candidate_retention"]["avg_shadow_candidate_count"], 1.0)
        self.assertEqual(scenario["candidate_retention"]["avg_shadow_exact_top_10_retention"], 0.5)
        self.assertEqual(scenario["candidate_retention"]["avg_shadow_exact_top_100_retention"], 0.5)
        self.assertEqual(scenario["candidate_retention"]["avg_shadow_exact_top_100_miss_count"], 1.0)

    def test_run_scenario_non_turboquant_keeps_existing_ordered_path(self):
        corpus = BENCHMARK_SUITE.Corpus(
            name="normalized_dense",
            dimension=2,
            rows=[
                BENCHMARK_SUITE.Row(1, (1.0, 0.0)),
                BENCHMARK_SUITE.Row(2, (0.0, 1.0)),
            ],
            queries=[(1.0, 0.0)],
            metadata={"normalized": True},
        )
        pgvector_spec = BENCHMARK_SUITE.method_spec("pgvector_ivfflat", corpus)

        with (
            mock.patch.object(BENCHMARK_SUITE, "load_corpus"),
            mock.patch.object(
                BENCHMARK_SUITE,
                "build_index",
                return_value=(0.01, 2048, 128, pgvector_spec),
            ),
            mock.patch.object(BENCHMARK_SUITE, "fetch_index_metadata", return_value={"format_version": 0}),
            mock.patch.object(BENCHMARK_SUITE, "fetch_simd_metadata", return_value={"selected_kernel": "scalar"}),
            mock.patch.object(BENCHMARK_SUITE, "query_psql", return_value="8192"),
            mock.patch.object(BENCHMARK_SUITE, "measure_insert_wal", return_value=(0, 0)),
            mock.patch.object(
                BENCHMARK_SUITE,
                "measure_concurrent_insert_rows_per_second",
                return_value=(0.0, 0, 0),
            ),
            mock.patch.object(BENCHMARK_SUITE, "measure_maintenance_wal", return_value=(0, 0)),
            mock.patch.object(BENCHMARK_SUITE, "run_psql"),
            mock.patch.object(
                BENCHMARK_SUITE,
                "query_top_ids",
                return_value=[1, 2],
            ) as query_top_ids_mock,
            mock.patch.object(
                BENCHMARK_SUITE,
                "query_turboquant_ordered_ids_and_scan_stats",
            ) as helper_mock,
        ):
            scenario = BENCHMARK_SUITE.run_scenario(
                ["psql"],
                corpus,
                "pgvector_ivfflat",
                repetitions=3,
                scenario_index=2,
                turboquant_probes=None,
                turboquant_oversample_factor=None,
                turboquant_max_visited_codes=None,
                turboquant_max_visited_pages=None,
                turboquant_shadow_decode_diagnostics=False,
                turboquant_force_decode_score_diagnostics=False,
                turboquant_decode_rescore_factor=1,
                turboquant_decode_rescore_extra_candidates=0,
                requested_rerank_candidate_limit=32,
            )

        self.assertEqual(query_top_ids_mock.call_count, 3)
        helper_mock.assert_not_called()
        self.assertEqual(scenario["scan_stats"]["mode"], "none")
        self.assertEqual(scenario["scan_stats"]["score_mode"], "none")

    def test_run_scenario_turboquant_three_repetitions_issue_three_batches_not_six(self):
        corpus = BENCHMARK_SUITE.Corpus(
            name="normalized_dense",
            dimension=2,
            rows=[
                BENCHMARK_SUITE.Row(1, (1.0, 0.0)),
                BENCHMARK_SUITE.Row(2, (0.0, 1.0)),
            ],
            queries=[(1.0, 0.0)],
            metadata={"normalized": True},
        )
        turbo_spec = BENCHMARK_SUITE.method_spec("turboquant_ivf", corpus)

        with (
            mock.patch.object(BENCHMARK_SUITE, "load_corpus"),
            mock.patch.object(
                BENCHMARK_SUITE,
                "build_index",
                return_value=(0.01, 4096, 256, turbo_spec),
            ),
            mock.patch.object(BENCHMARK_SUITE, "fetch_index_metadata", return_value={"format_version": 6}),
            mock.patch.object(BENCHMARK_SUITE, "fetch_simd_metadata", return_value={"selected_kernel": "scalar"}),
            mock.patch.object(BENCHMARK_SUITE, "query_psql", return_value="8192"),
            mock.patch.object(BENCHMARK_SUITE, "measure_insert_wal", return_value=(0, 0)),
            mock.patch.object(
                BENCHMARK_SUITE,
                "measure_concurrent_insert_rows_per_second",
                return_value=(0.0, 0, 0),
            ),
            mock.patch.object(BENCHMARK_SUITE, "measure_maintenance_wal", return_value=(0, 0)),
            mock.patch.object(BENCHMARK_SUITE, "run_psql"),
            mock.patch.object(
                BENCHMARK_SUITE,
                "query_psql_commands",
                return_value=[
                    '{"reranked_ids":[1,2],"approx_candidate_ids":[1,2]}',
                    '{"mode":"ivf","score_mode":"code_domain","score_kernel":"scalar","visited_code_count":8}',
                    '[1,2]',
                ],
            ) as query_psql_commands_mock,
        ):
            BENCHMARK_SUITE.run_scenario(
                ["psql"],
                corpus,
                "turboquant_ivf",
                repetitions=3,
                scenario_index=3,
                turboquant_probes=None,
                turboquant_oversample_factor=None,
                turboquant_max_visited_codes=None,
                turboquant_max_visited_pages=None,
                turboquant_shadow_decode_diagnostics=False,
                turboquant_force_decode_score_diagnostics=False,
                turboquant_decode_rescore_factor=1,
                turboquant_decode_rescore_extra_candidates=0,
                requested_rerank_candidate_limit=32,
            )

        self.assertEqual(query_psql_commands_mock.call_count, 3)

    def test_run_scenario_uses_all_repetition_latencies_for_percentiles(self):
        corpus = BENCHMARK_SUITE.Corpus(
            name="normalized_dense",
            dimension=2,
            rows=[
                BENCHMARK_SUITE.Row(1, (1.0, 0.0)),
                BENCHMARK_SUITE.Row(2, (0.0, 1.0)),
            ],
            queries=[(1.0, 0.0)],
            metadata={"normalized": True},
        )
        pgvector_spec = BENCHMARK_SUITE.method_spec("pgvector_ivfflat", corpus)

        with (
            mock.patch.object(BENCHMARK_SUITE, "load_corpus"),
            mock.patch.object(
                BENCHMARK_SUITE,
                "build_index",
                return_value=(0.01, 2048, 128, pgvector_spec),
            ),
            mock.patch.object(BENCHMARK_SUITE, "fetch_index_metadata", return_value={"format_version": 0}),
            mock.patch.object(BENCHMARK_SUITE, "fetch_simd_metadata", return_value={"selected_kernel": "scalar"}),
            mock.patch.object(BENCHMARK_SUITE, "query_psql", return_value="8192"),
            mock.patch.object(BENCHMARK_SUITE, "measure_insert_wal", return_value=(0, 0)),
            mock.patch.object(
                BENCHMARK_SUITE,
                "measure_concurrent_insert_rows_per_second",
                return_value=(0.0, 0, 0),
            ),
            mock.patch.object(BENCHMARK_SUITE, "measure_maintenance_wal", return_value=(0, 0)),
            mock.patch.object(BENCHMARK_SUITE, "run_psql"),
            mock.patch.object(BENCHMARK_SUITE, "query_top_ids", return_value=[1, 2]) as query_top_ids_mock,
            mock.patch.object(BENCHMARK_SUITE.time, "perf_counter", side_effect=[0.0, 0.010, 1.0, 1.050, 2.0, 2.090]),
        ):
            scenario = BENCHMARK_SUITE.run_scenario(
                ["psql"],
                corpus,
                "pgvector_ivfflat",
                repetitions=3,
                scenario_index=4,
                turboquant_probes=None,
                turboquant_oversample_factor=None,
                turboquant_max_visited_codes=None,
                turboquant_max_visited_pages=None,
                turboquant_shadow_decode_diagnostics=False,
                turboquant_force_decode_score_diagnostics=False,
                turboquant_decode_rescore_factor=1,
                turboquant_decode_rescore_extra_candidates=0,
                requested_rerank_candidate_limit=32,
            )

        self.assertEqual(query_top_ids_mock.call_count, 3)
        self.assertEqual(scenario["metrics"]["p50_ms"], 50.0)
        self.assertEqual(scenario["metrics"]["p95_ms"], 90.0)


if __name__ == "__main__":
    unittest.main()
