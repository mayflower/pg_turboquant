import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
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
        self.assertIn("ground_truth", scenario)
        self.assertIn("metrics", scenario)
        self.assertIn("index", scenario)
        self.assertIn("query_knobs", scenario)
        self.assertIn("query_api", scenario)
        self.assertIn("index_metadata", scenario)
        self.assertIn("simd", scenario)
        self.assertIn("scan_stats", scenario)

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
        self.assertIn("turboquant.probes", scenario["query_knobs"])
        self.assertEqual(scenario["query_api"]["helper"], "tq_rerank_candidates")
        self.assertIn("candidate_limit", scenario["query_api"])
        self.assertIn("final_limit", scenario["query_api"])
        self.assertIn("format_version", scenario["index_metadata"])
        self.assertIn("metric", scenario["index_metadata"])
        self.assertIn("list_count", scenario["index_metadata"])
        self.assertIn("capabilities", scenario["index_metadata"])
        self.assertIn("index_only_scan", scenario["index_metadata"]["capabilities"])
        self.assertIn("multicolumn", scenario["index_metadata"]["capabilities"])
        self.assertIn("include_columns", scenario["index_metadata"]["capabilities"])
        self.assertIn("bitmap_scan", scenario["index_metadata"]["capabilities"])
        self.assertIn("preferred_kernel", scenario["simd"])
        self.assertIn("compiled", scenario["simd"])
        self.assertIn("runtime_available", scenario["simd"])
        self.assertIn("selected_kernel", scenario["simd"])
        self.assertIn("code_domain_kernel", scenario["simd"])
        self.assertIn("mode", scenario["scan_stats"])
        self.assertIn("score_mode", scenario["scan_stats"])
        self.assertIn("score_kernel", scenario["scan_stats"])
        self.assertIn("selected_list_count", scenario["scan_stats"])
        self.assertIn("selected_live_count", scenario["scan_stats"])
        self.assertIn("visited_page_count", scenario["scan_stats"])
        self.assertIn("visited_code_count", scenario["scan_stats"])
        self.assertIn("nominal_probe_count", scenario["scan_stats"])
        self.assertIn("effective_probe_count", scenario["scan_stats"])
        self.assertIn("max_visited_codes", scenario["scan_stats"])
        self.assertIn("max_visited_pages", scenario["scan_stats"])
        self.assertIn("candidate_heap_count", scenario["scan_stats"])
        if scenario["method"] in {"turboquant_flat", "turboquant_ivf"}:
            self.assertEqual(scenario["scan_stats"]["score_mode"], "code_domain")
            self.assertEqual(scenario["scan_stats"]["decoded_vector_count"], 0)
        self.assertIn("python_version", payload["environment"])
        self.assertIn("platform", payload["environment"])
        self.assertIn("cpu_arch", payload["environment"])
        self.assertIn("methods", payload["scenario_matrix"])
        self.assertIn("corpora", payload["scenario_matrix"])
        self.assertIn("profiles", payload["scenario_matrix"])

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
        self.assertIn("comparisons", payload["report"])
        self.assertIn("leaderboards", payload["report"])
        self.assertIn("generated_at", payload["report"])
        self.assertIn("report_json", payload["artifacts"])
        self.assertIn("report_markdown", payload["artifacts"])

        summary = payload["report"]["summary"]
        self.assertGreater(summary["scenario_count"], 0)
        self.assertIn("methods", summary)
        self.assertIn("corpora", summary)

        comparison = payload["report"]["comparisons"][0]
        self.assertIn("corpus", comparison)
        self.assertIn("baseline_method", comparison)
        self.assertIn("candidate_method", comparison)
        self.assertIn("metrics", comparison)
        self.assertIn("recall_at_10_delta", comparison["metrics"])
        self.assertIn("p95_ms_delta", comparison["metrics"])
        self.assertIn("build_seconds_delta", comparison["metrics"])
        self.assertIn("index_size_bytes_delta", comparison["metrics"])
        self.assertIn("build_wal_bytes_delta", comparison["metrics"])

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
        self.assertFalse(scenario["index_metadata"]["capabilities"]["index_only_scan"])
        self.assertEqual(scenario["simd"]["selected_kernel"], scenario["simd"]["preferred_kernel"])
        self.assertEqual(scenario["scan_stats"]["score_kernel"], "none")
        self.assertEqual(scenario["simd"]["code_domain_kernel"], "none")

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
        self.assertIn("visited_page_count", scenario["scan_stats"])
        self.assertIn("visited_code_count", scenario["scan_stats"])
        self.assertIn("nominal_probe_count", scenario["scan_stats"])
        self.assertIn("effective_probe_count", scenario["scan_stats"])
        self.assertIn("max_visited_codes", scenario["scan_stats"])
        self.assertIn("max_visited_pages", scenario["scan_stats"])
        self.assertLessEqual(
            scenario["scan_stats"]["effective_probe_count"],
            scenario["scan_stats"]["nominal_probe_count"],
        )
        self.assertEqual(scenario["scan_stats"]["score_mode"], "code_domain")
        self.assertIn(scenario["scan_stats"]["score_kernel"], ("scalar", "avx2"))
        self.assertEqual(scenario["scan_stats"]["decoded_vector_count"], 0)
        self.assertEqual(scenario["simd"]["code_domain_kernel"], scenario["scan_stats"]["score_kernel"])

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


if __name__ == "__main__":
    unittest.main()
