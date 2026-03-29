import unittest

from benchmarks.rag.dataset_pack import (
    DATASET_CONFIG_DIR,
    dataset_config_paths,
    load_dataset_config,
    load_primary_dataset_pack,
    resolve_benchmark_plan,
)


class RagDatasetPackContractTest(unittest.TestCase):
    def test_primary_dataset_config_schema_validation(self):
        config = load_dataset_config(DATASET_CONFIG_DIR / "kilt_nq.json")

        self.assertEqual(config["dataset_id"], "kilt_nq")
        self.assertEqual(config["retrieval_profile"]["top_k_default"], 20)
        self.assertTrue(config["evidence"]["enabled"])
        self.assertIn("factual_retrieval", config["capabilities"])
        self.assertIn("answer_exact_match", config["answer_metrics"])

    def test_all_primary_dataset_configs_resolve_to_usable_plans(self):
        plans = [resolve_benchmark_plan(load_dataset_config(path)) for path in dataset_config_paths()]

        dataset_ids = [plan["dataset_id"] for plan in plans]
        self.assertEqual(
            dataset_ids,
            ["asqa", "kilt_hotpotqa", "kilt_nq", "kilt_triviaqa", "popqa"],
        )

        for plan in plans:
            self.assertIn("dataset_id", plan)
            self.assertIn("source", plan)
            self.assertIn("retrieval_top_k", plan)
            self.assertIn("answer_metrics", plan)
            self.assertIn("stable_passage_id_fields", plan)
            self.assertGreater(plan["retrieval_top_k"], 0)

    def test_primary_dataset_pack_loads_expected_ids(self):
        pack = load_primary_dataset_pack()
        self.assertEqual(
            sorted(pack.keys()),
            ["asqa", "kilt_hotpotqa", "kilt_nq", "kilt_triviaqa", "popqa"],
        )
        self.assertEqual(pack["kilt_hotpotqa"]["stress_focus"], "multi_hop_retrieval")
        self.assertEqual(pack["asqa"]["stress_focus"], "ambiguous_long_form_support")
        self.assertIn("supporting_fact_alignment", pack["kilt_hotpotqa"]["capabilities"])


if __name__ == "__main__":
    unittest.main()
