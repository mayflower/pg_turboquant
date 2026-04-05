import unittest
from pathlib import Path

from benchmarks.rag.ingestion_pipeline import load_campaign_config


ROOT = Path(__file__).resolve().parents[1]
POPQA_CONFIG = ROOT / "benchmarks" / "rag" / "configs" / "live" / "popqa_small_live.json"
KILT_NQ_CONFIG = ROOT / "benchmarks" / "rag" / "configs" / "live" / "kilt_nq_small_live.json"
KILT_HOTPOTQA_CONFIG = ROOT / "benchmarks" / "rag" / "configs" / "live" / "kilt_hotpotqa_small_live.json"
KILT_HOTPOTQA_IVF_CONFIG = ROOT / "benchmarks" / "rag" / "configs" / "live" / "kilt_hotpotqa_ivf_live.json"
KILT_HOTPOTQA_FILTERED_CHURN_CONFIG = (
    ROOT / "benchmarks" / "rag" / "configs" / "live" / "kilt_hotpotqa_filtered_churn_live.json"
)
KILT_HOTPOTQA_FILTERED_EXTERNAL_DELTA_CONFIG = (
    ROOT / "benchmarks" / "rag" / "configs" / "live" / "kilt_hotpotqa_filtered_external_delta_live.json"
)


class RagLiveConfigContractTest(unittest.TestCase):
    def test_popqa_small_live_config_parses(self):
        config = load_campaign_config(POPQA_CONFIG)

        self.assertEqual(config.dataset.name, "popqa_small_live")
        self.assertEqual(config.embedding.model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.embedding.dimension, 384)
        self.assertTrue(config.embedding.normalized)
        self.assertEqual(config.schema["passages_table"], "rag_passages")
        self.assertEqual(config.backends[0]["kind"], "pg_turboquant")
        self.assertEqual(config.backends[0]["options"]["lists"], 64)
        self.assertEqual(config.backends[0]["options"]["router_restarts"], 3)
        self.assertEqual(config.backends[0]["ann"]["probes"], 8)
        self.assertEqual(config.backends[0]["ann"]["oversampling"], 4)
        self.assertEqual(len(config.backends), 3)

    def test_kilt_nq_small_live_config_parses(self):
        config = load_campaign_config(KILT_NQ_CONFIG)

        self.assertEqual(config.dataset.name, "kilt_nq_small_live")
        self.assertEqual(config.embedding.model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.embedding.dimension, 384)
        self.assertTrue(config.embedding.normalized)
        self.assertEqual(config.schema["passages_table"], "rag_passages")
        self.assertEqual(config.backends[0]["kind"], "pg_turboquant")
        self.assertEqual(config.backends[0]["options"]["lists"], 64)
        self.assertEqual(config.backends[0]["options"]["router_restarts"], 3)
        self.assertEqual(config.backends[0]["ann"]["probes"], 8)
        self.assertEqual(config.backends[0]["ann"]["oversampling"], 4)
        self.assertEqual(len(config.backends), 3)

    def test_kilt_hotpotqa_small_live_config_parses(self):
        config = load_campaign_config(KILT_HOTPOTQA_CONFIG)

        self.assertEqual(config.dataset.name, "kilt_hotpotqa_small_live")
        self.assertEqual(config.embedding.model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.embedding.dimension, 384)
        self.assertTrue(config.embedding.normalized)
        self.assertEqual(config.schema["passages_table"], "rag_passages")
        self.assertEqual(config.backends[0]["kind"], "pg_turboquant")
        self.assertEqual(config.backends[0]["options"]["lists"], 64)
        self.assertEqual(config.backends[0]["options"]["router_restarts"], 3)
        self.assertEqual(config.backends[0]["ann"]["probes"], 8)
        self.assertEqual(config.backends[0]["ann"]["oversampling"], 4)
        self.assertEqual(len(config.backends), 3)

    def test_kilt_hotpotqa_ivf_live_config_parses(self):
        config = load_campaign_config(KILT_HOTPOTQA_IVF_CONFIG)

        self.assertEqual(config.dataset.name, "kilt_hotpotqa_ivf_live")
        self.assertEqual(config.embedding.model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.embedding.dimension, 384)
        self.assertEqual(config.schema["passages_table"], "rag_passages")
        self.assertEqual(config.backends[0]["kind"], "pg_turboquant")
        self.assertEqual(config.backends[0]["options"]["lists"], 64)
        self.assertEqual(config.backends[0]["options"]["router_restarts"], 3)
        self.assertEqual(config.backends[0]["filter_columns"], ["tenant_id", "source_id", "lang_id"])
        self.assertEqual(config.backends[0]["include_columns"], ["doc_id_int", "chunk_id_int", "doc_version"])
        self.assertEqual(config.backends[0]["ann"]["probes"], 8)
        self.assertEqual(config.backends[0]["ann"]["oversampling"], 4)
        self.assertEqual(config.backends[0]["ann"]["max_visited_codes"], 4096)
        self.assertEqual(config.backends[0]["ann"]["max_visited_pages"], 0)
        self.assertEqual(config.backends[0]["ann"]["filters"], {"tenant_id": 1, "source_id": [1, 2, 3], "lang_id": 1})
        self.assertEqual(
            config.backends[0]["ann"]["stage1_payload_columns"],
            ["doc_id_int", "chunk_id_int", "tenant_id", "doc_version"],
        )
        self.assertEqual(config.backends[0]["ann"]["iterative_scan"], "strict_order")
        self.assertEqual(config.backends[0]["ann"]["min_rows_after_filter"], 20)
        self.assertEqual(config.regression_gate["dataset_id"], "kilt_hotpotqa")
        self.assertEqual(config.regression_gate["method_id"], "pg_turboquant_approx")
        self.assertEqual(config.regression_gate["recall_at_10_floor"], 0.90)
        self.assertEqual(config.regression_gate["max_visited_code_fraction"], 0.85)
        self.assertEqual(config.regression_gate["max_visited_page_fraction"], 0.60)
        self.assertEqual(config.regression_gate["expected_score_mode"], "code_domain")
        self.assertEqual(config.regression_gate["max_effective_probe_count"], 8)
        self.assertEqual(len(config.backends), 3)

    def test_kilt_hotpotqa_filtered_churn_live_config_parses(self):
        config = load_campaign_config(KILT_HOTPOTQA_FILTERED_CHURN_CONFIG)

        self.assertEqual(config.dataset.name, "kilt_hotpotqa_filtered_churn_live")
        self.assertEqual(config.embedding.model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.embedding.dimension, 384)
        self.assertEqual(config.backends[0]["kind"], "pg_turboquant")
        self.assertEqual(config.backends[0]["filter_columns"], ["tenant_id", "source_id", "lang_id", "doc_version"])
        self.assertTrue(config.backends[0]["ann"]["native_delta"])
        self.assertEqual(
            config.backends[0]["ann"]["churn_profile"],
            {
                "insert_rows": 500,
                "delete_fraction": 0.1,
                "reembed_fraction": 0.2,
                "maintenance_after_batches": 2,
            },
        )
        self.assertEqual(config.regression_gate["dataset_id"], "kilt_hotpotqa")
        self.assertEqual(config.regression_gate["expected_score_mode"], "code_domain")
        self.assertEqual(len(config.backends), 3)

    def test_kilt_hotpotqa_filtered_external_delta_live_config_parses(self):
        config = load_campaign_config(KILT_HOTPOTQA_FILTERED_EXTERNAL_DELTA_CONFIG)

        self.assertEqual(config.dataset.name, "kilt_hotpotqa_filtered_external_delta_live")
        self.assertEqual(config.embedding.model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.embedding.dimension, 384)
        self.assertEqual(config.backends[0]["kind"], "pg_turboquant")
        self.assertEqual(config.backends[0]["filter_columns"], ["tenant_id", "source_id", "lang_id", "doc_version"])
        self.assertEqual(config.backends[0]["ann"]["delta_table_name"], "rag_passages_delta")
        self.assertEqual(config.backends[0]["ann"]["delta_candidate_limit"], 64)
        self.assertNotIn("native_delta", config.backends[0]["ann"])
        self.assertEqual(config.regression_gate["dataset_id"], "kilt_hotpotqa")
        self.assertEqual(config.regression_gate["expected_score_mode"], "code_domain")
        self.assertEqual(len(config.backends), 3)


if __name__ == "__main__":
    unittest.main()
