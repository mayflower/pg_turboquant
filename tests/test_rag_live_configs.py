import unittest
from pathlib import Path

from benchmarks.rag.ingestion_pipeline import load_campaign_config


ROOT = Path(__file__).resolve().parents[1]
POPQA_CONFIG = ROOT / "benchmarks" / "rag" / "configs" / "live" / "popqa_small_live.json"
KILT_NQ_CONFIG = ROOT / "benchmarks" / "rag" / "configs" / "live" / "kilt_nq_small_live.json"
KILT_HOTPOTQA_CONFIG = ROOT / "benchmarks" / "rag" / "configs" / "live" / "kilt_hotpotqa_small_live.json"
KILT_HOTPOTQA_IVF_CONFIG = ROOT / "benchmarks" / "rag" / "configs" / "live" / "kilt_hotpotqa_ivf_live.json"


class RagLiveConfigContractTest(unittest.TestCase):
    def test_popqa_small_live_config_parses(self):
        config = load_campaign_config(POPQA_CONFIG)

        self.assertEqual(config.dataset.name, "popqa_small_live")
        self.assertEqual(config.embedding.model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.embedding.dimension, 384)
        self.assertEqual(config.schema["passages_table"], "rag_passages")
        self.assertEqual(len(config.backends), 3)

    def test_kilt_nq_small_live_config_parses(self):
        config = load_campaign_config(KILT_NQ_CONFIG)

        self.assertEqual(config.dataset.name, "kilt_nq_small_live")
        self.assertEqual(config.embedding.model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.embedding.dimension, 384)
        self.assertEqual(config.schema["passages_table"], "rag_passages")
        self.assertEqual(len(config.backends), 3)

    def test_kilt_hotpotqa_small_live_config_parses(self):
        config = load_campaign_config(KILT_HOTPOTQA_CONFIG)

        self.assertEqual(config.dataset.name, "kilt_hotpotqa_small_live")
        self.assertEqual(config.embedding.model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.embedding.dimension, 384)
        self.assertEqual(config.schema["passages_table"], "rag_passages")
        self.assertEqual(len(config.backends), 3)

    def test_kilt_hotpotqa_ivf_live_config_parses(self):
        config = load_campaign_config(KILT_HOTPOTQA_IVF_CONFIG)

        self.assertEqual(config.dataset.name, "kilt_hotpotqa_ivf_live")
        self.assertEqual(config.embedding.model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.embedding.dimension, 384)
        self.assertEqual(config.schema["passages_table"], "rag_passages")
        self.assertEqual(config.backends[0]["kind"], "pg_turboquant")
        self.assertEqual(config.backends[0]["options"]["lists"], 64)
        self.assertEqual(len(config.backends), 3)


if __name__ == "__main__":
    unittest.main()
