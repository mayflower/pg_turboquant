import json
import os
import sys
import tempfile
import unittest
from types import ModuleType
from pathlib import Path
from unittest import mock

from benchmarks.rag.ingestion_pipeline import (
    CampaignConfig,
    ChunkingConfig,
    DatasetConfig,
    EmbeddingConfig,
    build_hf_corpus,
    build_backend_indexes,
    _default_hf_dataset_loader,
    ensure_schema,
    load_campaign_config,
    parse_hf_source_path,
    run_hf_ingestion,
    run_campaign,
    build_index_sql,
)
from benchmarks.rag.run_ingestion_pipeline import build_embedder


class FakeCursor:
    def __init__(self):
        self.executed = []
        self.rows = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def executemany(self, sql, seq_of_params):
        for params in seq_of_params:
            self.executed.append((sql, params))

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return (None,)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self):
        self.cursors = []
        self.commits = 0

    def cursor(self):
        cursor = FakeCursor()
        self.cursors.append(cursor)
        return cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class RagIngestionPipelineContractTest(unittest.TestCase):
    def write_config(self, payload):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "campaign.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_config_parsing_records_manifest_fields(self):
        config_path = self.write_config(
            {
                "dataset": {
                    "name": "tiny_corpus",
                    "version": "2026-03-28",
                    "source_path": "fixtures/tiny.jsonl",
                },
                "embedding": {
                    "model": "test-embed-small",
                    "dimension": 4,
                    "normalized": True,
                },
                "chunking": {
                    "strategy": "fixed_tokens",
                    "chunk_size": 128,
                    "chunk_overlap": 16,
                },
                "schema": {
                    "documents_table": "rag_documents",
                    "passages_table": "rag_passages",
                },
                "backends": [
                    {
                        "kind": "pg_turboquant",
                        "index_name": "rag_passages_tq_idx",
                        "metric": "cosine",
                        "mode": "approx",
                        "options": {"lists": 0},
                    },
                    {
                        "kind": "pgvector_hnsw",
                        "index_name": "rag_passages_hnsw_idx",
                        "metric": "cosine",
                        "mode": "approx",
                    },
                ],
            }
        )

        config = load_campaign_config(config_path)
        self.assertEqual(config.dataset.name, "tiny_corpus")
        self.assertEqual(config.dataset.version, "2026-03-28")
        self.assertEqual(config.embedding.model, "test-embed-small")
        self.assertEqual(config.embedding.dimension, 4)
        self.assertTrue(config.embedding.normalized)
        self.assertEqual(config.chunking.chunk_size, 128)
        self.assertEqual(config.chunking.chunk_overlap, 16)
        self.assertEqual(config.schema["documents_table"], "rag_documents")
        self.assertEqual(config.backends[0]["kind"], "pg_turboquant")

    def test_idempotent_schema_setup_uses_if_not_exists(self):
        config = CampaignConfig(
            dataset=DatasetConfig(name="tiny", version="v1", source_path="tiny.jsonl"),
            embedding=EmbeddingConfig(model="embed", dimension=3, normalized=True),
            chunking=ChunkingConfig(strategy="fixed", chunk_size=64, chunk_overlap=8),
            schema={"documents_table": "rag_documents", "passages_table": "rag_passages"},
            backends=[],
        )
        connection = FakeConnection()

        ensure_schema(connection, config)

        executed_sql = "\n".join(sql for cursor in connection.cursors for sql, _ in cursor.executed)
        self.assertIn("CREATE TABLE IF NOT EXISTS rag_documents", executed_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS rag_passages", executed_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS rag_campaign_manifest", executed_sql)

    def test_tiny_corpus_ingests_and_builds_all_configured_backends(self):
        config = CampaignConfig(
            dataset=DatasetConfig(name="tiny", version="v1", source_path="tiny.jsonl"),
            embedding=EmbeddingConfig(model="embed-small", dimension=3, normalized=True),
            chunking=ChunkingConfig(strategy="fixed", chunk_size=64, chunk_overlap=0),
            schema={"documents_table": "rag_documents", "passages_table": "rag_passages"},
            backends=[
                {
                    "kind": "pg_turboquant",
                    "index_name": "rag_passages_tq_idx",
                    "metric": "cosine",
                    "mode": "approx",
                    "options": {"lists": 0},
                },
                {
                    "kind": "pgvector_hnsw",
                    "index_name": "rag_passages_hnsw_idx",
                    "metric": "cosine",
                    "mode": "approx",
                },
                {
                    "kind": "pgvector_ivfflat",
                    "index_name": "rag_passages_ivf_idx",
                    "metric": "cosine",
                    "mode": "approx",
                },
            ],
        )
        corpus = [
            {
                "document_id": "doc-1",
                "title": "Doc 1",
                "passages": [
                    {"passage_id": "doc-1:p1", "text": "alpha"},
                    {"passage_id": "doc-1:p2", "text": "beta"},
                ],
            },
            {
                "document_id": "doc-2",
                "title": "Doc 2",
                "passages": [{"passage_id": "doc-2:p1", "text": "gamma"}],
            },
        ]

        def fake_embedder(texts):
            vectors = []
            for index, text in enumerate(texts):
                length = float(len(text))
                vectors.append([length, float(index), 1.0])
            return vectors

        connection = FakeConnection()
        manifest = run_campaign(connection, config, corpus, fake_embedder)

        self.assertEqual(manifest["dataset_version"], "v1")
        self.assertEqual(manifest["embedding_model"], "embed-small")
        self.assertEqual(manifest["dimension"], 3)
        self.assertEqual(manifest["normalization"], True)
        self.assertEqual(manifest["chunking"]["chunk_size"], 64)

        executed_sql = [(sql, params) for cursor in connection.cursors for sql, params in cursor.executed]
        merged_sql = "\n".join(sql for sql, _ in executed_sql)
        self.assertIn("INSERT INTO rag_campaign_manifest", merged_sql)
        self.assertIn("INSERT INTO rag_documents", merged_sql)
        self.assertIn("INSERT INTO rag_passages", merged_sql)
        self.assertIn("CREATE INDEX IF NOT EXISTS rag_passages_tq_idx", merged_sql)
        self.assertIn("CREATE INDEX IF NOT EXISTS rag_passages_hnsw_idx", merged_sql)
        self.assertIn("CREATE INDEX IF NOT EXISTS rag_passages_ivf_idx", merged_sql)

        insert_params = [params for sql, params in executed_sql if "INSERT INTO rag_passages" in sql]
        self.assertEqual(len(insert_params), 3)
        self.assertEqual(insert_params[0][0], "doc-1:p1")
        self.assertEqual(insert_params[0][1], "doc-1")
        self.assertEqual(insert_params[0][-1], "[5.0,0.0,1.0]")

    def test_build_backend_indexes_rejects_unknown_backend_kind(self):
        config = CampaignConfig(
            dataset=DatasetConfig(name="tiny", version="v1", source_path="tiny.jsonl"),
            embedding=EmbeddingConfig(model="embed-small", dimension=3, normalized=True),
            chunking=ChunkingConfig(strategy="fixed", chunk_size=64, chunk_overlap=0),
            schema={"documents_table": "rag_documents", "passages_table": "rag_passages"},
            backends=[{"kind": "mystery_backend", "index_name": "bad_idx", "metric": "cosine"}],
        )

        connection = FakeConnection()
        with self.assertRaisesRegex(ValueError, "unsupported backend kind"):
            build_backend_indexes(connection, config)

    def test_schema_includes_rag_filter_and_covering_columns(self):
        config = CampaignConfig(
            dataset=DatasetConfig(name="tiny", version="v1", source_path="tiny.jsonl"),
            embedding=EmbeddingConfig(model="embed", dimension=3, normalized=True),
            chunking=ChunkingConfig(strategy="fixed", chunk_size=64, chunk_overlap=8),
            schema={"documents_table": "rag_documents", "passages_table": "rag_passages"},
            backends=[],
        )
        connection = FakeConnection()

        ensure_schema(connection, config)

        executed_sql = "\n".join(sql for cursor in connection.cursors for sql, _ in cursor.executed)
        self.assertIn("tenant_id integer NOT NULL", executed_sql)
        self.assertIn("source_id integer NOT NULL", executed_sql)
        self.assertIn("lang_id integer NOT NULL", executed_sql)
        self.assertIn("doc_version integer NOT NULL", executed_sql)
        self.assertIn("doc_id_int integer NOT NULL", executed_sql)
        self.assertIn("chunk_id_int integer NOT NULL", executed_sql)

    def test_turboquant_index_sql_supports_multiple_filter_keys_and_include_payload(self):
        sql = build_index_sql(
            "rag_passages",
            "pg_turboquant",
            "rag_passages_tq_idx",
            "cosine",
            {
                "kind": "pg_turboquant",
                "index_name": "rag_passages_tq_idx",
                "metric": "cosine",
                "filter_columns": ["tenant_id", "source_id", "lang_id"],
                "include_columns": ["doc_id_int", "chunk_id_int", "doc_version"],
                "options": {"lists": 32, "normalized": True},
            },
        )

        self.assertIn("USING turboquant (embedding tq_cosine_ops, tenant_id tq_int4_filter_ops, source_id tq_int4_filter_ops, lang_id tq_int4_filter_ops)", sql)
        self.assertIn("INCLUDE (doc_id_int, chunk_id_int, doc_version)", sql)

    def test_delta_merge_backend_config_emits_base_and_delta_indexes(self):
        sql = build_index_sql(
            "rag_passages_delta",
            "pg_turboquant",
            "rag_passages_delta_tq_idx",
            "cosine",
            {
                "kind": "pg_turboquant",
                "index_name": "rag_passages_delta_tq_idx",
                "metric": "cosine",
                "filter_columns": ["tenant_id", "source_id", "lang_id"],
                "include_columns": ["doc_id_int", "chunk_id_int", "doc_version"],
                "options": {"lists": 0, "normalized": True},
            },
        )

        self.assertIn("CREATE INDEX IF NOT EXISTS rag_passages_delta_tq_idx", sql)
        self.assertIn("INCLUDE (doc_id_int, chunk_id_int, doc_version)", sql)

    def test_parse_hf_source_path_supports_subset_split_and_limit(self):
        parsed = parse_hf_source_path("hf://kilt_tasks/nq[validation][:10000]")

        self.assertEqual(parsed["dataset_name"], "kilt_tasks")
        self.assertEqual(parsed["subset_name"], "nq")
        self.assertEqual(parsed["split_name"], "validation")
        self.assertEqual(parsed["limit"], 10000)

    def test_build_hf_corpus_supports_popqa(self):
        config = CampaignConfig(
            dataset=DatasetConfig(
                name="popqa_small_live",
                version="2026-03-29",
                source_path="hf://akariasai/PopQA[test][:2]",
            ),
            embedding=EmbeddingConfig(model="BAAI/bge-small-en-v1.5", dimension=384, normalized=False),
            chunking=ChunkingConfig(strategy="row_subject_fact", chunk_size=1, chunk_overlap=0),
            schema={"documents_table": "rag_documents", "passages_table": "rag_passages"},
            backends=[],
        )

        fake_rows = [
            {
                "subj_id": 1850297,
                "subj": "George Rankin",
                "prop": "occupation",
                "obj": "politician",
                "question": "What is George Rankin's occupation?",
            },
            {
                "subj_id": 2079053,
                "subj": "John Mayne",
                "prop": "occupation",
                "obj": "journalist",
                "question": "What is John Mayne's occupation?",
            },
        ]

        corpus = build_hf_corpus(
            config,
            dataset_loader=lambda dataset_name, subset_name, split_name: fake_rows,
        )

        self.assertEqual(len(corpus), 2)
        self.assertEqual(corpus[0]["document_id"], "1850297")
        self.assertEqual(corpus[0]["passages"][0]["passage_id"], "1850297:0")
        self.assertIn("George Rankin", corpus[0]["passages"][0]["text"])

    def test_build_hf_corpus_supports_kilt_nq(self):
        config = CampaignConfig(
            dataset=DatasetConfig(
                name="kilt_nq_small_live",
                version="2026-03-29",
                source_path="hf://kilt_tasks/nq[validation][:2]",
            ),
            embedding=EmbeddingConfig(model="BAAI/bge-small-en-v1.5", dimension=384, normalized=False),
            chunking=ChunkingConfig(strategy="question_answer_seed", chunk_size=1, chunk_overlap=0),
            schema={"documents_table": "rag_documents", "passages_table": "rag_passages"},
            backends=[],
        )

        fake_rows = [
            {
                "input": "Who wrote Hamlet?",
                "output": [
                    {
                        "answer": "William Shakespeare",
                        "provenance": [{"wikipedia_id": "123"}],
                    }
                ],
            },
            {
                "input": "What is the capital of France?",
                "output": [
                    {
                        "answer": "Paris",
                        "provenance": [{"wikipedia_id": "456"}],
                    }
                ],
            },
        ]

        corpus = build_hf_corpus(
            config,
            dataset_loader=lambda dataset_name, subset_name, split_name: fake_rows,
        )

        self.assertEqual(len(corpus), 2)
        self.assertEqual(corpus[0]["document_id"], "123")
        self.assertEqual(corpus[0]["passages"][0]["passage_id"], "123:0")
        self.assertIn("William Shakespeare", corpus[0]["passages"][0]["text"])

    def test_build_hf_corpus_supports_kilt_hotpotqa(self):
        config = CampaignConfig(
            dataset=DatasetConfig(
                name="kilt_hotpotqa_small_live",
                version="2026-03-29",
                source_path="hf://kilt_tasks/hotpotqa[validation][:2]",
            ),
            embedding=EmbeddingConfig(model="BAAI/bge-small-en-v1.5", dimension=384, normalized=False),
            chunking=ChunkingConfig(strategy="question_answer_seed", chunk_size=1, chunk_overlap=0),
            schema={"documents_table": "rag_documents", "passages_table": "rag_passages"},
            backends=[],
        )

        fake_rows = [
            {
                "input": "Where was the author of Hamlet born?",
                "output": [
                    {
                        "answer": "Stratford-upon-Avon",
                        "provenance": [{"wikipedia_id": "789"}],
                    }
                ],
            },
            {
                "input": "What city is home to the Eiffel Tower?",
                "output": [
                    {
                        "answer": "Paris",
                        "provenance": [{"wikipedia_id": "456"}],
                    }
                ],
            },
        ]

        corpus = build_hf_corpus(
            config,
            dataset_loader=lambda dataset_name, subset_name, split_name: fake_rows,
        )

        self.assertEqual(len(corpus), 2)
        self.assertEqual(corpus[0]["document_id"], "789")
        self.assertEqual(corpus[0]["passages"][0]["passage_id"], "789:0")
        self.assertIn("Stratford-upon-Avon", corpus[0]["passages"][0]["text"])

    def test_run_hf_ingestion_prepares_fixed_dimension_embedding_column(self):
        config = CampaignConfig(
            dataset=DatasetConfig(
                name="popqa_small_live",
                version="2026-03-29",
                source_path="hf://akariasai/PopQA[test][:2]",
            ),
            embedding=EmbeddingConfig(model="BAAI/bge-small-en-v1.5", dimension=384, normalized=False),
            chunking=ChunkingConfig(strategy="row_subject_fact", chunk_size=1, chunk_overlap=0),
            schema={"documents_table": "rag_documents", "passages_table": "rag_passages"},
            backends=[
                {
                    "kind": "pg_turboquant",
                    "index_name": "rag_passages_tq_idx",
                    "metric": "cosine",
                    "mode": "approx",
                    "options": {"lists": 0},
                }
            ],
        )
        connection = FakeConnection()

        manifest = run_hf_ingestion(
            connection,
            config,
            dataset_loader=lambda dataset_name, subset_name, split_name: [
                {
                    "subj_id": 1850297,
                    "subj": "George Rankin",
                    "prop": "occupation",
                    "obj": "politician",
                    "question": "What is George Rankin's occupation?",
                }
            ],
            embedder=lambda texts: [[1.0, 0.0, 0.0]],
        )

        executed_sql = "\n".join(sql for cursor in connection.cursors for sql, _ in cursor.executed)
        self.assertEqual(manifest["dataset_name"], "popqa_small_live")
        self.assertIn("ALTER TABLE rag_passages ALTER COLUMN embedding TYPE vector(384)", executed_sql)
        self.assertIn("INSERT INTO rag_passages", executed_sql)
        self.assertIn("CREATE INDEX IF NOT EXISTS rag_passages_tq_idx", executed_sql)
        self.assertEqual(connection.commits, 1)

    def test_default_hf_dataset_loader_passes_hf_token_to_datasets(self):
        class FakeDatasetModule:
            def __init__(self):
                self.calls = []

            def load_dataset(self, dataset_name, subset_name=None, **kwargs):
                self.calls.append((dataset_name, subset_name, kwargs))
                return {"test": [{"id": 1}]}

        fake_datasets = FakeDatasetModule()
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}, clear=False):
            with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
                rows = _default_hf_dataset_loader("akariasai/PopQA", None, "test")

        self.assertEqual(rows, [{"id": 1}])
        self.assertEqual(
            fake_datasets.calls,
            [("akariasai/PopQA", None, {"token": "hf_test_token"})],
        )

    def test_build_embedder_passes_hf_token_to_transformers(self):
        class FakeModelInstance:
            def to(self, device):
                self.device = device
                return self

            def eval(self):
                return None

        class FakeAutoTokenizer:
            calls = []

            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                cls.calls.append((model_name, kwargs))
                return object()

        class FakeAutoModel:
            calls = []

            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                cls.calls.append((model_name, kwargs))
                return FakeModelInstance()

        fake_torch = ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_torch.cuda = type("FakeCuda", (), {"is_available": staticmethod(lambda: False)})
        fake_torch.device = lambda name: name

        fake_torch_nn = ModuleType("torch.nn")
        fake_torch_nn_functional = ModuleType("torch.nn.functional")

        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        fake_transformers.AutoModel = FakeAutoModel

        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}, clear=False):
            with mock.patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "torch.nn": fake_torch_nn,
                    "torch.nn.functional": fake_torch_nn_functional,
                    "transformers": fake_transformers,
                },
            ):
                embedder = build_embedder("BAAI/bge-small-en-v1.5", normalized=False)

        self.assertTrue(callable(embedder))
        self.assertEqual(
            FakeAutoTokenizer.calls,
            [("BAAI/bge-small-en-v1.5", {"trust_remote_code": True, "token": "hf_test_token"})],
        )
        self.assertEqual(
            FakeAutoModel.calls,
            [
                (
                    "BAAI/bge-small-en-v1.5",
                    {
                        "torch_dtype": "float32",
                        "trust_remote_code": True,
                        "token": "hf_test_token",
                    },
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
