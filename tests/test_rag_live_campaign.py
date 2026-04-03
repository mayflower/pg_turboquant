import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks.rag.ingestion_pipeline import CampaignConfig, ChunkingConfig, DatasetConfig, EmbeddingConfig
from benchmarks.rag.bergen_adapter import PassageTable
from benchmarks.rag.live_campaign import (
    QuerySample,
    _build_default_generator_runner,
    _isolated_relation_name,
    _prepare_dataset_source_layout,
    _run_retrieval_scenario,
    _rewrite_indexdef_for_clone,
    build_live_campaign_runtime,
    run_live_campaign,
)


class FakeCursor:
    def __init__(self):
        self.executed = []
        self.rows = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        if "pg_relation_size" in sql:
            self.rows = [(4096,)]
            return

        self.rows = [
            ("wiki-1:0", 0.10, "Alpha is the first letter."),
            ("wiki-2:0", 0.20, "Beta follows alpha."),
        ]

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        if not self.rows:
            return None
        return self.rows[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self):
        self.cursors = []

    def cursor(self):
        cursor = FakeCursor()
        self.cursors.append(cursor)
        return cursor

    def rollback(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class RagLiveCampaignContractTest(unittest.TestCase):
    def test_prepare_dataset_source_layout_rebuilds_a_dataset_specific_live_corpus(self):
        fake_config = CampaignConfig(
            dataset=DatasetConfig(
                name="kilt_nq_small_live",
                version="2026-03-29",
                source_path="hf://kilt_tasks/nq[validation][:10000]",
            ),
            embedding=EmbeddingConfig(model="BAAI/bge-small-en-v1.5", dimension=384, normalized=False),
            chunking=ChunkingConfig(strategy="question_answer_seed", chunk_size=1, chunk_overlap=0),
            schema={"documents_table": "rag_documents", "passages_table": "rag_passages"},
            backends=[
                {
                    "kind": "pg_turboquant",
                    "index_name": "rag_passages_tq_idx",
                    "metric": "cosine",
                    "mode": "approx",
                    "options": {"lists": 0, "bits": 4},
                },
                {
                    "kind": "pgvector_hnsw",
                    "index_name": "rag_passages_hnsw_idx",
                    "metric": "cosine",
                    "mode": "approx",
                    "options": {"m": 16, "ef_construction": 64},
                },
                {
                    "kind": "pgvector_ivfflat",
                    "index_name": "rag_passages_ivf_idx",
                    "metric": "cosine",
                    "mode": "approx",
                    "options": {"lists": 64},
                },
            ],
        )

        connection = FakeConnection()

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "benchmarks.rag.live_campaign.load_campaign_config",
                return_value=fake_config,
            ), mock.patch(
                "benchmarks.rag.live_campaign.build_embedder",
                return_value=lambda texts: [[1.0, 0.0] for _ in texts],
            ) as build_embedder_mock, mock.patch(
                "benchmarks.rag.live_campaign.run_hf_ingestion",
                return_value={"dataset_name": "kilt_nq_small_live"},
            ) as run_hf_ingestion_mock:
                layout = _prepare_dataset_source_layout(
                    dataset_id="kilt_nq",
                    output_dir=Path(tmpdir),
                    dsn="postgresql://fake",
                    connect_fn=lambda _dsn: connection,
                    metric="cosine",
                    query_vector_cast="vector",
                    passage_table=PassageTable(
                        table_name="rag_passages",
                        id_column="passage_id",
                        text_column="passage_text",
                        embedding_column="embedding",
                        query_vector_cast="vector",
                    ),
                )

        self.assertEqual(layout["passage_table"].id_column, "passage_id")
        self.assertIn("kilt_nq", layout["passage_table"].table_name)
        self.assertIn("kilt_nq", layout["index_names"]["pg_turboquant"])
        self.assertEqual(layout["manifest"]["dataset_name"], "kilt_nq_small_live")
        self.assertEqual(
            layout["backend_ann_defaults"],
            {"pg_turboquant": {}, "pgvector_hnsw": {}, "pgvector_ivfflat": {}},
        )
        build_embedder_mock.assert_called_once_with("BAAI/bge-small-en-v1.5", False)
        self.assertEqual(run_hf_ingestion_mock.call_count, 1)

        executed_sql = "\n".join(sql for cursor in connection.cursors for sql, _ in cursor.executed)
        self.assertIn("CREATE EXTENSION IF NOT EXISTS vector", executed_sql)
        self.assertIn("CREATE EXTENSION IF NOT EXISTS pg_turboquant", executed_sql)
        self.assertIn("DROP TABLE IF EXISTS", executed_sql)

    def test_retrieval_scenario_merges_backend_ann_defaults(self):
        class RecordingBackend:
            name = "pg_turboquant"

            def __init__(self):
                self.requests = []

            def build_plan(self, table, request):
                from benchmarks.rag.bergen_adapter import RetrievalPlan

                self.requests.append(request)
                return RetrievalPlan(
                    sql="SELECT %s::text AS doc_id, %s::float8 AS score, %s::text AS passage_text",
                    params=("wiki-1:0", 0.1, "Alpha"),
                    session_statements=[],
                )

        recording_backend = RecordingBackend()

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "benchmarks.rag.live_campaign._make_backend",
                return_value=recording_backend,
            ), mock.patch(
                "benchmarks.rag.live_campaign._fetch_relation_size_bytes",
                return_value=4096,
            ), mock.patch(
                "benchmarks.rag.live_campaign._fetch_turboquant_index_metadata",
                return_value={"capabilities": {"index_only_scan": True}},
            ):
                _run_retrieval_scenario(
                    dsn="postgresql://fake",
                    connect_fn=lambda _dsn: FakeConnection(),
                    passage_table=PassageTable(
                        table_name="rag_passages",
                        id_column="passage_id",
                        text_column="passage_text",
                        embedding_column="embedding",
                        query_vector_cast="vector",
                    ),
                    dataset_config={
                        "dataset_id": "kilt_nq",
                        "retrieval_profile": {"top_k_default": 5, "rerank_top_k_default": 10},
                    },
                    dataset_samples=[
                        QuerySample(
                            query_id="q1",
                            question="What is alpha?",
                            answers=["Alpha"],
                            relevant_ids=["wiki-1"],
                            evidence_ids=["wiki-1"],
                        )
                    ],
                    method_id="pg_turboquant_approx",
                    metric="cosine",
                    query_encoder=lambda texts: [[1.0, 0.0] for _ in texts],
                    dataset_top_k=5,
                    rerank_top_k=10,
                    eval_ks=(1, 5),
                    retrieval_root=Path(tmpdir),
                    clock_fn=lambda: 0.0,
                    turboquant_index_name="rag_passages_tq_idx",
                    hnsw_index_name="rag_passages_hnsw_idx",
                    ivfflat_index_name="rag_passages_ivf_idx",
                    backend_ann_defaults={
                        "pg_turboquant": {
                            "filters": {"tenant_id": 1, "lang_id": 1},
                            "stage1_payload_columns": ["doc_id_int", "chunk_id_int", "tenant_id"],
                            "iterative_scan": "strict_order",
                            "min_rows_after_filter": 12,
                        }
                    },
                    turboquant_probes=8,
                    turboquant_oversampling=4,
                    turboquant_max_visited_codes=4096,
                    turboquant_max_visited_pages=0,
                    hnsw_ef_search=80,
                    ivfflat_probes=8,
                )

        self.assertEqual(len(recording_backend.requests), 1)
        self.assertEqual(recording_backend.requests[0].ann["probes"], 8)
        self.assertEqual(recording_backend.requests[0].ann["filters"], {"tenant_id": 1, "lang_id": 1})
        self.assertEqual(
            recording_backend.requests[0].ann["stage1_payload_columns"],
            ["doc_id_int", "chunk_id_int", "tenant_id"],
        )
        self.assertEqual(recording_backend.requests[0].ann["iterative_scan"], "strict_order")
        self.assertEqual(recording_backend.requests[0].ann["min_rows_after_filter"], 12)

    def test_default_generator_runner_supports_oracle_answer_without_hitting_abstract_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bergen_root = Path(tmpdir)
            (bergen_root / "config" / "generator").mkdir(parents=True)
            (bergen_root / "config" / "prompt").mkdir(parents=True)
            (bergen_root / "config" / "generator" / "oracle_answer.yaml").write_text(
                "init_args:\n"
                "  _target_: models.generators.oracle_answer.OracleAnswer\n"
                "  model_name: oracle_answer\n"
                "  batch_size: 4\n",
                encoding="utf-8",
            )
            (bergen_root / "config" / "prompt" / "basic.yaml").write_text("system: test\n", encoding="utf-8")

            with mock.patch("benchmarks.rag.live_campaign._ensure_bergen_import_path"), mock.patch(
                "benchmarks.rag.live_campaign._instantiate_from_target",
                side_effect=AssertionError("oracle_answer fallback should not instantiate BERGEN generator"),
            ):
                runner = _build_default_generator_runner(
                    bergen_root=bergen_root,
                    generator_name="oracle_answer",
                    prompt_name="basic",
                )

            result = runner(
                QuerySample(
                    query_id="q1",
                    question="What is alpha?",
                    answers=["Alpha"],
                    relevant_ids=["wiki-1"],
                    evidence_ids=["wiki-1"],
                ),
                [{"id": "wiki-1:0", "text": "Alpha is the first letter."}],
                {"dataset_id": "kilt_nq"},
            )
            self.assertEqual(result["answer"], "Alpha")
            self.assertIn("Question:", result["prompt"])

    def test_runtime_builder_resolves_required_datasets_and_methods(self):
        runtime = build_live_campaign_runtime(
            dataset_ids=["kilt_nq", "kilt_hotpotqa", "popqa"],
            generator_name="oracle_answer",
            retriever_name="bge-small-en-v1.5",
        )

        self.assertEqual(runtime["plan"]["campaign_kind"], "comparative_rag")
        self.assertEqual(runtime["plan"]["datasets"], ["kilt_nq", "kilt_hotpotqa", "popqa"])
        self.assertEqual(len(runtime["plan"]["method_variants"]), 6)
        self.assertEqual(
            sorted(runtime["dataset_configs"].keys()),
            ["kilt_hotpotqa", "kilt_nq", "popqa"],
        )

    def test_index_definition_rewrite_targets_cloned_table_and_index(self):
        rewritten = _rewrite_indexdef_for_clone(
            "CREATE INDEX rag_passages_hnsw_idx ON public.rag_passages USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64')",
            new_table_name="rag_passages__cmp__hnsw",
            new_index_name="rag_passages__cmp__hnsw_idx",
        )

        self.assertEqual(
            rewritten,
            "CREATE INDEX rag_passages__cmp__hnsw_idx ON rag_passages__cmp__hnsw USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64')",
        )

    def test_isolated_relation_name_stays_unique_when_long_inputs_truncate(self):
        source_name = _isolated_relation_name(
            "rag_passages_tq_idx__live_rag_e2e_20260330_fair__kilt_nq_source",
            "pg_turboquant",
            "kilt_nq",
        )
        clone_name = _isolated_relation_name(
            "rag_passages_tq_idx__live_rag_e2e_20260330_fair__kilt_nq_source",
            "pg_turboquant_idx",
            "kilt_nq",
        )

        self.assertNotEqual(source_name, clone_name)
        self.assertLessEqual(len(source_name), 63)
        self.assertLessEqual(len(clone_name), 63)

    def test_live_runner_smoke_emits_per_scenario_and_campaign_artifacts_with_fakes(self):
        runtime = build_live_campaign_runtime(
            dataset_ids=["kilt_nq"],
            generator_name="oracle_answer",
            retriever_name="bge-small-en-v1.5",
        )

        def fake_connect(_dsn):
            return FakeConnection()

        def fake_dataset_loader(dataset_id, dataset_config, query_limit):
            self.assertEqual(dataset_id, "kilt_nq")
            self.assertEqual(query_limit, 1)
            self.assertEqual(dataset_config["dataset_id"], "kilt_nq")
            return [
                QuerySample(
                    query_id="q1",
                    question="What is alpha?",
                    answers=["Alpha"],
                    relevant_ids=["wiki-1"],
                    evidence_ids=["wiki-1"],
                )
            ]

        def fake_query_encoder(texts):
            return [[1.0, 0.0] for _ in texts]

        def fake_generator_runner(query_sample, contexts, dataset_config):
            self.assertEqual(dataset_config["dataset_id"], "kilt_nq")
            return {
                "prompt": f"Question: {query_sample.question}\nContext: {contexts[0]['text']}",
                "answer": query_sample.answers[0],
            }

        def fake_prepare_dataset_source_layout(**kwargs):
            self.assertEqual(kwargs["dataset_id"], "kilt_nq")
            return {
                "passage_table": kwargs["passage_table"].__class__(
                    table_name="rag_passages__kilt_nq_source",
                    id_column="passage_id",
                    text_column="passage_text",
                    embedding_column="embedding",
                    query_vector_cast="vector",
                ),
                "index_names": {
                    "pg_turboquant": "rag_passages__kilt_nq_tq_idx",
                    "pgvector_hnsw": "rag_passages__kilt_nq_hnsw_idx",
                    "pgvector_ivfflat": "rag_passages__kilt_nq_ivf_idx",
                },
                "manifest": {"dataset_name": "kilt_nq_small_live"},
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "benchmarks.rag.live_campaign._prepare_dataset_source_layout",
                side_effect=fake_prepare_dataset_source_layout,
            ):
                result = run_live_campaign(
                    output_dir=Path(tmpdir),
                    runtime=runtime,
                    dsn="postgresql://fake",
                    table_name="rag_passages",
                    id_column="passage_id",
                    text_column="passage_text",
                    embedding_column="embedding",
                    metric="cosine",
                    turboquant_index_name="rag_passages_tq_idx",
                    hnsw_index_name="rag_passages_hnsw_idx",
                    ivfflat_index_name="rag_passages_ivf_idx",
                    query_limit=1,
                    generation_top_k=1,
                    backend_isolation=False,
                    connect_fn=fake_connect,
                    dataset_loader=fake_dataset_loader,
                    query_encoder=fake_query_encoder,
                    generator_runner=fake_generator_runner,
                )

            self.assertIn("artifacts", result)
            campaign_json = Path(tmpdir) / result["artifacts"]["campaign_json"]
            report_html_path = Path(tmpdir) / result["artifacts"]["report_html"]
            self.assertTrue(campaign_json.exists())
            self.assertTrue(report_html_path.exists())

            payload = json.loads(campaign_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["plan"]["datasets"], ["kilt_nq"])
            self.assertEqual(len(payload["tables"]["retrieval_only"]), 6)
            self.assertEqual(len(payload["tables"]["end_to_end"]), 6)
            self.assertIn("TurboQuant Outcome", report_html_path.read_text(encoding="utf-8"))

            retrieval_path = (
                Path(tmpdir)
                / "scenarios"
                / "kilt_nq"
                / "pg_turboquant_approx"
                / "retrieval"
                / "retrieval-results.json"
            )
            rerank_path = (
                Path(tmpdir)
                / "scenarios"
                / "kilt_nq"
                / "pg_turboquant_rerank"
                / "retrieval"
                / "two-stage-retrieval-results.json"
            )
            end_to_end_path = (
                Path(tmpdir)
                / "scenarios"
                / "kilt_nq"
                / "pgvector_hnsw_approx"
                / "end_to_end"
                / "end-to-end-results.json"
            )
            run_config_path = Path(tmpdir) / "live-campaign-config.json"

            self.assertTrue(retrieval_path.exists())
            self.assertTrue(rerank_path.exists())
            self.assertTrue(end_to_end_path.exists())
            self.assertTrue(run_config_path.exists())

            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                run_config["dataset_source_layouts"]["kilt_nq"]["passage_table"]["table_name"],
                "rag_passages__kilt_nq_source",
            )
            self.assertEqual(
                run_config["dataset_source_layouts"]["kilt_nq"]["manifest"]["dataset_name"],
                "kilt_nq_small_live",
            )


if __name__ == "__main__":
    unittest.main()
