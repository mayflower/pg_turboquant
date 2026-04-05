import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks.rag.ingestion_pipeline import CampaignConfig, ChunkingConfig, DatasetConfig, EmbeddingConfig
from benchmarks.rag.bergen_adapter import PassageTable
from benchmarks.rag.live_campaign import (
    LIVE_CONFIG_PATHS,
    QuerySample,
    _build_default_generator_runner,
    _isolated_relation_name,
    _prepare_backend_isolated_layout,
    _prepare_dataset_source_layout,
    _provided_dataset_source_layout,
    _resolve_method_rerank_top_k,
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
        if "FROM rag_passages AS p" in sql and "ORDER BY p.embedding" in sql:
            self.rows = [
                ("wiki-1:0", 0.10),
                ("wiki-2:0", 0.20),
            ]
            return
        if "FROM rag_passages" in sql and "passage_text" in sql:
            self.rows = [
                ("wiki-1:0", "Alpha is the first letter."),
                ("wiki-2:0", "Beta follows alpha."),
            ]
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
    def test_default_live_campaign_uses_normal_hotpotqa_config(self):
        self.assertEqual(LIVE_CONFIG_PATHS["kilt_hotpotqa"].name, "kilt_hotpotqa_small_live.json")

    def test_turboquant_rerank_candidate_pool_scales_above_dataset_default(self):
        rerank_top_k = _resolve_method_rerank_top_k(
            method_id="pg_turboquant_rerank",
            dataset_top_k=20,
            dataset_rerank_top_k=100,
            source_live_count=2474,
        )

        self.assertEqual(rerank_top_k, 1280)

    def test_turboquant_rerank_candidate_pool_caps_at_live_count(self):
        rerank_top_k = _resolve_method_rerank_top_k(
            method_id="pg_turboquant_rerank",
            dataset_top_k=20,
            dataset_rerank_top_k=100,
            source_live_count=600,
        )

        self.assertEqual(rerank_top_k, 600)

    def test_non_turboquant_rerank_candidate_pool_uses_dataset_default(self):
        rerank_top_k = _resolve_method_rerank_top_k(
            method_id="pgvector_hnsw_rerank",
            dataset_top_k=20,
            dataset_rerank_top_k=100,
            source_live_count=2474,
        )

        self.assertEqual(rerank_top_k, 100)

    def test_provided_dataset_source_layout_uses_explicit_live_relations(self):
        layout = _provided_dataset_source_layout(
            passage_table=PassageTable(
                table_name="public.rag_passages_normcmp",
                id_column="passage_id",
                text_column="passage_text",
                embedding_column="embedding",
                query_vector_cast="vector",
            ),
            turboquant_index_name="public.rag_passages_normcmp_tq_idx",
            hnsw_index_name="public.rag_passages_normcmp_hnsw_idx",
            ivfflat_index_name="public.rag_passages_normcmp_ivf_idx",
        )

        self.assertEqual(layout["passage_table"].table_name, "public.rag_passages_normcmp")
        self.assertEqual(layout["index_names"]["pg_turboquant"], "public.rag_passages_normcmp_tq_idx")
        self.assertEqual(layout["index_names"]["pgvector_hnsw"], "public.rag_passages_normcmp_hnsw_idx")
        self.assertEqual(layout["index_names"]["pgvector_ivfflat"], "public.rag_passages_normcmp_ivf_idx")
        self.assertEqual(layout["manifest"]["source_mode"], "provided")
        self.assertEqual(layout["manifest"]["table_name"], "public.rag_passages_normcmp")
        self.assertEqual(layout["backend_ann_defaults"], {})

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
                    sql="SELECT %s::text AS doc_id, %s::float8 AS score",
                    params=("wiki-1:0", 0.1),
                    session_statements=[],
                )

            def serialize_run_metadata(self, plan):
                return {
                    "retrieval_execution_mode": "approx_stage1_only",
                    "context_fetch_mode": "post_limit_text_fetch",
                    "stage1_covering": True,
                }

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
                    query_encoder=lambda texts: [[3.0, 4.0] for _ in texts],
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
        self.assertAlmostEqual(recording_backend.requests[0].query_vector[0], 0.6)
        self.assertAlmostEqual(recording_backend.requests[0].query_vector[1], 0.8)
        self.assertEqual(recording_backend.requests[0].ann["probes"], 8)
        self.assertEqual(recording_backend.requests[0].ann["filters"], {"tenant_id": 1, "lang_id": 1})
        self.assertEqual(
            recording_backend.requests[0].ann["stage1_payload_columns"],
            ["doc_id_int", "chunk_id_int", "tenant_id"],
        )
        self.assertEqual(recording_backend.requests[0].ann["iterative_scan"], "strict_order")
        self.assertEqual(recording_backend.requests[0].ann["min_rows_after_filter"], 12)

    def test_end_to_end_fetches_text_after_limit_for_covering_stage1_rows(self):
        retrieval_result = {
            "metrics": {"recall@10": 1.0},
            "query_results": [
                {
                    "query_id": "q1",
                    "question": "What is alpha?",
                    "retrieved_rows": [
                        {"id": "wiki-1:0", "score": 0.1, "text": None},
                        {"id": "wiki-2:0", "score": 0.2, "text": None},
                    ],
                    "retrieval_latency_ms": 3.0,
                }
            ],
            "run_metadata": {
                "retrieval_execution_mode": "approx_stage1_only",
                "context_fetch_mode": "post_limit_text_fetch",
            },
        }

        captured_contexts = []

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_live_campaign.__globals__["_run_end_to_end_scenario"](
                dataset_config={"dataset_id": "kilt_nq"},
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
                generation_top_k=2,
                retrieval_result=retrieval_result,
                end_to_end_root=Path(tmpdir),
                generator_runner=lambda sample, contexts, _config: (
                    captured_contexts.append(contexts) or {"prompt": "P", "answer": "Alpha"}
                ),
                answer_metrics_fn=lambda preds, refs, questions: {"answer_exact_match": 1.0},
                clock_fn=lambda: 0.0,
                dsn="postgresql://fake",
                connect_fn=lambda _dsn: FakeConnection(),
                passage_table=PassageTable(
                    table_name="rag_passages",
                    id_column="passage_id",
                    text_column="passage_text",
                    embedding_column="embedding",
                    query_vector_cast="vector",
                ),
            )

        self.assertEqual(captured_contexts[0][0]["text"], "Alpha is the first letter.")
        self.assertEqual(captured_contexts[0][1]["text"], "Beta follows alpha.")
        self.assertEqual(
            result["operational_summary"]["latency_ms"]["context_fetch"]["p50"],
            0.0,
        )

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

        self.assertEqual(runtime["plan"]["campaign_kind"], "rag_benchmark")
        self.assertEqual(runtime["plan"]["datasets"], ["kilt_nq", "kilt_hotpotqa", "popqa"])
        self.assertEqual(len(runtime["plan"]["system_variants"]), 6)
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

    def test_backend_isolation_accepts_schema_qualified_source_index_names(self):
        class IndexLookupCursor(FakeCursor):
            def execute(self, sql, params=()):
                self.executed.append((sql, params))
                if "SELECT indexdef FROM pg_indexes" in sql:
                    if params == ("public", "rag_normcmp_tq_idx"):
                        self.rows = [
                            (
                                "CREATE INDEX rag_normcmp_tq_idx "
                                "ON public.rag_passages_normcmp USING turboquant (embedding tq_cosine_ops)",
                            )
                        ]
                    elif params == ("public", "rag_normcmp_hnsw_idx"):
                        self.rows = [
                            (
                                "CREATE INDEX rag_normcmp_hnsw_idx "
                                "ON public.rag_passages_normcmp USING hnsw (embedding vector_cosine_ops)",
                            )
                        ]
                    elif params == ("public", "rag_normcmp_ivf_idx"):
                        self.rows = [
                            (
                                "CREATE INDEX rag_normcmp_ivf_idx "
                                "ON public.rag_passages_normcmp USING ivfflat (embedding vector_cosine_ops)",
                            )
                        ]
                    else:
                        self.rows = []
                    return
                self.rows = []

        class IndexLookupConnection(FakeConnection):
            def cursor(self):
                cursor = IndexLookupCursor()
                self.cursors.append(cursor)
                return cursor

            def commit(self):
                pass

        connection = IndexLookupConnection()

        with tempfile.TemporaryDirectory() as tmpdir:
            layout = _prepare_backend_isolated_layout(
                dsn="postgresql://fake",
                connect_fn=lambda _dsn: connection,
                source_table=PassageTable(
                    table_name="public.rag_passages_normcmp",
                    id_column="passage_id",
                    text_column="passage_text",
                    embedding_column="embedding",
                    query_vector_cast="vector",
                ),
                output_dir=Path(tmpdir),
                turboquant_index_name="public.rag_normcmp_tq_idx",
                hnsw_index_name="public.rag_normcmp_hnsw_idx",
                ivfflat_index_name="public.rag_normcmp_ivf_idx",
            )

        self.assertIn("pg_turboquant", layout)
        lookup_calls = [
            params[-1]
            for cursor in connection.cursors
            for sql, params in cursor.executed
            if "SELECT indexdef FROM pg_indexes" in sql
        ]
        self.assertEqual(
            lookup_calls,
            ["rag_normcmp_tq_idx", "rag_normcmp_hnsw_idx", "rag_normcmp_ivf_idx"],
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

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "benchmarks.rag.live_campaign._prepare_dataset_source_layout",
                side_effect=AssertionError("run_live_campaign should use the provided live source layout"),
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
                    turboquant_index_name="rag_passages_shared_idx",
                    hnsw_index_name="rag_passages_shared_idx",
                    ivfflat_index_name="rag_passages_shared_idx",
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
            self.assertEqual(len(payload["tables"]["retrieval_benchmark"]), 6)
            self.assertEqual(len(payload["tables"]["end_to_end_benchmark"]), 6)
            self.assertEqual(len(payload["tables"]["retrieval_diagnostics"]), 6)
            self.assertIn("RAG Benchmark Outcome", report_html_path.read_text(encoding="utf-8"))

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
                "rag_passages",
            )
            self.assertEqual(
                run_config["dataset_source_layouts"]["kilt_nq"]["manifest"]["source_mode"],
                "provided",
            )

    def test_live_runner_rejects_shared_table_mode_for_distinct_backend_indexes(self):
        runtime = build_live_campaign_runtime(
            dataset_ids=["kilt_nq"],
            generator_name="oracle_answer",
            retriever_name="bge-small-en-v1.5",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "backend_isolation=False is unsafe"):
                run_live_campaign(
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
                    backend_isolation=False,
                    connect_fn=lambda _dsn: FakeConnection(),
                    dataset_loader=lambda *_args: [],
                )


if __name__ == "__main__":
    unittest.main()
