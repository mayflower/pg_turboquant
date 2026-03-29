import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks.rag.live_campaign import (
    QuerySample,
    _build_default_generator_runner,
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class RagLiveCampaignContractTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
