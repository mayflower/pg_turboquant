import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from benchmarks.rag.bergen_adapter import (
    ExactMetricBackend,
    PassageTable,
    PostgresRetrieverAdapter,
)
from benchmarks.rag.regression_gate import (
    REGRESSION_CONFIG_DIR,
    BeirFixtureQuery,
    load_regression_config,
    load_regression_fixture,
    resolve_regression_harness,
    run_beir_smoke_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "benchmarks" / "rag" / "README.md"


class SequencedCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params=()):
        return None

    def fetchall(self):
        return list(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class SequencedConnection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return SequencedCursor(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class SequencedConnectionFactory:
    def __init__(self, rows_per_call):
        self.rows_per_call = list(rows_per_call)

    def __call__(self, dsn):
        rows = self.rows_per_call.pop(0)
        return SequencedConnection(rows)


class RagRegressionGateContractTest(unittest.TestCase):
    def test_beir_style_smoke_evaluation_uses_postgres_retriever_adapter(self):
        adapter = PostgresRetrieverAdapter(
            dsn="postgresql://example.invalid/db",
            table=PassageTable(
                table_name="rag_passages",
                id_column="doc_id",
                text_column="passage_text",
                embedding_column="embedding",
            ),
            backend=ExactMetricBackend(),
            connect_fn=SequencedConnectionFactory(
                [
                    [
                        ("d1", Decimal("0.01"), "alpha"),
                        ("d3", Decimal("0.02"), "noise"),
                    ],
                    [
                        ("x2", Decimal("0.01"), "right"),
                        ("x9", Decimal("0.03"), "other"),
                    ],
                ]
            ),
        )
        fixture = [
            BeirFixtureQuery(
                query_id="q1",
                query_text="alpha?",
                query_vector=[1.0, 0.0],
                relevant_ids=["d1"],
            ),
            BeirFixtureQuery(
                query_id="q2",
                query_text="x2?",
                query_vector=[0.0, 1.0],
                relevant_ids=["x2"],
            ),
        ]

        result = run_beir_smoke_evaluation(
            adapter=adapter,
            fixture=fixture,
            top_k=2,
            metric="cosine",
        )

        self.assertEqual(result["run_metadata"]["harness"], "beir")
        self.assertEqual(result["run_metadata"]["result_kind"], "retrieval_only")
        self.assertEqual(result["metrics"]["recall@1"], 1.0)
        self.assertEqual(result["metrics"]["recall@2"], 1.0)

    def test_harness_selection_config_supports_beir_and_lotte_scaffolding(self):
        beir_config = load_regression_config(REGRESSION_CONFIG_DIR / "beir_tiny_smoke.json")
        self.assertEqual(beir_config["harness"], "beir")
        self.assertEqual(beir_config["dataset_id"], "beir_tiny_smoke")

        beir_harness = resolve_regression_harness(beir_config)
        self.assertEqual(beir_harness["runner"], "beir_smoke")
        self.assertTrue(beir_harness["available"])

        lotte_harness = resolve_regression_harness(
            {
                "harness": "lotte",
                "dataset_id": "lotte_lifestyle_dev",
                "top_k": 10,
                "metric": "cosine",
            }
        )
        self.assertEqual(lotte_harness["runner"], "lotte_adapter")
        self.assertFalse(lotte_harness["available"])

    def test_documentation_includes_harness_decision_matrix(self):
        readme = README.read_text(encoding="utf-8")
        self.assertIn("Decision Matrix", readme)
        self.assertIn("BERGEN", readme)
        self.assertIn("BEIR", readme)
        self.assertIn("LoTTE", readme)
        self.assertIn("retrieval-only regression gate", readme)

    def test_local_beir_fixture_is_loadable(self):
        fixture = load_regression_fixture(REGRESSION_CONFIG_DIR / "beir_tiny_smoke_fixture.json")
        self.assertEqual([entry.query_id for entry in fixture], ["q1", "q2"])
        self.assertEqual(fixture[0].relevant_ids, ["d1"])


if __name__ == "__main__":
    unittest.main()
