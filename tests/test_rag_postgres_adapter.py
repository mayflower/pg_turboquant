import unittest
from decimal import Decimal

from benchmarks.rag.bergen_adapter import (
    ExactMetricBackend,
    PassageTable,
    PostgresRetrieverAdapter,
    RetrievalRequest,
    StaticAnnBackend,
)
from benchmarks.rag.bergen_adapter.adapter import validate_ann_backend_request


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def execute(self, sql, params=()):
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
    def __init__(self, rows):
        self.rows = rows
        self.cursors = []

    def cursor(self):
        cursor = FakeCursor(self.rows)
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


class PostgresRetrieverAdapterContractTest(unittest.TestCase):
    def setUp(self):
        self.table = PassageTable(
            table_name="synthetic_passages",
            id_column="doc_id",
            text_column="passage_text",
            embedding_column="embedding",
        )

    def test_adapter_initialization_rejects_unknown_metric(self):
        adapter = PostgresRetrieverAdapter(
            dsn="postgresql://example.invalid/db",
            table=self.table,
            backend=ExactMetricBackend(),
            connect_fn=lambda dsn: FakeConnection([]),
        )

        with self.assertRaisesRegex(ValueError, "unsupported metric"):
            adapter.build_plan(
                RetrievalRequest(
                    query_vector=[1.0, 0.0],
                    top_k=3,
                    metric="manhattan",
                )
            )

    def test_validate_ann_backend_request_consolidates_shared_mode_checks(self):
        metric = validate_ann_backend_request(
            backend_name="pgvector",
            configured_metric="cosine",
            requested_metric="cosine",
            mode="approx",
            top_k=4,
        )
        self.assertEqual(metric, "cosine")

        metric = validate_ann_backend_request(
            backend_name="pg_turboquant",
            configured_metric="l2",
            requested_metric="l2",
            mode="approx_rerank",
            rerank_k=12,
            top_k=8,
        )
        self.assertEqual(metric, "l2")

        with self.assertRaisesRegex(ValueError, "metric mismatch"):
            validate_ann_backend_request(
                backend_name="pgvector",
                configured_metric="cosine",
                requested_metric="l2",
                mode="approx",
                top_k=3,
            )

        with self.assertRaisesRegex(ValueError, "unsupported pg_turboquant mode"):
            validate_ann_backend_request(
                backend_name="pg_turboquant",
                configured_metric="cosine",
                requested_metric="cosine",
                mode="exact",
                top_k=3,
            )

        with self.assertRaisesRegex(ValueError, "rerank_k >= top_k"):
            validate_ann_backend_request(
                backend_name="pgvector",
                configured_metric="cosine",
                requested_metric="cosine",
                mode="approx_rerank",
                rerank_k=2,
                top_k=3,
            )

    def test_sql_generation_uses_explicit_metric_mapping_and_ann_settings(self):
        adapter = PostgresRetrieverAdapter(
            dsn="postgresql://example.invalid/db",
            table=self.table,
            backend=StaticAnnBackend(
                name="fake_ann",
                metric_operators={
                    "cosine": "<=>",
                    "inner_product": "<#>",
                    "l2": "<->",
                },
                ann_setting_gucs={
                    "probes": "fake.probes",
                    "ef_search": "fake.ef_search",
                    "oversampling": "fake.oversampling",
                },
            ),
            connect_fn=lambda dsn: FakeConnection([]),
        )

        plan = adapter.build_plan(
            RetrievalRequest(
                query_vector=[0.25, -0.75],
                top_k=5,
                metric="cosine",
                ann={"probes": 4, "ef_search": 40, "oversampling": 2.5},
            )
        )

        self.assertEqual(
            plan.session_statements,
            [
                ("SET LOCAL fake.probes = %s", (4,)),
                ("SET LOCAL fake.ef_search = %s", (40,)),
                ("SET LOCAL fake.oversampling = %s", (2.5,)),
            ],
        )
        self.assertIn("FROM synthetic_passages", plan.sql)
        self.assertIn("WITH query_vector AS (SELECT %s::vector AS embedding)", plan.sql)
        self.assertIn("embedding <=> query_vector.embedding", plan.sql)
        self.assertIn("LIMIT %s", plan.sql)
        self.assertEqual(plan.params, ("[0.25,-0.75]", 5))

    def test_retrieve_renders_set_local_statements_without_bound_parameters(self):
        fake_connection = FakeConnection([("doc-1", Decimal("0.01"), "alpha")])
        adapter = PostgresRetrieverAdapter(
            dsn="postgresql://example.invalid/db",
            table=self.table,
            backend=StaticAnnBackend(
                name="fake_ann",
                metric_operators={"cosine": "<=>"},
                ann_setting_gucs={"probes": "fake.probes"},
            ),
            connect_fn=lambda dsn: fake_connection,
        )

        adapter.retrieve(
            RetrievalRequest(
                query_vector=[1.0, 0.0],
                top_k=1,
                metric="cosine",
                ann={"probes": 7},
            )
        )

        executed = fake_connection.cursors[0].executed
        self.assertEqual(executed[0], ("SET LOCAL fake.probes = 7", ()))
        self.assertEqual(executed[1][1], ("[1.0,0.0]", 1))

    def test_result_normalization_converts_ids_scores_and_text(self):
        adapter = PostgresRetrieverAdapter(
            dsn="postgresql://example.invalid/db",
            table=self.table,
            backend=ExactMetricBackend(),
            connect_fn=lambda dsn: FakeConnection([]),
        )

        normalized = adapter.normalize_rows(
            [
                (123, Decimal("0.125"), "alpha"),
                ("doc-2", 1, b"beta"),
            ]
        )

        self.assertEqual(
            normalized,
            [
                {"id": "123", "score": 0.125, "text": "alpha"},
                {"id": "doc-2", "score": 1.0, "text": "beta"},
            ],
        )

    def test_fake_exact_backend_integration_smoke(self):
        fake_connection = FakeConnection(
            [
                ("doc-1", Decimal("0.01"), "First synthetic passage"),
                ("doc-3", Decimal("0.05"), "Third synthetic passage"),
            ]
        )

        adapter = PostgresRetrieverAdapter(
            dsn="postgresql://example.invalid/db",
            table=self.table,
            backend=ExactMetricBackend(),
            connect_fn=lambda dsn: fake_connection,
        )

        results = adapter.retrieve(
            RetrievalRequest(
                query_vector=[1.0, 0.0, 0.0],
                top_k=2,
                metric="l2",
            )
        )

        self.assertEqual(
            results,
            [
                {"id": "doc-1", "score": 0.01, "text": "First synthetic passage"},
                {"id": "doc-3", "score": 0.05, "text": "Third synthetic passage"},
            ],
        )
        executed = fake_connection.cursors[0].executed
        self.assertEqual(len(executed), 1)
        self.assertIn("ORDER BY embedding <-> query_vector.embedding ASC", executed[0][0])
        self.assertEqual(executed[0][1], ("[1.0,0.0,0.0]", 2))


if __name__ == "__main__":
    unittest.main()
