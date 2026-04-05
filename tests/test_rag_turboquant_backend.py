import json
import unittest

from benchmarks.rag.bergen_adapter import PassageTable, RetrievalRequest
from benchmarks.rag.bergen_adapter.turboquant_backend import PgTurboquantBackend


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


class PgTurboquantBackendContractTest(unittest.TestCase):
    def setUp(self):
        self.table = PassageTable(
            table_name="rag_docs",
            id_column="doc_id",
            text_column="passage_text",
            embedding_column="embedding",
        )

    def test_approx_template_and_metadata_are_stable(self):
        backend = PgTurboquantBackend(
            index_name="rag_docs_embedding_tq_idx",
            metric="cosine",
            normalized=True,
            mode="approx",
        )

        plan = backend.build_plan(
            self.table,
            RetrievalRequest(
                query_vector=[1.0, 0.0, 0.0],
                top_k=3,
                metric="cosine",
                ann={"probes": 4, "oversampling": 2.0},
            ),
        )

        self.assertEqual(
            plan.session_statements,
            [
                ("SET LOCAL turboquant.probes = %s", (4,)),
                ("SET LOCAL turboquant.oversample_factor = %s", (2.0,)),
            ],
        )
        self.assertIn("WITH query_vector AS", plan.sql)
        self.assertIn("FROM rag_docs AS p", plan.sql)
        self.assertIn("ORDER BY p.embedding <=> query_vector.embedding ASC", plan.sql)
        self.assertNotIn("tq_approx_candidates", plan.sql)
        self.assertNotIn("tq_rerank_candidates", plan.sql)
        self.assertEqual(plan.params[-1], 3)

        metadata = backend.serialize_run_metadata(plan)
        self.assertEqual(metadata["index_kind"], "pg_turboquant")
        self.assertEqual(metadata["metric"], "cosine")
        self.assertTrue(metadata["normalized"])
        self.assertEqual(metadata["probes"], 4)
        self.assertEqual(metadata["oversample_factor"], 2.0)
        self.assertIsNone(metadata["rerank_k"])
        self.assertEqual(metadata["mode"], "approx")
        self.assertEqual(len(metadata["sql_template_hash"]), 64)
        json.dumps(metadata)

    def test_rerank_template_uses_helper_and_serializes_rerank_k(self):
        backend = PgTurboquantBackend(
            index_name="rag_docs_embedding_tq_idx",
            metric="l2",
            normalized=False,
            mode="approx_rerank",
            rerank_k=25,
        )

        plan = backend.build_plan(
            self.table,
            RetrievalRequest(
                query_vector=[0.25, -0.25],
                top_k=5,
                metric="l2",
                ann={"probes": 2, "oversampling": 3.0},
            ),
        )

        self.assertIn("approx_candidates", plan.sql)
        self.assertIn("ORDER BY", plan.sql)
        self.assertEqual(plan.params[-2:], (25, 5))

        metadata = backend.serialize_run_metadata(plan)
        self.assertEqual(metadata["mode"], "approx_rerank")
        self.assertEqual(metadata["rerank_k"], 25)
        self.assertEqual(metadata["metric"], "l2")
        self.assertFalse(metadata["normalized"])

    def test_filtered_covering_plan_uses_stage1_payload_projection(self):
        backend = PgTurboquantBackend(
            index_name="rag_docs_embedding_tq_idx",
            metric="cosine",
            normalized=True,
            mode="approx",
        )

        plan = backend.build_plan(
            self.table,
            RetrievalRequest(
                query_vector=[1.0, 0.0, 0.0],
                top_k=3,
                metric="cosine",
                ann={
                    "probes": 2,
                    "oversampling": 2.0,
                    "filters": {
                        "tenant_id": 7,
                        "source_id": [10, 11],
                        "lang_id": 1,
                    },
                    "stage1_payload_columns": [
                        "doc_id",
                        "chunk_id",
                        "tenant_id",
                        "doc_version",
                    ],
                    "text_join_column": "passage_id",
                },
            ),
        )

        self.assertIn("stage1_candidates AS", plan.sql)
        self.assertIn("source_id = ANY", plan.sql)
        self.assertIn("SELECT p.passage_id AS id, p.doc_id, p.chunk_id, p.tenant_id, p.doc_version", plan.sql)
        self.assertIn("JOIN rag_docs AS text_source ON text_source.passage_id = stage1_candidates.id", plan.sql)
        self.assertEqual(plan.session_statements[:2], [
            ("SET LOCAL turboquant.probes = %s", (2,)),
            ("SET LOCAL turboquant.oversample_factor = %s", (2.0,)),
        ])

        metadata = backend.serialize_run_metadata(plan)
        self.assertEqual(metadata["filters"], {"tenant_id": 7, "source_id": [10, 11], "lang_id": 1})
        self.assertEqual(
            metadata["stage1_payload_columns"],
            ["doc_id", "chunk_id", "tenant_id", "doc_version"],
        )

    def test_delta_union_plan_queries_base_and_delta_indexes(self):
        backend = PgTurboquantBackend(
            index_name="rag_docs_embedding_tq_idx",
            metric="cosine",
            normalized=True,
            mode="approx",
        )

        table = PassageTable(
            table_name="rag_docs",
            id_column="passage_id",
            text_column="passage_text",
            embedding_column="embedding",
        )

        plan = backend.build_plan(
            table,
            RetrievalRequest(
                query_vector=[1.0, 0.0, 0.0],
                top_k=2,
                metric="cosine",
                ann={
                    "delta_table_name": "rag_docs_delta",
                    "delta_candidate_limit": 8,
                },
            ),
        )

        self.assertIn("FROM rag_docs AS p", plan.sql)
        self.assertIn("FROM rag_docs_delta AS p", plan.sql)
        self.assertIn("UNION ALL", plan.sql)
        metadata = backend.serialize_run_metadata(plan)
        self.assertEqual(metadata["delta_mode"], "union")
        self.assertEqual(metadata["delta_table_name"], "rag_docs_delta")
        self.assertEqual(metadata["delta_candidate_limit"], 8)

    def test_native_delta_plan_stays_on_one_index_path(self):
        backend = PgTurboquantBackend(
            index_name="rag_docs_embedding_tq_idx",
            metric="cosine",
            normalized=True,
            mode="approx",
        )

        plan = backend.build_plan(
            self.table,
            RetrievalRequest(
                query_vector=[1.0, 0.0, 0.0],
                top_k=2,
                metric="cosine",
                ann={
                    "native_delta": True,
                    "filters": {"tenant_id": 7, "doc_version": [7, 8]},
                    "stage1_payload_columns": ["doc_id", "doc_version"],
                },
            ),
        )

        self.assertIn("stage1_candidates AS", plan.sql)
        self.assertNotIn("UNION ALL", plan.sql)
        metadata = backend.serialize_run_metadata(plan)
        self.assertEqual(metadata["delta_mode"], "native")
        self.assertIsNone(metadata["delta_table_name"])

    def test_inner_product_requires_normalized_vectors(self):
        backend = PgTurboquantBackend(
            index_name="rag_docs_embedding_tq_idx",
            metric="inner_product",
            normalized=False,
            mode="approx",
        )

        with self.assertRaisesRegex(ValueError, "requires normalized vectors"):
            backend.build_plan(
                self.table,
                RetrievalRequest(
                    query_vector=[1.0, 0.0],
                    top_k=2,
                    metric="inner_product",
                ),
            )

    def test_tiny_integration_smoke_for_both_modes(self):
        from benchmarks.rag.bergen_adapter.adapter import PostgresRetrieverAdapter

        approx_backend = PgTurboquantBackend(
            index_name="rag_docs_embedding_tq_idx",
            metric="cosine",
            normalized=True,
            mode="approx",
        )
        rerank_backend = PgTurboquantBackend(
            index_name="rag_docs_embedding_tq_idx",
            metric="cosine",
            normalized=True,
            mode="approx_rerank",
            rerank_k=10,
        )

        approx_connection = FakeConnection(
            [("doc-1", 0.125, "Approx result"), ("doc-2", 0.250, "Approx result 2")]
        )
        rerank_connection = FakeConnection(
            [("doc-1", 0.010, "Reranked result"), ("doc-3", 0.040, "Reranked result 2")]
        )

        approx_adapter = PostgresRetrieverAdapter(
            dsn="postgresql://example.invalid/db",
            table=self.table,
            backend=approx_backend,
            connect_fn=lambda dsn: approx_connection,
        )
        rerank_adapter = PostgresRetrieverAdapter(
            dsn="postgresql://example.invalid/db",
            table=self.table,
            backend=rerank_backend,
            connect_fn=lambda dsn: rerank_connection,
        )

        approx_results = approx_adapter.retrieve(
            RetrievalRequest(
                query_vector=[1.0, 0.0, 0.0],
                top_k=2,
                metric="cosine",
                ann={"probes": 1, "oversampling": 1.5},
            )
        )
        rerank_results = rerank_adapter.retrieve(
            RetrievalRequest(
                query_vector=[1.0, 0.0, 0.0],
                top_k=2,
                metric="cosine",
                ann={"probes": 2, "oversampling": 2.0},
            )
        )

        self.assertEqual(len(approx_results), 2)
        self.assertEqual(len(rerank_results), 2)
        self.assertEqual(approx_results[0]["id"], "doc-1")
        self.assertEqual(rerank_results[0]["id"], "doc-1")
        self.assertIn("ORDER BY p.embedding <=> query_vector.embedding ASC", approx_connection.cursors[0].executed[-1][0])
        self.assertNotIn("tq_approx_candidates", approx_connection.cursors[0].executed[-1][0])
        self.assertIn("approx_candidates", rerank_connection.cursors[0].executed[-1][0])


if __name__ == "__main__":
    unittest.main()
