import unittest

from benchmarks.rag.bergen_adapter import PassageTable, RetrievalRequest
from benchmarks.rag.bergen_adapter.pgvector_backends import (
    PgvectorHnswBackend,
    PgvectorIvfflatBackend,
)


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


class PgvectorBackendContractTest(unittest.TestCase):
    def setUp(self):
        self.table = PassageTable(
            table_name="rag_docs",
            id_column="doc_id",
            text_column="passage_text",
            embedding_column="embedding",
        )

    def test_hnsw_sql_generation_and_metadata(self):
        backend = PgvectorHnswBackend(
            index_name="rag_docs_embedding_hnsw_idx",
            metric="cosine",
            mode="approx",
        )

        plan = backend.build_plan(
            self.table,
            RetrievalRequest(
                query_vector=[1.0, 0.0, 0.0],
                top_k=4,
                metric="cosine",
                ann={"ef_search": 80},
            ),
        )

        self.assertEqual(plan.session_statements, [("SET LOCAL hnsw.ef_search = %s", (80,))])
        self.assertIn("FROM rag_docs AS p", plan.sql)
        self.assertIn("ORDER BY p.embedding <=> query_vector.embedding ASC", plan.sql)
        self.assertNotIn("approx_candidates", plan.sql)

        metadata = backend.serialize_run_metadata(plan)
        self.assertEqual(metadata["index_kind"], "pgvector_hnsw")
        self.assertEqual(metadata["metric"], "cosine")
        self.assertEqual(metadata["ef_search"], 80)
        self.assertIsNone(metadata["rerank_k"])
        self.assertEqual(len(metadata["sql_template_hash"]), 64)

    def test_ivfflat_rerank_sql_generation_and_metadata(self):
        backend = PgvectorIvfflatBackend(
            index_name="rag_docs_embedding_ivfflat_idx",
            metric="l2",
            mode="approx_rerank",
            rerank_k=20,
        )

        plan = backend.build_plan(
            self.table,
            RetrievalRequest(
                query_vector=[0.5, -0.25],
                top_k=5,
                metric="l2",
                ann={"probes": 6},
            ),
        )

        self.assertEqual(plan.session_statements, [("SET LOCAL ivfflat.probes = %s", (6,))])
        self.assertIn("WITH query_vector AS", plan.sql)
        self.assertIn("approx_candidates AS", plan.sql)
        self.assertIn("SELECT p.doc_id AS id, p.passage_text AS text, p.embedding AS embedding", plan.sql)
        self.assertIn("SELECT id, approx_candidates.embedding <-> query_vector.embedding AS score, text", plan.sql)
        self.assertIn("ORDER BY score ASC", plan.sql)
        self.assertEqual(plan.params[-2:], (20, 5))

        metadata = backend.serialize_run_metadata(plan)
        self.assertEqual(metadata["index_kind"], "pgvector_ivfflat")
        self.assertEqual(metadata["probes"], 6)
        self.assertEqual(metadata["rerank_k"], 20)

    def test_invalid_metric_or_rerank_configuration_fails_early(self):
        backend = PgvectorHnswBackend(
            index_name="rag_docs_embedding_hnsw_idx",
            metric="cosine",
            mode="approx_rerank",
            rerank_k=3,
        )

        with self.assertRaisesRegex(ValueError, "metric mismatch"):
            backend.build_plan(
                self.table,
                RetrievalRequest(
                    query_vector=[1.0, 0.0],
                    top_k=2,
                    metric="l2",
                ),
            )

        with self.assertRaisesRegex(ValueError, "rerank_k >= top_k"):
            backend.build_plan(
                self.table,
                RetrievalRequest(
                    query_vector=[1.0, 0.0],
                    top_k=4,
                    metric="cosine",
                ),
            )

    def test_tiny_integration_smoke_for_hnsw_and_ivfflat(self):
        from benchmarks.rag.bergen_adapter.adapter import PostgresRetrieverAdapter

        hnsw_backend = PgvectorHnswBackend(
            index_name="rag_docs_embedding_hnsw_idx",
            metric="inner_product",
            mode="approx",
        )
        ivfflat_backend = PgvectorIvfflatBackend(
            index_name="rag_docs_embedding_ivfflat_idx",
            metric="inner_product",
            mode="approx_rerank",
            rerank_k=8,
        )

        hnsw_connection = FakeConnection(
            [("doc-1", 0.25, "HNSW result"), ("doc-2", 0.5, "HNSW result 2")]
        )
        ivfflat_connection = FakeConnection(
            [("doc-3", 0.1, "IVFFlat result"), ("doc-1", 0.2, "IVFFlat result 2")]
        )

        hnsw_adapter = PostgresRetrieverAdapter(
            dsn="postgresql://example.invalid/db",
            table=self.table,
            backend=hnsw_backend,
            connect_fn=lambda dsn: hnsw_connection,
        )
        ivfflat_adapter = PostgresRetrieverAdapter(
            dsn="postgresql://example.invalid/db",
            table=self.table,
            backend=ivfflat_backend,
            connect_fn=lambda dsn: ivfflat_connection,
        )

        hnsw_results = hnsw_adapter.retrieve(
            RetrievalRequest(
                query_vector=[1.0, 0.0],
                top_k=2,
                metric="inner_product",
                ann={"ef_search": 32},
            )
        )
        ivfflat_results = ivfflat_adapter.retrieve(
            RetrievalRequest(
                query_vector=[1.0, 0.0],
                top_k=2,
                metric="inner_product",
                ann={"probes": 4},
            )
        )

        self.assertEqual(hnsw_results[0]["id"], "doc-1")
        self.assertEqual(ivfflat_results[0]["id"], "doc-3")
        self.assertIn("hnsw.ef_search", hnsw_connection.cursors[0].executed[0][0])
        self.assertIn("ivfflat.probes", ivfflat_connection.cursors[0].executed[0][0])
        self.assertIn(
            "ORDER BY p.embedding <#> query_vector.embedding ASC",
            hnsw_connection.cursors[0].executed[-1][0],
        )
        self.assertIn("approx_candidates AS", ivfflat_connection.cursors[0].executed[-1][0])


if __name__ == "__main__":
    unittest.main()
