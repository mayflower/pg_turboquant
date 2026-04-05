from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .adapter import (
    MODE_APPROX,
    MODE_APPROX_RERANK,
    PassageTable,
    RetrievalPlan,
    RetrievalRequest,
    validate_ann_backend_request,
    vector_literal,
)
METRIC_OPERATORS = {
    "cosine": "<=>",
    "inner_product": "<#>",
    "l2": "<->",
}


@dataclass(frozen=True)
class PgvectorBackendBase:
    index_name: str
    metric: str
    mode: str
    rerank_k: int | None = None

    index_kind: str = ""
    ann_key: str = ""
    ann_guc: str = ""

    @property
    def name(self) -> str:
        return self.index_kind

    def build_plan(self, table: PassageTable, request: RetrievalRequest) -> RetrievalPlan:
        request_metric = validate_ann_backend_request(
            backend_name="pgvector",
            configured_metric=self.metric,
            requested_metric=request.metric,
            mode=self.mode,
            rerank_k=self.rerank_k,
            top_k=request.top_k,
        )

        operator = METRIC_OPERATORS[request_metric]
        session_statements = []
        ann_value = request.ann.get(self.ann_key)
        if ann_value is not None:
            session_statements.append((f"SET LOCAL {self.ann_guc} = %s", (ann_value,)))

        query_literal = vector_literal(request.query_vector)
        if self.mode == MODE_APPROX:
            sql = (
                f"WITH query_vector AS (SELECT %s::{table.query_vector_cast} AS embedding) "
                f"SELECT p.{table.id_column} AS id, "
                f"p.{table.embedding_column} {operator} query_vector.embedding AS score, "
                f"p.{table.text_column} AS text "
                f"FROM {table.table_name} AS p "
                f"CROSS JOIN query_vector "
                f"ORDER BY p.{table.embedding_column} {operator} query_vector.embedding ASC "
                f"LIMIT %s"
            )
            params = (query_literal, request.top_k)
        else:
            sql = (
                f"WITH query_vector AS (SELECT %s::{table.query_vector_cast} AS embedding), "
                f"approx_candidates AS ("
                f"SELECT p.{table.id_column} AS id, p.{table.text_column} AS text, "
                f"p.{table.embedding_column} AS embedding "
                f"FROM {table.table_name} AS p "
                f"CROSS JOIN query_vector "
                f"ORDER BY p.{table.embedding_column} {operator} query_vector.embedding ASC "
                f"LIMIT %s"
                f") "
                f"SELECT id, approx_candidates.embedding {operator} query_vector.embedding AS score, text "
                f"FROM approx_candidates "
                f"CROSS JOIN query_vector "
                f"ORDER BY score ASC "
                f"LIMIT %s"
            )
            params = (query_literal, self.rerank_k, request.top_k)

        return RetrievalPlan(sql=sql, params=params, session_statements=session_statements)

    def serialize_run_metadata(self, plan: RetrievalPlan) -> dict[str, Any]:
        ann_value = None
        for sql, params in plan.session_statements:
            if self.ann_guc in sql:
                ann_value = params[0]
                break

        metadata = {
            "index_kind": self.index_kind,
            "index_name": self.index_name,
            "metric": self.metric,
            "mode": self.mode,
            "rerank_k": self.rerank_k if self.mode == MODE_APPROX_RERANK else None,
            "sql_template_hash": hashlib.sha256(plan.sql.encode("utf-8")).hexdigest(),
        }
        if self.ann_key == "ef_search":
            metadata["ef_search"] = ann_value
            metadata["probes"] = None
        else:
            metadata["probes"] = ann_value
            metadata["ef_search"] = None
        return metadata


@dataclass(frozen=True)
class PgvectorHnswBackend(PgvectorBackendBase):
    index_kind: str = "pgvector_hnsw"
    ann_key: str = "ef_search"
    ann_guc: str = "hnsw.ef_search"


@dataclass(frozen=True)
class PgvectorIvfflatBackend(PgvectorBackendBase):
    index_kind: str = "pgvector_ivfflat"
    ann_key: str = "probes"
    ann_guc: str = "ivfflat.probes"
