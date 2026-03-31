from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .adapter import PassageTable, RetrievalPlan, RetrievalRequest, validate_metric, vector_literal


MODE_APPROX = "approx"
MODE_APPROX_RERANK = "approx_rerank"
METRIC_OPERATORS = {
    "cosine": "<=>",
    "inner_product": "<#>",
    "l2": "<->",
}


@dataclass(frozen=True)
class PgTurboquantBackend:
    index_name: str
    metric: str
    normalized: bool
    mode: str
    rerank_k: int | None = None
    probes_guc: str = "turboquant.probes"
    oversample_guc: str = "turboquant.oversample_factor"
    max_visited_codes_guc: str = "turboquant.max_visited_codes"
    max_visited_pages_guc: str = "turboquant.max_visited_pages"
    helper_schema: str = "public"

    @property
    def name(self) -> str:
        return "pg_turboquant"

    def build_plan(self, table: PassageTable, request: RetrievalRequest) -> RetrievalPlan:
        request_metric = validate_metric(request.metric)
        if request_metric != self.metric:
            raise ValueError(
                f"backend metric mismatch: configured {self.metric}, requested {request_metric}"
            )
        if request_metric == "inner_product" and not self.normalized:
            raise ValueError("pg_turboquant inner_product backend requires normalized vectors")
        if self.mode == MODE_APPROX_RERANK and (self.rerank_k is None or self.rerank_k < request.top_k):
            raise ValueError("approx_rerank mode requires rerank_k >= top_k")
        if self.mode not in {MODE_APPROX, MODE_APPROX_RERANK}:
            raise ValueError(f"unsupported pg_turboquant mode: {self.mode}")

        session_statements = []
        probes = request.ann.get("probes")
        if probes is not None:
            session_statements.append((f"SET LOCAL {self.probes_guc} = %s", (probes,)))
        oversampling = request.ann.get("oversampling")
        if oversampling is not None:
            session_statements.append((f"SET LOCAL {self.oversample_guc} = %s", (oversampling,)))
        max_visited_codes = request.ann.get("max_visited_codes")
        if max_visited_codes is not None:
            session_statements.append((f"SET LOCAL {self.max_visited_codes_guc} = %s", (max_visited_codes,)))
        max_visited_pages = request.ann.get("max_visited_pages")
        if max_visited_pages is not None:
            session_statements.append((f"SET LOCAL {self.max_visited_pages_guc} = %s", (max_visited_pages,)))

        query_literal = vector_literal(request.query_vector)
        operator = METRIC_OPERATORS[request_metric]
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

        return RetrievalPlan(
            sql=sql,
            params=params,
            session_statements=session_statements,
        )

    def metric_name_for_helper(self) -> str:
        if self.metric == "inner_product":
            return "ip"
        return self.metric

    def serialize_run_metadata(self, plan: RetrievalPlan) -> dict[str, Any]:
        probes = None
        oversample_factor = None
        max_visited_codes = None
        max_visited_pages = None
        for sql, params in plan.session_statements:
            if self.probes_guc in sql:
                probes = params[0]
            elif self.oversample_guc in sql:
                oversample_factor = params[0]
            elif self.max_visited_codes_guc in sql:
                max_visited_codes = params[0]
            elif self.max_visited_pages_guc in sql:
                max_visited_pages = params[0]

        sql_template_hash = hashlib.sha256(plan.sql.encode("utf-8")).hexdigest()
        return {
            "index_kind": "pg_turboquant",
            "index_name": self.index_name,
            "metric": self.metric,
            "normalized": self.normalized,
            "mode": self.mode,
            "probes": probes,
            "oversample_factor": oversample_factor,
            "max_visited_codes": max_visited_codes,
            "max_visited_pages": max_visited_pages,
            "rerank_k": self.rerank_k if self.mode == MODE_APPROX_RERANK else None,
            "sql_template_hash": sql_template_hash,
        }
