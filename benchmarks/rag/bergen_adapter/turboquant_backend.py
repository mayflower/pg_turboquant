from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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
    iterative_scan_guc: str = "turboquant.iterative_scan"
    min_rows_after_filter_guc: str = "turboquant.min_rows_after_filter"
    helper_schema: str = "public"

    @property
    def name(self) -> str:
        return "pg_turboquant"

    def build_plan(self, table: PassageTable, request: RetrievalRequest) -> RetrievalPlan:
        request_metric = validate_ann_backend_request(
            backend_name="pg_turboquant",
            configured_metric=self.metric,
            requested_metric=request.metric,
            mode=self.mode,
            rerank_k=self.rerank_k,
            top_k=request.top_k,
        )
        if request_metric == "inner_product" and not self.normalized:
            raise ValueError("pg_turboquant inner_product backend requires normalized vectors")

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
        iterative_scan = request.ann.get("iterative_scan")
        if iterative_scan is not None:
            session_statements.append((f"SET LOCAL {self.iterative_scan_guc} = %s", (iterative_scan,)))
        min_rows_after_filter = request.ann.get("min_rows_after_filter")
        if min_rows_after_filter is not None:
            session_statements.append(
                (f"SET LOCAL {self.min_rows_after_filter_guc} = %s", (min_rows_after_filter,))
            )

        query_literal = vector_literal(request.query_vector)
        operator = METRIC_OPERATORS[request_metric]
        filters = _normalize_filters(request.ann.get("filters"))
        stage1_payload_columns = _normalize_identifier_list(request.ann.get("stage1_payload_columns"))
        text_join_column = str(request.ann.get("text_join_column") or table.id_column)
        delta_table_name = request.ann.get("delta_table_name")
        native_delta = bool(request.ann.get("native_delta"))
        delta_candidate_limit = int(request.ann.get("delta_candidate_limit") or max(request.top_k, 1))
        stage1_limit = int(request.ann.get("stage1_candidate_limit") or request.top_k)
        if self.mode == MODE_APPROX:
            if filters or stage1_payload_columns or delta_table_name or native_delta:
                sql, params = self._build_filtered_stage1_plan(
                    table=table,
                    query_literal=query_literal,
                    operator=operator,
                    request_top_k=request.top_k,
                    filters=filters,
                    stage1_payload_columns=stage1_payload_columns,
                    text_join_column=text_join_column,
                    delta_table_name=str(delta_table_name) if delta_table_name else None,
                    native_delta=native_delta,
                    delta_candidate_limit=delta_candidate_limit,
                    stage1_limit=stage1_limit,
                    exact_rerank=False,
                )
            else:
                sql = (
                    f"WITH query_vector AS (SELECT %s::{table.query_vector_cast} AS embedding) "
                    f"SELECT p.{table.id_column} AS id, "
                    f"p.{table.embedding_column} {operator} query_vector.embedding AS score "
                    f"FROM {table.table_name} AS p "
                    f"CROSS JOIN query_vector "
                    f"ORDER BY p.{table.embedding_column} {operator} query_vector.embedding ASC "
                    f"LIMIT %s"
                )
                params = (query_literal, request.top_k)
        elif filters or stage1_payload_columns or delta_table_name or native_delta:
            sql, params = self._build_filtered_stage1_plan(
                table=table,
                query_literal=query_literal,
                operator=operator,
                request_top_k=request.top_k,
                filters=filters,
                stage1_payload_columns=stage1_payload_columns,
                text_join_column=text_join_column,
                delta_table_name=str(delta_table_name) if delta_table_name else None,
                native_delta=native_delta,
                delta_candidate_limit=delta_candidate_limit,
                stage1_limit=self.rerank_k or stage1_limit,
                exact_rerank=True,
            )
        else:
            sql = (
                f"WITH query_vector AS (SELECT %s::{table.query_vector_cast} AS embedding), "
                f"approx_candidates AS ("
                f"SELECT p.{table.id_column} AS id, p.{table.embedding_column} AS embedding "
                f"FROM {table.table_name} AS p "
                f"CROSS JOIN query_vector "
                f"ORDER BY p.{table.embedding_column} {operator} query_vector.embedding ASC "
                f"LIMIT %s"
                f") "
                f"SELECT approx_candidates.id AS id, "
                f"text_source.{table.embedding_column} {operator} query_vector.embedding AS score "
                f"FROM approx_candidates "
                f"JOIN {table.table_name} AS text_source ON text_source.{table.id_column} = approx_candidates.id "
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

    def _build_filtered_stage1_plan(
        self,
        *,
        table: PassageTable,
        query_literal: str,
        operator: str,
        request_top_k: int,
        filters: Mapping[str, int | Sequence[int]],
        stage1_payload_columns: Sequence[str],
        text_join_column: str,
        delta_table_name: str | None,
        native_delta: bool,
        delta_candidate_limit: int,
        stage1_limit: int,
        exact_rerank: bool,
    ) -> tuple[str, tuple[Any, ...]]:
        params: list[Any] = [query_literal]
        where_sql, where_params = _render_filter_clause("p", filters)
        params.extend(where_params)

        stage1_projection = [
            f"p.{text_join_column} AS id",
            f"p.{table.embedding_column} {operator} query_vector.embedding AS ann_score",
        ]
        for column in stage1_payload_columns:
            stage1_projection.append(f"p.{column}")
        projection_sql = ", ".join(stage1_projection)

        stage1_ctes = [
            "query_vector AS (SELECT %s::%s AS embedding)" % ( "%s", table.query_vector_cast),
            (
                "base_stage1 AS ("
                f"SELECT {projection_sql} "
                f"FROM {table.table_name} AS p "
                "CROSS JOIN query_vector "
                f"{where_sql} "
                f"ORDER BY p.{table.embedding_column} {operator} query_vector.embedding ASC "
                "LIMIT %s)"
            ),
        ]
        params.append(stage1_limit)

        union_source = "SELECT * FROM base_stage1"
        if delta_table_name and not native_delta:
            delta_where_sql, delta_where_params = _render_filter_clause("p", filters)
            stage1_ctes.append(
                "delta_stage1 AS ("
                f"SELECT {projection_sql} "
                f"FROM {delta_table_name} AS p "
                "CROSS JOIN query_vector "
                f"{delta_where_sql} "
                f"ORDER BY p.{table.embedding_column} {operator} query_vector.embedding ASC "
                "LIMIT %s)"
            )
            params.extend(delta_where_params)
            params.append(delta_candidate_limit)
            union_source = "SELECT * FROM base_stage1 UNION ALL SELECT * FROM delta_stage1"

        stage1_ctes.append(
            "stage1_candidates AS ("
            f"{union_source}"
            ")"
        )

        final_projection = ["stage1_candidates.id"]
        for column in stage1_payload_columns:
            final_projection.append(f"stage1_candidates.{column}")

        if exact_rerank:
            final_projection.insert(
                1,
                f"text_source.{table.embedding_column} {operator} query_vector.embedding AS score",
            )
        else:
            final_projection.insert(1, "stage1_candidates.ann_score AS score")

        final_from = "FROM stage1_candidates "
        if exact_rerank:
            final_from += (
                f"JOIN {table.table_name} AS text_source "
                f"ON text_source.{text_join_column} = stage1_candidates.id "
            )
        final_from += "CROSS JOIN query_vector "

        sql = (
            f"/* tq_filters: {json.dumps(filters, sort_keys=True)} */ "
            f"/* tq_stage1_payload_columns: {json.dumps(list(stage1_payload_columns))} */ "
            f"/* tq_delta_table_name: {json.dumps(delta_table_name)} */ "
            f"/* tq_delta_mode: {json.dumps('native' if native_delta else ('union' if delta_table_name else None))} */ "
            f"/* tq_delta_candidate_limit: {json.dumps(delta_candidate_limit if delta_table_name else None)} */ "
            "WITH "
            + ", ".join(stage1_ctes)
            + " "
            + "SELECT "
            + ", ".join(final_projection)
            + " "
            + final_from
            + "ORDER BY score ASC "
            + "LIMIT %s"
        )
        params.append(request_top_k)
        return sql, tuple(params)

    def metric_name_for_helper(self) -> str:
        if self.metric == "inner_product":
            return "ip"
        return self.metric

    def serialize_run_metadata(self, plan: RetrievalPlan) -> dict[str, Any]:
        probes = None
        oversample_factor = None
        max_visited_codes = None
        max_visited_pages = None
        iterative_scan = None
        min_rows_after_filter = None
        for sql, params in plan.session_statements:
            if self.probes_guc in sql:
                probes = params[0]
            elif self.oversample_guc in sql:
                oversample_factor = params[0]
            elif self.max_visited_codes_guc in sql:
                max_visited_codes = params[0]
            elif self.max_visited_pages_guc in sql:
                max_visited_pages = params[0]
            elif self.iterative_scan_guc in sql:
                iterative_scan = params[0]
            elif self.min_rows_after_filter_guc in sql:
                min_rows_after_filter = params[0]

        sql_template_hash = hashlib.sha256(plan.sql.encode("utf-8")).hexdigest()
        filters = _extract_json_comment(plan.sql, "tq_filters")
        stage1_payload_columns = _extract_json_comment(plan.sql, "tq_stage1_payload_columns")
        delta_table_name = _extract_json_comment(plan.sql, "tq_delta_table_name")
        delta_mode = _extract_json_comment(plan.sql, "tq_delta_mode")
        delta_candidate_limit = _extract_json_comment(plan.sql, "tq_delta_candidate_limit")
        return {
            "index_kind": "pg_turboquant",
            "index_name": self.index_name,
            "metric": self.metric,
            "normalized": self.normalized,
            "mode": self.mode,
            "retrieval_execution_mode": (
                "approx_exact_rerank" if self.mode == MODE_APPROX_RERANK else "approx_stage1_only"
            ),
            "context_fetch_mode": "post_limit_text_fetch",
            "exact_rerank_enabled": self.mode == MODE_APPROX_RERANK,
            "stage1_covering": bool(stage1_payload_columns) and self.mode == MODE_APPROX,
            "probes": probes,
            "oversample_factor": oversample_factor,
            "max_visited_codes": max_visited_codes,
            "max_visited_pages": max_visited_pages,
            "iterative_scan": iterative_scan,
            "min_rows_after_filter": min_rows_after_filter,
            "rerank_k": self.rerank_k if self.mode == MODE_APPROX_RERANK else None,
            "filters": filters,
            "stage1_payload_columns": stage1_payload_columns,
            "delta_table_name": delta_table_name,
            "delta_mode": delta_mode,
            "delta_surface": (
                "native_index_delta"
                if delta_mode == "native"
                else ("external_union_delta" if delta_mode == "union" else "none")
            ),
            "delta_candidate_limit": delta_candidate_limit,
            "sql_template_hash": sql_template_hash,
        }


def _normalize_identifier_list(value: Any) -> list[str]:
    if not value:
        return []
    return [str(item) for item in value]


def _normalize_filters(value: Any) -> dict[str, int | list[int]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, int | list[int]] = {}
    for key, raw in value.items():
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            normalized[str(key)] = [int(item) for item in raw]
        else:
            normalized[str(key)] = int(raw)
    return normalized


def _render_filter_clause(
    table_alias: str, filters: Mapping[str, int | Sequence[int]]
) -> tuple[str, list[Any]]:
    if not filters:
        return "", []

    clauses: list[str] = []
    params: list[Any] = []
    for column, value in filters.items():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            clauses.append(f"{table_alias}.{column} = ANY (%s::int4[])")
            params.append(list(int(item) for item in value))
        else:
            clauses.append(f"{table_alias}.{column} = %s")
            params.append(int(value))

    return "WHERE " + " AND ".join(clauses), params


def _extract_json_comment(sql: str, key: str) -> Any:
    marker = f"/* {key}:"
    start = sql.find(marker)
    if start < 0:
        return None
    payload_start = start + len(marker)
    payload_end = sql.find("*/", payload_start)
    if payload_end < 0:
        return None
    raw = sql[payload_start:payload_end].strip()
    if not raw:
        return None
    return json.loads(raw)
