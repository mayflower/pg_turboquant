from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence


VALID_METRICS = frozenset({"cosine", "inner_product", "l2"})
MODE_APPROX = "approx"
MODE_APPROX_RERANK = "approx_rerank"
VALID_APPROX_MODES = frozenset({MODE_APPROX, MODE_APPROX_RERANK})


@dataclass(frozen=True)
class PassageTable:
    table_name: str
    id_column: str
    text_column: str
    embedding_column: str
    query_vector_cast: str = "vector"


@dataclass(frozen=True)
class RetrievalRequest:
    query_vector: Sequence[float]
    top_k: int
    metric: str
    ann: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalPlan:
    sql: str
    params: tuple[Any, ...]
    session_statements: list[tuple[str, tuple[Any, ...]]]


class PostgresBackend(Protocol):
    name: str

    def build_plan(self, table: PassageTable, request: RetrievalRequest) -> RetrievalPlan:
        ...


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def render_session_statement(sql: str, params: Sequence[Any]) -> str:
    rendered = sql
    for value in params:
        rendered = rendered.replace("%s", render_sql_literal(value), 1)
    return rendered


def render_sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    raise TypeError(f"unsupported SQL literal type for session statement: {type(value).__name__}")


@dataclass(frozen=True)
class StaticAnnBackend:
    name: str
    metric_operators: Mapping[str, str]
    ann_setting_gucs: Mapping[str, str] = field(default_factory=dict)

    def build_plan(self, table: PassageTable, request: RetrievalRequest) -> RetrievalPlan:
        metric = validate_metric(request.metric)
        operator = self.metric_operators.get(metric)
        if operator is None:
            raise ValueError(f"backend {self.name} does not support metric: {metric}")
        session_statements = []
        for key, value in request.ann.items():
            guc = self.ann_setting_gucs.get(key)
            if guc is None or value is None:
                continue
            session_statements.append((f"SET LOCAL {guc} = %s", (value,)))

        sql = (
            f"WITH query_vector AS (SELECT %s::{table.query_vector_cast} AS embedding) "
            f"SELECT {table.id_column} AS doc_id, "
            f"{table.embedding_column} {operator} query_vector.embedding AS score, "
            f"{table.text_column} AS passage_text "
            f"FROM {table.table_name} "
            f"CROSS JOIN query_vector "
            f"ORDER BY {table.embedding_column} {operator} query_vector.embedding ASC "
            f"LIMIT %s"
        )
        literal = vector_literal(request.query_vector)
        return RetrievalPlan(
            sql=sql,
            params=(literal, request.top_k),
            session_statements=session_statements,
        )


class ExactMetricBackend(StaticAnnBackend):
    def __init__(self) -> None:
        super().__init__(
            name="exact_metric",
            metric_operators={
                "cosine": "<=>",
                "inner_product": "<#>",
                "l2": "<->",
            },
        )


class PostgresRetrieverAdapter:
    def __init__(
        self,
        *,
        dsn: str,
        table: PassageTable,
        backend: PostgresBackend,
        connect_fn: Callable[[str], Any],
    ) -> None:
        self.dsn = dsn
        self.table = table
        self.backend = backend
        self.connect_fn = connect_fn
        self._connection: Any = None

    def _get_connection(self) -> Any:
        if self._connection is None:
            self._connection = self.connect_fn(self.dsn)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def build_plan(self, request: RetrievalRequest) -> RetrievalPlan:
        return self.backend.build_plan(self.table, request)

    def normalize_rows(self, rows: Sequence[Any]) -> list[dict[str, Any]]:
        normalized = []
        for row in rows:
            if isinstance(row, Mapping):
                doc_id = row["doc_id"]
                score = row["score"]
                text = row["passage_text"]
            else:
                doc_id, score, text = row

            if isinstance(doc_id, bytes):
                doc_id = doc_id.decode("utf-8")
            if isinstance(text, bytes):
                text = text.decode("utf-8")

            normalized.append(
                {
                    "id": str(doc_id),
                    "score": float(score),
                    "text": str(text),
                }
            )
        return normalized

    def retrieve(self, request: RetrievalRequest) -> list[dict[str, Any]]:
        rows, _ = self.retrieve_with_metadata(request)
        return rows

    def retrieve_with_metadata(
        self, request: RetrievalRequest
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        rows = self._execute_retrieval(request)
        scan_stats = self._fetch_scan_stats()
        return rows, scan_stats

    def _execute_retrieval(self, request: RetrievalRequest) -> list[dict[str, Any]]:
        plan = self.build_plan(request)
        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                for sql, params in plan.session_statements:
                    cursor.execute(render_session_statement(sql, params))
                cursor.execute(plan.sql, plan.params)
                return self.normalize_rows(cursor.fetchall())
        finally:
            connection.rollback()

    def _fetch_scan_stats(self) -> dict[str, Any] | None:
        if getattr(self.backend, "name", None) != "pg_turboquant":
            return None
        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT public.tq_last_scan_stats()::text")
                raw = cursor.fetchone()[0]
                if raw:
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        pass
        finally:
            connection.rollback()
        return None


def validate_metric(metric: str) -> str:
    if metric not in VALID_METRICS:
        raise ValueError(f"unsupported metric: {metric}")
    return metric


def validate_ann_backend_request(
    *,
    backend_name: str,
    configured_metric: str,
    requested_metric: str,
    mode: str,
    top_k: int,
    rerank_k: int | None = None,
) -> str:
    metric = validate_metric(requested_metric)
    if metric != configured_metric:
        raise ValueError(
            f"backend metric mismatch: configured {configured_metric}, requested {metric}"
        )
    if mode not in VALID_APPROX_MODES:
        raise ValueError(f"unsupported {backend_name} mode: {mode}")
    if mode == MODE_APPROX_RERANK and (rerank_k is None or rerank_k < top_k):
        raise ValueError("approx_rerank mode requires rerank_k >= top_k")
    return metric
