SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

SET enable_seqscan = off;
SET enable_bitmapscan = off;
SET turboquant.probes = 2;
SET turboquant.oversample_factor = 1;
SET turboquant.max_visited_codes = 0;
SET turboquant.max_visited_pages = 0;

CREATE FUNCTION tq_vec64(a float8, b float8)
RETURNS vector
LANGUAGE SQL
IMMUTABLE
AS $$
	SELECT ('[' || a::text || ',' || b::text || repeat(',0', 62) || ']')::vector(64);
$$;

CREATE TABLE tq_filtered_page_pruning_docs (
	id int4 PRIMARY KEY,
	tenant_id int4 NOT NULL,
	embedding vector(64)
);

INSERT INTO tq_filtered_page_pruning_docs (id, tenant_id, embedding)
SELECT gs, 1, tq_vec64(cos(gs * 0.0008), sin(gs * 0.0008))
FROM generate_series(1, 128) AS gs;

INSERT INTO tq_filtered_page_pruning_docs (id, tenant_id, embedding)
SELECT 128 + gs, 2, tq_vec64(cos(0.48 + (gs * 0.0012)), sin(0.48 + (gs * 0.0012)))
FROM generate_series(1, 128) AS gs;

INSERT INTO tq_filtered_page_pruning_docs (id, tenant_id, embedding)
SELECT 256 + gs, 3, tq_vec64(cos(3.0 - (gs * 0.0012)), sin(3.0 - (gs * 0.0012)))
FROM generate_series(1, 128) AS gs;

CREATE INDEX tq_filtered_page_pruning_idx
	ON tq_filtered_page_pruning_docs
	USING turboquant (embedding tq_cosine_ops, tenant_id tq_int4_filter_ops)
	WITH (
		bits = 4,
		lists = 3,
		lanes = auto,
		transform = 'hadamard',
		normalized = true,
		router_samples = 192,
		router_iterations = 8,
		router_seed = 19
	);

CREATE TEMP TABLE tq_filtered_page_pruning_runs (
	label text PRIMARY KEY,
	approx_ids int4[],
	visited_page_count int4,
	visited_code_count int4,
	early_stop_count int4
);

SET turboquant.enable_summary_bounds = on;
INSERT INTO tq_filtered_page_pruning_runs (label, approx_ids)
SELECT 'summary_on', array_agg(id ORDER BY id)
FROM (
	SELECT id
	FROM tq_filtered_page_pruning_docs
	WHERE tenant_id = 2
	ORDER BY embedding <=> tq_vec64(1.0, 0.0)
	LIMIT 5
) ranked;
UPDATE tq_filtered_page_pruning_runs
SET visited_page_count = (tq_last_scan_stats()->>'visited_page_count')::int,
	visited_code_count = (tq_last_scan_stats()->>'visited_code_count')::int,
	early_stop_count = (tq_last_scan_stats()->>'early_stop_count')::int
WHERE label = 'summary_on';

SET turboquant.enable_summary_bounds = off;
INSERT INTO tq_filtered_page_pruning_runs (label, approx_ids)
SELECT 'summary_off', array_agg(id ORDER BY id)
FROM (
	SELECT id
	FROM tq_filtered_page_pruning_docs
	WHERE tenant_id = 2
	ORDER BY embedding <=> tq_vec64(1.0, 0.0)
	LIMIT 5
) ranked;
UPDATE tq_filtered_page_pruning_runs
SET visited_page_count = (tq_last_scan_stats()->>'visited_page_count')::int,
	visited_code_count = (tq_last_scan_stats()->>'visited_code_count')::int,
	early_stop_count = (tq_last_scan_stats()->>'early_stop_count')::int
WHERE label = 'summary_off';

WITH runs AS (
	SELECT
		max(approx_ids) FILTER (WHERE label = 'summary_on') AS summary_on_ids,
		max(approx_ids) FILTER (WHERE label = 'summary_off') AS summary_off_ids,
		max(visited_page_count) FILTER (WHERE label = 'summary_on') AS summary_on_visited_pages,
		max(visited_page_count) FILTER (WHERE label = 'summary_off') AS summary_off_visited_pages,
		max(visited_code_count) FILTER (WHERE label = 'summary_on') AS summary_on_visited_codes,
		max(visited_code_count) FILTER (WHERE label = 'summary_off') AS summary_off_visited_codes,
		max(early_stop_count) FILTER (WHERE label = 'summary_on') AS summary_on_early_stops
	FROM tq_filtered_page_pruning_runs
)
SELECT
	summary_on_ids = summary_off_ids AS candidate_ids_stable,
	summary_on_visited_pages < summary_off_visited_pages AS filtered_summary_skips_pages,
	summary_on_visited_codes <= summary_off_visited_codes AS filtered_summary_never_adds_code_work,
	summary_on_early_stops >= 0 AS filtered_summary_tracks_early_stop_counter
FROM runs;
