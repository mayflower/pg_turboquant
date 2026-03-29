SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

SET enable_seqscan = off;
SET enable_bitmapscan = off;
SET turboquant.probes = 3;
SET turboquant.oversample_factor = 1;

CREATE FUNCTION tq_vec64(a float8, b float8)
RETURNS vector
LANGUAGE SQL
IMMUTABLE
AS $$
	SELECT ('[' || a::text || ',' || b::text || repeat(',0', 62) || ']')::vector(64);
$$;

CREATE TABLE tq_adaptive_probing_docs (
	id int4 PRIMARY KEY,
	embedding vector(64)
);

INSERT INTO tq_adaptive_probing_docs (id, embedding)
SELECT gs, tq_vec64(cos(gs * 0.0005), sin(gs * 0.0005))
FROM generate_series(1, 96) AS gs;

INSERT INTO tq_adaptive_probing_docs (id, embedding)
SELECT 96 + gs, tq_vec64(cos(0.35 + (gs * 0.002)), sin(0.35 + (gs * 0.002)))
FROM generate_series(1, 16) AS gs;

INSERT INTO tq_adaptive_probing_docs (id, embedding)
SELECT 112 + gs, tq_vec64(cos(2.8 + (gs * 0.002)), sin(2.8 + (gs * 0.002)))
FROM generate_series(1, 16) AS gs;

CREATE INDEX tq_adaptive_probing_idx
	ON tq_adaptive_probing_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 3,
		lanes = auto,
		transform = 'hadamard',
		normalized = true,
		router_samples = 128,
		router_iterations = 8,
		router_seed = 13
	);

SET turboquant.max_visited_codes = 0;
SET turboquant.max_visited_pages = 0;

SELECT array_agg(id ORDER BY id) AS adaptive_off_ids
FROM (
	SELECT id
	FROM tq_adaptive_probing_docs
	ORDER BY embedding <=> tq_vec64(1.0, 0.0)
	LIMIT 3
) ranked;

WITH stats AS (
	SELECT tq_last_scan_stats() AS stats
)
SELECT
	(stats->>'nominal_probe_count')::int AS nominal_probe_count,
	(stats->>'effective_probe_count')::int AS effective_probe_count,
	(stats->>'selected_live_count')::int AS selected_live_count
FROM stats;

SET turboquant.max_visited_codes = 104;
SET turboquant.max_visited_pages = 0;

SELECT array_agg(id ORDER BY id) AS adaptive_on_ids
FROM (
	SELECT id
	FROM tq_adaptive_probing_docs
	ORDER BY embedding <=> tq_vec64(1.0, 0.0)
	LIMIT 3
) ranked;

WITH stats AS (
	SELECT tq_last_scan_stats() AS stats
)
SELECT
	(stats->>'effective_probe_count')::int < (stats->>'nominal_probe_count')::int AS effective_probes_reduced,
	(stats->>'selected_live_count')::int < 128 AS selected_live_reduced,
	(stats->>'max_visited_codes')::int = 104 AS code_budget_visible,
	(stats->>'visited_code_count')::int <= 104 AS visited_codes_within_budget
FROM stats;

SELECT
	probes > 0 AS probes_positive,
	oversample_factor > 0 AS oversample_positive,
	max_visited_codes >= 24 AS recommended_codes_positive,
	max_visited_pages = 0 AS recommended_pages_disabled
FROM tq_recommended_query_knobs(24, 10);
