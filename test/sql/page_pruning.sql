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

CREATE TABLE tq_page_pruning_docs (
	id int4 PRIMARY KEY,
	embedding vector(64)
);

INSERT INTO tq_page_pruning_docs (id, embedding)
SELECT gs, tq_vec64(cos(gs * 0.001), sin(gs * 0.001))
FROM generate_series(1, 48) AS gs;

INSERT INTO tq_page_pruning_docs (id, embedding)
SELECT 48 + gs, tq_vec64(cos(1.2 + (gs * 0.001)), sin(1.2 + (gs * 0.001)))
FROM generate_series(1, 48) AS gs;

INSERT INTO tq_page_pruning_docs (id, embedding)
SELECT 96 + gs, tq_vec64(cos(3.0 - (gs * 0.001)), sin(3.0 - (gs * 0.001)))
FROM generate_series(1, 48) AS gs;

CREATE INDEX tq_page_pruning_idx
	ON tq_page_pruning_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 3,
		lanes = auto,
		transform = 'hadamard',
		normalized = true,
		router_samples = 144,
		router_iterations = 8,
		router_seed = 11
	);

SELECT array_agg(id ORDER BY id) AS approx_ids
FROM (
	SELECT id
	FROM tq_page_pruning_docs
	ORDER BY embedding <=> tq_vec64(1.0, 0.0)
	LIMIT 3
) ranked;

WITH stats AS (
	SELECT tq_last_scan_stats() AS stats
)
SELECT
	(stats->>'visited_page_count')::int < 9 AS visited_pages_reduced,
	(stats->>'page_prune_count')::int > 0 AS page_prunes_positive,
	(stats->>'early_stop_count')::int > 0 AS early_stops_positive,
	(stats->>'visited_code_count')::int < (stats->>'selected_live_count')::int AS visited_codes_below_selected_live
FROM stats;
