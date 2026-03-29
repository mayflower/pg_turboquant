SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

SET enable_seqscan = off;
SET enable_bitmapscan = off;
SET turboquant.probes = 2;
SET turboquant.oversample_factor = 2;

CREATE TABLE tq_scan_stats_docs (
	id int4 PRIMARY KEY,
	embedding vector(2)
);

INSERT INTO tq_scan_stats_docs (id, embedding) VALUES
	(1, '[1.0,0.0]'),
	(2, '[0.98,0.02]'),
	(3, '[0.0,1.0]'),
	(4, '[0.02,0.98]'),
	(5, '[-1.0,0.0]'),
	(6, '[-0.98,0.02]'),
	(7, '[0.0,-1.0]'),
	(8, '[0.02,-0.98]');

CREATE INDEX tq_scan_stats_idx
	ON tq_scan_stats_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 4,
		lanes = auto,
		transform = 'hadamard',
		normalized = true,
		router_samples = 8,
		router_iterations = 6,
		router_seed = 7
	);

SELECT array_agg(id ORDER BY id) AS approx_ids
FROM (
	SELECT id
	FROM tq_scan_stats_docs
	ORDER BY embedding <=> '[1.0,0.0]'
	LIMIT 2
) ranked;

WITH stats AS (
	SELECT tq_last_scan_stats() AS stats
)
SELECT
	stats->>'mode' AS mode,
	stats->>'score_mode' AS score_mode,
	(stats->>'visited_code_count')::int > 0 AS visited_codes_positive,
	(stats->>'visited_page_count')::int > 0 AS visited_pages_positive,
	(stats->>'selected_list_count')::int <= 2 AS selected_lists_within_probes,
	(stats->>'selected_live_count')::int > 0 AS selected_live_positive,
	(stats->>'candidate_heap_capacity')::int >= (stats->>'candidate_heap_count')::int AS heap_bounded
FROM stats;
