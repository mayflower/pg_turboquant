SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE TABLE tq_planner_small_docs (
	id int4 PRIMARY KEY,
	embedding vector(4)
);

INSERT INTO tq_planner_small_docs (id, embedding)
SELECT i,
	   format('[%s,%s,0,0]',
			  round(cos(i::float8 / 4.0)::numeric, 6),
			  round(sin(i::float8 / 4.0)::numeric, 6))::vector(4)
FROM generate_series(1, 32) AS g(i);

CREATE INDEX tq_planner_small_flat_idx
	ON tq_planner_small_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = true);

ANALYZE tq_planner_small_docs;

CREATE TABLE tq_planner_ivf_docs (
	id int4 PRIMARY KEY,
	embedding vector(4)
);

INSERT INTO tq_planner_ivf_docs (id, embedding)
SELECT i,
	   format('[%s,%s,%s,%s]',
			  round(cos(i::float8 / 16.0)::numeric, 6),
			  round(sin(i::float8 / 16.0)::numeric, 6),
			  round(cos(i::float8 / 11.0)::numeric, 6),
			  round(sin(i::float8 / 11.0)::numeric, 6))::vector(4)
FROM generate_series(1, 8192) AS g(i);

CREATE INDEX tq_planner_ivf_idx
	ON tq_planner_ivf_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 16, lanes = auto, transform = 'hadamard', normalized = true);

ANALYZE tq_planner_ivf_docs;

SET enable_seqscan = off;
SET enable_bitmapscan = off;

SET turboquant.probes = 1;
SET turboquant.oversample_factor = 4;

EXPLAIN (COSTS OFF)
SELECT id
FROM tq_planner_small_docs
ORDER BY embedding <=> '[1,0,0,0]'
LIMIT 10;

EXPLAIN
SELECT id
FROM tq_planner_ivf_docs
ORDER BY embedding <=> '[1,0,0,0]'
LIMIT 10;

SET turboquant.probes = 16;
SET turboquant.max_visited_codes = 0;
SET turboquant.max_visited_pages = 0;

EXPLAIN
SELECT id
FROM tq_planner_ivf_docs
ORDER BY embedding <=> '[1,0,0,0]'
LIMIT 10;

SET turboquant.max_visited_codes = 256;

EXPLAIN
SELECT id
FROM tq_planner_ivf_docs
ORDER BY embedding <=> '[1,0,0,0]'
LIMIT 10;

SET turboquant.max_visited_codes = 0;
SET turboquant.max_visited_pages = 32;

EXPLAIN
SELECT id
FROM tq_planner_ivf_docs
ORDER BY embedding <=> '[1,0,0,0]'
LIMIT 10;

SET turboquant.max_visited_pages = 0;

EXPLAIN
SELECT id
FROM tq_planner_ivf_docs
WHERE id <= 8
ORDER BY embedding <=> '[1,0,0,0]'
LIMIT 5;
