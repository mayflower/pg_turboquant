DO $$
BEGIN
	IF to_regclass('public.tq_admin_docs') IS NOT NULL THEN
		EXECUTE 'DROP TABLE public.tq_admin_docs CASCADE';
	END IF;
END;
$$;

SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_turboquant;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
RESET client_min_messages;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

SET enable_seqscan = off;
SET enable_bitmapscan = off;

CREATE TABLE tq_query_helper_docs (
	id int4 PRIMARY KEY,
	embedding vector(4)
);

INSERT INTO tq_query_helper_docs (id, embedding) VALUES
	(1, '[1,0,0,0]'),
	(2, '[0.9,0.1,0,0]'),
	(3, '[0.6,0.8,0,0]'),
	(4, '[0,1,0,0]');

CREATE INDEX tq_query_helper_docs_idx
	ON tq_query_helper_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 0,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

SELECT * FROM tq_recommended_query_knobs(24, 10);
SELECT * FROM tq_resolve_query_knobs(24, 10, 6, 4);

SELECT *
FROM tq_approx_candidates(
	'tq_query_helper_docs'::regclass,
	'id',
	'embedding',
	'[1,0,0,0]'::vector(4),
	'cosine',
	3,
	1,
	4
);

WITH approx AS MATERIALIZED (
	SELECT
		id::text AS candidate_id,
		row_number() OVER (ORDER BY embedding <=> '[1,0,0,0]'::vector(4), id) AS approximate_rank,
		round((embedding <=> '[1,0,0,0]'::vector(4))::numeric, 6)::float8 AS approximate_distance
	FROM tq_query_helper_docs
	ORDER BY embedding <=> '[1,0,0,0]'::vector(4), id
	LIMIT 3
)
SELECT candidate_id, approximate_rank, approximate_distance
FROM approx
ORDER BY approximate_rank;

SELECT *
FROM tq_rerank_candidates(
	'tq_query_helper_docs'::regclass,
	'id',
	'embedding',
	'[1,0,0,0]'::vector(4),
	'cosine',
	3,
	2,
	1,
	4
);

WITH approx AS MATERIALIZED (
	SELECT
		id::text AS candidate_id,
		id AS candidate_key,
		embedding AS candidate_embedding,
		row_number() OVER (ORDER BY embedding <=> '[1,0,0,0]'::vector(4), id) AS approximate_rank,
		round((embedding <=> '[1,0,0,0]'::vector(4))::numeric, 6)::float8 AS approximate_distance
	FROM tq_query_helper_docs
	ORDER BY embedding <=> '[1,0,0,0]'::vector(4), id
	LIMIT 3
), reranked AS (
	SELECT
		candidate_id,
		approximate_rank,
		approximate_distance,
		row_number() OVER (ORDER BY candidate_embedding <=> '[1,0,0,0]'::vector(4), candidate_key) AS exact_rank,
		round((candidate_embedding <=> '[1,0,0,0]'::vector(4))::numeric, 6)::float8 AS exact_distance
	FROM approx
)
SELECT candidate_id, approximate_rank, approximate_distance, exact_rank, exact_distance
FROM reranked
WHERE exact_rank <= 2
ORDER BY exact_rank;

SELECT * FROM tq_recommended_query_knobs(0, 1);
SELECT * FROM tq_recommended_query_knobs(4, 8);

SELECT *
FROM tq_approx_candidates(
	'tq_query_helper_docs'::regclass,
	'id',
	'embedding',
	'[1,0,0,0]'::vector(4),
	'angular',
	3
);

SELECT *
FROM tq_rerank_candidates(
	'tq_query_helper_docs'::regclass,
	'id',
	'embedding',
	'[1,0,0,0]'::vector(4),
	'cosine',
	2,
	3
);
