SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE TABLE tq_ordered_ios_docs (
	id int4 PRIMARY KEY,
	category int4 NOT NULL,
	embedding vector(4)
);

INSERT INTO tq_ordered_ios_docs (id, category, embedding) VALUES
	(1, 1, '[1,0,0,0]'),
	(2, 1, '[0.99,0.01,0,0]'),
	(3, 2, '[0,1,0,0]');

CREATE INDEX tq_ordered_ios_docs_idx
	ON tq_ordered_ios_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 0,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

VACUUM (FREEZE, ANALYZE) tq_ordered_ios_docs;

SET enable_seqscan = off;
SET enable_bitmapscan = off;

EXPLAIN (ANALYZE, BUFFERS, COSTS OFF, TIMING OFF, SUMMARY OFF)
SELECT embedding
FROM tq_ordered_ios_docs
ORDER BY embedding <=> '[1,0,0,0]'::vector(4)
LIMIT 2;

CREATE INDEX tq_ordered_ios_docs_filtered_idx
	ON tq_ordered_ios_docs
	USING turboquant (
		embedding tq_cosine_ops,
		category tq_int4_filter_ops
	)
	WITH (
		bits = 4,
		lists = 0,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

EXPLAIN (ANALYZE, BUFFERS, COSTS OFF, TIMING OFF, SUMMARY OFF)
SELECT embedding
FROM tq_ordered_ios_docs
WHERE category = 1
ORDER BY embedding <=> '[1,0,0,0]'::vector(4)
LIMIT 2;
