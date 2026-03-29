DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE TABLE tq_flat_docs (
	id int4 PRIMARY KEY,
	embedding vector(4)
);

INSERT INTO tq_flat_docs (id, embedding) VALUES
	(1, '[1,0,0,0]'),
	(2, '[0,1,0,0]'),
	(3, '[0,0,1,0]'),
	(4, '[0.70710678,0.70710678,0,0]');

CREATE INDEX tq_flat_docs_embedding_tq_idx
	ON tq_flat_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 0,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

SET enable_seqscan = off;
SET enable_bitmapscan = off;

EXPLAIN (COSTS OFF)
SELECT id
FROM tq_flat_docs
ORDER BY embedding <=> '[1,0,0,0]'
LIMIT 1;

SELECT id
FROM tq_flat_docs
ORDER BY embedding <=> '[1,0,0,0]'
LIMIT 1;
