DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE TABLE tq_ivf_docs (
	id int4 PRIMARY KEY,
	embedding vector(4)
);

INSERT INTO tq_ivf_docs (id, embedding) VALUES
	(1, '[1,0,0,0]'),
	(2, '[0,1,0,0]'),
	(3, '[0.98,0.02,0,0]'),
	(4, '[0.02,0.98,0,0]'),
	(5, '[0.96,0.04,0,0]'),
	(6, '[0.04,0.96,0,0]');

CREATE INDEX tq_ivf_docs_embedding_tq_idx
	ON tq_ivf_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 2,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

SET enable_seqscan = off;
SET enable_bitmapscan = off;

BEGIN;
SET LOCAL turboquant.probes = 1;
EXPLAIN (COSTS OFF)
SELECT id
FROM tq_ivf_docs
ORDER BY embedding <=> '[1,0,0,0]'
LIMIT 1;

SELECT id
FROM tq_ivf_docs
ORDER BY embedding <=> '[1,0,0,0]'
LIMIT 1;
ROLLBACK;

BEGIN;
SET LOCAL turboquant.probes = 2;
SELECT id
FROM tq_ivf_docs
ORDER BY embedding <=> '[0,1,0,0]'
LIMIT 1;
ROLLBACK;
