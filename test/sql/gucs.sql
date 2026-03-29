SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

SHOW turboquant.probes;
SHOW turboquant.oversample_factor;

BEGIN;
SET LOCAL turboquant.probes = 3;
SHOW turboquant.probes;
SET LOCAL turboquant.oversample_factor = 5;
SHOW turboquant.oversample_factor;
ROLLBACK;

SHOW turboquant.probes;
SHOW turboquant.oversample_factor;

SET turboquant.probes = 0;
SET turboquant.oversample_factor = 0;

CREATE TABLE tq_guc_docs (
	id int4 PRIMARY KEY,
	embedding vector(4)
);

INSERT INTO tq_guc_docs (id, embedding) VALUES
	(1, '[1,0,0,0]'),
	(2, '[0,1,0,0]'),
	(3, '[0,0,1,0]'),
	(4, '[0.70710678,0.70710678,0,0]');

CREATE INDEX tq_guc_docs_embedding_tq_idx
	ON tq_guc_docs
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

BEGIN;
SET LOCAL turboquant.probes = 2;
SET LOCAL turboquant.oversample_factor = 3;
EXPLAIN (COSTS OFF)
SELECT id
FROM tq_guc_docs
ORDER BY embedding <=> '[1,0,0,0]'
LIMIT 1;
ROLLBACK;
