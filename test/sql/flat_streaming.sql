SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE TABLE tq_flat_stream_docs (
	id int4 PRIMARY KEY,
	embedding vector(4)
);

INSERT INTO tq_flat_stream_docs (id, embedding) VALUES
	(1, '[1,0,0,0]'),
	(2, '[0.92387953,0.38268343,0,0]'),
	(3, '[0.70710678,0.70710678,0,0]'),
	(4, '[0.38268343,0.92387953,0,0]'),
	(5, '[0,1,0,0]'),
	(6, '[-0.38268343,0.92387953,0,0]'),
	(7, '[-0.70710678,0.70710678,0,0]'),
	(8, '[-1,0,0,0]');

CREATE INDEX tq_flat_stream_docs_embedding_tq_idx
	ON tq_flat_stream_docs
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
SET turboquant.probes = 1;
SET turboquant.oversample_factor = 4;

SELECT id
FROM tq_flat_stream_docs
ORDER BY embedding <=> '[1,0,0,0]'
LIMIT 4;
