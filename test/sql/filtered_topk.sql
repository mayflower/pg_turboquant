SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE TABLE tq_filtered_docs (
	id int4 PRIMARY KEY,
	category int4 NOT NULL,
	payload text NOT NULL,
	embedding vector(4)
);

INSERT INTO tq_filtered_docs (id, category, payload, embedding) VALUES
	(1, 1, 'tenant-a-1', '[1,0,0,0]'),
	(2, 1, 'tenant-a-2', '[0.98,0.02,0,0]'),
	(3, 1, 'tenant-a-3', '[0,1,0,0]'),
	(4, 2, 'tenant-b-1', '[1,0,0,0]'),
	(5, 2, 'tenant-b-2', '[0.97,0.03,0,0]'),
	(6, 2, 'tenant-b-3', '[0,1,0,0]');

CREATE INDEX tq_filtered_docs_embedding_category_idx
	ON tq_filtered_docs
	USING turboquant (embedding tq_cosine_ops, category tq_int4_filter_ops)
	WITH (
		bits = 4,
		lists = 0,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

VACUUM (ANALYZE) tq_filtered_docs;

SET enable_seqscan = off;
SET enable_bitmapscan = off;

EXPLAIN (COSTS OFF)
SELECT id, category
FROM tq_filtered_docs
WHERE category = 1
ORDER BY embedding <=> '[1,0,0,0]'::vector(4)
LIMIT 2;

SELECT id, category
FROM tq_filtered_docs
WHERE category = 1
ORDER BY embedding <=> '[1,0,0,0]'::vector(4)
LIMIT 2;

SELECT id, category
FROM tq_filtered_docs
WHERE category = 2
ORDER BY embedding <=> '[1,0,0,0]'::vector(4)
LIMIT 2;

SELECT id
FROM tq_filtered_docs
ORDER BY embedding <=> '[1,0,0,0]'::vector(4)
LIMIT 3;
