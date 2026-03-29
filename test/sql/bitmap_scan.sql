SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE TABLE tq_bitmap_docs (
	id int4 PRIMARY KEY,
	category int4 NOT NULL,
	embedding vector(4)
);

INSERT INTO tq_bitmap_docs (id, category, embedding) VALUES
	(1, 1, '[1,0,0,0]'),
	(2, 1, '[0.98,0.02,0,0]'),
	(3, 1, '[0,1,0,0]'),
	(4, 2, '[1,0,0,0]'),
	(5, 2, '[0,1,0,0]');

CREATE INDEX tq_bitmap_docs_embedding_tq_idx
	ON tq_bitmap_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 0,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

CREATE INDEX tq_bitmap_docs_category_idx
	ON tq_bitmap_docs (category);

ANALYZE tq_bitmap_docs;

SET enable_seqscan = off;
SET enable_indexscan = off;
SET enable_tidscan = off;

EXPLAIN (COSTS OFF)
SELECT id
FROM tq_bitmap_docs
WHERE embedding <?=> tq_bitmap_cosine_filter(
	'[1,0,0,0]'::vector(4),
	0.20
)
ORDER BY id;

SELECT id
FROM tq_bitmap_docs
WHERE embedding <?=> tq_bitmap_cosine_filter(
	'[1,0,0,0]'::vector(4),
	0.20
)
ORDER BY id;

EXPLAIN (COSTS OFF)
SELECT id
FROM tq_bitmap_docs
WHERE category = 1
  AND embedding <?=> tq_bitmap_cosine_filter(
	'[1,0,0,0]'::vector(4),
	0.20
  )
ORDER BY id;

SELECT id
FROM tq_bitmap_docs
WHERE category = 1
  AND embedding <?=> tq_bitmap_cosine_filter(
	'[1,0,0,0]'::vector(4),
	0.20
  )
ORDER BY id;
