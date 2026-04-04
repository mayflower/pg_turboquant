SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE TABLE tq_capability_docs (
	id int4 PRIMARY KEY,
	category int4 NOT NULL,
	f1 int4 NOT NULL,
	f2 int4 NOT NULL,
	f3 int4 NOT NULL,
	f4 int4 NOT NULL,
	f5 int4 NOT NULL,
	f6 int4 NOT NULL,
	f7 int4 NOT NULL,
	payload text NOT NULL,
	embedding vector(4)
);

INSERT INTO tq_capability_docs
	(id, category, f1, f2, f3, f4, f5, f6, f7, payload, embedding)
VALUES
	(1, 1, 11, 12, 13, 14, 15, 16, 17, 'alpha', '[1,0,0,0]'),
	(2, 1, 21, 22, 23, 24, 25, 26, 27, 'beta', '[0.98,0.02,0,0]'),
	(3, 2, 31, 32, 33, 34, 35, 36, 37, 'gamma', '[0,1,0,0]');

CREATE INDEX tq_capability_embedding_idx
	ON tq_capability_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 0,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

VACUUM (ANALYZE) tq_capability_docs;

SELECT
	meta #>> '{capabilities,index_only_scan}' AS index_only_scan,
	meta #>> '{capabilities,vector_key_returnable}' AS vector_key_returnable,
	meta #>> '{capabilities,ordered_vector_key_index_only_scan}' AS ordered_vector_key_index_only_scan,
	meta #>> '{capabilities,multicolumn}' AS multicolumn,
	meta #>> '{capabilities,include_columns}' AS include_columns,
	meta #>> '{capabilities,bitmap_scan}' AS bitmap_scan,
	meta #>> '{operability,parallel_scan}' AS parallel_scan,
	meta #>> '{operability,parallel_vacuum}' AS parallel_vacuum,
	meta #>> '{operability,maintenance_work_mem_aware}' AS maintenance_work_mem_aware
FROM (SELECT tq_index_metadata('tq_capability_embedding_idx'::regclass) AS meta) AS s;

SET enable_seqscan = off;
SET enable_bitmapscan = off;

EXPLAIN (COSTS OFF)
SELECT embedding
FROM tq_capability_docs
ORDER BY embedding <=> '[1,0,0,0]'::vector(4)
LIMIT 2;

SELECT id, embedding
FROM tq_capability_docs
ORDER BY embedding <=> '[1,0,0,0]'::vector(4)
LIMIT 2;

CREATE INDEX tq_capability_multicol_idx
	ON tq_capability_docs
	USING turboquant (embedding tq_cosine_ops, category tq_int4_filter_ops)
	WITH (
		bits = 4,
		lists = 0,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

EXPLAIN (COSTS OFF)
SELECT id
FROM tq_capability_docs
WHERE category = 1
ORDER BY embedding <=> '[1,0,0,0]'::vector(4)
LIMIT 2;

SELECT id
FROM tq_capability_docs
WHERE category = 1
ORDER BY embedding <=> '[1,0,0,0]'::vector(4)
LIMIT 2;

CREATE INDEX tq_capability_too_wide_idx
	ON tq_capability_docs
	USING turboquant (
		embedding tq_cosine_ops,
		category tq_int4_filter_ops,
		id tq_int4_filter_ops,
		f1 tq_int4_filter_ops,
		f2 tq_int4_filter_ops,
		f3 tq_int4_filter_ops,
		f4 tq_int4_filter_ops,
		f5 tq_int4_filter_ops,
		f6 tq_int4_filter_ops,
		f7 tq_int4_filter_ops
	);

CREATE INDEX tq_capability_include_idx
	ON tq_capability_docs
	USING turboquant (embedding tq_cosine_ops)
	INCLUDE (payload);
