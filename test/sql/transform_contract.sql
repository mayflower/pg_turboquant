CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_turboquant;
CREATE EXTENSION IF NOT EXISTS pg_turboquant_test_support;

CREATE TABLE tq_transform_contract_docs (
	id int4 PRIMARY KEY,
	embedding vector(5)
);

INSERT INTO tq_transform_contract_docs (id, embedding) VALUES
	(1, '[1,0,0,0,0]'),
	(2, '[0,1,0,0,0]'),
	(3, '[0,0,1,0,0]'),
	(4, '[0,0,0,1,0]'),
	(5, '[0,0,0,0,1]');

CREATE INDEX tq_transform_contract_idx
ON tq_transform_contract_docs
USING turboquant (embedding tq_cosine_ops)
WITH (
	bits = 4,
	lists = 0,
	lanes = auto,
	transform = 'hadamard',
	normalized = true
);

SELECT tq_debug_transform_metadata('tq_transform_contract_idx'::regclass);

SET enable_seqscan = off;
SET enable_bitmapscan = off;

SELECT id
FROM tq_transform_contract_docs
ORDER BY embedding <=> '[1,0,0,0,0]'
LIMIT 3;

REINDEX INDEX tq_transform_contract_idx;

SELECT tq_debug_transform_metadata('tq_transform_contract_idx'::regclass);

SELECT id
FROM tq_transform_contract_docs
ORDER BY embedding <=> '[1,0,0,0,0]'
LIMIT 3;

DROP EXTENSION pg_turboquant_test_support;
