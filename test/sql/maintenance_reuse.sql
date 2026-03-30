SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_turboquant;
DROP TABLE IF EXISTS tq_maintenance_reuse_docs CASCADE;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
RESET client_min_messages;

CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE TABLE tq_maintenance_reuse_docs (
	id int4 PRIMARY KEY,
	embedding vector(4)
);

INSERT INTO tq_maintenance_reuse_docs (id, embedding)
SELECT
	i,
	CASE
		WHEN i <= 750 THEN format('[1,%s,0,0]', (i % 10) * 0.01)
		ELSE format('[0,1,%s,0]', (i % 10) * 0.01)
	END::vector(4)
FROM generate_series(1, 1500) AS i;

CREATE INDEX tq_maintenance_reuse_idx
	ON tq_maintenance_reuse_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 2,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

CREATE TEMP TABLE tq_maintenance_reuse_baseline AS
SELECT
	(meta->>'batch_page_count')::int AS batch_page_count,
	pg_relation_size('tq_maintenance_reuse_idx'::regclass) AS relation_size,
	current_setting('block_size')::int AS block_size
FROM (SELECT tq_index_metadata('tq_maintenance_reuse_idx'::regclass) AS meta) AS s;

DELETE FROM tq_maintenance_reuse_docs
WHERE id > 300;

VACUUM tq_maintenance_reuse_docs;

INSERT INTO tq_maintenance_reuse_docs (id, embedding) VALUES
	(2001, '[0,0,1,0]');

INSERT INTO tq_maintenance_reuse_docs (id, embedding)
SELECT
	i,
	CASE
		WHEN i % 2 = 0 THEN format('[1,0,%s,0]', (i % 10) * 0.01)
		ELSE format('[0,1,%s,0]', (i % 10) * 0.01)
	END::vector(4)
FROM generate_series(2002, 3200) AS i;

SELECT
	(meta->>'live_count')::int AS live_count,
	(meta->>'dead_count')::int AS dead_count,
	(meta->>'batch_page_count')::int <= baseline.batch_page_count + 1 AS reused_without_runaway_growth,
	pg_relation_size('tq_maintenance_reuse_idx'::regclass) <= baseline.relation_size + (baseline.block_size * 3) AS size_without_runaway_growth
FROM
	(SELECT tq_index_metadata('tq_maintenance_reuse_idx'::regclass) AS meta) AS s,
	tq_maintenance_reuse_baseline AS baseline;

SET enable_seqscan = off;
SET enable_bitmapscan = off;

SELECT id
FROM tq_maintenance_reuse_docs
ORDER BY embedding <=> '[0,0,1,0]'::vector(4), id
LIMIT 1;

SELECT
	(meta #>> '{list_distribution,min_live_count}')::int >= 1 AS lists_remain_searchable,
	jsonb_array_length(meta->'lists') = 2 AS list_count_is_stable
FROM (SELECT tq_index_metadata('tq_maintenance_reuse_idx'::regclass) AS meta) AS s;
