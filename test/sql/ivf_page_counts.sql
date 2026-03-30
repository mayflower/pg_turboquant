SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_turboquant;
DROP TABLE IF EXISTS tq_ivf_page_count_docs CASCADE;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
RESET client_min_messages;

CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE TABLE tq_ivf_page_count_docs (
	id int4 PRIMARY KEY,
	embedding vector(4)
);

INSERT INTO tq_ivf_page_count_docs (id, embedding)
SELECT
	i,
	CASE
		WHEN i % 2 = 0 THEN '[1,0,0,0]'::vector(4)
		ELSE '[0.999,0.001,0,0]'::vector(4)
	END
FROM generate_series(1, 3000) AS i;

CREATE INDEX tq_ivf_page_count_idx
	ON tq_ivf_page_count_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 1,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

SELECT
	jsonb_array_length(meta->'lists') = 1 AS one_list_row,
	(meta->>'batch_page_count')::int = (meta #>> '{lists,0,batch_page_count}')::int AS total_matches_list_page_count,
	(meta #>> '{lists,0,batch_page_count}')::int > 1 AS build_persisted_multiple_pages
FROM (SELECT tq_index_metadata('tq_ivf_page_count_idx'::regclass) AS meta) AS s;

CREATE TEMP TABLE tq_ivf_page_count_baseline AS
SELECT
	(meta #>> '{lists,0,batch_page_count}')::int AS list_page_count,
	meta->'lists' AS list_rows
FROM (SELECT tq_index_metadata('tq_ivf_page_count_idx'::regclass) AS meta) AS s;

INSERT INTO tq_ivf_page_count_docs (id, embedding)
SELECT
	i,
	CASE
		WHEN i % 2 = 0 THEN '[1,0,0,0]'::vector(4)
		ELSE '[0.999,0.001,0,0]'::vector(4)
	END
FROM generate_series(3001, 9000) AS i;

SELECT
	(meta #>> '{lists,0,batch_page_count}')::int > baseline.list_page_count AS insert_updates_list_page_count,
	(meta->>'batch_page_count')::int = (meta #>> '{lists,0,batch_page_count}')::int AS total_stays_in_sync
FROM
	(SELECT tq_index_metadata('tq_ivf_page_count_idx'::regclass) AS meta) AS s,
	tq_ivf_page_count_baseline AS baseline;

CREATE TEMP TABLE tq_ivf_page_count_after_insert AS
SELECT tq_index_metadata('tq_ivf_page_count_idx'::regclass)->'lists' AS list_rows;

VACUUM tq_ivf_page_count_docs;

SELECT
	meta->'lists' = after_insert.list_rows AS maintenance_keeps_page_counts_stable,
	(meta->>'batch_page_count')::int = (meta #>> '{lists,0,batch_page_count}')::int AS total_matches_after_maintenance
FROM
	(SELECT tq_index_metadata('tq_ivf_page_count_idx'::regclass) AS meta) AS s,
	tq_ivf_page_count_after_insert AS after_insert;
