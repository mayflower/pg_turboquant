DO $$
BEGIN
	IF to_regclass('public.tq_ivf_training_docs') IS NOT NULL THEN
		EXECUTE 'DROP TABLE public.tq_ivf_training_docs CASCADE';
	END IF;
	IF to_regclass('public.tq_ivf_sparse_docs') IS NOT NULL THEN
		EXECUTE 'DROP TABLE public.tq_ivf_sparse_docs CASCADE';
	END IF;
	IF to_regclass('public.tq_transform_contract_docs') IS NOT NULL THEN
		EXECUTE 'DROP TABLE public.tq_transform_contract_docs CASCADE';
	END IF;
END;
$$;

SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_turboquant;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
RESET client_min_messages;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE TABLE tq_admin_docs (
	id int4 PRIMARY KEY,
	embedding vector(4)
);

INSERT INTO tq_admin_docs (id, embedding) VALUES
	(1, '[1,0,0,0]'),
	(2, '[0.9,0.1,0,0]'),
	(3, '[0,1,0,0]'),
	(4, '[0,0,1,0]');

CREATE INDEX tq_admin_flat_idx
	ON tq_admin_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 0,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

SELECT
	(meta->>'format_version')::int AS format_version,
	meta->>'metric' AS metric,
	meta->>'opclass' AS opclass,
	(meta->>'list_count')::int AS list_count,
	(meta->>'live_count')::int AS live_count,
	(meta->>'dead_count')::int AS dead_count,
	(meta->>'heap_live_rows')::int AS heap_live_rows,
	meta #>> '{transform,kind}' AS transform_kind,
	(meta #>> '{transform,version}')::int AS transform_version
FROM (SELECT tq_index_metadata('tq_admin_flat_idx'::regclass) AS meta) AS s;

INSERT INTO tq_admin_docs (id, embedding) VALUES
	(5, '[0,0,0,1]');

SELECT
	(meta->>'live_count')::int AS live_count,
	(meta->>'dead_count')::int AS dead_count,
	(meta->>'heap_live_rows')::int AS heap_live_rows
FROM (SELECT tq_index_metadata('tq_admin_flat_idx'::regclass) AS meta) AS s;

DELETE FROM tq_admin_docs WHERE id = 5;

SELECT
	(meta->>'live_count')::int AS live_count,
	(meta->>'dead_count')::int AS dead_count,
	(meta->>'heap_live_rows')::int AS heap_live_rows
FROM (SELECT tq_index_metadata('tq_admin_flat_idx'::regclass) AS meta) AS s;

VACUUM tq_admin_docs;

SELECT
	(meta->>'live_count')::int AS live_count,
	(meta->>'dead_count')::int AS dead_count,
	(meta->>'heap_live_rows')::int AS heap_live_rows,
	(meta->>'reclaimable_pages')::int AS reclaimable_pages
FROM (SELECT tq_index_metadata('tq_admin_flat_idx'::regclass) AS meta) AS s;

REINDEX INDEX tq_admin_flat_idx;

SELECT
	(meta->>'live_count')::int AS live_count,
	(meta->>'dead_count')::int AS dead_count,
	(meta->>'heap_live_rows')::int AS heap_live_rows
FROM (SELECT tq_index_metadata('tq_admin_flat_idx'::regclass) AS meta) AS s;

CREATE INDEX tq_admin_ivf_idx
	ON tq_admin_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 2,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

SELECT
	(meta->>'list_count')::int AS list_count,
	meta #>> '{router,algorithm}' AS router_algorithm,
	(meta #>> '{router,seed}')::int AS router_seed,
	(meta #>> '{router,trained_vector_count}')::int AS router_trained_vector_count,
	(meta #>> '{list_distribution,min_live_count}')::int AS min_live_count,
	(meta #>> '{list_distribution,max_live_count}')::int AS max_live_count,
	(meta #>> '{list_distribution,avg_live_count}')::numeric(10,2) AS avg_live_count,
	jsonb_array_length(meta->'lists') AS list_rows
FROM (SELECT tq_index_metadata('tq_admin_ivf_idx'::regclass) AS meta) AS s;

WITH meta AS (
	SELECT tq_index_metadata('tq_admin_ivf_idx'::regclass) AS meta
)
SELECT
	sum((value->>'live_count')::int) AS summed_live_count,
	sum((value->>'dead_count')::int) AS summed_dead_count,
	sum((value->>'batch_page_count')::int) AS summed_batch_pages
FROM meta, LATERAL jsonb_array_elements(meta.meta->'lists') AS value;
