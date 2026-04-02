SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;
CREATE EXTENSION pg_turboquant_test_support;

SET enable_seqscan = off;
SET enable_bitmapscan = off;
SET turboquant.probes = 2;

CREATE TABLE tq_ivf_training_docs (
	id int4 PRIMARY KEY,
	embedding vector(2)
);

INSERT INTO tq_ivf_training_docs (id, embedding) VALUES
	(1, '[1.0,0.0]'),
	(2, '[0.98,0.02]'),
	(3, '[0.0,1.0]'),
	(4, '[0.02,0.98]'),
	(5, '[-1.0,0.0]'),
	(6, '[-0.98,0.02]'),
	(7, '[0.0,-1.0]'),
	(8, '[0.02,-0.98]');

CREATE INDEX tq_ivf_training_idx
	ON tq_ivf_training_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 4,
		lanes = auto,
		transform = 'hadamard',
		normalized = true,
		router_samples = 8,
		router_iterations = 6,
		router_restarts = 3,
		router_seed = 7
	);

SELECT tq_debug_router_metadata('tq_ivf_training_idx'::regclass);

SELECT
	(meta->'router'->>'restart_count')::int AS router_restarts,
	(meta->'router'->>'selected_restart')::int AS selected_restart,
	round((meta->'router'->>'balance_penalty')::numeric, 4) AS balance_penalty,
	round((meta->'list_distribution'->>'avg_list_size')::numeric, 2) AS avg_list_size,
	(meta->'list_distribution'->>'max_list_size')::int AS max_list_size,
	(meta->'list_distribution'->>'p95_list_size')::int AS p95_list_size,
	round((meta->'list_distribution'->>'coeff_var')::numeric, 4) AS coeff_var
FROM (SELECT tq_index_metadata('tq_ivf_training_idx'::regclass) AS meta) AS s;

SELECT array_agg(id) AS approx_clustered_ids
FROM (
	SELECT id
	FROM tq_ivf_training_docs
	ORDER BY embedding <=> '[1.0,0.0]'
	LIMIT 4
) ranked;

CREATE TABLE tq_ivf_sparse_docs (
	id int4 PRIMARY KEY,
	embedding vector(2)
);

INSERT INTO tq_ivf_sparse_docs (id, embedding) VALUES
	(1, '[1.0,0.0]'),
	(2, '[0.0,1.0]');

CREATE INDEX tq_ivf_sparse_idx
	ON tq_ivf_sparse_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 4,
		lanes = auto,
		transform = 'hadamard',
		normalized = true,
		router_samples = 2,
		router_iterations = 4,
		router_restarts = 3,
		router_seed = 11
	);

SELECT tq_debug_router_metadata('tq_ivf_sparse_idx'::regclass);

SELECT
	(meta->'router'->>'restart_count')::int AS router_restarts,
	(meta->'router'->>'selected_restart')::int AS selected_restart,
	round((meta->'router'->>'balance_penalty')::numeric, 4) AS balance_penalty,
	round((meta->'list_distribution'->>'avg_list_size')::numeric, 2) AS avg_list_size,
	(meta->'list_distribution'->>'max_list_size')::int AS max_list_size,
	(meta->'list_distribution'->>'p95_list_size')::int AS p95_list_size,
	round((meta->'list_distribution'->>'coeff_var')::numeric, 4) AS coeff_var
FROM (SELECT tq_index_metadata('tq_ivf_sparse_idx'::regclass) AS meta) AS s;

SELECT array_agg(id) AS sparse_ids
FROM (
	SELECT id
	FROM tq_ivf_sparse_docs
	ORDER BY embedding <=> '[1.0,0.0]'
	LIMIT 2
) ranked;

DROP EXTENSION pg_turboquant_test_support;
