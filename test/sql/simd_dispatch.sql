SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

SELECT
	meta ? 'preferred_kernel' AS has_preferred_kernel,
	meta ? 'compiled' AS has_compiled,
	meta ? 'runtime_available' AS has_runtime_available,
	meta ? 'force_disabled' AS has_force_disabled,
	(meta->>'preferred_kernel') IN ('scalar', 'avx2', 'avx512', 'neon') AS preferred_kernel_known,
	(meta #>> '{compiled,scalar}')::boolean AS scalar_compiled,
	(meta #>> '{runtime_available,scalar}')::boolean AS scalar_runtime_available,
	(meta->>'force_disabled')::boolean AS force_disabled
FROM (SELECT tq_runtime_simd_features() AS meta) AS s;

SELECT
	CASE
		WHEN meta->>'preferred_kernel' = 'avx512' THEN (meta #>> '{runtime_available,avx512}')::boolean
		WHEN meta->>'preferred_kernel' = 'avx2' THEN (meta #>> '{runtime_available,avx2}')::boolean
		WHEN meta->>'preferred_kernel' = 'neon' THEN (meta #>> '{runtime_available,neon}')::boolean
		ELSE (meta #>> '{runtime_available,scalar}')::boolean
	END AS preferred_kernel_runtime_available
FROM (SELECT tq_runtime_simd_features() AS meta) AS s;

SET enable_bitmapscan = off;
SET enable_seqscan = off;

CREATE TABLE tq_simd_supported_docs (
	id int4 PRIMARY KEY,
	embedding vector(8)
);

INSERT INTO tq_simd_supported_docs (id, embedding) VALUES
	(1, '[1,0,0,0,0,0,0,0]'),
	(2, '[0.92388,0.382683,0,0,0,0,0,0]'),
	(3, '[0,1,0,0,0,0,0,0]');

CREATE INDEX tq_simd_supported_idx
	ON tq_simd_supported_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = true);

SELECT array_agg(id) AS supported_ids
FROM (
	SELECT id
	FROM tq_simd_supported_docs
	ORDER BY embedding <=> '[1,0,0,0,0,0,0,0]'
	LIMIT 3
) ranked;

WITH simd AS (
	SELECT tq_runtime_simd_features() AS meta
),
stats AS (
	SELECT tq_last_scan_stats() AS stats
)
SELECT
	stats->>'score_kernel' IN ('scalar', 'avx2') AS supported_kernel_known,
	CASE
		WHEN (simd.meta #>> '{runtime_available,avx2}')::boolean THEN stats->>'score_kernel' = 'avx2'
		ELSE stats->>'score_kernel' = 'scalar'
	END AS supported_kernel_matches_runtime,
	stats->>'score_mode' = 'code_domain' AS supported_mode_is_code_domain
FROM simd, stats;

CREATE TABLE tq_simd_unsupported_docs (
	id int4 PRIMARY KEY,
	embedding vector(6)
);

INSERT INTO tq_simd_unsupported_docs (id, embedding) VALUES
	(1, '[1,0,0,0,0,0]'),
	(2, '[0.92388,0.382683,0,0,0,0]'),
	(3, '[0,1,0,0,0,0]');

CREATE INDEX tq_simd_unsupported_idx
	ON tq_simd_unsupported_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = true);

SELECT array_agg(id) AS unsupported_ids
FROM (
	SELECT id
	FROM tq_simd_unsupported_docs
	ORDER BY embedding <=> '[1,0,0,0,0,0]'
	LIMIT 3
) ranked;

SELECT
	(tq_last_scan_stats()->>'score_kernel') = 'scalar' AS unsupported_kernel_scalar,
	(tq_last_scan_stats()->>'score_mode') = 'code_domain' AS unsupported_mode_is_code_domain;
