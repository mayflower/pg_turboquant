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
