CREATE FUNCTION tq_runtime_simd_features_core()
RETURNS text
LANGUAGE C STRICT
AS 'MODULE_PATHNAME', 'tq_runtime_simd_features_core';

CREATE FUNCTION tq_runtime_simd_features()
RETURNS jsonb
LANGUAGE sql
STABLE
AS $$
	SELECT tq_runtime_simd_features_core()::jsonb;
$$;

COMMENT ON FUNCTION tq_runtime_simd_features() IS 'Returns compile-time and runtime SIMD availability plus the preferred turboquant score kernel.';
