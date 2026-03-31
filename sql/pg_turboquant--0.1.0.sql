CREATE FUNCTION tq_smoke()
RETURNS text
AS 'MODULE_PATHNAME', 'tq_smoke'
LANGUAGE C
STRICT;

COMMENT ON FUNCTION tq_smoke() IS 'Bootstrap smoke-test function for pg_turboquant';

CREATE FUNCTION tq_debug_validate_reloptions(text[])
RETURNS text
AS 'MODULE_PATHNAME', 'tq_debug_validate_reloptions'
LANGUAGE C
STRICT;

COMMENT ON FUNCTION tq_debug_validate_reloptions(text[]) IS 'Test-only helper that validates turboquant reloptions';

CREATE FUNCTION tq_debug_router_metadata(regclass)
RETURNS text
LANGUAGE C STRICT
AS 'MODULE_PATHNAME', 'tq_debug_router_metadata';

COMMENT ON FUNCTION tq_debug_router_metadata(regclass) IS 'Test-only helper that reads persisted turboquant router metadata';

CREATE FUNCTION tq_debug_transform_metadata(regclass)
RETURNS text
LANGUAGE C STRICT
AS 'MODULE_PATHNAME', 'tq_debug_transform_metadata';

COMMENT ON FUNCTION tq_debug_transform_metadata(regclass) IS 'Test-only helper that reads persisted turboquant transform metadata';

CREATE FUNCTION tq_index_metadata_core(regclass)
RETURNS text
LANGUAGE C STRICT
AS 'MODULE_PATHNAME', 'tq_index_metadata_core';

CREATE FUNCTION tq_last_scan_stats_core()
RETURNS text
LANGUAGE C
AS 'MODULE_PATHNAME', 'tq_last_scan_stats_core';

CREATE FUNCTION tq_last_shadow_decode_candidate_tids_core()
RETURNS text[]
LANGUAGE C
AS 'MODULE_PATHNAME', 'tq_last_shadow_decode_candidate_tids_core';

CREATE FUNCTION tq_runtime_simd_features_core()
RETURNS text
LANGUAGE C STRICT
AS 'MODULE_PATHNAME', 'tq_runtime_simd_features_core';

CREATE FUNCTION tq_last_scan_stats()
RETURNS jsonb
LANGUAGE sql
VOLATILE
AS $$
	SELECT tq_last_scan_stats_core()::jsonb;
$$;

COMMENT ON FUNCTION tq_last_scan_stats() IS 'Returns backend-local JSON metrics for the last turboquant scan in the current session.';

COMMENT ON FUNCTION tq_last_shadow_decode_candidate_tids_core() IS 'Diagnostic helper that returns backend-local shadow decode candidate CTIDs from the last turboquant scan in the current session.';

CREATE FUNCTION tq_runtime_simd_features()
RETURNS jsonb
LANGUAGE sql
STABLE
AS $$
	SELECT tq_runtime_simd_features_core()::jsonb;
$$;

COMMENT ON FUNCTION tq_runtime_simd_features() IS 'Returns compile-time and runtime SIMD availability plus the preferred turboquant score kernel.';

CREATE FUNCTION tq_index_metadata(indexed_index regclass)
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
	meta jsonb;
	heap_relation regclass;
	heap_live_rows bigint;
	opclass_name text;
	input_type text;
BEGIN
	meta := tq_index_metadata_core(indexed_index)::jsonb;

	SELECT
		i.indrelid::regclass,
		opc.opcname,
		opc.opcintype::regtype::text
	INTO
		heap_relation,
		opclass_name,
		input_type
	FROM pg_index AS i
	JOIN pg_opclass AS opc
		ON opc.oid = i.indclass[0]
	WHERE i.indexrelid = indexed_index;

	EXECUTE format('SELECT count(*) FROM %s', heap_relation)
	INTO heap_live_rows;

	RETURN meta || jsonb_build_object(
		'access_method', 'turboquant',
		'opclass', opclass_name,
		'input_type', input_type,
		'heap_relation', heap_relation::text,
		'heap_live_rows', heap_live_rows,
		'capabilities', jsonb_build_object(
			'ordered_scan', true,
			'bitmap_scan', true,
			'index_only_scan', false,
			'multicolumn', false,
			'include_columns', false
		)
	);
END;
$$;

COMMENT ON FUNCTION tq_index_metadata(regclass) IS 'Returns stable JSON metadata, capability flags, and maintenance stats for a turboquant index.';

CREATE FUNCTION tq_metric_order_operator(metric text)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
BEGIN
	CASE lower(metric)
		WHEN 'cosine' THEN
			RETURN '<=>';
		WHEN 'ip' THEN
			RETURN '<#>';
		WHEN 'l2' THEN
			RETURN '<->';
		ELSE
			RAISE EXCEPTION 'invalid turboquant metric: %', metric
				USING ERRCODE = '22023',
					  HINT = 'Use one of: cosine, ip, l2.';
	END CASE;
END;
$$;

CREATE FUNCTION tq_resolve_query_knobs(candidate_limit integer,
									   final_limit integer,
									   requested_probes integer DEFAULT NULL,
									   requested_oversample_factor integer DEFAULT NULL)
RETURNS TABLE(probes integer,
			  oversample_factor integer,
			  max_visited_codes integer,
			  max_visited_pages integer)
LANGUAGE plpgsql
AS $$
BEGIN
	IF candidate_limit IS NULL OR candidate_limit < 1 THEN
		RAISE EXCEPTION 'invalid turboquant candidate_limit: %', candidate_limit
			USING ERRCODE = '22023',
				  HINT = 'candidate_limit must be at least 1.';
	END IF;

	IF final_limit IS NULL THEN
		final_limit := candidate_limit;
	END IF;

	IF final_limit < 1 THEN
		RAISE EXCEPTION 'invalid turboquant final_limit: %', final_limit
			USING ERRCODE = '22023',
				  HINT = 'final_limit must be at least 1.';
	END IF;

	IF final_limit > candidate_limit THEN
		RAISE EXCEPTION 'invalid turboquant limits: final_limit (%) exceeds candidate_limit (%)', final_limit, candidate_limit
			USING ERRCODE = '22023',
				  HINT = 'final_limit must be less than or equal to candidate_limit.';
	END IF;

	IF requested_probes IS NOT NULL AND requested_probes < 1 THEN
		RAISE EXCEPTION 'invalid turboquant probes: %', requested_probes
			USING ERRCODE = '22023',
				  HINT = 'probes must be at least 1.';
	END IF;

	IF requested_oversample_factor IS NOT NULL AND requested_oversample_factor < 1 THEN
		RAISE EXCEPTION 'invalid turboquant oversample_factor: %', requested_oversample_factor
			USING ERRCODE = '22023',
				  HINT = 'oversample_factor must be at least 1.';
	END IF;

	IF requested_probes IS NULL AND requested_oversample_factor IS NULL THEN
		oversample_factor := GREATEST(8, CEIL(candidate_limit::numeric / final_limit)::integer);
		probes := GREATEST(1, CEIL(candidate_limit::numeric / oversample_factor)::integer);
	ELSIF requested_probes IS NULL THEN
		oversample_factor := requested_oversample_factor;
		probes := GREATEST(1, CEIL(candidate_limit::numeric / oversample_factor)::integer);
	ELSIF requested_oversample_factor IS NULL THEN
		probes := requested_probes;
		oversample_factor := GREATEST(1, CEIL(candidate_limit::numeric / probes)::integer);
	ELSE
		probes := requested_probes;
		oversample_factor := requested_oversample_factor;
	END IF;

	max_visited_codes := LEAST(
		2147483647::bigint,
		GREATEST(
			candidate_limit::bigint,
			candidate_limit::bigint * oversample_factor::bigint
		)
	)::integer;
	max_visited_pages := 0;

	RETURN QUERY
	SELECT tq_resolve_query_knobs.probes,
		   tq_resolve_query_knobs.oversample_factor,
		   tq_resolve_query_knobs.max_visited_codes,
		   tq_resolve_query_knobs.max_visited_pages;
END;
$$;

CREATE FUNCTION tq_recommended_query_knobs(candidate_limit integer,
										   final_limit integer DEFAULT NULL)
RETURNS TABLE(probes integer,
			  oversample_factor integer,
			  max_visited_codes integer,
			  max_visited_pages integer)
LANGUAGE SQL
AS $$
	SELECT probes, oversample_factor, max_visited_codes, max_visited_pages
	FROM tq_resolve_query_knobs(candidate_limit, final_limit, NULL, NULL);
$$;

COMMENT ON FUNCTION tq_recommended_query_knobs(integer, integer) IS 'Returns recommended turboquant probes, oversample_factor, and visit budgets for a two-stage query.';

CREATE FUNCTION tq_effective_rerank_candidate_limit(candidate_limit integer,
													final_limit integer)
RETURNS integer
LANGUAGE plpgsql
AS $$
DECLARE
	decode_rescore_factor integer;
	decode_rescore_extra_candidates integer;
	base_candidate_limit integer;
BEGIN
	IF candidate_limit IS NULL OR candidate_limit < 1 THEN
		RAISE EXCEPTION 'invalid turboquant candidate_limit: %', candidate_limit
			USING ERRCODE = '22023',
				  HINT = 'candidate_limit must be at least 1.';
	END IF;

	IF final_limit IS NULL THEN
		final_limit := candidate_limit;
	END IF;

	IF final_limit < 1 THEN
		RAISE EXCEPTION 'invalid turboquant final_limit: %', final_limit
			USING ERRCODE = '22023',
				  HINT = 'final_limit must be at least 1.';
	END IF;

	base_candidate_limit := GREATEST(candidate_limit, final_limit);
	decode_rescore_factor := COALESCE(current_setting('turboquant.decode_rescore_factor', true), '1')::integer;

	IF decode_rescore_factor <= 1 THEN
		RETURN base_candidate_limit;
	END IF;

	decode_rescore_extra_candidates := COALESCE(current_setting('turboquant.decode_rescore_extra_candidates', true), '-1')::integer;

	IF decode_rescore_extra_candidates < 0 THEN
		decode_rescore_extra_candidates := LEAST(512, GREATEST(128, candidate_limit / 2));
	END IF;

	RETURN base_candidate_limit + decode_rescore_extra_candidates;
END;
$$;

COMMENT ON FUNCTION tq_effective_rerank_candidate_limit(integer, integer) IS 'Returns the effective approximate candidate limit used by tq_rerank_candidates() after decode-rescore boundary-band expansion.';

CREATE FUNCTION tq_approx_candidates(indexed_table regclass,
									 id_column name,
									 embedding_column name,
									 query_vector vector,
									 metric text,
									 candidate_limit integer,
									 probes integer DEFAULT NULL,
									 oversample_factor integer DEFAULT NULL)
RETURNS TABLE(candidate_id text,
			  approximate_rank integer,
			  approximate_distance double precision)
LANGUAGE plpgsql
AS $$
DECLARE
	order_operator text;
	effective_candidate_limit integer;
	resolved_probes integer;
	resolved_oversample integer;
	resolved_max_visited_codes integer;
	resolved_max_visited_pages integer;
	helper_sql text;
BEGIN
	order_operator := tq_metric_order_operator(metric);

	SELECT knobs.probes, knobs.oversample_factor, knobs.max_visited_codes, knobs.max_visited_pages
	INTO resolved_probes, resolved_oversample, resolved_max_visited_codes, resolved_max_visited_pages
	FROM tq_resolve_query_knobs(candidate_limit, candidate_limit, probes, oversample_factor) AS knobs;

	EXECUTE format('SET LOCAL turboquant.probes = %s', resolved_probes);
	EXECUTE format('SET LOCAL turboquant.oversample_factor = %s', resolved_oversample);
	EXECUTE format('SET LOCAL turboquant.max_visited_codes = %s', resolved_max_visited_codes);
	EXECUTE format('SET LOCAL turboquant.max_visited_pages = %s', resolved_max_visited_pages);

	helper_sql := format($fmt$
		SELECT candidate_id, approximate_rank, approximate_distance
		FROM (
			SELECT %1$I::text AS candidate_id,
				row_number() OVER (ORDER BY %2$I %3$s %5$L::vector, %1$I)::integer AS approximate_rank,
				round((%2$I %3$s %5$L::vector)::numeric, 6)::double precision AS approximate_distance
			FROM %4$s
			ORDER BY %2$I %3$s %5$L::vector
			LIMIT %6$s
		) approx
		ORDER BY approximate_rank
	$fmt$, id_column, embedding_column, order_operator, indexed_table, query_vector::text, candidate_limit);

	RETURN QUERY EXECUTE helper_sql;
END;
$$;

CREATE FUNCTION tq_approx_candidates(indexed_table regclass,
									 id_column name,
									 embedding_column name,
									 query_vector halfvec,
									 metric text,
									 candidate_limit integer,
									 probes integer DEFAULT NULL,
									 oversample_factor integer DEFAULT NULL)
RETURNS TABLE(candidate_id text,
			  approximate_rank integer,
			  approximate_distance double precision)
LANGUAGE plpgsql
AS $$
DECLARE
	order_operator text;
	resolved_probes integer;
	resolved_oversample integer;
	resolved_max_visited_codes integer;
	resolved_max_visited_pages integer;
	helper_sql text;
BEGIN
	order_operator := tq_metric_order_operator(metric);

	SELECT knobs.probes, knobs.oversample_factor, knobs.max_visited_codes, knobs.max_visited_pages
	INTO resolved_probes, resolved_oversample, resolved_max_visited_codes, resolved_max_visited_pages
	FROM tq_resolve_query_knobs(candidate_limit, candidate_limit, probes, oversample_factor) AS knobs;

	EXECUTE format('SET LOCAL turboquant.probes = %s', resolved_probes);
	EXECUTE format('SET LOCAL turboquant.oversample_factor = %s', resolved_oversample);
	EXECUTE format('SET LOCAL turboquant.max_visited_codes = %s', resolved_max_visited_codes);
	EXECUTE format('SET LOCAL turboquant.max_visited_pages = %s', resolved_max_visited_pages);

	helper_sql := format($fmt$
		SELECT candidate_id, approximate_rank, approximate_distance
		FROM (
			SELECT %1$I::text AS candidate_id,
				row_number() OVER (ORDER BY %2$I %3$s %5$L::halfvec, %1$I)::integer AS approximate_rank,
				round((%2$I %3$s %5$L::halfvec)::numeric, 6)::double precision AS approximate_distance
			FROM %4$s
			ORDER BY %2$I %3$s %5$L::halfvec
			LIMIT %6$s
		) approx
		ORDER BY approximate_rank
	$fmt$, id_column, embedding_column, order_operator, indexed_table, query_vector::text, candidate_limit);

	RETURN QUERY EXECUTE helper_sql;
END;
$$;

COMMENT ON FUNCTION tq_approx_candidates(regclass, name, name, vector, text, integer, integer, integer) IS 'Returns approximate turboquant candidate IDs, ranks, and approximate distances.';
COMMENT ON FUNCTION tq_approx_candidates(regclass, name, name, halfvec, text, integer, integer, integer) IS 'Returns approximate turboquant candidate IDs, ranks, and approximate distances.';

CREATE FUNCTION tq_rerank_candidates(indexed_table regclass,
									 id_column name,
									 embedding_column name,
									 query_vector vector,
									 metric text,
									 candidate_limit integer,
									 final_limit integer,
									 probes integer DEFAULT NULL,
									 oversample_factor integer DEFAULT NULL)
RETURNS TABLE(candidate_id text,
			  approximate_rank integer,
			  approximate_distance double precision,
			  exact_rank integer,
			  exact_distance double precision)
LANGUAGE plpgsql
AS $$
DECLARE
	order_operator text;
	effective_candidate_limit integer;
	resolved_probes integer;
	resolved_oversample integer;
	resolved_max_visited_codes integer;
	resolved_max_visited_pages integer;
	helper_sql text;
BEGIN
	order_operator := tq_metric_order_operator(metric);
	effective_candidate_limit := tq_effective_rerank_candidate_limit(candidate_limit, final_limit);

	SELECT knobs.probes, knobs.oversample_factor, knobs.max_visited_codes, knobs.max_visited_pages
	INTO resolved_probes, resolved_oversample, resolved_max_visited_codes, resolved_max_visited_pages
	FROM tq_resolve_query_knobs(effective_candidate_limit, final_limit, probes, oversample_factor) AS knobs;

	EXECUTE format('SET LOCAL turboquant.probes = %s', resolved_probes);
	EXECUTE format('SET LOCAL turboquant.oversample_factor = %s', resolved_oversample);
	EXECUTE format('SET LOCAL turboquant.max_visited_codes = %s', resolved_max_visited_codes);
	EXECUTE format('SET LOCAL turboquant.max_visited_pages = %s', resolved_max_visited_pages);

	helper_sql := format($fmt$
		SELECT candidate_id, approximate_rank, approximate_distance, exact_rank, exact_distance
		FROM (
			SELECT candidate_id,
				row_number() OVER (ORDER BY approx_distance, candidate_key)::integer AS approximate_rank,
				round(approx_distance::numeric, 6)::double precision AS approximate_distance,
				row_number() OVER (ORDER BY exact_distance, candidate_key)::integer AS exact_rank,
				round(exact_distance::numeric, 6)::double precision AS exact_distance
			FROM (
				SELECT %1$I::text AS candidate_id,
					%1$I AS candidate_key,
					%2$I %3$s %5$L::vector AS approx_distance,
					%2$I %3$s %5$L::vector AS exact_distance
				FROM %4$s
				ORDER BY %2$I %3$s %5$L::vector
				LIMIT %6$s
			) approx
		) reranked
		WHERE exact_rank <= %7$s
		ORDER BY exact_rank
	$fmt$, id_column, embedding_column, order_operator, indexed_table, query_vector::text, effective_candidate_limit, final_limit);

	RETURN QUERY EXECUTE helper_sql;
END;
$$;

CREATE FUNCTION tq_rerank_candidates(indexed_table regclass,
									 id_column name,
									 embedding_column name,
									 query_vector halfvec,
									 metric text,
									 candidate_limit integer,
									 final_limit integer,
									 probes integer DEFAULT NULL,
									 oversample_factor integer DEFAULT NULL)
RETURNS TABLE(candidate_id text,
			  approximate_rank integer,
			  approximate_distance double precision,
			  exact_rank integer,
			  exact_distance double precision)
LANGUAGE plpgsql
AS $$
DECLARE
	order_operator text;
	effective_candidate_limit integer;
	resolved_probes integer;
	resolved_oversample integer;
	resolved_max_visited_codes integer;
	resolved_max_visited_pages integer;
	helper_sql text;
BEGIN
	order_operator := tq_metric_order_operator(metric);
	effective_candidate_limit := tq_effective_rerank_candidate_limit(candidate_limit, final_limit);

	SELECT knobs.probes, knobs.oversample_factor, knobs.max_visited_codes, knobs.max_visited_pages
	INTO resolved_probes, resolved_oversample, resolved_max_visited_codes, resolved_max_visited_pages
	FROM tq_resolve_query_knobs(effective_candidate_limit, final_limit, probes, oversample_factor) AS knobs;

	EXECUTE format('SET LOCAL turboquant.probes = %s', resolved_probes);
	EXECUTE format('SET LOCAL turboquant.oversample_factor = %s', resolved_oversample);
	EXECUTE format('SET LOCAL turboquant.max_visited_codes = %s', resolved_max_visited_codes);
	EXECUTE format('SET LOCAL turboquant.max_visited_pages = %s', resolved_max_visited_pages);

	helper_sql := format($fmt$
		SELECT candidate_id, approximate_rank, approximate_distance, exact_rank, exact_distance
		FROM (
			SELECT candidate_id,
				row_number() OVER (ORDER BY approx_distance, candidate_key)::integer AS approximate_rank,
				round(approx_distance::numeric, 6)::double precision AS approximate_distance,
				row_number() OVER (ORDER BY exact_distance, candidate_key)::integer AS exact_rank,
				round(exact_distance::numeric, 6)::double precision AS exact_distance
			FROM (
				SELECT %1$I::text AS candidate_id,
					%1$I AS candidate_key,
					%2$I %3$s %5$L::halfvec AS approx_distance,
					%2$I %3$s %5$L::halfvec AS exact_distance
				FROM %4$s
				ORDER BY %2$I %3$s %5$L::halfvec
				LIMIT %6$s
			) approx
		) reranked
		WHERE exact_rank <= %7$s
		ORDER BY exact_rank
	$fmt$, id_column, embedding_column, order_operator, indexed_table, query_vector::text, effective_candidate_limit, final_limit);

	RETURN QUERY EXECUTE helper_sql;
END;
$$;

COMMENT ON FUNCTION tq_rerank_candidates(regclass, name, name, vector, text, integer, integer, integer, integer) IS 'Returns approximate candidates reranked exactly within SQL over the candidate set.';
COMMENT ON FUNCTION tq_rerank_candidates(regclass, name, name, halfvec, text, integer, integer, integer, integer) IS 'Returns approximate candidates reranked exactly within SQL over the candidate set.';

CREATE FUNCTION tq_bitmap_cosine_filter(query_vector vector,
										distance_threshold double precision)
RETURNS bytea
AS 'MODULE_PATHNAME', 'tq_bitmap_cosine_filter'
LANGUAGE C
STRICT
IMMUTABLE;

CREATE FUNCTION tq_bitmap_cosine_filter(query_vector halfvec,
										distance_threshold double precision)
RETURNS bytea
AS 'MODULE_PATHNAME', 'tq_bitmap_cosine_filter_halfvec'
LANGUAGE C
STRICT
IMMUTABLE;

COMMENT ON FUNCTION tq_bitmap_cosine_filter(vector, double precision) IS 'Builds an internal turboquant cosine bitmap-filter payload.';
COMMENT ON FUNCTION tq_bitmap_cosine_filter(halfvec, double precision) IS 'Builds an internal turboquant cosine bitmap-filter payload.';

CREATE FUNCTION tq_bitmap_cosine_match(left_vector vector,
									   filter bytea)
RETURNS boolean
AS 'MODULE_PATHNAME', 'tq_bitmap_cosine_match'
LANGUAGE C
STRICT
IMMUTABLE;

CREATE FUNCTION tq_bitmap_cosine_match(left_vector halfvec,
									   filter bytea)
RETURNS boolean
AS 'MODULE_PATHNAME', 'tq_bitmap_cosine_match_halfvec'
LANGUAGE C
STRICT
IMMUTABLE;

CREATE OPERATOR <?=> (
	LEFTARG = vector,
	RIGHTARG = bytea,
	PROCEDURE = tq_bitmap_cosine_match
);

CREATE OPERATOR <?=> (
	LEFTARG = halfvec,
	RIGHTARG = bytea,
	PROCEDURE = tq_bitmap_cosine_match
);

CREATE FUNCTION turboquanthandler(internal)
RETURNS index_am_handler
AS 'MODULE_PATHNAME', 'turboquanthandler'
LANGUAGE C;

CREATE ACCESS METHOD turboquant TYPE INDEX HANDLER turboquanthandler;
COMMENT ON ACCESS METHOD turboquant IS 'pg_turboquant index access method skeleton';

CREATE OPERATOR FAMILY tq_vector_cosine_turboquant_ops USING turboquant;
CREATE OPERATOR FAMILY tq_vector_ip_turboquant_ops USING turboquant;
CREATE OPERATOR FAMILY tq_vector_l2_turboquant_ops USING turboquant;
CREATE OPERATOR FAMILY tq_halfvec_cosine_turboquant_ops USING turboquant;
CREATE OPERATOR FAMILY tq_halfvec_ip_turboquant_ops USING turboquant;
CREATE OPERATOR FAMILY tq_halfvec_l2_turboquant_ops USING turboquant;

CREATE OPERATOR CLASS tq_cosine_ops
DEFAULT FOR TYPE vector USING turboquant FAMILY tq_vector_cosine_turboquant_ops AS
	OPERATOR 1 <?=> (vector, bytea),
	OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 vector_negative_inner_product(vector, vector),
	FUNCTION 2 vector_norm(vector);

CREATE OPERATOR CLASS tq_ip_ops
FOR TYPE vector USING turboquant FAMILY tq_vector_ip_turboquant_ops AS
	OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 vector_negative_inner_product(vector, vector);

CREATE OPERATOR CLASS tq_l2_ops
FOR TYPE vector USING turboquant FAMILY tq_vector_l2_turboquant_ops AS
	OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 vector_l2_squared_distance(vector, vector);

CREATE OPERATOR CLASS tq_halfvec_cosine_ops
DEFAULT FOR TYPE halfvec USING turboquant FAMILY tq_halfvec_cosine_turboquant_ops AS
	OPERATOR 1 <?=> (halfvec, bytea),
	OPERATOR 1 <=> (halfvec, halfvec) FOR ORDER BY float_ops,
	FUNCTION 1 halfvec_negative_inner_product(halfvec, halfvec),
	FUNCTION 2 l2_norm(halfvec);

CREATE OPERATOR CLASS tq_halfvec_ip_ops
FOR TYPE halfvec USING turboquant FAMILY tq_halfvec_ip_turboquant_ops AS
	OPERATOR 1 <#> (halfvec, halfvec) FOR ORDER BY float_ops,
	FUNCTION 1 halfvec_negative_inner_product(halfvec, halfvec);

CREATE OPERATOR CLASS tq_halfvec_l2_ops
FOR TYPE halfvec USING turboquant FAMILY tq_halfvec_l2_turboquant_ops AS
	OPERATOR 1 <-> (halfvec, halfvec) FOR ORDER BY float_ops,
	FUNCTION 1 halfvec_l2_squared_distance(halfvec, halfvec);
