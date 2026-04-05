CREATE FUNCTION tq_smoke()
RETURNS text
AS 'MODULE_PATHNAME', 'tq_smoke'
LANGUAGE C
STRICT;

COMMENT ON FUNCTION tq_smoke() IS 'Bootstrap smoke-test function for pg_turboquant';

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

CREATE FUNCTION tq_maintain_index_core(regclass)
RETURNS text
LANGUAGE C STRICT
AS 'MODULE_PATHNAME', 'tq_maintain_index_core';

CREATE FUNCTION tq_last_scan_stats()
RETURNS jsonb
LANGUAGE C
VOLATILE
AS 'MODULE_PATHNAME', 'tq_last_scan_stats';

COMMENT ON FUNCTION tq_last_scan_stats() IS 'Returns backend-local JSON metrics for the last turboquant scan in the current session.';

CREATE FUNCTION tq_last_shadow_decode_candidate_tids()
RETURNS text[]
LANGUAGE C
VOLATILE
AS 'MODULE_PATHNAME', 'tq_last_shadow_decode_candidate_tids';

COMMENT ON FUNCTION tq_last_shadow_decode_candidate_tids() IS 'Returns backend-local shadow decode candidate CTIDs from the last turboquant scan in the current session.';

CREATE FUNCTION tq_runtime_simd_features()
RETURNS jsonb
LANGUAGE C
STABLE
AS 'MODULE_PATHNAME', 'tq_runtime_simd_features';

COMMENT ON FUNCTION tq_runtime_simd_features() IS 'Returns compile-time and runtime SIMD availability plus the preferred turboquant score kernel.';

REVOKE EXECUTE ON FUNCTION tq_index_metadata_core(regclass) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION tq_last_scan_stats_core() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION tq_last_shadow_decode_candidate_tids_core() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION tq_runtime_simd_features_core() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION tq_maintain_index_core(regclass) FROM PUBLIC;

CREATE FUNCTION tq_index_metadata(indexed_index regclass)
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
	meta jsonb;
	heap_relation regclass;
	heap_live_rows_estimate bigint;
	opclass_name text;
	input_type text;
BEGIN
	meta := tq_index_metadata_core(indexed_index)::jsonb;

	SELECT
		i.indrelid::regclass,
		opc.opcname,
		opc.opcintype::regtype::text,
		CASE
			WHEN heap_class.reltuples < 0 THEN NULL
			ELSE round(heap_class.reltuples)::bigint
		END
	INTO
		heap_relation,
		opclass_name,
		input_type,
		heap_live_rows_estimate
	FROM pg_index AS i
	JOIN pg_opclass AS opc
		ON opc.oid = i.indclass[0]
	JOIN pg_class AS heap_class
		ON heap_class.oid = i.indrelid
	WHERE i.indexrelid = indexed_index;

	RETURN meta || jsonb_build_object(
		'access_method', 'turboquant',
		'opclass', opclass_name,
		'input_type', input_type,
		'heap_relation', heap_relation::text,
		'heap_live_rows_estimate', heap_live_rows_estimate,
		'capabilities', jsonb_build_object(
			'ordered_scan', true,
			'bitmap_scan', true,
			'index_only_scan', true,
			'vector_key_returnable', true,
			'ordered_vector_key_index_only_scan', true,
			'multicolumn', true,
			'include_columns', true
		),
		'operability', jsonb_build_object(
			'parallel_scan', false,
			'parallel_vacuum', false,
			'maintenance_work_mem_aware', false
		)
	);
END;
$$;

COMMENT ON FUNCTION tq_index_metadata(regclass) IS 'Returns stable JSON metadata, capability flags, and cheap estimated heap stats for a turboquant index.';

CREATE FUNCTION tq_maintain_index(indexed_index regclass)
RETURNS jsonb
LANGUAGE C
VOLATILE
AS 'MODULE_PATHNAME', 'tq_maintain_index';

COMMENT ON FUNCTION tq_maintain_index(regclass) IS 'Performs lightweight turboquant maintenance and returns delta/maintenance counters.';

CREATE FUNCTION tq_index_heap_stats(indexed_index regclass)
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
	heap_relation regclass;
	heap_live_rows_exact bigint;
BEGIN
	SELECT i.indrelid::regclass
	INTO heap_relation
	FROM pg_index AS i
	WHERE i.indexrelid = indexed_index;

	IF heap_relation IS NULL THEN
		RAISE EXCEPTION 'relation % is not a valid turboquant index', indexed_index
			USING ERRCODE = '22023';
	END IF;

	EXECUTE format('SELECT count(*) FROM %s', heap_relation)
	INTO heap_live_rows_exact;

	RETURN jsonb_build_object(
		'heap_relation', heap_relation::text,
		'heap_live_rows_exact', heap_live_rows_exact
	);
END;
$$;

COMMENT ON FUNCTION tq_index_heap_stats(regclass) IS 'Returns exact heap statistics for a turboquant index. This helper is intentionally expensive.';

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

CREATE FUNCTION tq_recommended_query_knobs(indexed_index regclass,
										   candidate_limit integer,
										   final_limit integer DEFAULT NULL,
										   filter_selectivity double precision DEFAULT NULL)
RETURNS TABLE(probes integer,
			  oversample_factor integer,
			  max_visited_codes integer,
			  max_visited_pages integer)
LANGUAGE plpgsql
AS $$
DECLARE
	meta jsonb;
	last_stats jsonb;
	base_probes integer;
	base_oversample integer;
	base_max_visited_codes integer;
	base_max_visited_pages integer;
	list_count integer;
	batch_page_count integer;
	live_count bigint;
	max_list_over_avg double precision;
	effective_selectivity double precision;
	budget_pressure_multiplier double precision;
	imbalance_multiplier double precision;
	recent_mode text;
	last_max_visited_codes integer;
	last_max_visited_pages integer;
	visited_code_count integer;
	visited_page_count integer;
	selected_live_count integer;
	candidate_heap_count integer;
	desired_candidate_limit bigint;
	desired_probes bigint;
	desired_oversample bigint;
	desired_code_budget bigint;
	desired_page_budget bigint;
	pages_per_probe double precision;
BEGIN
	SELECT knobs.probes, knobs.oversample_factor, knobs.max_visited_codes, knobs.max_visited_pages
	INTO base_probes, base_oversample, base_max_visited_codes, base_max_visited_pages
	FROM tq_resolve_query_knobs(candidate_limit, final_limit, NULL, NULL) AS knobs;

	IF filter_selectivity IS NOT NULL
	   AND (filter_selectivity <= 0.0 OR filter_selectivity > 1.0) THEN
		RAISE EXCEPTION 'invalid turboquant filter_selectivity: %', filter_selectivity
			USING ERRCODE = '22023',
				  HINT = 'filter_selectivity must be greater than 0 and less than or equal to 1.';
	END IF;

	meta := tq_index_metadata(indexed_index);
	last_stats := tq_last_scan_stats();
	list_count := GREATEST(COALESCE((meta->>'list_count')::integer, 0), 0);
	batch_page_count := GREATEST(COALESCE((meta->>'batch_page_count')::integer, 0), 0);
	live_count := GREATEST(COALESCE((meta->>'live_count')::bigint, 0), 0);
	max_list_over_avg := GREATEST(COALESCE((meta #>> '{list_distribution,max_list_over_avg}')::double precision, 1.0), 1.0);
	effective_selectivity := GREATEST(0.05, LEAST(COALESCE(filter_selectivity, 1.0), 1.0));
	budget_pressure_multiplier := 1.0;
	imbalance_multiplier := LEAST(2.0, GREATEST(1.0, sqrt(max_list_over_avg)));
	recent_mode := COALESCE(last_stats->>'mode', 'none');
	last_max_visited_codes := GREATEST(COALESCE((last_stats->>'max_visited_codes')::integer, 0), 0);
	last_max_visited_pages := GREATEST(COALESCE((last_stats->>'max_visited_pages')::integer, 0), 0);
	visited_code_count := GREATEST(COALESCE((last_stats->>'visited_code_count')::integer, 0), 0);
	visited_page_count := GREATEST(COALESCE((last_stats->>'visited_page_count')::integer, 0), 0);
	selected_live_count := GREATEST(COALESCE((last_stats->>'selected_live_count')::integer, 0), 0);
	candidate_heap_count := GREATEST(COALESCE((last_stats->>'candidate_heap_count')::integer, 0), 0);

	IF (list_count = 0 AND recent_mode = 'flat')
	   OR (list_count > 0 AND recent_mode = 'ivf') THEN
		IF last_max_visited_codes > 0 AND visited_code_count >= last_max_visited_codes THEN
			budget_pressure_multiplier := GREATEST(budget_pressure_multiplier, 2.0);
		END IF;
		IF last_max_visited_pages > 0 AND visited_page_count >= last_max_visited_pages THEN
			budget_pressure_multiplier := GREATEST(budget_pressure_multiplier, 2.0);
		END IF;
		IF filter_selectivity IS NULL
		   AND budget_pressure_multiplier > 1.0
		   AND selected_live_count > 0
		   AND candidate_heap_count > 0
		   AND candidate_heap_count < COALESCE(final_limit, candidate_limit) THEN
			effective_selectivity := GREATEST(
				0.05,
				LEAST(
					1.0,
					candidate_heap_count::double precision / selected_live_count::double precision
				)
			);
		END IF;
	END IF;

	desired_candidate_limit := CEIL(candidate_limit::numeric
									* budget_pressure_multiplier
									/ effective_selectivity)::bigint;
	desired_candidate_limit := GREATEST(desired_candidate_limit, candidate_limit::bigint);
	IF live_count > 0 THEN
		desired_candidate_limit := LEAST(desired_candidate_limit, live_count);
	END IF;

	IF list_count <= 0 THEN
		probes := base_probes;
		desired_oversample := GREATEST(
			base_oversample::bigint,
			CEIL(desired_candidate_limit::numeric
				 / GREATEST(COALESCE(final_limit, candidate_limit), 1))::bigint
		);
		oversample_factor := LEAST(1024, desired_oversample)::integer;
		desired_code_budget := GREATEST(
			base_max_visited_codes::bigint,
			desired_candidate_limit * GREATEST(2, LEAST(oversample_factor, 16))
		);
		max_visited_codes := LEAST(2147483647::bigint, desired_code_budget)::integer;
		max_visited_pages := 0;
	ELSE
		desired_probes := CEIL(base_probes::numeric
							   * imbalance_multiplier
							   * sqrt(1.0 / effective_selectivity))::bigint;
		desired_probes := GREATEST(base_probes::bigint, desired_probes);
		desired_probes := LEAST(list_count::bigint, desired_probes);
		probes := desired_probes::integer;

		desired_oversample := GREATEST(
			base_oversample::bigint,
			CEIL(desired_candidate_limit::numeric
				 / GREATEST(COALESCE(final_limit, candidate_limit), 1)
			/ GREATEST(probes, 1))::bigint
		);
		oversample_factor := LEAST(1024, GREATEST(1, desired_oversample))::integer;

		desired_code_budget := GREATEST(
			base_max_visited_codes::bigint,
			desired_candidate_limit * GREATEST(2, LEAST(oversample_factor, 16))
		);
		IF last_max_visited_codes > 0 AND visited_code_count >= last_max_visited_codes THEN
			desired_code_budget := GREATEST(desired_code_budget, (visited_code_count * 2)::bigint);
		END IF;
		max_visited_codes := LEAST(2147483647::bigint, desired_code_budget)::integer;

		pages_per_probe := GREATEST(1.0, batch_page_count::double precision / list_count::double precision);
		desired_page_budget := CEIL(pages_per_probe * probes * imbalance_multiplier)::bigint;
		desired_page_budget := GREATEST(base_max_visited_pages::bigint, desired_page_budget);
		desired_page_budget := GREATEST(desired_page_budget, probes::bigint);
		IF last_max_visited_pages > 0 AND visited_page_count >= last_max_visited_pages THEN
			desired_page_budget := GREATEST(desired_page_budget, (visited_page_count * 2)::bigint);
		END IF;
		IF batch_page_count > 0 THEN
			desired_page_budget := LEAST(desired_page_budget, batch_page_count::bigint);
		END IF;
		max_visited_pages := LEAST(2147483647::bigint, desired_page_budget)::integer;
	END IF;

	RETURN NEXT;
END;
$$;

COMMENT ON FUNCTION tq_recommended_query_knobs(regclass, integer, integer, double precision) IS 'Returns index-aware turboquant probes, oversample_factor, and visit budgets using index health, optional filter selectivity, and recent scan pressure.';

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
				approximate_rank,
				approximate_distance,
				row_number() OVER (ORDER BY approximate_distance, candidate_key)::integer AS exact_rank,
				approximate_distance AS exact_distance
			FROM (
				SELECT %1$I::text AS candidate_id,
					%1$I AS candidate_key,
					row_number() OVER ()::integer AS approximate_rank,
					round((%2$I %3$s %5$L::vector)::numeric, 6)::double precision AS approximate_distance
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
				approximate_rank,
				approximate_distance,
				row_number() OVER (ORDER BY approximate_distance, candidate_key)::integer AS exact_rank,
				approximate_distance AS exact_distance
			FROM (
				SELECT %1$I::text AS candidate_id,
					%1$I AS candidate_key,
					row_number() OVER ()::integer AS approximate_rank,
					round((%2$I %3$s %5$L::halfvec)::numeric, 6)::double precision AS approximate_distance
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

CREATE FUNCTION tq_vector_negative_inner_product(left_vector vector, right_vector vector)
RETURNS double precision
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
AS $$
	SELECT -((left_vector <#> right_vector)::double precision);
$$;

CREATE FUNCTION tq_vector_l2_squared_distance(left_vector vector, right_vector vector)
RETURNS double precision
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
AS $$
	SELECT power((left_vector <-> right_vector)::double precision, 2);
$$;

CREATE FUNCTION tq_vector_norm(input_vector vector)
RETURNS double precision
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
AS $$
	SELECT sqrt(GREATEST(-((input_vector <#> input_vector)::double precision), 0.0));
$$;

CREATE FUNCTION tq_halfvec_negative_inner_product(left_vector halfvec, right_vector halfvec)
RETURNS double precision
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
AS $$
	SELECT -((left_vector <#> right_vector)::double precision);
$$;

CREATE FUNCTION tq_halfvec_l2_squared_distance(left_vector halfvec, right_vector halfvec)
RETURNS double precision
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
AS $$
	SELECT power((left_vector <-> right_vector)::double precision, 2);
$$;

CREATE FUNCTION tq_halfvec_norm(input_vector halfvec)
RETURNS double precision
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
AS $$
	SELECT sqrt(GREATEST(-((input_vector <#> input_vector)::double precision), 0.0));
$$;

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
CREATE OPERATOR FAMILY tq_bool_filter_turboquant_ops USING turboquant;
CREATE OPERATOR FAMILY tq_int2_filter_turboquant_ops USING turboquant;
CREATE OPERATOR FAMILY tq_int4_filter_turboquant_ops USING turboquant;
CREATE OPERATOR FAMILY tq_int8_filter_turboquant_ops USING turboquant;
CREATE OPERATOR FAMILY tq_date_filter_turboquant_ops USING turboquant;
CREATE OPERATOR FAMILY tq_timestamptz_filter_turboquant_ops USING turboquant;
CREATE OPERATOR FAMILY tq_uuid_filter_turboquant_ops USING turboquant;

CREATE OPERATOR CLASS tq_cosine_ops
DEFAULT FOR TYPE vector USING turboquant FAMILY tq_vector_cosine_turboquant_ops AS
	OPERATOR 1 <?=> (vector, bytea),
	OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 tq_vector_negative_inner_product(vector, vector),
	FUNCTION 2 tq_vector_norm(vector);

CREATE OPERATOR CLASS tq_ip_ops
FOR TYPE vector USING turboquant FAMILY tq_vector_ip_turboquant_ops AS
	OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 tq_vector_negative_inner_product(vector, vector);

CREATE OPERATOR CLASS tq_l2_ops
FOR TYPE vector USING turboquant FAMILY tq_vector_l2_turboquant_ops AS
	OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 tq_vector_l2_squared_distance(vector, vector);

CREATE OPERATOR CLASS tq_halfvec_cosine_ops
DEFAULT FOR TYPE halfvec USING turboquant FAMILY tq_halfvec_cosine_turboquant_ops AS
	OPERATOR 1 <?=> (halfvec, bytea),
	OPERATOR 1 <=> (halfvec, halfvec) FOR ORDER BY float_ops,
	FUNCTION 1 tq_halfvec_negative_inner_product(halfvec, halfvec),
	FUNCTION 2 tq_halfvec_norm(halfvec);

CREATE OPERATOR CLASS tq_halfvec_ip_ops
FOR TYPE halfvec USING turboquant FAMILY tq_halfvec_ip_turboquant_ops AS
	OPERATOR 1 <#> (halfvec, halfvec) FOR ORDER BY float_ops,
	FUNCTION 1 tq_halfvec_negative_inner_product(halfvec, halfvec);

CREATE OPERATOR CLASS tq_halfvec_l2_ops
FOR TYPE halfvec USING turboquant FAMILY tq_halfvec_l2_turboquant_ops AS
	OPERATOR 1 <-> (halfvec, halfvec) FOR ORDER BY float_ops,
	FUNCTION 1 tq_halfvec_l2_squared_distance(halfvec, halfvec);

CREATE OPERATOR CLASS tq_int4_filter_ops
FOR TYPE int4 USING turboquant FAMILY tq_int4_filter_turboquant_ops AS
	OPERATOR 1 = (int4, int4),
	FUNCTION 1 btint4cmp(int4, int4);

CREATE OPERATOR CLASS tq_bool_filter_ops
FOR TYPE bool USING turboquant FAMILY tq_bool_filter_turboquant_ops AS
	OPERATOR 1 = (bool, bool),
	FUNCTION 1 btboolcmp(bool, bool);

CREATE OPERATOR CLASS tq_int2_filter_ops
FOR TYPE int2 USING turboquant FAMILY tq_int2_filter_turboquant_ops AS
	OPERATOR 1 = (int2, int2),
	FUNCTION 1 btint2cmp(int2, int2);

CREATE OPERATOR CLASS tq_int8_filter_ops
FOR TYPE int8 USING turboquant FAMILY tq_int8_filter_turboquant_ops AS
	OPERATOR 1 = (int8, int8),
	FUNCTION 1 btint8cmp(int8, int8);

CREATE OPERATOR CLASS tq_date_filter_ops
FOR TYPE date USING turboquant FAMILY tq_date_filter_turboquant_ops AS
	OPERATOR 1 = (date, date),
	FUNCTION 1 date_cmp(date, date);

CREATE OPERATOR CLASS tq_timestamptz_filter_ops
FOR TYPE timestamptz USING turboquant FAMILY tq_timestamptz_filter_turboquant_ops AS
	OPERATOR 1 = (timestamptz, timestamptz),
	FUNCTION 1 timestamptz_cmp(timestamptz, timestamptz);

CREATE OPERATOR CLASS tq_uuid_filter_ops
FOR TYPE uuid USING turboquant FAMILY tq_uuid_filter_turboquant_ops AS
	OPERATOR 1 = (uuid, uuid),
	FUNCTION 1 uuid_cmp(uuid, uuid);
