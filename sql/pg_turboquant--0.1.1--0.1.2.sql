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

ALTER OPERATOR FAMILY tq_vector_cosine_turboquant_ops USING turboquant
	ADD OPERATOR 1 <?=> (vector, bytea);

ALTER OPERATOR FAMILY tq_halfvec_cosine_turboquant_ops USING turboquant
	ADD OPERATOR 1 <?=> (halfvec, bytea);
