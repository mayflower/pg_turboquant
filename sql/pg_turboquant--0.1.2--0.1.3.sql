CREATE OR REPLACE FUNCTION tq_index_metadata(indexed_index regclass)
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
