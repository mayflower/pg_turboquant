CREATE FUNCTION tq_debug_validate_reloptions(text[])
RETURNS text
AS 'MODULE_PATHNAME', 'tq_debug_validate_reloptions'
LANGUAGE C
STRICT;

COMMENT ON FUNCTION tq_debug_validate_reloptions(text[]) IS 'Regression-only helper that validates turboquant reloptions.';

CREATE FUNCTION tq_debug_router_metadata(regclass)
RETURNS text
LANGUAGE C
STRICT
AS 'MODULE_PATHNAME', 'tq_debug_router_metadata';

COMMENT ON FUNCTION tq_debug_router_metadata(regclass) IS 'Regression-only helper that reads persisted turboquant router metadata.';

CREATE FUNCTION tq_debug_transform_metadata(regclass)
RETURNS text
LANGUAGE C
STRICT
AS 'MODULE_PATHNAME', 'tq_debug_transform_metadata';

COMMENT ON FUNCTION tq_debug_transform_metadata(regclass) IS 'Regression-only helper that reads persisted turboquant transform metadata.';

CREATE FUNCTION tq_test_corrupt_meta_magic(regclass)
RETURNS void
LANGUAGE C
STRICT
AS 'MODULE_PATHNAME', 'tq_test_corrupt_meta_magic';

CREATE FUNCTION tq_test_corrupt_meta_format_version(regclass, integer)
RETURNS void
LANGUAGE C
STRICT
AS 'MODULE_PATHNAME', 'tq_test_corrupt_meta_format_version';

CREATE FUNCTION tq_test_corrupt_meta_transform_contract(regclass)
RETURNS void
LANGUAGE C
STRICT
AS 'MODULE_PATHNAME', 'tq_test_corrupt_meta_transform_contract';

CREATE FUNCTION tq_test_corrupt_first_list_head_to_directory_root(regclass)
RETURNS void
LANGUAGE C
STRICT
AS 'MODULE_PATHNAME', 'tq_test_corrupt_first_list_head_to_directory_root';

CREATE FUNCTION tq_test_corrupt_first_batch_occupied_count(regclass)
RETURNS void
LANGUAGE C
STRICT
AS 'MODULE_PATHNAME', 'tq_test_corrupt_first_batch_occupied_count';
