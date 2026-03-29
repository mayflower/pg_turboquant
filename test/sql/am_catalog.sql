DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

SELECT amname, amtype
FROM pg_am
WHERE amname = 'turboquant';

SELECT proname, pg_get_function_result(oid) AS result_type
FROM pg_proc
WHERE proname = 'turboquanthandler';
