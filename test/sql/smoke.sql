DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

SELECT extname = 'pg_turboquant' AS extension_present
FROM pg_extension
WHERE extname = 'pg_turboquant';

SELECT tq_smoke();
