SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;
SET client_min_messages = notice;

CREATE TABLE tq_fast_lane_build_docs (
	id int4 PRIMARY KEY,
	embedding vector(2)
);

INSERT INTO tq_fast_lane_build_docs (id, embedding) VALUES
	(1, '[2,0]'),
	(2, '[0,1]');

DO $$
BEGIN
	BEGIN
		CREATE INDEX tq_fast_lane_build_idx
			ON tq_fast_lane_build_docs
			USING turboquant (embedding tq_cosine_ops)
			WITH (
				bits = 4,
				lists = 0,
				lanes = auto,
				transform = 'hadamard',
				normalized = true
			);
	EXCEPTION
		WHEN OTHERS THEN
			RAISE NOTICE 'build_error=%', SQLERRM;
	END;
END;
$$;

CREATE TABLE tq_fast_lane_query_docs (
	id int4 PRIMARY KEY,
	embedding vector(2)
);

INSERT INTO tq_fast_lane_query_docs (id, embedding) VALUES
	(1, '[1,0]'),
	(2, '[0,1]');

CREATE INDEX tq_fast_lane_query_idx
	ON tq_fast_lane_query_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (
		bits = 4,
		lists = 0,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

SET enable_seqscan = off;
SET enable_bitmapscan = off;

DO $$
BEGIN
	BEGIN
		PERFORM id
		FROM tq_fast_lane_query_docs
		ORDER BY embedding <=> '[2,0]'::vector(2)
		LIMIT 1;
	EXCEPTION
		WHEN OTHERS THEN
			RAISE NOTICE 'query_error=%', SQLERRM;
	END;
END;
$$;

SELECT
	meta #>> '{fast_lane,metric}' AS fast_lane_metric,
	(meta #>> '{fast_lane,strict_normalization}')::boolean AS strict_normalization,
	meta #>> '{fast_lane,fallback_reason}' AS fallback_reason
FROM (
	SELECT tq_index_metadata('tq_fast_lane_query_idx'::regclass) AS meta
) AS s;
