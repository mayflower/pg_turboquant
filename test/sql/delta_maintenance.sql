SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE TABLE tq_delta_docs (
	passage_id int4 PRIMARY KEY,
	tenant_id int4 NOT NULL,
	doc_version int4 NOT NULL,
	embedding vector(4)
);

INSERT INTO tq_delta_docs (passage_id, tenant_id, doc_version, embedding) VALUES
	(1, 1, 1, '[0.98,0.02,0,0]'),
	(2, 1, 1, '[0.95,0.05,0,0]'),
	(3, 2, 1, '[0,1,0,0]'),
	(4, 2, 1, '[0,0.95,0.05,0]');

CREATE INDEX tq_delta_docs_idx
	ON tq_delta_docs
	USING turboquant (
		embedding tq_cosine_ops,
		tenant_id tq_int4_filter_ops
	)
	INCLUDE (doc_version)
	WITH (
		bits = 4,
		lists = 2,
		lanes = auto,
		transform = 'hadamard',
		normalized = true
	);

VACUUM (FREEZE, ANALYZE) tq_delta_docs;

INSERT INTO tq_delta_docs (passage_id, tenant_id, doc_version, embedding) VALUES
	(5, 1, 2, '[1,0,0,0]');

SET enable_seqscan = off;
SET enable_bitmapscan = off;
SET turboquant.probes = 1;

SELECT passage_id, tenant_id, doc_version
FROM tq_delta_docs
WHERE tenant_id = 1
ORDER BY embedding <=> '[1,0,0,0]'::vector(4)
LIMIT 2;

SELECT
	(meta->>'delta_enabled')::boolean AS delta_enabled,
	(meta->>'delta_live_count')::int AS delta_live_count,
	(meta->>'delta_batch_page_count')::int AS delta_batch_page_count,
	(meta->>'delta_head_block')::int AS delta_head_block,
	(meta->>'delta_tail_block')::int AS delta_tail_block,
	(meta->>'maintenance_required')::boolean AS maintenance_required,
	meta #>> '{delta_health,page_depth}' AS delta_page_depth,
	meta #>> '{delta_health,merge_recommended}' AS delta_merge_recommended,
	meta #>> '{maintenance,dead_fraction}' AS dead_fraction,
	meta #>> '{maintenance,compaction_recommended}' AS compaction_recommended,
	meta->>'maintenance_action_recommended' AS maintenance_action_recommended
FROM (
	SELECT tq_index_metadata('tq_delta_docs_idx'::regclass) AS meta
) AS s;

SELECT
	result->>'action' AS action,
	(result->>'delta_merge_performed')::boolean AS delta_merge_performed,
	(result->>'compaction_pass_performed')::boolean AS compaction_pass_performed,
	(result->>'merged_delta_count')::int AS merged_delta_count,
	(result->>'rewritten_list_count')::int AS rewritten_list_count,
	(result->>'recycled_delta_page_count')::int AS recycled_delta_page_count,
	(result->>'post_maintenance_required')::boolean AS post_maintenance_required
FROM (
	SELECT tq_maintain_index('tq_delta_docs_idx'::regclass) AS result
) AS s;

SELECT
	(meta->>'delta_live_count')::int AS delta_live_count,
	(meta->>'delta_batch_page_count')::int AS delta_batch_page_count,
	(meta->>'delta_head_block')::int AS delta_head_block,
	(meta->>'delta_tail_block')::int AS delta_tail_block,
	(meta->>'live_count')::int AS live_count,
	(meta->>'maintenance_required')::boolean AS maintenance_required,
	meta #>> '{delta_health,merge_recommended}' AS delta_merge_recommended,
	meta #>> '{maintenance,compaction_recommended}' AS compaction_recommended,
	meta->>'maintenance_action_recommended' AS maintenance_action_recommended
FROM (
	SELECT tq_index_metadata('tq_delta_docs_idx'::regclass) AS meta
) AS s;
