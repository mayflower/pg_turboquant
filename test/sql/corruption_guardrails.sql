\set VERBOSITY terse
SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant_test_support CASCADE;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
RESET client_min_messages;

CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;
CREATE EXTENSION pg_turboquant_test_support;

CREATE TABLE tq_corrupt_docs (
	id int4 PRIMARY KEY,
	embedding vector(4)
);

INSERT INTO tq_corrupt_docs (id, embedding) VALUES
	(1, '[1,0,0,0]'),
	(2, '[0,1,0,0]'),
	(3, '[0,0,1,0]'),
	(4, '[0,0,0,1]');

CREATE INDEX tq_corrupt_meta_magic_idx
	ON tq_corrupt_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 1, lanes = auto, transform = 'hadamard', normalized = true);

SELECT tq_test_corrupt_meta_magic('tq_corrupt_meta_magic_idx'::regclass);
SELECT tq_index_metadata('tq_corrupt_meta_magic_idx'::regclass);

DROP INDEX tq_corrupt_meta_magic_idx;

CREATE INDEX tq_corrupt_meta_version_idx
	ON tq_corrupt_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 1, lanes = auto, transform = 'hadamard', normalized = true);

SELECT tq_test_corrupt_meta_format_version('tq_corrupt_meta_version_idx'::regclass, 999);
SELECT tq_index_metadata('tq_corrupt_meta_version_idx'::regclass);

DROP INDEX tq_corrupt_meta_version_idx;

CREATE INDEX tq_corrupt_transform_idx
	ON tq_corrupt_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 1, lanes = auto, transform = 'hadamard', normalized = true);

SELECT tq_test_corrupt_meta_transform_contract('tq_corrupt_transform_idx'::regclass);
SELECT tq_index_metadata('tq_corrupt_transform_idx'::regclass);

DROP INDEX tq_corrupt_transform_idx;

CREATE INDEX tq_corrupt_batch_idx
	ON tq_corrupt_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 1, lanes = auto, transform = 'hadamard', normalized = true);

SELECT tq_test_corrupt_first_batch_occupied_count('tq_corrupt_batch_idx'::regclass);
SELECT tq_index_metadata('tq_corrupt_batch_idx'::regclass);

DROP INDEX tq_corrupt_batch_idx;

CREATE INDEX tq_corrupt_list_idx
	ON tq_corrupt_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 1, lanes = auto, transform = 'hadamard', normalized = true);

SELECT tq_test_corrupt_first_list_head_to_directory_root('tq_corrupt_list_idx'::regclass);
SELECT tq_index_metadata('tq_corrupt_list_idx'::regclass);

DROP EXTENSION pg_turboquant_test_support;
