SET client_min_messages = warning;
DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

SET enable_bitmapscan = off;

CREATE TABLE tq_metric_normalized_docs (
	id int4 PRIMARY KEY,
	embedding vector(2)
);

INSERT INTO tq_metric_normalized_docs (id, embedding) VALUES
	(1, '[1,0]'),
	(2, '[0.70710678,0.70710678]'),
	(3, '[0,1]');

CREATE INDEX tq_metric_normalized_cosine_idx
	ON tq_metric_normalized_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = true);

CREATE INDEX tq_metric_normalized_ip_idx
	ON tq_metric_normalized_docs
	USING turboquant (embedding tq_ip_ops)
	WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = true);

CREATE INDEX tq_metric_normalized_l2_idx
	ON tq_metric_normalized_docs
	USING turboquant (embedding tq_l2_ops)
	WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = true);

SET enable_indexscan = off;
SET enable_seqscan = on;

SELECT array_agg(id) AS exact_normalized_cosine
FROM (
	SELECT id
	FROM tq_metric_normalized_docs
	ORDER BY embedding <=> '[1,0]'
	LIMIT 3
) ranked;

SELECT array_agg(id) AS exact_normalized_ip
FROM (
	SELECT id
	FROM tq_metric_normalized_docs
	ORDER BY embedding <#> '[1,0]'
	LIMIT 3
) ranked;

SELECT array_agg(id) AS exact_normalized_l2
FROM (
	SELECT id
	FROM tq_metric_normalized_docs
	ORDER BY embedding <-> '[1,0]'
	LIMIT 3
) ranked;

SET enable_indexscan = on;
SET enable_seqscan = off;

SELECT array_agg(id) AS approx_normalized_cosine
FROM (
	SELECT id
	FROM tq_metric_normalized_docs
	ORDER BY embedding <=> '[1,0]'
	LIMIT 3
) ranked;

SELECT array_agg(id) AS approx_normalized_ip
FROM (
	SELECT id
	FROM tq_metric_normalized_docs
	ORDER BY embedding <#> '[1,0]'
	LIMIT 3
) ranked;

SELECT array_agg(id) AS approx_normalized_l2
FROM (
	SELECT id
	FROM tq_metric_normalized_docs
	ORDER BY embedding <-> '[1,0]'
	LIMIT 3
) ranked;

CREATE TABLE tq_metric_cosine_docs (
	id int4 PRIMARY KEY,
	embedding vector(2)
);

INSERT INTO tq_metric_cosine_docs (id, embedding) VALUES
	(1, '[0.8,0.6]'),
	(2, '[3,4]'),
	(3, '[0,1]');

CREATE INDEX tq_metric_cosine_idx
	ON tq_metric_cosine_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = false);

SET enable_indexscan = off;
SET enable_seqscan = on;

SELECT array_agg(id) AS exact_non_normalized_cosine
FROM (
	SELECT id
	FROM tq_metric_cosine_docs
	ORDER BY embedding <=> '[1,0]'
	LIMIT 3
) ranked;

SET enable_indexscan = on;
SET enable_seqscan = off;

SELECT array_agg(id) AS approx_non_normalized_cosine
FROM (
	SELECT id
	FROM tq_metric_cosine_docs
	ORDER BY embedding <=> '[1,0]'
	LIMIT 3
) ranked;

CREATE TABLE tq_metric_ip_docs (
	id int4 PRIMARY KEY,
	embedding vector(2)
);

INSERT INTO tq_metric_ip_docs (id, embedding) VALUES
	(1, '[0.8,0.6]'),
	(2, '[3,4]'),
	(3, '[0,1]');

CREATE INDEX tq_metric_ip_idx
	ON tq_metric_ip_docs
	USING turboquant (embedding tq_ip_ops)
	WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = false);

SET enable_indexscan = off;
SET enable_seqscan = on;

SELECT array_agg(id) AS exact_non_normalized_ip
FROM (
	SELECT id
	FROM tq_metric_ip_docs
	ORDER BY embedding <#> '[1,0]'
	LIMIT 3
) ranked;

SET enable_indexscan = on;
SET enable_seqscan = off;

SELECT array_agg(id) AS approx_non_normalized_ip
FROM (
	SELECT id
	FROM tq_metric_ip_docs
	ORDER BY embedding <#> '[1,0]'
	LIMIT 3
) ranked;

CREATE TABLE tq_metric_l2_docs (
	id int4 PRIMARY KEY,
	embedding vector(2)
);

INSERT INTO tq_metric_l2_docs (id, embedding) VALUES
	(1, '[1.1,0]'),
	(2, '[3,0]'),
	(3, '[0,2]');

CREATE INDEX tq_metric_l2_idx
	ON tq_metric_l2_docs
	USING turboquant (embedding tq_l2_ops)
	WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = false);

SET enable_indexscan = off;
SET enable_seqscan = on;

SELECT array_agg(id) AS exact_non_normalized_l2
FROM (
	SELECT id
	FROM tq_metric_l2_docs
	ORDER BY embedding <-> '[1,0]'
	LIMIT 3
) ranked;

SET enable_indexscan = on;
SET enable_seqscan = off;

SELECT array_agg(id) AS approx_non_normalized_l2
FROM (
	SELECT id
	FROM tq_metric_l2_docs
	ORDER BY embedding <-> '[1,0]'
	LIMIT 3
) ranked;
