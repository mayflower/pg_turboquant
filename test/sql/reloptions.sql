DROP EXTENSION IF EXISTS pg_turboquant CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION vector;
CREATE EXTENSION pg_turboquant;

CREATE OPERATOR FAMILY tq_int4_test_fam USING turboquant;

CREATE OPERATOR CLASS tq_int4_test_ops
DEFAULT FOR TYPE int4 USING turboquant FAMILY tq_int4_test_fam AS
  OPERATOR 1 <(int4, int4),
  FUNCTION 1 btint4cmp(int4, int4);

CREATE TABLE tq_relopt_test (id int4);

CREATE INDEX tq_relopt_unknown_idx
ON tq_relopt_test
USING turboquant (id tq_int4_test_ops)
WITH (codec = 'prod');

CREATE INDEX tq_relopt_bits_idx
ON tq_relopt_test
USING turboquant (id tq_int4_test_ops)
WITH (bits = 1);

CREATE INDEX tq_relopt_transform_idx
ON tq_relopt_test
USING turboquant (id tq_int4_test_ops)
WITH (transform = 'dense');

CREATE INDEX tq_relopt_lanes_idx
ON tq_relopt_test
USING turboquant (id tq_int4_test_ops)
WITH (lanes = 8);

SELECT tq_debug_validate_reloptions(
  ARRAY['bits=4', 'lists=0', 'lanes=auto', 'transform=hadamard', 'normalized=true']
);
