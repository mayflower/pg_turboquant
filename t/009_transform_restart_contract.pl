use strict;
use warnings FATAL => 'all';

use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('transform_restart_contract');
$node->init;
$node->append_conf(
	'postgresql.conf', q{
fsync = on
full_page_writes = on
restart_after_crash = on
synchronous_commit = on
wal_level = replica
}
);
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant_test_support;');

$node->safe_psql(
	'postgres', q{
		CREATE TABLE tq_transform_restart_docs (
			id int4 PRIMARY KEY,
			embedding vector(5)
		);

		INSERT INTO tq_transform_restart_docs (id, embedding) VALUES
			(1, '[1,0,0,0,0]'),
			(2, '[0,1,0,0,0]'),
			(3, '[0,0,1,0,0]'),
			(4, '[0,0,0,1,0]');

		CREATE INDEX tq_transform_restart_idx
		ON tq_transform_restart_docs
		USING turboquant (embedding tq_cosine_ops)
		WITH (
			bits = 4,
			lists = 0,
			lanes = auto,
			transform = 'hadamard',
			normalized = true
		);
	}
);

my $metadata_before = $node->safe_psql(
	'postgres',
	q{SELECT tq_debug_transform_metadata('tq_transform_restart_idx'::regclass);}
);
my $query_before = $node->safe_psql(
	'postgres', q{
		SET enable_seqscan = off;
		SET enable_bitmapscan = off;
		SELECT string_agg(id::text, ',' ORDER BY ord)
		FROM (
			SELECT id, row_number() OVER () AS ord
			FROM (
				SELECT id
				FROM tq_transform_restart_docs
				ORDER BY embedding <=> '[1,0,0,0,0]'
				LIMIT 3
			) ranked
		) ordered;
	}
);

is(
	$metadata_before,
	'transform_version=1 input_dimension=5 output_dimension=8 seed=0',
	'transform metadata exposes explicit padded contract before restart'
);
is($query_before, '1,2,4', 'query order is deterministic before restart');

$node->restart;

is(
	$node->safe_psql(
		'postgres',
		q{SELECT tq_debug_transform_metadata('tq_transform_restart_idx'::regclass);}
	),
	$metadata_before,
	'transform metadata is stable across restart'
);
is(
	$node->safe_psql(
		'postgres', q{
			SET enable_seqscan = off;
			SET enable_bitmapscan = off;
			SELECT string_agg(id::text, ',' ORDER BY ord)
			FROM (
				SELECT id, row_number() OVER () AS ord
				FROM (
					SELECT id
					FROM tq_transform_restart_docs
					ORDER BY embedding <=> '[1,0,0,0,0]'
					LIMIT 3
				) ranked
			) ordered;
		}
	),
	$query_before,
	'query order is stable across restart'
);

$node->stop;

done_testing();
