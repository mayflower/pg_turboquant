use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('ext_upgrade');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');

$node->safe_psql(
	'postgres', q{
		CREATE EXTENSION pg_turboquant VERSION '0.1.0';
		CREATE TABLE tq_upgrade_docs (
			id int4 PRIMARY KEY,
			embedding vector(4)
		);
		INSERT INTO tq_upgrade_docs (id, embedding) VALUES
			(1, '[1,0,0,0]'),
			(2, '[0,1,0,0]'),
			(3, '[0,0,1,0]');
		CREATE INDEX tq_upgrade_idx
			ON tq_upgrade_docs
			USING turboquant (embedding tq_cosine_ops)
			WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = true);
	}
);

$node->safe_psql('postgres', q{ALTER EXTENSION pg_turboquant UPDATE TO '0.1.1';});
$node->safe_psql('postgres', q{ALTER EXTENSION pg_turboquant UPDATE TO '0.1.2';});
$node->safe_psql('postgres', q{ALTER EXTENSION pg_turboquant UPDATE TO '0.1.3';});
$node->safe_psql('postgres', q{ALTER EXTENSION pg_turboquant UPDATE TO '0.1.4';});

is(
	$node->safe_psql(
		'postgres',
		q{SELECT extversion FROM pg_extension WHERE extname = 'pg_turboquant';}
	),
	'0.1.4',
	'extension upgrade reaches 0.1.4'
);

my $meta = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_upgrade_idx'::regclass)::text;})
);

is($meta->{format_version}, 4, 'upgraded extension preserves readable index metadata');
ok(exists $meta->{capabilities}, 'upgraded extension exposes capability metadata');
is($meta->{capabilities}{index_only_scan}, JSON::PP::false, 'capability metadata marks index-only scans unsupported');
my $simd = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_runtime_simd_features()::text;})
);
ok(exists $simd->{preferred_kernel}, 'upgraded extension exposes SIMD runtime metadata');
is(
	$node->safe_psql(
		'postgres', q{
			SET enable_seqscan = off;
			SET enable_bitmapscan = off;
			SELECT id
			FROM tq_upgrade_docs
			ORDER BY embedding <=> '[1,0,0,0]'::vector(4), id
			LIMIT 1;
		}
	),
	'1',
	'upgraded extension keeps existing index query-correct'
);

$node->safe_psql(
	'postgres', q{
		DROP EXTENSION pg_turboquant CASCADE;
		CREATE EXTENSION pg_turboquant;
	}
);

is(
	$node->safe_psql(
		'postgres',
		q{SELECT extversion FROM pg_extension WHERE extname = 'pg_turboquant';}
	),
	'0.1.4',
	'fresh install uses the current default extension version'
);

$node->stop;

done_testing();
