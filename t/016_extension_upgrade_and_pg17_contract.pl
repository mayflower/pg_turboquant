use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('ext_pg17_contract');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');

$node->safe_psql(
	'postgres', q{
		CREATE EXTENSION pg_turboquant;
		CREATE TABLE tq_docs (
			id int4 PRIMARY KEY,
			embedding vector(4)
		);
		INSERT INTO tq_docs (id, embedding) VALUES
			(1, '[1,0,0,0]'),
			(2, '[0,1,0,0]'),
			(3, '[0,0,1,0]');
		CREATE INDEX tq_idx
			ON tq_docs
			USING turboquant (embedding tq_cosine_ops)
			WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = true);
	}
);

is(
	$node->safe_psql(
		'postgres',
		q{SELECT extversion FROM pg_extension WHERE extname = 'pg_turboquant';}
	),
	'0.1.0',
	'fresh install uses the public extension version'
);

my $meta = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_idx'::regclass)::text;})
);

is($meta->{format_version}, 12, 'extension exposes readable index metadata');
ok(exists $meta->{capabilities}, 'extension exposes capability metadata');
is($meta->{capabilities}{index_only_scan}, JSON::PP::false, 'capability metadata marks index-only scans unsupported');
my $simd = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_runtime_simd_features()::text;})
);
ok(exists $simd->{preferred_kernel}, 'extension exposes SIMD runtime metadata');
is(
	$node->safe_psql(
		'postgres', q{
			SET enable_seqscan = off;
			SET enable_bitmapscan = off;
			SELECT id
			FROM tq_docs
			ORDER BY embedding <=> '[1,0,0,0]'::vector(4), id
			LIMIT 1;
		}
	),
	'1',
	'extension keeps index query-correct'
);

$node->stop;

done_testing();
