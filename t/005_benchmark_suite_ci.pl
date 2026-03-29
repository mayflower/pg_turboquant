use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('benchmark_suite_ci');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

my $output_path = $node->data_dir . '/benchmark-suite.json';

$node->command_ok(
	[
		'python3',
		'scripts/benchmark_suite.py',
		'--host=' . $node->host,
		'--port=' . $node->port,
		'--dbname=postgres',
		'--profile=tiny',
		'--corpus=normalized_dense',
		'--methods=turboquant_flat,pgvector_ivfflat',
		'--output=' . $output_path
	],
	'benchmark suite script completes for ci-sized scenario'
);

my $file_json = slurp_file($output_path);

my $file_payload = decode_json($file_json);

is($file_payload->{profile}, 'tiny', 'benchmark suite reports tiny profile');
is_deeply(
	$file_payload->{methods},
	['turboquant_flat', 'pgvector_ivfflat'],
	'benchmark suite reports compared methods'
);
ok(@{$file_payload->{scenarios}} == 2, 'benchmark suite emits two scenario results');
ok(exists $file_payload->{scenarios}[0]{metrics}{recall_at_10}, 'scenario reports recall_at_10');
ok(exists $file_payload->{scenarios}[0]{metrics}{p50_ms}, 'scenario reports p50 latency');
ok(exists $file_payload->{scenarios}[0]{metrics}{p95_ms}, 'scenario reports p95 latency');
ok(exists $file_payload->{scenarios}[0]{metrics}{build_seconds}, 'scenario reports build time');
ok(exists $file_payload->{scenarios}[0]{metrics}{index_size_bytes}, 'scenario reports index size');
ok(exists $file_payload->{scenarios}[0]{metrics}{concurrent_insert_rows_per_second}, 'scenario reports concurrent insert throughput');
ok(exists $file_payload->{scenarios}[0]{index_metadata}, 'scenario reports attached index metadata');
ok(exists $file_payload->{scenarios}[0]{index_metadata}{metric}, 'index metadata reports metric');

$node->stop;

done_testing();
