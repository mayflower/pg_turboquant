use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('flat_streaming_medium');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

my $output_path = $node->data_dir . '/flat-streaming-medium.json';

$node->command_ok(
	[
		'python3',
		'scripts/benchmark_suite.py',
		'--host=' . $node->host,
		'--port=' . $node->port,
		'--dbname=postgres',
		'--profile=medium',
		'--corpus=normalized_dense',
		'--methods=turboquant_flat',
		'--output=' . $output_path
	],
	'medium benchmark completes for turboquant flat'
);

my $payload = decode_json(slurp_file($output_path));
my $scenario = $payload->{scenarios}[0];

is($scenario->{corpus}, 'normalized_dense', 'normalized_dense benchmark scenario reported');
is($scenario->{method}, 'turboquant_flat', 'turboquant flat benchmark scenario reported');
is($scenario->{metrics}{candidate_slots_bound}, 16, 'flat streaming benchmark reports bounded candidate slots');
cmp_ok($scenario->{metrics}{p95_ms}, '<', 250.0, 'medium flat benchmark stays within latency guardrail');

$node->stop;

done_testing();
