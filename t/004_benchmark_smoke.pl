use strict;
use warnings FATAL => 'all';

use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('benchmark_smoke');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

my $output_path = $node->data_dir . '/benchmark-smoke.json';

$node->command_ok(
	[
		'python3',
		'scripts/benchmark_suite.py',
		'--host=' . $node->host,
		'--port=' . $node->port,
		'--dbname=postgres',
		'--profile=tiny',
		'--corpus=normalized_dense',
		'--methods=turboquant_flat',
		'--output=' . $output_path
	],
	'benchmark suite smoke run completes'
);

my $json = slurp_file($output_path);

like($json, qr/"profile"\s*:\s*"tiny"/, 'benchmark output reports profile');
like($json, qr/"recall_at_10"\s*:/, 'benchmark output reports recall');
like($json, qr/"p50_ms"\s*:/, 'benchmark output reports latency');
like($json, qr/"index_size_bytes"\s*:/, 'benchmark output reports index footprint');

$node->stop;

done_testing();
