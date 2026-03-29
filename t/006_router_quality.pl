use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('router_quality_ci');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

my $output_path = $node->data_dir . '/router-quality.json';

$node->command_ok(
	[
		'python3',
		'scripts/benchmark_suite.py',
		'--host=' . $node->host,
		'--port=' . $node->port,
		'--dbname=postgres',
		'--profile=tiny',
		'--corpus=clustered',
		'--methods=turboquant_ivf',
		'--output=' . $output_path
	],
	'clustered benchmark completes for turboquant ivf'
);

my $payload = decode_json(slurp_file($output_path));
my $scenario = $payload->{scenarios}[0];

is($scenario->{corpus}, 'clustered', 'clustered benchmark scenario reported');
is($scenario->{method}, 'turboquant_ivf', 'turboquant ivf benchmark scenario reported');
cmp_ok($scenario->{metrics}{recall_at_10}, '>=', 0.56, 'clustered ivf recall_at_10 stays above regression floor');
cmp_ok($scenario->{metrics}{recall_at_100}, '>=', 0.16, 'clustered ivf recall_at_100 stays above regression floor');

$node->stop;

done_testing();
