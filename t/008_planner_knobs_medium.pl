use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('planner_knobs_medium');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

my $low_output_path = $node->data_dir . '/planner-knobs-low.json';
my $high_output_path = $node->data_dir . '/planner-knobs-high.json';

$node->command_ok(
	[
		'python3',
		'scripts/benchmark_suite.py',
		'--host=' . $node->host,
		'--port=' . $node->port,
		'--dbname=postgres',
		'--profile=medium',
		'--corpus=normalized_dense',
		'--methods=turboquant_ivf',
		'--turboquant-probes=1',
		'--output=' . $low_output_path
	],
	'medium benchmark completes for low-probe turboquant ivf'
);

$node->command_ok(
	[
		'python3',
		'scripts/benchmark_suite.py',
		'--host=' . $node->host,
		'--port=' . $node->port,
		'--dbname=postgres',
		'--profile=medium',
		'--corpus=normalized_dense',
		'--methods=turboquant_ivf',
		'--turboquant-probes=4',
		'--output=' . $high_output_path
	],
	'medium benchmark completes for high-probe turboquant ivf'
);

my $low = decode_json(slurp_file($low_output_path))->{scenarios}[0];
my $high = decode_json(slurp_file($high_output_path))->{scenarios}[0];

is($low->{query_knobs}{'turboquant.probes'}, 1, 'low-probe scenario reports probes');
is($high->{query_knobs}{'turboquant.probes'}, 4, 'high-probe scenario reports probes');
cmp_ok($high->{metrics}{candidate_slots_bound}, '>', $low->{metrics}{candidate_slots_bound}, 'higher probes raise bounded candidate window');
cmp_ok($high->{metrics}{recall_at_10}, '>=', $low->{metrics}{recall_at_10}, 'higher probes do not reduce recall_at_10');

$node->stop;

done_testing();
