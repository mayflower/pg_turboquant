use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('adaptive_probing_hotpot');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

my $output_path = $node->data_dir . '/adaptive-probing-hotpot.json';

$node->command_ok(
	[
		'python3',
		'scripts/benchmark_suite.py',
		'--host=' . $node->host,
		'--port=' . $node->port,
		'--dbname=postgres',
		'--profile=tiny',
		'--corpus=hotpot_skewed',
		'--methods=turboquant_ivf',
		'--turboquant-probes=4',
		'--turboquant-max-visited-codes=256',
		'--output=' . $output_path,
	],
	'benchmark suite emits adaptive probing stats for the hotpot skew profile'
);

my $payload = decode_json(slurp_file($output_path));
my $scenario = $payload->{scenarios}[0];

cmp_ok(
	$scenario->{scan_stats}{effective_probe_count},
	'<=',
	$scenario->{scan_stats}{nominal_probe_count},
	'effective probes do not exceed nominal probes'
);
cmp_ok(
	$scenario->{scan_stats}{visited_code_count},
	'<=',
	$scenario->{scan_stats}{max_visited_codes} + 64,
	'visited codes respect the configured code budget with first-list tolerance'
);
cmp_ok($scenario->{metrics}{recall_at_10}, '>=', 0.90, 'recall floor is preserved');
is($scenario->{query_knobs}{'turboquant.probes'}, 4, 'benchmark JSON records nominal probes');
ok(exists $scenario->{scan_stats}{effective_probe_count}, 'benchmark JSON records effective probes');

$node->stop;

done_testing();
