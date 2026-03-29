use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('scan_stats_hotpot_baseline');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

my $output_path = $node->data_dir . '/scan-stats-hotpot.json';

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
		'--output=' . $output_path,
	],
	'benchmark suite emits scan stats for the skewed hotpot baseline'
);

my $payload = decode_json(slurp_file($output_path));
is(scalar @{$payload->{scenarios}}, 1, 'one hotpot baseline scenario recorded');

my $scenario = $payload->{scenarios}[0];
ok(exists $scenario->{scan_stats}, 'scenario reports scan stats');
ok(exists $scenario->{scan_stats}{selected_list_count}, 'selected list count is present');
ok(exists $scenario->{scan_stats}{selected_live_count}, 'selected live count is present');
ok(exists $scenario->{scan_stats}{visited_page_count}, 'visited page count is present');
ok(exists $scenario->{scan_stats}{visited_code_count}, 'visited code count is present');
ok(exists $scenario->{scan_stats}{candidate_heap_count}, 'candidate heap count is present');
ok(exists $scenario->{scan_stats}{score_mode}, 'score mode is present');
cmp_ok(
	$scenario->{scan_stats}{visited_code_count},
	'>',
	$scenario->{scan_stats}{candidate_heap_count},
	'visited code count exceeds retained candidates'
);

$node->stop;

done_testing();
