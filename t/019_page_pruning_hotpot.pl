use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('page_pruning_hotpot');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

my $output_path = $node->data_dir . '/page-pruning-hotpot.json';

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
	'benchmark suite emits page-pruning stats for the hotpot skew profile'
);

my $payload = decode_json(slurp_file($output_path));
my $scenario = $payload->{scenarios}[0];

cmp_ok($scenario->{scan_stats}{page_prune_count}, '>', 0, 'page pruning is reported');
cmp_ok($scenario->{scan_stats}{early_stop_count}, '>', 0, 'early stop is reported');
cmp_ok(
	$scenario->{scan_stats}{visited_code_count},
	'<',
	$scenario->{scan_stats}{selected_live_count},
	'visited codes stay below selected live count'
);
cmp_ok($scenario->{metrics}{recall_at_10}, '>=', 0.90, 'recall floor is preserved');

$node->stop;

done_testing();
