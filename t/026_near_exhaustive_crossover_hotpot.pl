use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('near_exhaustive_crossover_hotpot');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

my $overlap_output_path = $node->data_dir . '/near-exhaustive-overlap.json';
my $skew_output_path = $node->data_dir . '/near-exhaustive-skew.json';

$node->command_ok(
	[
		'python3',
		'scripts/benchmark_suite.py',
		'--host=' . $node->host,
		'--port=' . $node->port,
		'--dbname=postgres',
		'--profile=tiny',
		'--corpus=hotpot_overlap',
		'--methods=turboquant_ivf',
		'--turboquant-probes=64',
		'--turboquant-max-visited-codes=0',
		'--turboquant-max-visited-pages=0',
		'--output=' . $overlap_output_path,
	],
	'benchmark suite emits near-exhaustive scan stats for the hotpot overlap profile'
);

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
		'--turboquant-probes=1',
		'--turboquant-max-visited-codes=0',
		'--turboquant-max-visited-pages=0',
		'--output=' . $skew_output_path,
	],
	'benchmark suite keeps the bounded-page path on the narrow hotpot skew profile'
);

my $overlap_payload = decode_json(slurp_file($overlap_output_path));
my $overlap = $overlap_payload->{scenarios}[0];
ok($overlap->{scan_stats}{near_exhaustive_crossover}, 'hotpot overlap activates the near-exhaustive crossover');
is(
	$overlap->{scan_stats}{scan_orchestration},
	'ivf_near_exhaustive',
	'hotpot overlap switches to the near-exhaustive scan orchestration'
);
cmp_ok(
	$overlap->{scan_stats}{visited_page_count},
	'==',
	$overlap->{scan_stats}{selected_page_count},
	'near-exhaustive scans visit every selected page'
);
cmp_ok($overlap->{metrics}{recall_at_10}, '>=', 0.06, 'near-exhaustive overlap path keeps the measured recall floor');

my $skew_payload = decode_json(slurp_file($skew_output_path));
my $skew = $skew_payload->{scenarios}[0];
ok(!$skew->{scan_stats}{near_exhaustive_crossover}, 'hotpot skew stays on the bounded-page path');
is(
	$skew->{scan_stats}{scan_orchestration},
	'ivf_bounded_pages',
	'hotpot skew reports the bounded-page scan orchestration'
);

$node->stop;

done_testing();
