use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('prod_code_domain_simd_smoke');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

my $quick_output = $node->data_dir . '/prod-code-domain-simd-quick.json';
my $tiny_output = $node->data_dir . '/prod-code-domain-simd-tiny.json';

$node->command_ok(
	[
		'python3',
		'scripts/benchmark_suite.py',
		'--host=' . $node->host,
		'--port=' . $node->port,
		'--dbname=postgres',
		'--profile=quick',
		'--corpus=hotpot_skewed',
		'--methods=turboquant_ivf',
		'--output=' . $quick_output,
	],
	'benchmark suite runs the supported-shape SIMD scenario'
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
		'--output=' . $tiny_output,
	],
	'benchmark suite runs the unsupported-shape fallback scenario'
);

my $quick_payload = decode_json(slurp_file($quick_output));
my $tiny_payload = decode_json(slurp_file($tiny_output));
my $quick = $quick_payload->{scenarios}[0];
my $tiny = $tiny_payload->{scenarios}[0];
my $avx2_runtime = $quick->{simd}{runtime_available}{avx2} ? 1 : 0;
my $neon_runtime = $quick->{simd}{runtime_available}{neon} ? 1 : 0;
my $expected_supported_kernel = $avx2_runtime ? 'avx2' : $neon_runtime ? 'neon' : 'scalar';

ok(exists $quick->{scan_stats}{score_kernel}, 'benchmark JSON includes selected score kernel');
ok(exists $quick->{simd}{code_domain_kernel}, 'benchmark JSON includes code-domain kernel metadata');
is($quick->{simd}{code_domain_kernel}, $quick->{scan_stats}{score_kernel}, 'benchmark metadata matches scan stats for supported shape');
is($quick->{scan_stats}{score_mode}, 'code_domain', 'supported shape still uses code-domain scoring');
is(
	$quick->{scan_stats}{score_kernel},
	$expected_supported_kernel,
	'supported shape uses the best available code-domain SIMD kernel and otherwise reports scalar fallback'
);

is($tiny->{scan_stats}{score_mode}, 'code_domain', 'unsupported shape still uses code-domain scoring');
is($tiny->{scan_stats}{score_kernel}, 'scalar', 'unsupported shape falls back to scalar');
is($tiny->{simd}{code_domain_kernel}, 'scalar', 'benchmark metadata reports scalar fallback for unsupported shape');

$node->stop;

done_testing();
