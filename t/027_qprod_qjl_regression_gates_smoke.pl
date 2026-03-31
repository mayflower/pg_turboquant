use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Utils;
use Test::More;

my $output_path = 'tmp/qprod-qjl-regression-gates.json';

command_ok(
	[
		'python3',
		'scripts/benchmark_suite.py',
		'--dry-run',
		'--profile=tiny',
		'--corpus=normalized_dense',
		'--methods=turboquant_flat',
		'--microbench',
		'--report',
		'--output=' . $output_path,
	],
	'qprod/qjl benchmark suite emits regression-gate output in dry-run mode'
);

ok(-e $output_path, 'benchmark suite JSON output file exists');

my $payload = decode_json(slurp_file($output_path));
ok(exists $payload->{microbenchmarks}, 'payload reports microbenchmark section');
ok(exists $payload->{report}, 'payload reports benchmark report section');
ok(exists $payload->{microbenchmarks}->{comparisons}, 'payload reports microbenchmark comparisons');
ok(exists $payload->{microbenchmarks}->{regression_gates}, 'payload reports microbenchmark regression gates');
ok(exists $payload->{microbenchmarks}->{interpretation_notes}, 'payload reports microbenchmark interpretation notes');
cmp_ok(scalar @{$payload->{microbenchmarks}->{comparisons}}, '>=', 4,
	   'payload includes the expected microbenchmark comparison rows');
cmp_ok(scalar @{$payload->{microbenchmarks}->{regression_gates}}, '>=', 4,
	   'payload includes the expected regression gate rows');

my %comparisons = map { $_->{comparison} => $_ } @{$payload->{microbenchmarks}->{comparisons}};
ok(exists $comparisons{score_code_from_lut_avx2_vs_scalar},
   'payload includes avx2-vs-scalar comparison');
ok(exists $comparisons{score_code_from_lut_neon_vs_scalar},
   'payload includes neon-vs-scalar comparison');
ok(exists $comparisons{qjl_lut_quantized_vs_float_reference},
   'payload includes quantized-vs-float LUT comparison');
ok(exists $comparisons{page_scan_block_local_vs_global_heap},
   'payload includes block-local-vs-global-heap comparison');

ok(exists $comparisons{page_scan_block_local_vs_global_heap}->{metrics}->{codes_per_second_ratio},
   'block-local selection comparison reports throughput ratio');
ok(exists $comparisons{page_scan_block_local_vs_global_heap}->{metrics}->{candidate_heap_insert_delta},
   'block-local selection comparison reports global heap insert delta');
ok(exists $comparisons{qjl_lut_quantized_vs_float_reference}->{metrics}->{codes_per_second_ratio},
   'quantized LUT comparison reports throughput ratio');

my %gates = map { $_->{gate} => $_ } @{$payload->{microbenchmarks}->{regression_gates}};
ok(exists $gates{avx2_kernel_speedup_signal}, 'payload includes avx2 gate');
ok(exists $gates{neon_kernel_speedup_signal}, 'payload includes neon gate');
ok(exists $gates{quantized_qjl_lut_signal}, 'payload includes quantized LUT gate');
ok(exists $gates{block_local_selection_signal}, 'payload includes block-local selection gate');
ok($gates{block_local_selection_signal}->{status} eq 'pass'
   || $gates{block_local_selection_signal}->{status} eq 'warn'
   || $gates{block_local_selection_signal}->{status} eq 'not_applicable',
   'block-local selection gate reports a stable status enum');
ok(exists $gates{block_local_selection_signal}->{checks}->{throughput_directional_signal},
   'block-local selection gate reports directional throughput check');

ok(exists $payload->{report}->{microbenchmark_regression},
   'report includes microbenchmark regression summary');
ok(exists $payload->{report}->{microbenchmark_regression}->{comparisons},
   'report includes microbenchmark comparison summary');
ok(exists $payload->{report}->{microbenchmark_regression}->{regression_gates},
   'report includes regression gate summary');

my $report_json = 'tmp/benchmark-report.json';
my $report_md = 'tmp/benchmark-report.md';
my $report_html = 'tmp/benchmark-report.html';
ok(-e $report_json, 'report JSON artifact exists');
ok(-e $report_md, 'report markdown artifact exists');
ok(-e $report_html, 'report HTML artifact exists');

my $markdown = slurp_file($report_md);
my $html = slurp_file($report_html);
like($markdown, qr/Microbenchmark Regression/, 'report markdown includes microbenchmark regression section');
like($markdown, qr/block_local_selection_signal/, 'report markdown includes block-local gate');
like($html, qr/Microbenchmark Regression/, 'report html includes microbenchmark regression section');
like($html, qr/block_local_selection_signal/, 'report html includes block-local gate');

done_testing();
