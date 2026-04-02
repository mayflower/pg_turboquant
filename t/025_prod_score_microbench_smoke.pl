use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Utils;
use Test::More;

my $output_path = 'tmp/prod-score-microbench.json';

command_ok(
	[
		'python3',
		'scripts/prod_score_microbench.py',
		'--output=' . $output_path,
	],
	'prod score microbenchmark helper emits JSON output'
);

ok(-e $output_path, 'microbenchmark JSON output file exists');

my $payload = decode_json(slurp_file($output_path));
ok(exists $payload->{architecture}, 'microbenchmark payload reports architecture');
ok(exists $payload->{simd}, 'microbenchmark payload reports simd metadata');
ok(exists $payload->{results}, 'microbenchmark payload reports result rows');
cmp_ok(scalar @{$payload->{results}}, '>=', 2, 'microbenchmark payload contains multiple result rows');

for my $row (@{$payload->{results}})
{
	ok(exists $row->{lookup_style}, "result row '$row->{benchmark}' reports lookup_style");
	ok(exists $row->{block_width}, "result row '$row->{benchmark}' reports block_width");
	ok(exists $row->{qjl_path}, "result row '$row->{benchmark}' reports qjl_path");
	ok(exists $row->{gamma_path}, "result row '$row->{benchmark}' reports gamma_path");
}

my %kernels = map { $_->{kernel} => 1 } @{$payload->{results}};
ok($kernels{scalar}, 'microbenchmark payload includes scalar kernel timing');
ok($kernels{avx2} || $kernels{neon} || $kernels{scalar}, 'microbenchmark payload reports an available execution kernel');

my ($page_scan) = grep { $_->{benchmark} eq 'page_scan' } @{$payload->{results}};
my ($router_full_sort) = grep { $_->{benchmark} eq 'router_top_probes_full_sort' } @{$payload->{results}};
my ($router_partial) = grep { $_->{benchmark} eq 'router_top_probes_partial' } @{$payload->{results}};
my ($lut16_reference) = grep { $_->{benchmark} eq 'score_lut16_reference' } @{$payload->{results}};
ok(defined $lut16_reference, 'microbenchmark payload includes lut16 reference timing');
is($lut16_reference->{lookup_style}, 'lut16_scalar',
   'lut16 reference row reports lut16_scalar lookup style');
is($lut16_reference->{block_width}, 1, 'lut16 reference row reports block_width of 1');
is($lut16_reference->{gamma_path}, 'float32_scalar',
   'lut16 reference row reports float32_scalar gamma path');

my ($lut16_dispatch) = grep { $_->{benchmark} eq 'score_lut16_dispatch' } @{$payload->{results}};
ok(defined $lut16_dispatch, 'microbenchmark payload includes lut16 dispatch timing');
ok($lut16_dispatch->{lookup_style} =~ /^lut16_/,
   'lut16 dispatch row reports a lut16 lookup style');
ok($lut16_dispatch->{kernel} eq 'scalar' || $lut16_dispatch->{kernel} eq 'avx2' || $lut16_dispatch->{kernel} eq 'neon',
   'lut16 dispatch row reports a valid kernel');

my ($lut16_quantized_fused) = grep { $_->{benchmark} eq 'score_lut16_quantized_fused' } @{$payload->{results}};
ok(defined $lut16_quantized_fused, 'microbenchmark payload includes lut16 quantized fused timing');
is($lut16_quantized_fused->{qjl_path}, 'lut16_quantized',
   'quantized fused row reports lut16_quantized qjl path');
is($lut16_quantized_fused->{qjl_lut_mode}, 'quantized',
   'quantized fused row reports quantized qjl lut mode');

my ($transpose_only) = grep { $_->{benchmark} eq 'page_transpose_only' } @{$payload->{results}};
my ($transpose_block16) = grep { $_->{benchmark} eq 'page_transpose_block16_scalar' } @{$payload->{results}};
ok(defined $transpose_only, 'microbenchmark payload includes page-transpose-only timing');
ok(defined $transpose_block16, 'microbenchmark payload includes page-transpose+block16 timing');
is($transpose_only->{scan_layout}, 'scratch_transposed_block16',
   'transpose-only row reports scratch_transposed_block16 layout');
is($transpose_block16->{scan_layout}, 'scratch_transposed_block16',
   'transpose+block16 row reports scratch_transposed_block16 layout');
is($transpose_block16->{lookup_style}, 'lut16_scalar',
   'transpose+block16 row reports lut16_scalar lookup style');
cmp_ok($transpose_block16->{block_width}, '>', 0,
       'transpose+block16 row reports positive block width');

my ($block16_topm) = grep { $_->{benchmark} eq 'page_transpose_block16_topm' } @{$payload->{results}};
ok(defined $block16_topm, 'microbenchmark payload includes block16+topM timing');
cmp_ok($block16_topm->{candidate_heap_reject_count}, '>', 0,
       'block16+topM row shows rejected candidates from block-local selection');
cmp_ok($block16_topm->{candidate_heap_insert_count}, '<', $block16_topm->{visited_code_count},
       'block16+topM row reduces heap inserts below visited code count');

my ($avx2_requested) = grep {
	$_->{benchmark} eq 'score_code_from_lut'
	&& exists $_->{requested_kernel}
	&& $_->{requested_kernel} eq 'avx2'
} @{$payload->{results}};
my ($neon_requested) = grep {
	$_->{benchmark} eq 'score_code_from_lut'
	&& exists $_->{requested_kernel}
	&& $_->{requested_kernel} eq 'neon'
} @{$payload->{results}};
ok(defined $page_scan, 'microbenchmark payload includes page-scan timing');
ok(defined $router_full_sort, 'microbenchmark payload includes full-sort router timing');
ok(defined $router_partial, 'microbenchmark payload includes partial-selection router timing');
ok(defined $avx2_requested, 'microbenchmark payload includes explicit avx2-requested timing');
ok(defined $neon_requested, 'microbenchmark payload includes explicit neon-requested timing');
ok(exists $page_scan->{visited_code_count}, 'page-scan row reports visited code count');
ok(exists $page_scan->{visited_page_count}, 'page-scan row reports visited page count');
ok(exists $page_scan->{candidate_heap_insert_count}, 'page-scan row reports heap insert count');
ok(exists $page_scan->{candidate_heap_replace_count}, 'page-scan row reports heap replace count');
ok(exists $page_scan->{candidate_heap_reject_count}, 'page-scan row reports heap reject count');
ok(exists $page_scan->{local_candidate_heap_insert_count}, 'page-scan row reports local heap insert count');
ok(exists $page_scan->{local_candidate_heap_replace_count}, 'page-scan row reports local heap replace count');
ok(exists $page_scan->{local_candidate_heap_reject_count}, 'page-scan row reports local heap reject count');
ok(exists $page_scan->{local_candidate_merge_count}, 'page-scan row reports local candidate merge count');
ok(exists $page_scan->{qjl_lut_mode}, 'page-scan row reports qjl lut mode');
ok(exists $page_scan->{scan_layout}, 'page-scan row reports scan layout');
ok(exists $page_scan->{lookup_style}, 'page-scan row reports lookup style');
ok(exists $page_scan->{block_width}, 'page-scan row reports block width');
ok(exists $page_scan->{qjl_path}, 'page-scan row reports qjl path');
ok(exists $page_scan->{gamma_path}, 'page-scan row reports gamma path');
ok(exists $page_scan->{codes_per_second}, 'page-scan row reports code throughput');
ok(exists $page_scan->{pages_per_second}, 'page-scan row reports page throughput');
ok(exists $page_scan->{scratch_allocations}, 'page-scan row reports scratch allocations');
ok(exists $page_scan->{decoded_buffer_reuses}, 'page-scan row reports decoded-buffer reuse');
ok(exists $page_scan->{code_view_uses}, 'page-scan row reports direct code-view uses');
ok(exists $page_scan->{code_copy_uses}, 'page-scan row reports code-copy uses');
ok(exists $router_full_sort->{list_count}, 'router full-sort row reports list count');
ok(exists $router_full_sort->{probe_count}, 'router full-sort row reports probe count');
ok(exists $router_partial->{list_count}, 'router partial row reports list count');
ok(exists $router_partial->{probe_count}, 'router partial row reports probe count');
ok(exists $avx2_requested->{requested_kernel}, 'explicit avx2 row reports requested kernel');
ok(exists $avx2_requested->{requested_kernel_honored}, 'explicit avx2 row reports whether the requested kernel was honored');
ok(exists $avx2_requested->{qjl_lut_mode}, 'explicit avx2 row reports qjl lut mode');
ok(exists $neon_requested->{requested_kernel}, 'explicit neon row reports requested kernel');
ok(exists $neon_requested->{requested_kernel_honored}, 'explicit neon row reports whether the requested kernel was honored');
ok(exists $neon_requested->{qjl_lut_mode}, 'explicit neon row reports qjl lut mode');
cmp_ok($page_scan->{visited_code_count}, '>', 0, 'page-scan row visits at least one code');
cmp_ok($page_scan->{visited_page_count}, '>', 0, 'page-scan row visits at least one page');
cmp_ok($page_scan->{candidate_heap_insert_count}, '>', 0, 'page-scan row records heap inserts');
cmp_ok($page_scan->{local_candidate_heap_insert_count}, '>', 0, 'page-scan row records local heap inserts');
cmp_ok($page_scan->{local_candidate_merge_count}, '>', 0, 'page-scan row records local candidate merges');
cmp_ok($page_scan->{candidate_heap_insert_count}, '<', $page_scan->{visited_code_count},
	   'page-scan row reduces global heap inserts below visited code count');
is($page_scan->{local_candidate_heap_insert_count}
   + $page_scan->{local_candidate_heap_replace_count}
   + $page_scan->{local_candidate_heap_reject_count},
   $page_scan->{visited_code_count},
   'page-scan row accounts for every visited code in local selection counters');
cmp_ok($page_scan->{codes_per_second}, '>', 0, 'page-scan row reports positive code throughput');
cmp_ok($page_scan->{pages_per_second}, '>', 0, 'page-scan row reports positive page throughput');
cmp_ok($page_scan->{code_view_uses}, '>', 0, 'page-scan row uses direct page-code views');
is($page_scan->{code_copy_uses}, 0, 'page-scan row avoids code copies on the steady-state path');
cmp_ok($router_full_sort->{visited_code_count}, '>', 0, 'router full-sort row scores at least one centroid');
cmp_ok($router_partial->{visited_code_count}, '>', 0, 'router partial row scores at least one centroid');
cmp_ok($router_full_sort->{list_count}, '>', $router_full_sort->{probe_count},
	   'router full-sort row uses more centroids than requested probes');
is($router_partial->{list_count}, $router_full_sort->{list_count},
   'router partial row uses the same list count as the full-sort row');
is($router_partial->{probe_count}, $router_full_sort->{probe_count},
   'router partial row uses the same probe count as the full-sort row');
is($router_full_sort->{visited_page_count}, 0, 'router full-sort row does not touch pages');
is($router_partial->{visited_page_count}, 0, 'router partial row does not touch pages');
ok($page_scan->{qjl_lut_mode} eq 'float' || $page_scan->{qjl_lut_mode} eq 'quantized',
   'page-scan row reports float or quantized qjl lut mode');
is($page_scan->{scan_layout}, 'row_major',
   'page-scan row reports the row-major baseline layout');
ok($page_scan->{lookup_style} eq 'scalar_loop' || $page_scan->{lookup_style} eq 'float_gather',
   'page-scan row reports scalar_loop or float_gather lookup style');
is($page_scan->{block_width}, 1, 'page-scan row reports block_width of 1');
ok($page_scan->{qjl_path} eq 'float' || $page_scan->{qjl_path} eq 'int16_quantized',
   'page-scan row reports float or int16_quantized qjl path');
is($page_scan->{gamma_path}, 'float32_scalar',
   'page-scan row reports float32_scalar gamma path');
is($avx2_requested->{requested_kernel}, 'avx2', 'explicit avx2 row records avx2 as the requested kernel');
ok($avx2_requested->{kernel} eq 'scalar' || $avx2_requested->{kernel} eq 'avx2',
   'explicit avx2 row reports scalar fallback or avx2 execution');
ok($avx2_requested->{qjl_lut_mode} eq 'float' || $avx2_requested->{qjl_lut_mode} eq 'quantized',
   'explicit avx2 row reports float or quantized qjl lut mode');
is($avx2_requested->{requested_kernel_honored} ? JSON::PP::true : JSON::PP::false,
   $avx2_requested->{kernel} eq 'avx2' ? JSON::PP::true : JSON::PP::false,
   'explicit avx2 row truthfully reports whether avx2 actually ran');
is($neon_requested->{requested_kernel}, 'neon', 'explicit neon row records neon as the requested kernel');
ok($neon_requested->{kernel} eq 'scalar' || $neon_requested->{kernel} eq 'neon',
   'explicit neon row reports scalar fallback or neon execution');
ok($neon_requested->{qjl_lut_mode} eq 'float' || $neon_requested->{qjl_lut_mode} eq 'quantized',
   'explicit neon row reports float or quantized qjl lut mode');
is($neon_requested->{requested_kernel_honored} ? JSON::PP::true : JSON::PP::false,
   $neon_requested->{kernel} eq 'neon' ? JSON::PP::true : JSON::PP::false,
   'explicit neon row truthfully reports whether neon actually ran');

# Gate validation
ok(exists $payload->{gates}, 'microbenchmark payload includes gates section');
cmp_ok(scalar @{$payload->{gates}}, '>=', 1, 'microbenchmark payload contains at least one gate');

for my $gate (@{$payload->{gates}})
{
	ok(exists $gate->{gate}, "gate entry has a gate name");
	ok(exists $gate->{passed}, "gate '$gate->{gate}' has a passed field");
	ok(exists $gate->{reason}, "gate '$gate->{gate}' has a reason field");

	# All gates must pass — fail the TAP test if any gate reports failure
	ok($gate->{passed}, "gate '$gate->{gate}' passed: $gate->{reason}");
}

# Architecture-aware: on supported hosts, lut16_dispatch must not silently fall back
my ($dispatch_gate) = grep { $_->{gate} eq 'lut16_dispatch_kernel_selection' } @{$payload->{gates}};
if (defined $dispatch_gate)
{
	if ($dispatch_gate->{simd_available})
	{
		isnt($dispatch_gate->{selected_kernel}, 'scalar',
			 'supported host does not silently fall back to scalar for lut16 dispatch');
	}
	else
	{
		is($dispatch_gate->{selected_kernel}, 'scalar',
		   'unsupported host correctly uses scalar fallback for lut16 dispatch');
	}
}

done_testing();
