use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('prod_code_domain_smoke');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

my $output_path = $node->data_dir . '/prod-code-domain.json';

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
	'benchmark suite runs the normalized turboquant IVF code-domain scenario'
);

my $payload = decode_json(slurp_file($output_path));
my $scenario = $payload->{scenarios}[0];

is($scenario->{scan_stats}{score_mode}, 'code_domain', 'benchmark reports code-domain score mode');
cmp_ok($scenario->{scan_stats}{visited_code_count}, '>', 0, 'benchmark visits codes');
is($scenario->{scan_stats}{decoded_vector_count}, 0, 'benchmark reports no decoded vectors for normalized cosine/IP');
cmp_ok($scenario->{metrics}{recall_at_10}, '>=', 0.07, 'recall remains stable on the measured ANN fixture');

$node->stop;

done_testing();
