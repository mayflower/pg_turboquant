use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('wal_benchmark_reduction');
$node->init;
$node->append_conf(
	'postgresql.conf', q{
wal_level = replica
fsync = on
full_page_writes = on
synchronous_commit = on
}
);
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

my $output_path = $node->data_dir . '/wal-benchmark.json';

$node->command_ok(
	[
		'python3',
		'scripts/benchmark_suite.py',
		'--host=' . $node->host,
		'--port=' . $node->port,
		'--dbname=postgres',
		'--profile=tiny',
		'--corpus=normalized_dense,mixed_live_dead',
		'--methods=turboquant_flat',
		'--output=' . $output_path,
	],
	'benchmark suite emits wal-aware results for turboquant scenarios'
);

my $payload = decode_json(slurp_file($output_path));

ok(@{$payload->{scenarios}} == 2, 'two turboquant wal scenarios recorded');

for my $scenario (@{$payload->{scenarios}})
{
	ok(exists $scenario->{metrics}{build_wal_bytes}, 'scenario reports build wal bytes');
	ok(exists $scenario->{metrics}{insert_wal_bytes}, 'scenario reports insert wal bytes');
	ok(exists $scenario->{metrics}{maintenance_wal_bytes}, 'scenario reports maintenance wal bytes');
	ok(exists $scenario->{metrics}{sealed_baseline_build_wal_bytes}, 'scenario reports sealed baseline build wal bytes');
	ok(exists $scenario->{metrics}{sealed_baseline_insert_wal_bytes}, 'scenario reports sealed baseline insert wal bytes');
	ok(exists $scenario->{metrics}{sealed_baseline_maintenance_wal_bytes}, 'scenario reports sealed baseline maintenance wal bytes');
	cmp_ok($scenario->{metrics}{build_wal_bytes}, '<', $scenario->{metrics}{sealed_baseline_build_wal_bytes}, 'build wal bytes beat sealed baseline');
	cmp_ok($scenario->{metrics}{insert_wal_bytes}, '<', $scenario->{metrics}{sealed_baseline_insert_wal_bytes}, 'insert wal bytes beat sealed baseline');
}

my ($mixed_live_dead) = grep { $_->{corpus} eq 'mixed_live_dead' } @{$payload->{scenarios}};
ok(defined $mixed_live_dead, 'mixed_live_dead scenario is present');
cmp_ok($mixed_live_dead->{metrics}{maintenance_wal_bytes}, '>', 0, 'maintenance workload emits measurable wal');
cmp_ok(
	$mixed_live_dead->{metrics}{maintenance_wal_bytes},
	'<',
	$mixed_live_dead->{metrics}{sealed_baseline_maintenance_wal_bytes},
	'maintenance wal bytes beat sealed baseline'
);

$node->stop;

done_testing();
