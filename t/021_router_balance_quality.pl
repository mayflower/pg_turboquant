use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('router_balance_hotpot');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant_test_support;');

$node->safe_psql(
	'postgres',
	q{
		CREATE TABLE tq_router_balance_baseline (id int4 PRIMARY KEY, embedding vector(2));
		CREATE TABLE tq_router_balance_balanced (id int4 PRIMARY KEY, embedding vector(2));
		INSERT INTO tq_router_balance_baseline (id, embedding) VALUES
			(1, '[1.18,0.02]'), (2, '[0.59,-0.01]'), (3, '[0.72,-0.01]'), (4, '[1.10,-0.01]'),
			(5, '[0.67,0.01]'), (6, '[1.27,0.00]'), (7, '[0.39,0.02]'), (8, '[0.87,-0.01]'),
			(9, '[4.12,4.14]'), (10, '[4.09,4.12]'), (11, '[3.94,4.07]'), (12, '[4.12,4.06]'),
			(13, '[-4.01,3.88]'), (14, '[-4.02,4.03]'), (15, '[-3.88,4.14]'), (16, '[-4.01,4.11]'),
			(17, '[-0.07,-3.91]'), (18, '[0.01,-4.15]');
		INSERT INTO tq_router_balance_balanced
		SELECT * FROM tq_router_balance_baseline;
		CREATE INDEX tq_router_balance_baseline_idx
			ON tq_router_balance_baseline
			USING turboquant (embedding tq_cosine_ops)
			WITH (bits = 4, lists = 4, normalized = true, router_samples = 18, router_iterations = 8, router_restarts = 1, router_seed = 17);
		CREATE INDEX tq_router_balance_balanced_idx
			ON tq_router_balance_balanced
			USING turboquant (embedding tq_cosine_ops)
			WITH (bits = 4, lists = 4, normalized = true, router_samples = 18, router_iterations = 8, router_restarts = 4, router_seed = 17);
	}
);

my $baseline_meta = decode_json(
	$node->safe_psql(
		'postgres',
		q{SELECT tq_index_metadata('tq_router_balance_baseline_idx'::regclass)::text}
	)
);
my $balanced_meta = decode_json(
	$node->safe_psql(
		'postgres',
		q{SELECT tq_index_metadata('tq_router_balance_balanced_idx'::regclass)::text}
	)
);

sub fetch_scan_stats {
	my ($table_name) = @_;

	return decode_json(
		$node->safe_psql(
			'postgres',
			qq{
				SET enable_seqscan = off;
				SET enable_bitmapscan = off;
				SET turboquant.probes = 2;
				WITH ranked AS MATERIALIZED (
					SELECT id
					FROM $table_name
					ORDER BY embedding <=> '[1.0,0.0]'
					LIMIT 4
				)
				SELECT tq_last_scan_stats()::text
				FROM (SELECT count(*) FROM ranked) AS _;
			}
		)
	);
}

cmp_ok(
	$balanced_meta->{list_distribution}{max_list_size},
	'<',
	$baseline_meta->{list_distribution}{max_list_size},
	'balanced router training reduces max list size'
);
cmp_ok(
	$balanced_meta->{list_distribution}{coeff_var},
	'<',
	$baseline_meta->{list_distribution}{coeff_var},
	'balanced router training reduces list-size coefficient of variation'
);
is($balanced_meta->{router}{restart_count}, 4, 'router metadata exposes restart count');
ok(exists $balanced_meta->{router}{balance_penalty}, 'router metadata exposes balance penalty');

my $baseline_scan = fetch_scan_stats('tq_router_balance_baseline');
my $balanced_scan = fetch_scan_stats('tq_router_balance_balanced');

cmp_ok(
	$balanced_scan->{visited_code_count},
	'<',
	$baseline_scan->{visited_code_count},
	'balanced router training reduces visited-code work on the seeded skew query'
);

$node->stop;

done_testing();
