use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('maintenance_reuse_cycle');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

$node->safe_psql(
	'postgres', q{
		CREATE TABLE tq_reuse_cycle_docs (
			id int4 PRIMARY KEY,
			embedding vector(4)
		);

		INSERT INTO tq_reuse_cycle_docs (id, embedding)
		SELECT
			i,
			CASE
				WHEN i <= 750 THEN format('[1,%s,0,0]', (i % 10) * 0.01)
				ELSE format('[0,1,%s,0]', (i % 10) * 0.01)
			END::vector(4)
		FROM generate_series(1, 1500) AS i;

		CREATE INDEX tq_reuse_cycle_idx
			ON tq_reuse_cycle_docs
			USING turboquant (embedding tq_cosine_ops)
			WITH (bits = 4, lists = 2, lanes = auto, transform = 'hadamard', normalized = true);
	}
);

my $initial = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_reuse_cycle_idx'::regclass)::text;})
);
my $initial_size = $node->safe_psql('postgres', q{SELECT pg_relation_size('tq_reuse_cycle_idx'::regclass);});
my $block_size = $node->safe_psql('postgres', q{SELECT current_setting('block_size');});

$node->safe_psql(
	'postgres', q{
		DELETE FROM tq_reuse_cycle_docs WHERE id > 300;
		VACUUM tq_reuse_cycle_docs;
		INSERT INTO tq_reuse_cycle_docs (id, embedding) VALUES (2001, '[0,0,1,0]');
		INSERT INTO tq_reuse_cycle_docs (id, embedding)
		SELECT
			i,
			CASE
				WHEN i % 2 = 0 THEN format('[1,0,%s,0]', (i % 10) * 0.01)
				ELSE format('[0,1,%s,0]', (i % 10) * 0.01)
			END::vector(4)
		FROM generate_series(2002, 3200) AS i;
	}
);

my $after_first = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_reuse_cycle_idx'::regclass)::text;})
);
my $after_first_size = $node->safe_psql('postgres', q{SELECT pg_relation_size('tq_reuse_cycle_idx'::regclass);});

cmp_ok($after_first->{batch_page_count}, '<=', $initial->{batch_page_count} + 1, 'first maintenance cycle stays within one batch page of the original distribution');
cmp_ok($after_first_size, '<=', $initial_size + $block_size, 'first maintenance cycle stays within one block of the original relation size');
is($after_first->{dead_count}, 0, 'first maintenance cycle leaves no dead tuples in metadata');

$node->safe_psql(
	'postgres', q{
		DELETE FROM tq_reuse_cycle_docs WHERE id BETWEEN 2001 AND 2600;
		VACUUM tq_reuse_cycle_docs;
		INSERT INTO tq_reuse_cycle_docs (id, embedding) VALUES (4001, '[0,0,1,0]');
		INSERT INTO tq_reuse_cycle_docs (id, embedding)
		SELECT
			i,
			CASE
				WHEN i % 2 = 0 THEN format('[1,0,%s,0]', (i % 10) * 0.01)
				ELSE format('[0,1,%s,0]', (i % 10) * 0.01)
			END::vector(4)
		FROM generate_series(4002, 4600) AS i;
	}
);

my $after_second = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_reuse_cycle_idx'::regclass)::text;})
);
my $after_second_size = $node->safe_psql('postgres', q{SELECT pg_relation_size('tq_reuse_cycle_idx'::regclass);});

cmp_ok($after_second->{batch_page_count}, '<=', $after_first->{batch_page_count}, 'second maintenance cycle does not keep growing active batch pages');
cmp_ok($after_second_size, '<=', $after_first_size, 'second maintenance cycle does not keep growing relation size');
is(
	$node->safe_psql(
		'postgres', q{
			SET enable_seqscan = off;
			SET enable_bitmapscan = off;
			SELECT id
			FROM tq_reuse_cycle_docs
			ORDER BY embedding <=> '[0,0,1,0]'::vector(4), id
			LIMIT 1;
		}
	),
	'4001',
	'search remains correct after repeated maintenance reuse cycles'
);

$node->stop;

done_testing();
