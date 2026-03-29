use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('maintenance_reuse_restart');
$node->init;
$node->append_conf(
	'postgresql.conf', q{
fsync = on
full_page_writes = on
restart_after_crash = on
synchronous_commit = on
wal_level = replica
}
);
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

$node->safe_psql(
	'postgres', q{
		CREATE TABLE tq_reuse_restart_docs (
			id int4 PRIMARY KEY,
			embedding vector(4)
		);

		INSERT INTO tq_reuse_restart_docs (id, embedding)
		SELECT
			i,
			CASE
				WHEN i <= 750 THEN format('[1,%s,0,0]', (i % 10) * 0.01)
				ELSE format('[0,1,%s,0]', (i % 10) * 0.01)
			END::vector(4)
		FROM generate_series(1, 1500) AS i;

		CREATE INDEX tq_reuse_restart_idx
			ON tq_reuse_restart_docs
			USING turboquant (embedding tq_cosine_ops)
			WITH (bits = 4, lists = 2, lanes = auto, transform = 'hadamard', normalized = true);

		DELETE FROM tq_reuse_restart_docs WHERE id > 300;
		VACUUM tq_reuse_restart_docs;
		INSERT INTO tq_reuse_restart_docs (id, embedding) VALUES (5001, '[0,0,1,0]');
		INSERT INTO tq_reuse_restart_docs (id, embedding)
		SELECT
			i,
			CASE
				WHEN i % 2 = 0 THEN format('[1,0,%s,0]', (i % 10) * 0.01)
				ELSE format('[0,1,%s,0]', (i % 10) * 0.01)
			END::vector(4)
		FROM generate_series(5002, 6200) AS i;
	}
);

my $before = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_reuse_restart_idx'::regclass)::text;})
);

is($before->{dead_count}, 0, 'metadata is compacted before restart');

$node->stop('immediate');
$node->start;

my $after = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_reuse_restart_idx'::regclass)::text;})
);

is($after->{batch_page_count}, $before->{batch_page_count}, 'batch page count survives restart after reuse');
is($after->{dead_count}, 0, 'dead count remains compacted after restart');
is(
	$node->safe_psql(
		'postgres', q{
			SET enable_seqscan = off;
			SET enable_bitmapscan = off;
			SELECT id
			FROM tq_reuse_restart_docs
			ORDER BY embedding <=> '[0,0,1,0]'::vector(4), id
			LIMIT 1;
		}
	),
	'5001',
	'search remains correct after restart on reused pages'
);

$node->stop;

done_testing();
