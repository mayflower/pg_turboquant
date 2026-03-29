use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('wal_sparse_restart');
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
		CREATE TABLE tq_wal_sparse_docs (
			id int4 PRIMARY KEY,
			embedding vector(4)
		);

		INSERT INTO tq_wal_sparse_docs (id, embedding)
		SELECT
			i,
			CASE
				WHEN i % 3 = 0 THEN '[1,0,0,0]'::vector(4)
				WHEN i % 3 = 1 THEN '[0,1,0,0]'::vector(4)
				ELSE '[0,0,1,0]'::vector(4)
			END
		FROM generate_series(1, 256) AS i;

		CREATE INDEX tq_wal_sparse_idx
			ON tq_wal_sparse_docs
			USING turboquant (embedding tq_cosine_ops)
			WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = true);

		DELETE FROM tq_wal_sparse_docs WHERE id % 2 = 0;
		VACUUM tq_wal_sparse_docs;

		INSERT INTO tq_wal_sparse_docs (id, embedding) VALUES
			(1001, '[0,0,0,1]'),
			(1002, '[0,0,0.99,0.01]');
	}
);

my $before = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_wal_sparse_idx'::regclass)::text;})
);

ok($before->{dead_count} == 0, 'compacted sparse-page metadata is clean before restart');

$node->stop('immediate');
$node->start;

my $after = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_wal_sparse_idx'::regclass)::text;})
);

is($after->{dead_count}, 0, 'dead count stays clean after restart');
is($after->{batch_page_count}, $before->{batch_page_count}, 'batch page count survives sparse-page restart');
is(
	$node->safe_psql(
		'postgres', q{
			SET enable_seqscan = off;
			SET enable_bitmapscan = off;
			SELECT id
			FROM tq_wal_sparse_docs
			ORDER BY embedding <=> '[0,0,0,1]'::vector(4), id
			LIMIT 1;
		}
	),
	'1001',
	'sparse-page payload remains query-correct after restart'
);

$node->stop;

done_testing();
