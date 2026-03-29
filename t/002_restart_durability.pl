use strict;
use warnings FATAL => 'all';

use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('restart_durability');
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
		CREATE TABLE tq_restart_docs (
			id int4 PRIMARY KEY,
			embedding vector(4)
		);

		INSERT INTO tq_restart_docs (id, embedding) VALUES
			(1, '[1,0,0,0]'),
			(2, '[0,1,0,0]'),
			(3, '[0,0,1,0]');

		CREATE INDEX tq_restart_docs_embedding_tq_idx
		ON tq_restart_docs
		USING turboquant (embedding tq_cosine_ops)
		WITH (
			bits = 4,
			lists = 0,
			lanes = auto,
			transform = 'hadamard',
			normalized = true
		);

		INSERT INTO tq_restart_docs (id, embedding)
		VALUES (4, '[0,0,0,1]');
	}
);

is(
	$node->safe_psql(
		'postgres', q{
			SET enable_seqscan = off;
			SET enable_bitmapscan = off;
			SELECT id
			FROM tq_restart_docs
			ORDER BY embedding <=> '[0,0,0,1]'
			LIMIT 1;
		}
	),
	'4',
	'newly inserted row is visible before crash restart'
);

$node->stop('immediate');
$node->start;

is(
	$node->safe_psql(
		'postgres', q{
			SET enable_seqscan = off;
			SET enable_bitmapscan = off;
			SELECT id
			FROM tq_restart_docs
			ORDER BY embedding <=> '[0,0,0,1]'
			LIMIT 1;
		}
	),
	'4',
	'index build and append path survive immediate restart'
);

$node->stop;

done_testing();
