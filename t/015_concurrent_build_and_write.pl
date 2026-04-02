use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

sub nearest_id_sql
{
	my ($table_name, $query_vector) = @_;

	return qq{
		SET enable_seqscan = off;
		SET enable_bitmapscan = off;
		SELECT id
		FROM ${table_name}
		ORDER BY embedding <=> '${query_vector}'::vector(4), id
		LIMIT 1;
	};
}

sub wait_background_ok
{
	my ($session, $label) = @_;
	my $ok = eval
	{
		$session->query_safe('SELECT 1;');
		1;
	};
	my $err = $@;

	ok($ok, $label);
	diag($err) unless $ok;
	return $ok;
}

my $node = PostgreSQL::Test::Cluster->new('concurrent_build_write');
$node->init;
$node->append_conf(
	'postgresql.conf',
	'lock_timeout = ' . (1000 * $PostgreSQL::Test::Utils::timeout_default));
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
		CREATE TABLE tq_cic_docs (
			id int4 PRIMARY KEY,
			embedding vector(4)
		);

		INSERT INTO tq_cic_docs (id, embedding)
		SELECT
			i,
			CASE
				WHEN i % 4 = 0 THEN '[1,0,0,0]'::vector(4)
				WHEN i % 4 = 1 THEN '[0,1,0,0]'::vector(4)
				WHEN i % 4 = 2 THEN '[0,0,1,0]'::vector(4)
				ELSE '[0,0,0,1]'::vector(4)
			END
		FROM generate_series(1, 4000) AS i;
	}
);

my $overlap_insert = $node->background_psql('postgres');
$overlap_insert->query_safe(
	q{
		BEGIN;
		INSERT INTO tq_cic_docs (id, embedding) VALUES (900001, '[0.01,0.01,0.01,0.97]');
	}
);

my $cic = $node->background_psql('postgres');
$cic->query_until(
	qr/cic-start/,
	q{
\echo cic-start
CREATE INDEX CONCURRENTLY tq_cic_idx
	ON tq_cic_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = true);
});

$overlap_insert->query_safe('COMMIT;');
$overlap_insert->quit;
$cic->query_safe('SELECT 1;');
$cic->quit;

is(
	$node->safe_psql('postgres', nearest_id_sql('tq_cic_docs', '[0.01,0.01,0.01,0.97]')),
	'900001',
	'CREATE INDEX CONCURRENTLY captures rows committed during the build window'
);

my $cic_meta = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_cic_idx'::regclass)::text;})
);
my $cic_heap = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_heap_stats('tq_cic_idx'::regclass)::text;})
);
is($cic_meta->{live_count}, 4001, 'concurrent build metadata includes the overlapping row');
is($cic_heap->{heap_live_rows_exact}, 4001, 'exact heap stats match after concurrent build');

$node->stop('immediate');
$node->start;

my $cic_after_restart = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_cic_idx'::regclass)::text;})
);
my $cic_heap_after_restart = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_heap_stats('tq_cic_idx'::regclass)::text;})
);

is($cic_after_restart->{live_count}, 4001, 'concurrent build live count survives immediate restart');
is($cic_heap_after_restart->{heap_live_rows_exact}, 4001, 'exact heap stats survive immediate restart after concurrent build');
is(
	$node->safe_psql('postgres', nearest_id_sql('tq_cic_docs', '[0.01,0.01,0.01,0.97]')),
	'900001',
	'concurrent build query result survives immediate restart'
);

$node->safe_psql(
	'postgres', q{
		CREATE TABLE tq_concurrent_write_docs (
			id int4 PRIMARY KEY,
			embedding vector(4)
		);

		INSERT INTO tq_concurrent_write_docs (id, embedding)
		SELECT i, '[1,0,0,0]'::vector(4)
		FROM generate_series(1, 256) AS i;

		CREATE INDEX tq_concurrent_write_idx
			ON tq_concurrent_write_docs
			USING turboquant (embedding tq_cosine_ops)
			WITH (bits = 4, lists = 1, lanes = auto, transform = 'hadamard', normalized = true);
	}
);

my $barrier = $node->background_psql('postgres');
$barrier->query_safe('SELECT pg_advisory_lock(424242);');

my @writers = map { $node->background_psql('postgres') } 1 .. 3;

$writers[0]->query_until(
	qr/writer-1-start/,
	q{
\echo writer-1-start
SELECT pg_advisory_lock_shared(424242);
SELECT pg_advisory_unlock_shared(424242);
INSERT INTO tq_concurrent_write_docs (id, embedding)
SELECT g.i, '[1,0,0,0]'::vector(4)
FROM generate_series(1001, 1240) AS g(i)
CROSS JOIN LATERAL (SELECT pg_sleep(0.0005)) AS s;
});

$writers[1]->query_until(
	qr/writer-2-start/,
	q{
\echo writer-2-start
SELECT pg_advisory_lock_shared(424242);
SELECT pg_advisory_unlock_shared(424242);
INSERT INTO tq_concurrent_write_docs (id, embedding)
SELECT g.i, '[1,0,0,0]'::vector(4)
FROM generate_series(2001, 2240) AS g(i)
CROSS JOIN LATERAL (SELECT pg_sleep(0.0005)) AS s;
});

$writers[2]->query_until(
	qr/writer-3-start/,
	q{
\echo writer-3-start
SELECT pg_advisory_lock_shared(424242);
SELECT pg_advisory_unlock_shared(424242);
INSERT INTO tq_concurrent_write_docs (id, embedding)
	SELECT rows_to_insert.id, rows_to_insert.embedding
	FROM (
		SELECT i AS id, '[1,0,0,0]'::vector(4) AS embedding
		FROM generate_series(3001, 3240) AS g(i)
		UNION ALL
		SELECT 990001 AS id, '[0,0,0,1]'::vector(4)
	) AS rows_to_insert
	CROSS JOIN LATERAL (SELECT pg_sleep(0.0005)) AS s;
});

$barrier->query_safe('SELECT pg_advisory_unlock(424242);');
$barrier->quit;

for my $attempt (1 .. 5)
{
	my $probe = $node->safe_psql(
		'postgres',
		q{
			SET enable_seqscan = off;
			SET enable_bitmapscan = off;
			SELECT count(*)
			FROM (
				SELECT id
				FROM tq_concurrent_write_docs
				ORDER BY embedding <=> '[1,0,0,0]'::vector(4), id
				LIMIT 8
			) AS candidates;
		}
	);

	ok($probe =~ /^\d+$/, "search remains available while concurrent writers run (attempt $attempt)");
}

for my $writer (@writers)
{
	wait_background_ok($writer, 'writer session completed without SQL errors');
	$writer->quit;
}

my $write_meta = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_concurrent_write_idx'::regclass)::text;})
);
my $write_heap = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_heap_stats('tq_concurrent_write_idx'::regclass)::text;})
);
my $heap_rows = $node->safe_psql('postgres', q{SELECT count(*) FROM tq_concurrent_write_docs;});

is($write_heap->{heap_live_rows_exact}, int($heap_rows), 'exact heap stats remain correct after concurrent inserts');
is($write_meta->{live_count}, int($heap_rows), 'index live count matches heap after concurrent inserts');
is(
	$node->safe_psql('postgres', nearest_id_sql('tq_concurrent_write_docs', '[0,0,0,1]')),
	'990001',
	'concurrent inserts remain query-correct after writer overlap'
);

$node->stop('immediate');
$node->start;

my $write_meta_after_restart = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_metadata('tq_concurrent_write_idx'::regclass)::text;})
);
my $write_heap_after_restart = decode_json(
	$node->safe_psql('postgres', q{SELECT tq_index_heap_stats('tq_concurrent_write_idx'::regclass)::text;})
);
my $heap_rows_after_restart = $node->safe_psql('postgres', q{SELECT count(*) FROM tq_concurrent_write_docs;});

is(
	$write_heap_after_restart->{heap_live_rows_exact},
	int($heap_rows_after_restart),
	'exact heap stats survive immediate restart after concurrent inserts'
);
is(
	$write_meta_after_restart->{live_count},
	int($heap_rows_after_restart),
	'live count survives immediate restart after concurrent inserts'
);
is(
	$node->safe_psql('postgres', nearest_id_sql('tq_concurrent_write_docs', '[0,0,0,1]')),
	'990001',
	'concurrent-insert query result survives immediate restart'
);

$node->safe_psql(
	'postgres', q{
		CREATE TABLE tq_cic_cancel_docs (
			id int4 PRIMARY KEY,
			embedding vector(4)
		);

		INSERT INTO tq_cic_cancel_docs (id, embedding)
		SELECT
			i,
			CASE
				WHEN i % 4 = 0 THEN '[1,0,0,0]'::vector(4)
				WHEN i % 4 = 1 THEN '[0,1,0,0]'::vector(4)
				WHEN i % 4 = 2 THEN '[0,0,1,0]'::vector(4)
				ELSE '[0,0,0,1]'::vector(4)
			END
		FROM generate_series(1, 60000) AS i;
	}
);

my $cancel_cic = $node->background_psql('postgres');
my $cancel_pid = $cancel_cic->query_safe('SELECT pg_backend_pid();');

$cancel_cic->query_until(
	qr/cancel-start/,
	q{
\echo cancel-start
CREATE INDEX CONCURRENTLY tq_cic_cancel_idx
	ON tq_cic_cancel_docs
	USING turboquant (embedding tq_cosine_ops)
	WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = true);
});

ok(
	$node->poll_query_until(
		'postgres',
		qq{SELECT EXISTS (SELECT 1 FROM pg_stat_progress_create_index WHERE pid = ${cancel_pid});},
		't'
	),
	'cancel test observes active CREATE INDEX CONCURRENTLY progress'
);

is(
	$node->safe_psql('postgres', qq{SELECT pg_cancel_backend(${cancel_pid});}),
	't',
	'cancelled concurrent build backend'
);

my $cancel_error = '';
eval
{
	$cancel_cic->query_safe('SELECT 1;');
	1;
} or $cancel_error = $@;
$cancel_cic->quit;

like(
	$cancel_error,
	qr/(canceling statement due to user request|query failed:|process ended prematurely)/,
	'cancelled concurrent build surfaces a cancellation error'
);

my $cancel_regclass = $node->safe_psql(
	'postgres',
	q{SELECT to_regclass('public.tq_cic_cancel_idx') IS NOT NULL;}
);

if ($cancel_regclass eq 't')
{
	my ($indisvalid, $indisready, $indislive) = split /\|/,
	  $node->safe_psql(
		'postgres',
		q{
			SELECT indisvalid::int || '|' || indisready::int || '|' || indislive::int
			FROM pg_index
			WHERE indexrelid = 'tq_cic_cancel_idx'::regclass;
		}
	  );

	is($indisvalid, 0, 'cancelled concurrent build does not leave a valid index behind');
	ok($indisready == 0 || $indisready == 1, 'catalog row remains readable after cancellation');
	is($indislive, 1, 'cancelled concurrent build leaves a droppable catalog entry');
	$node->safe_psql('postgres', 'DROP INDEX CONCURRENTLY tq_cic_cancel_idx;');
}

is(
	$node->safe_psql('postgres', q{SELECT to_regclass('public.tq_cic_cancel_idx') IS NULL;}),
	't',
	'cancelled concurrent build leaves no stuck index after cleanup'
);

$node->stop;

done_testing();
