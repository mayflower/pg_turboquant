use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('admin_introspection_maintenance');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

$node->safe_psql(
	'postgres',
	q{
		CREATE TABLE admin_introspection_docs (
			id int4 PRIMARY KEY,
			embedding vector(4)
		);
		INSERT INTO admin_introspection_docs (id, embedding) VALUES
			(1, '[1,0,0,0]'),
			(2, '[0.9,0.1,0,0]'),
			(3, '[0,1,0,0]'),
			(4, '[0,0,1,0]');
		CREATE INDEX admin_introspection_idx
			ON admin_introspection_docs
			USING turboquant (embedding tq_cosine_ops)
			WITH (bits = 4, lists = 0, lanes = auto, transform = 'hadamard', normalized = true);
	}
);

my $built = decode_json(
	$node->safe_psql(
		'postgres',
		q{SELECT tq_index_metadata('admin_introspection_idx'::regclass)::text;}
	)
);
my $built_heap = decode_json(
	$node->safe_psql(
		'postgres',
		q{SELECT tq_index_heap_stats('admin_introspection_idx'::regclass)::text;}
	)
);

is($built->{live_count}, 4, 'introspection reports build live count');
is($built->{dead_count}, 0, 'introspection reports no dead tuples after build');
is($built_heap->{heap_live_rows_exact}, 4, 'exact heap stats report live rows after build');

$node->safe_psql('postgres', q{DELETE FROM admin_introspection_docs WHERE id = 4;});

my $deleted = decode_json(
	$node->safe_psql(
		'postgres',
		q{SELECT tq_index_metadata('admin_introspection_idx'::regclass)::text;}
	)
);
my $deleted_heap = decode_json(
	$node->safe_psql(
		'postgres',
		q{SELECT tq_index_heap_stats('admin_introspection_idx'::regclass)::text;}
	)
);

is($deleted->{live_count}, 4, 'index live count is unchanged before vacuum');
is($deleted_heap->{heap_live_rows_exact}, 3, 'exact heap stats update immediately after delete');

$node->safe_psql('postgres', q{VACUUM admin_introspection_docs;});

my $vacuumed = decode_json(
	$node->safe_psql(
		'postgres',
		q{SELECT tq_index_metadata('admin_introspection_idx'::regclass)::text;}
	)
);

is($vacuumed->{live_count}, 3, 'vacuum updates live tuple count');
is($vacuumed->{dead_count}, 0, 'vacuum compaction clears dead tuple count');

$node->safe_psql('postgres', q{REINDEX INDEX admin_introspection_idx;});

my $reindexed = decode_json(
	$node->safe_psql(
		'postgres',
		q{SELECT tq_index_metadata('admin_introspection_idx'::regclass)::text;}
	)
);

is($reindexed->{live_count}, 3, 'reindex preserves live count');
is($reindexed->{dead_count}, 0, 'reindex clears dead count');
is($reindexed->{format_version}, 12, 'introspection includes page format version');

$node->stop;

done_testing();
