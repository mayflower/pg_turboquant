use strict;
use warnings FATAL => 'all';

use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('smoke');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_turboquant;');

is(
	$node->safe_psql(
		'postgres',
		q{SELECT extname FROM pg_extension WHERE extname = 'pg_turboquant';}
	),
	'pg_turboquant',
	'extension is registered'
);

is(
	$node->safe_psql('postgres', q{SELECT tq_smoke();}),
	'pg_turboquant',
	'extension library loads and responds'
);

$node->stop;

done_testing();
