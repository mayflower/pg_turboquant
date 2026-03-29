use strict;
use warnings FATAL => 'all';

use JSON::PP qw(decode_json);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('hotpot_regression_gate');
$node->init;
$node->start;

my $output_dir = $node->data_dir . '/hotpot-regression-report';

$node->command_ok(
	[
		'python3',
		'-c',
		qq{
from pathlib import Path
from benchmarks.rag.campaign_report import build_comparative_campaign_plan, run_comparative_campaign

output_dir = Path(r"$output_dir")
plan = build_comparative_campaign_plan(
    dataset_ids=["kilt_hotpotqa"],
    generator_id="fixed-debug-generator",
)
plan["regression_gate"] = {
    "kilt_hotpotqa": {
        "dataset_id": "kilt_hotpotqa",
        "method_id": "pg_turboquant_approx",
        "recall_at_10_floor": 0.90,
        "max_visited_code_fraction": 0.85,
        "max_visited_page_fraction": 0.60,
        "expected_score_mode": "code_domain",
        "max_effective_probe_count": 8,
    }
}

def fake_retrieval_runner(scenario):
    rank = [
        "pg_turboquant_approx",
        "pg_turboquant_rerank",
        "pgvector_hnsw_approx",
        "pgvector_hnsw_rerank",
        "pgvector_ivfflat_approx",
        "pgvector_ivfflat_rerank",
    ].index(scenario["method_id"]) + 1
    is_turboquant = scenario["method_id"].startswith("pg_turboquant_")
    return {
        "run_metadata": {
            "dataset_id": scenario["dataset_id"],
            "method_id": scenario["method_id"],
            "result_kind": "retrieval_only",
            "footprint_bytes": 8192 * (10 + rank),
            "index_metadata": {
                "live_count": 200,
                "router": {
                    "restart_count": 3,
                    "balance_penalty": round(0.05 * rank, 4),
                },
                "list_distribution": {
                    "max_list_size": 10 + rank,
                    "coeff_var": round(0.1 * rank, 4),
                },
            },
	        },
	        "metrics": {
	            "recall\@10": 0.95 - (rank * 0.01),
	            "latency_p95_ms": 12.0 + rank,
	            "latency_p50_ms": 10.0 + rank,
	        },
        "operational_summary": {
            "scan_stats": {
                "score_mode": {
                    "uniform": "code_domain" if is_turboquant else "none",
                    "values": ["code_domain" if is_turboquant else "none"],
                    "count": 1,
                },
                "selected_list_count": {"avg": 2.0 + rank, "p50": 2.0 + rank, "p95": 3.0 + rank, "p99": 3.0 + rank},
                "selected_live_count": {"avg": 40.0 + rank, "p50": 40.0 + rank, "p95": 44.0 + rank, "p99": 44.0 + rank},
                "visited_page_count": {"avg": 1.0 + (rank / 10.0), "p50": 1.0 + (rank / 10.0), "p95": 2.0 + (rank / 10.0), "p99": 2.0 + (rank / 10.0)},
                "visited_code_count": {"avg": 10.0 + rank, "p50": 10.0 + rank, "p95": 12.0 + rank, "p99": 12.0 + rank},
                "effective_probe_count": {"avg": float(rank), "p50": float(rank), "p95": float(rank + 1), "p99": float(rank + 1)},
                "page_prune_count": {"avg": float(rank), "p50": float(rank), "p95": float(rank + 1), "p99": float(rank + 1)},
            }
        },
    }

def fake_end_to_end_runner(scenario, retrieval_result):
    rank = [
        "pg_turboquant_approx",
        "pg_turboquant_rerank",
        "pgvector_hnsw_approx",
        "pgvector_hnsw_rerank",
        "pgvector_ivfflat_approx",
        "pgvector_ivfflat_rerank",
    ].index(scenario["method_id"]) + 1
    return {
        "run_metadata": {
            "dataset_id": scenario["dataset_id"],
            "method_id": scenario["method_id"],
            "result_kind": "end_to_end",
        },
        "answer_metrics": {
            "answer_exact_match": 0.7 - (rank * 0.01),
            "answer_f1": 0.8 - (rank * 0.01),
        },
        "operational_summary": {
            "latency_ms": {
                "total": {"p50": 20.0 + rank, "p95": 25.0 + rank, "p99": 30.0 + rank}
            }
        },
    }

run_comparative_campaign(
    output_dir=output_dir,
    plan=plan,
    retrieval_runner=fake_retrieval_runner,
    end_to_end_runner=fake_end_to_end_runner,
)
		},
	],
	'comparative Hotpot report emits regression gate artifacts'
);

my $payload = decode_json(slurp_file($output_dir . '/rag-campaign.json'));
my $retrieval_csv = slurp_file($output_dir . '/retrieval-comparison.csv');
my $first_row = $payload->{tables}{retrieval_only}[0];

ok($payload->{report}{regression_gate}{passed}, 'regression gate passes on seeded baseline');
ok(exists $first_row->{avg_selected_list_count}, 'retrieval JSON includes avg selected list count');
ok(exists $first_row->{avg_selected_live_count}, 'retrieval JSON includes avg selected live count');
ok(exists $first_row->{avg_visited_page_count}, 'retrieval JSON includes avg visited page count');
ok(exists $first_row->{avg_visited_code_count}, 'retrieval JSON includes avg visited code count');
ok(exists $first_row->{avg_effective_probe_count}, 'retrieval JSON includes avg effective probe count');
ok(exists $first_row->{avg_page_prune_count}, 'retrieval JSON includes avg page prune count');
ok(exists $first_row->{score_mode}, 'retrieval JSON includes score mode');
like($retrieval_csv, qr/avg_selected_list_count/, 'retrieval CSV includes avg selected list count column');
like($retrieval_csv, qr/avg_visited_page_count/, 'retrieval CSV includes avg visited page count column');
like($retrieval_csv, qr/avg_effective_probe_count/, 'retrieval CSV includes avg effective probe count column');
like($retrieval_csv, qr/score_mode/, 'retrieval CSV includes score mode column');

$node->stop;

done_testing();
