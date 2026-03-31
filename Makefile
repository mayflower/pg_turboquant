EXTENSION = pg_turboquant
EXTVERSION = 0.1.4
MODULE_big = pg_turboquant
OBJS = src/tq_extension.o src/tq_am.o src/tq_am_routine.o src/tq_reloptions.o src/tq_options.o src/tq_page.o src/tq_transform.o src/tq_codec_mse.o src/tq_codec_prod.o src/tq_pgvector_compat.o src/tq_scan.o src/tq_query_tuning.o src/tq_guc.o src/tq_simd_avx2.o src/tq_router.o src/tq_wal.o src/tq_bitmap_filter.o src/tq_probe_input.o
DATA = $(wildcard sql/pg_turboquant--*.sql)
REGRESS = smoke am_catalog reloptions flat_scan flat_streaming planner_costs gucs ivf_scan maintenance maintenance_reuse opclasses metric_fidelity ivf_training transform_contract admin_introspection query_helpers bitmap_scan capability_boundaries simd_dispatch scan_stats page_pruning adaptive_probing ivf_page_counts
NO_INSTALLCHECK = 1
.DEFAULT_GOAL := all

PG_CONFIG ?= $(shell \
	if [ -x /opt/homebrew/opt/postgresql@16/bin/pg_config ]; then \
		echo /opt/homebrew/opt/postgresql@16/bin/pg_config; \
	elif [ -x /usr/lib/postgresql/16/bin/pg_config ]; then \
		echo /usr/lib/postgresql/16/bin/pg_config; \
	elif command -v pg_config >/dev/null 2>&1; then \
		command -v pg_config; \
	else \
		echo pg_config; \
	fi)
PGXS := $(shell $(PG_CONFIG) --pgxs 2>/dev/null)
PGBINDIR := $(shell $(PG_CONFIG) --bindir 2>/dev/null)
PGXS_DIR := $(dir $(PGXS))
PG_TEST_PERL := $(shell cd third_party/postgresql-source/src/test/perl 2>/dev/null && pwd)
PERL5_LOCAL_LIB := $(CURDIR)/third_party/perl5

PG_CPPFLAGS += -Wall -Werror

UNIT_TEST_BINS = tests/unit/test_smoke tests/unit/test_scan_stats tests/unit/test_prod_code_domain tests/unit/test_prod_code_domain_simd tests/unit/test_batch_bounds tests/unit/test_probe_budgeting tests/unit/test_router_balance tests/unit/test_probe_inputs tests/unit/test_router_top_probes
PERF_TEST_BINS = tests/perf/test_prod_code_domain_avx2 tests/perf/test_prod_score_microbench
UNIT_TEST_COMMON_SRCS = src/tq_am_routine.c src/tq_am_routine.h src/tq_options.c src/tq_options.h src/tq_page.c src/tq_page.h src/tq_transform.c src/tq_transform.h src/tq_codec_mse.c src/tq_codec_mse.h src/tq_codec_prod.c src/tq_codec_prod.h src/tq_pgvector_compat.c src/tq_pgvector_compat.h src/tq_scan.c src/tq_scan.h src/tq_query_tuning.c src/tq_query_tuning.h src/tq_guc.c src/tq_guc.h src/tq_simd_avx2.c src/tq_simd_avx2.h src/tq_router.c src/tq_router.h src/tq_probe_input.c src/tq_probe_input.h

.PHONY: unitcheck tapcheck clean-unit installcheck install-pgvector perf-prod-code-domain-avx2 perf-prod-score-microbench

install-pgvector:
	./scripts/install_pgvector.sh "$(PG_CONFIG)"

install: install-pgvector

tests/unit/%: tests/unit/%.c $(UNIT_TEST_COMMON_SRCS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DTQ_UNIT_TEST=1 -Wall -Werror -std=c11 -o $@ $< src/tq_am_routine.c src/tq_options.c src/tq_page.c src/tq_transform.c src/tq_codec_mse.c src/tq_codec_prod.c src/tq_pgvector_compat.c src/tq_scan.c src/tq_query_tuning.c src/tq_guc.c src/tq_simd_avx2.c src/tq_router.c src/tq_probe_input.c -lm

tests/perf/%: tests/perf/%.c $(UNIT_TEST_COMMON_SRCS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DTQ_UNIT_TEST=1 -Wall -Werror -std=c11 -o $@ $< src/tq_am_routine.c src/tq_options.c src/tq_page.c src/tq_transform.c src/tq_codec_mse.c src/tq_codec_prod.c src/tq_pgvector_compat.c src/tq_scan.c src/tq_query_tuning.c src/tq_guc.c src/tq_simd_avx2.c src/tq_router.c src/tq_probe_input.c -lm

unitcheck: $(UNIT_TEST_BINS)
	for test_bin in $(UNIT_TEST_BINS); do ./$$test_bin; done

perf-prod-code-domain-avx2: tests/perf/test_prod_code_domain_avx2
	./tests/perf/test_prod_code_domain_avx2

perf-prod-score-microbench: tests/perf/test_prod_score_microbench
	./tests/perf/test_prod_score_microbench

tapcheck: install-pgvector
	./scripts/fetch_postgres_test_libs.sh
	./scripts/install_perl_test_deps.sh
	./scripts/run_tapcheck.sh "$(PGXS)" "$(PGBINDIR)" "$(PG_TEST_PERL)" "$(PERL5_LOCAL_LIB)" "t/*.pl"

clean-unit:
	rm -f $(UNIT_TEST_BINS)

EXTRA_CLEAN += $(UNIT_TEST_BINS)
EXTRA_CLEAN += $(PERF_TEST_BINS)

ifeq ($(wildcard $(PGXS)),)
$(error Could not find a usable pgxs.mk via PG_CONFIG=$(PG_CONFIG). Install a full PostgreSQL server toolchain or set PG_CONFIG explicitly)
endif

include $(PGXS)

installcheck: install
	./scripts/run_installcheck.sh "$(PG_CONFIG)" "$(PGXS)" "$(PGBINDIR)" $(REGRESS)
SHLIB_LINK += -lm
