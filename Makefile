EXTENSION = pg_turboquant
EXTVERSION = 0.1.4
MODULE_big = pg_turboquant
OBJS = src/tq_extension.o src/tq_am.o src/tq_am_routine.o src/tq_reloptions.o src/tq_options.o src/tq_page.o src/tq_transform.o src/tq_codec_mse.o src/tq_codec_prod.o src/tq_pgvector_compat.o src/tq_scan.o src/tq_query_tuning.o src/tq_guc.o src/tq_simd_avx2.o src/tq_router.o src/tq_wal.o src/tq_bitmap_filter.o
DATA = $(wildcard sql/pg_turboquant--*.sql)
REGRESS = smoke am_catalog reloptions flat_scan flat_streaming planner_costs gucs ivf_scan maintenance maintenance_reuse opclasses metric_fidelity ivf_training transform_contract admin_introspection query_helpers bitmap_scan capability_boundaries simd_dispatch
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

UNIT_TEST_BIN = tests/unit/test_smoke

.PHONY: unitcheck tapcheck clean-unit installcheck install-pgvector

install-pgvector:
	./scripts/install_pgvector.sh "$(PG_CONFIG)"

install: install-pgvector

$(UNIT_TEST_BIN): tests/unit/test_smoke.c src/tq_am_routine.c src/tq_am_routine.h src/tq_options.c src/tq_options.h src/tq_page.c src/tq_page.h src/tq_transform.c src/tq_transform.h src/tq_codec_mse.c src/tq_codec_mse.h src/tq_codec_prod.c src/tq_codec_prod.h src/tq_pgvector_compat.c src/tq_pgvector_compat.h src/tq_scan.c src/tq_scan.h src/tq_query_tuning.c src/tq_query_tuning.h src/tq_simd_avx2.c src/tq_simd_avx2.h src/tq_router.c src/tq_router.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -DTQ_UNIT_TEST=1 -Wall -Werror -std=c11 -o $@ tests/unit/test_smoke.c src/tq_am_routine.c src/tq_options.c src/tq_page.c src/tq_transform.c src/tq_codec_mse.c src/tq_codec_prod.c src/tq_pgvector_compat.c src/tq_scan.c src/tq_query_tuning.c src/tq_simd_avx2.c src/tq_router.c -lm

unitcheck: $(UNIT_TEST_BIN)
	./$(UNIT_TEST_BIN)

tapcheck: install-pgvector
	./scripts/fetch_postgres_test_libs.sh
	./scripts/install_perl_test_deps.sh
	./scripts/run_tapcheck.sh "$(PGXS)" "$(PGBINDIR)" "$(PG_TEST_PERL)" "$(PERL5_LOCAL_LIB)" "t/*.pl"

clean-unit:
	rm -f $(UNIT_TEST_BIN)

EXTRA_CLEAN += $(UNIT_TEST_BIN)

ifeq ($(wildcard $(PGXS)),)
$(error Could not find a usable pgxs.mk via PG_CONFIG=$(PG_CONFIG). Install a full PostgreSQL server toolchain or set PG_CONFIG explicitly)
endif

include $(PGXS)

installcheck: install
	./scripts/run_installcheck.sh "$(PG_CONFIG)" "$(PGXS)" "$(PGBINDIR)" $(REGRESS)
SHLIB_LINK += -lm
