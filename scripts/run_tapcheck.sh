#!/usr/bin/env bash
set -euo pipefail

PGXS_PATH="${1:?pgxs path required}"
PG_BINDIR="${2:?postgres bindir required}"
PG_TEST_PERL="${3:?postgres test perl dir required}"
PERL5_LOCAL_LIB="${4:?local perl root required}"
TEST_GLOB="${5:?tap glob required}"

PG_REGRESS_BIN="$(cd "$(dirname "${PGXS_PATH}")/../test/regress" && pwd)/pg_regress"

if [[ ! -x "${PG_REGRESS_BIN}" ]]; then
  echo "pg_regress was not found at ${PG_REGRESS_BIN}"
  exit 1
fi

export PATH="${PG_BINDIR}:$PATH"
export PG_REGRESS="${PG_REGRESS_BIN}"
export PERL5LIB="${PG_TEST_PERL}:${PERL5_LOCAL_LIB}/lib/perl5:${PERL5LIB:-}"

rm -rf tmp_check

prove -I"${PG_TEST_PERL}" -I"${PERL5_LOCAL_LIB}/lib/perl5" ${TEST_GLOB}
