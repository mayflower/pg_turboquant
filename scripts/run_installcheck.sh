#!/usr/bin/env bash
set -euo pipefail

PG_CONFIG_BIN="${1:?pg_config path required}"
PGXS_PATH="${2:?pgxs path required}"
PG_BINDIR="${3:?postgres bindir required}"
shift 3
TESTS=("$@")

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/tmp_installcheck"
REGRESS_INPUT_DIR="${ROOT_DIR}/test"
PG_REGRESS_BIN="$(cd "$(dirname "${PGXS_PATH}")/../test/regress" && pwd)/pg_regress"

if [[ ! -x "${PG_REGRESS_BIN}" ]]; then
  echo "pg_regress was not found at ${PG_REGRESS_BIN}"
  exit 1
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "# +++ regress install-check in ${ROOT_DIR} +++"
"${PG_REGRESS_BIN}" \
  --inputdir="${REGRESS_INPUT_DIR}" \
  --expecteddir="${REGRESS_INPUT_DIR}/expected" \
  --outputdir="${OUTPUT_DIR}" \
  --bindir="${PG_BINDIR}" \
  --dlpath="$("${PG_CONFIG_BIN}" --pkglibdir)" \
  --temp-instance="${OUTPUT_DIR}/tmp_check" \
  --no-locale \
  --dbname=contrib_regression \
  "${TESTS[@]}"
