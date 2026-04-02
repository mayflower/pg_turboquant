#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PG_CONFIG:-}" ]]; then
  PG_CONFIG_BIN="${PG_CONFIG}"
elif [[ -x /opt/homebrew/opt/postgresql@16/bin/pg_config ]]; then
  PG_CONFIG_BIN=/opt/homebrew/opt/postgresql@16/bin/pg_config
elif [[ -x /usr/lib/postgresql/16/bin/pg_config ]]; then
  PG_CONFIG_BIN=/usr/lib/postgresql/16/bin/pg_config
else
  PG_CONFIG_BIN="$(command -v pg_config || true)"
fi

if [[ -z "${PG_CONFIG_BIN}" ]]; then
  echo "pg_config was not found. Install a full PostgreSQL server development toolchain first."
  exit 1
fi

echo "Using pg_config: ${PG_CONFIG_BIN}"
"${PG_CONFIG_BIN}" --version
echo "Export with: export PG_CONFIG=${PG_CONFIG_BIN}"
export PG_CONFIG="${PG_CONFIG_BIN}"

PG_BIN_DIR="$(dirname "${PG_CONFIG_BIN}")"
PGXS_PATH="$("${PG_CONFIG_BIN}" --pgxs)"
PG_REGRESS_BIN="$(cd "$(dirname "${PGXS_PATH}")/../test/regress" 2>/dev/null && pwd)/pg_regress"

"${ROOT_DIR}/scripts/fetch_pgvector.sh"
"${ROOT_DIR}/scripts/install_pgvector.sh" "${PG_CONFIG_BIN}"
"${ROOT_DIR}/scripts/fetch_postgres_test_libs.sh"

if [[ -x "${PG_BIN_DIR}/postgres" && -x "${PG_REGRESS_BIN}" ]]; then
  cat <<EOF
Local PostgreSQL server tooling was detected.
Next steps:
  export PG_CONFIG="${PG_CONFIG_BIN}"
  export PATH="${PG_BIN_DIR}:\$PATH"
  make
  make install
  make unitcheck
  make installcheck
  make tapcheck
EOF
else
  cat <<'EOF'
Full local PostgreSQL server tooling was not detected.
Use the PG16 dev container path instead:
  docker compose build
  docker compose run --rm dev bash -lc './scripts/bootstrap_dev.sh && make && make install && make unitcheck && make installcheck && make tapcheck'
EOF
fi
