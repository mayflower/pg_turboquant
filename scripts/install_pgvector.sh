#!/usr/bin/env bash
set -euo pipefail

PG_CONFIG_BIN="${1:?pg_config path required}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGVECTOR_DIR="${ROOT_DIR}/third_party/pgvector"

if [[ ! -d "${PGVECTOR_DIR}" ]]; then
  echo "pgvector source was not found at ${PGVECTOR_DIR}"
  exit 1
fi

make -C "${PGVECTOR_DIR}" PG_CONFIG="${PG_CONFIG_BIN}" install
