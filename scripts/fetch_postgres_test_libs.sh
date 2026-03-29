#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_CONFIG_BIN="${PG_CONFIG:-$(command -v pg_config || true)}"
DEST_DIR="${ROOT_DIR}/third_party/postgresql-source"
TMP_DIR="$(mktemp -d)"

trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ -d "${DEST_DIR}/src/test/perl/PostgreSQL/Test" ]]; then
  echo "PostgreSQL TAP libraries already present at ${DEST_DIR}"
  exit 0
fi

if [[ -z "${PG_CONFIG_BIN}" ]]; then
  echo "pg_config was not found; cannot determine PostgreSQL version for TAP libraries."
  exit 1
fi

PG_VERSION_STR="$("${PG_CONFIG_BIN}" --version | awk '{print $2}')"
ARCHIVE_URL="https://ftp.postgresql.org/pub/source/v${PG_VERSION_STR}/postgresql-${PG_VERSION_STR}.tar.bz2"

mkdir -p "${ROOT_DIR}/third_party"

echo "Fetching PostgreSQL source ${PG_VERSION_STR} from ${ARCHIVE_URL}"
curl -fsSL "${ARCHIVE_URL}" -o "${TMP_DIR}/postgresql.tar.bz2"
mkdir -p "${DEST_DIR}"
tar -xjf "${TMP_DIR}/postgresql.tar.bz2" -C "${DEST_DIR}" --strip-components=1

echo "Fetched PostgreSQL TAP libraries into ${DEST_DIR}"
