#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${ROOT_DIR}/third_party/pgvector"
PGVECTOR_REF="${PGVECTOR_REF:-v0.8.1}"
TMP_DIR="$(mktemp -d)"
ARCHIVE_URL="https://github.com/pgvector/pgvector/archive/refs/tags/${PGVECTOR_REF}.tar.gz"

trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ -d "${DEST_DIR}" ]]; then
  echo "pgvector already present at ${DEST_DIR}"
  exit 0
fi

mkdir -p "${ROOT_DIR}/third_party"

echo "Fetching pgvector ${PGVECTOR_REF} from ${ARCHIVE_URL}"
curl -fsSL "${ARCHIVE_URL}" -o "${TMP_DIR}/pgvector.tar.gz"
mkdir -p "${DEST_DIR}"
tar -xzf "${TMP_DIR}/pgvector.tar.gz" -C "${DEST_DIR}" --strip-components=1

echo "Fetched pgvector into ${DEST_DIR}"

