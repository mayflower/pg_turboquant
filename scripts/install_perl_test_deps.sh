#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_LIB_ROOT="${ROOT_DIR}/third_party/perl5"
CPANM_BIN="${LOCAL_LIB_ROOT}/bin/cpanm"
export PERL_MM_USE_DEFAULT=1

mkdir -p "${LOCAL_LIB_ROOT}"

if PERL5LIB="${LOCAL_LIB_ROOT}/lib/perl5" perl -MIPC::Run -e 1 >/dev/null 2>&1; then
  echo "IPC::Run is up to date. ($(PERL5LIB="${LOCAL_LIB_ROOT}/lib/perl5" perl -MIPC::Run -e 'print $IPC::Run::VERSION'))"
  exit 0
fi

if [[ ! -x "${CPANM_BIN}" ]]; then
  curl -fsSL https://cpanmin.us/ -o "${LOCAL_LIB_ROOT}/cpanm-installer"
  perl -I"${LOCAL_LIB_ROOT}/lib/perl5" "${LOCAL_LIB_ROOT}/cpanm-installer" \
    --local-lib-contained "${LOCAL_LIB_ROOT}" \
    App::cpanminus
fi

"${CPANM_BIN}" --notest --local-lib-contained "${LOCAL_LIB_ROOT}" IPC::Run
