#!/bin/sh
set -eu

usage() {
    cat <<'EOF'
Usage: bootstrap_bergen.sh [--dry-run] [--env-dir PATH]

Create an isolated Python environment for BERGEN-based RAG benchmarks.
EOF
}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

DRY_RUN=0
ENV_DIR="$SCRIPT_DIR/.venv"
VENDOR_ROOT="$SCRIPT_DIR/vendor"
VENDOR_DIR="$VENDOR_ROOT/bergen"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements-bergen.txt"
BERGEN_REPO_URL="https://github.com/naver/bergen.git"
BERGEN_REF="${BERGEN_REF:-v0.1}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --env-dir)
            if [ "$#" -lt 2 ]; then
                echo "bootstrap_bergen.sh: --env-dir requires a value" >&2
                exit 2
            fi
            ENV_DIR="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "bootstrap_bergen.sh: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

print_plan() {
    printf '%s\n' "DRY RUN: BERGEN benchmark environment bootstrap"
    printf '%s\n' "step 1: create uv environment at $ENV_DIR"
    printf '%s\n' "  uv venv $ENV_DIR"
    printf '%s\n' "step 2: ensure BERGEN checkout at $VENDOR_DIR"
    printf '%s\n' "  git clone --branch $BERGEN_REF --depth 1 $BERGEN_REPO_URL $VENDOR_DIR"
    printf '%s\n' "step 3: install benchmark requirements with uv"
    printf '%s\n' "  uv pip install --python $ENV_DIR/bin/python -r benchmarks/rag/requirements-bergen.txt"
    printf '%s\n' "step 5: run BERGEN experiments from $VENDOR_DIR"
}

if [ "$DRY_RUN" -eq 1 ]; then
    print_plan
    exit 0
fi

mkdir -p "$VENDOR_ROOT" "$SCRIPT_DIR/results" "$SCRIPT_DIR/configs"

if ! command -v uv >/dev/null 2>&1; then
    echo "bootstrap_bergen.sh: uv is required but was not found in PATH" >&2
    exit 2
fi

uv venv "$ENV_DIR"

if [ ! -d "$VENDOR_DIR/.git" ]; then
    git clone --branch "$BERGEN_REF" --depth 1 "$BERGEN_REPO_URL" "$VENDOR_DIR"
else
    git -C "$VENDOR_DIR" fetch --tags origin
    git -C "$VENDOR_DIR" checkout "$BERGEN_REF"
fi

(
    cd "$REPO_ROOT"
    uv pip install --python "$ENV_DIR/bin/python" -r "benchmarks/rag/requirements-bergen.txt"
)

printf '%s\n' "BERGEN environment ready at $ENV_DIR"
printf '%s\n' "Vendored BERGEN checkout: $VENDOR_DIR"
