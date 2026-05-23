#!/usr/bin/env bash
# Run llm-bench using the project virtualenv (no manual activation required).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_BENCH="$ROOT/.venv/bin/llm-bench"

if [[ ! -x "$LLM_BENCH" ]]; then
  cat >&2 <<EOF
error: $LLM_BENCH not found.

Run setup first:
  ./setup.sh
EOF
  exit 1
fi

exec "$LLM_BENCH" "$@"
