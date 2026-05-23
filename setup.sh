#!/usr/bin/env bash
# One-time setup: create .venv and install llm-bench in editable mode.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_VERSION=""
if [[ -f "$ROOT/.python-version" ]]; then
  PYTHON_VERSION="$(tr -d '[:space:]' < "$ROOT/.python-version")"
fi

resolve_python() {
  local candidate=""

  if [[ -n "$PYTHON_VERSION" ]] && command -v pyenv >/dev/null 2>&1; then
    candidate="$(pyenv root 2>/dev/null)/versions/${PYTHON_VERSION}/bin/python"
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi

    if PYENV_VERSION="$PYTHON_VERSION" pyenv exec python -c "import sys" >/dev/null 2>&1; then
      echo "pyenv-exec"
      return 0
    fi
  fi

  for candidate in python3.11 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
        echo "$candidate"
        return 0
      fi
    fi
  done

  return 1
}

PYTHON="$(resolve_python)" || {
  cat >&2 <<EOF
error: Python 3.11+ is required but was not found.

If you use pyenv, install the project version first:
  pyenv install ${PYTHON_VERSION:-3.11.11} --skip-existing

Then re-run:
  ./setup.sh
EOF
  exit 1
}

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Creating virtual environment in .venv ..."
  if [[ "$PYTHON" == "pyenv-exec" ]]; then
    PYENV_VERSION="$PYTHON_VERSION" pyenv exec python -m venv "$ROOT/.venv"
  else
    "$PYTHON" -m venv "$ROOT/.venv"
  fi
else
  echo "Using existing .venv"
fi

VENV_PYTHON="$ROOT/.venv/bin/python"
VENV_PIP="$ROOT/.venv/bin/pip"

echo "Upgrading pip ..."
"$VENV_PYTHON" -m pip install --upgrade pip

echo "Installing llm-bench (editable) ..."
"$VENV_PIP" install -e ".[dev]"

cat <<EOF

Setup complete.

Run a benchmark:
  ./bench.sh --host gpu-server --port 11434 --api ollama --model llama3.2:3b

Show CLI help:
  ./bench.sh --help
EOF
