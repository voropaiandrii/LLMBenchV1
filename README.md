# LLMBenchV1

Universal streaming benchmark for self-hosted LLM servers. Measures **output tokens/sec**, **time-to-first-token (TTFT)**, **time-between-tokens (TBT)**, and per-run latency against any host you can reach by IP, hostname, or URL.

## Requirements

- Python 3.11+
- Network access to the LLM host from the machine where you run the benchmark

## Install

### Shell scripts (easiest)

From the repo root:

```bash
./setup.sh    # once — creates .venv and installs llm-bench
./bench.sh --help
./bench.sh --host gpu-server --port 11434 --api ollama --model llama3.2:3b
```

`./bench.sh` forwards all flags to `llm-bench` (same options as the CLI reference below). No `source .venv/bin/activate` needed.

Requires Python 3.11+ available via pyenv or on `PATH`. If you use pyenv, install the version from [`.python-version`](.python-version) first:

```bash
pyenv install 3.11.11 --skip-existing
./setup.sh
```

### Option A — pyenv without global Python shims (manual)

Use pyenv to **install** Python 3.11+, but create the venv with an **explicit interpreter path**. This avoids adding pyenv shims to your shell `PATH` (no `eval "$(pyenv init -)"` in `.zshrc` / `.bashrc`), so system `python3` is not redirected globally.

**1. Install pyenv**

macOS (Homebrew):

```bash
brew update
brew install pyenv
```

Linux (pyenv installer):

```bash
curl -fsSL https://pyenv.run | bash
# Ensure ~/.pyenv/bin is on PATH for the pyenv command only (not shims):
export PATH="$HOME/.pyenv/bin:$PATH"
```

**2. Install Python 3.11 for this project**

```bash
cd /path/to/LLMBenchV1
pyenv install 3.11.11 --skip-existing
```

The repo includes [`.python-version`](.python-version) (`3.11.11`) for pyenv-aware tools; you do not need `pyenv local` if you use the explicit path below.

**3. Create venv with the pyenv interpreter directly**

```bash
"$(pyenv root)/versions/3.11.11/bin/python" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

After activation, `python` and `pip` refer to `.venv` only — pyenv shims are not involved.

**Alternative without hardcoding the version string:**

```bash
PYENV_VERSION=3.11.11 pyenv exec python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

`pyenv exec` needs the `pyenv` binary on `PATH`, but still does not install shims that intercept every `python` command.

### Option B — pyenv with per-directory shims

If you already use `eval "$(pyenv init -)"` in your shell, the usual flow works:

```bash
cd /path/to/LLMBenchV1   # .python-version selects 3.11.11 here
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Shims apply only when you are inside this directory (or when `PYENV_VERSION` is set), not system-wide outside pyenv-managed paths.

### Option C — without pyenv

Use any Python 3.11+ interpreter directly:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```


## Quick start

With shell wrappers (after `./setup.sh`):

```bash
# Auto-detect API from host (default port 8000)
./bench.sh --host llm-server.local --model llama3.2:3b

# Direct Ollama
./bench.sh --host gpu-server --port 11434 --api ollama --model llama3.2:3b
```

Or with `llm-bench` directly (venv activated):

```bash
# Auto-detect API from host (default port 8000)
llm-bench --host llm-server.local --model llama3.2:3b

# OpenAI-compatible proxy (vLLM, SGLang, LM Studio, LiteLLM)
llm-bench \
  --base-url http://gpu-server:8000/v1 \
  --api openai \
  --model llama3.2:3b

# Direct Ollama
llm-bench \
  --host gpu-server \
  --port 11434 \
  --api ollama \
  --model llama3.2:3b

# Repeat runs + custom output folder
llm-bench \
  --host gpu-server --port 11434 --api ollama --model mistral:7b \
  --rounds 3 --max-tokens 512 --temperature 0.2 \
  --output-dir ./benchmark-results
```

## Supported backends

| API | Flag | Typical setup |
|-----|------|----------------|
| OpenAI-compatible | `--api openai` | vLLM, SGLang, LM Studio, LiteLLM, Ollama OpenAI mode on `:8000` |
| Native Ollama | `--api ollama` | Direct Ollama on `:11434`, `:11436`, etc. |
| Auto | `--api auto` (default) | Probes `/v1/models` and `/api/tags`; infers from URL shape if probes fail |

Model name is passed through unchanged (`llama3.2:3b`, `mistral:7b`, …).

## URL resolution

Priority:

1. `--base-url` (e.g. `http://gpu-server:8000/v1`)
2. `--host` + `--port` + `--api`
3. Environment variable `LLM_BASE_URL` (see [`.env.example`](.env.example))

**Do not mix APIs:** port `8000` with `/v1` is OpenAI-compatible — use `--api openai`, not `ollama`.

## CLI reference

### Target

| Flag | Description |
|------|-------------|
| `--host` | IP or hostname |
| `--port` | Port (default: 8000 for openai hint, 11434 for ollama) |
| `--base-url` | Full API root |
| `--api` | `auto`, `openai`, or `ollama` |
| `--model` | **Required** model id |
| `--api-key` | Bearer token (default: `local`) |
| `--endpoint` | Override stream path |

### Generation

| Flag | Description |
|------|-------------|
| `--max-tokens` | Max output tokens (default: 512) |
| `--temperature`, `--top-p` | Sampling |
| `--system`, `--prompt`, `--prompt-file` | Input text |
| `--prompt-profile structured_extract` | Realistic prefill using a structured-extract prompt |
| `--param KEY=VALUE` | Extra JSON body fields (repeatable) |
| `--rounds` | Repeat benchmark (default: 1) |
| `--timeout` | Request timeout seconds |

### Output

| Flag | Description |
|------|-------------|
| `--output-dir` | Result folder (default: `<repo>/output/llm-benchmarks/`, or `$LLM_BENCH_OUTPUT_DIR`) |
| `--output-file` | Exact JSON path (`.txt` uses same stem) |
| `--no-save` | Stdout only |
| `--json` | Also print JSON summary to stdout |

## Saved results

Every successful run writes two files (unless `--no-save`) under the results directory.

Default location: `<repo>/output/llm-benchmarks/`. Override with `--output-dir ./benchmark-results` or env `LLM_BENCH_OUTPUT_DIR`.

Filename pattern:

```text
output/llm-benchmarks/20260101_120000_gpu-server-11434_ollama_llama3.2-3b.json
output/llm-benchmarks/20260101_120000_gpu-server-11434_ollama_llama3.2-3b.txt
```

- **`.json`** — structured metrics (schema version 2); no API keys or full model output
- **`.txt`** — human-readable summary (same as console)

The default folder is gitignored. Check the absolute `Results directory:` line printed after each run.

## Metrics glossary

| Metric | JSON field | Meaning |
|--------|------------|---------|
| **TTFT** | `ttft_ms`, `prefill_sec` | Prefill phase: request → first output token |
| **TBT** | `tbt_ms_mean`, `tbt_ms_median`, `tbt_ms_p95` | Inter-chunk decode cadence during streaming |
| **TPS** | `tps`, `tok_per_sec` | Decode throughput: `completion_tokens / decode_sec` |
| **Latency** | `t_lat_sec`, `total_sec` | End-to-end wall clock through final token |

TBT is measured between **stream chunks** (SSE/NDJSON events), not individual tokens unless the server emits one token per chunk. When Ollama buffers output, TBT is estimated as `decode_sec / (tokens − 1)`.

For concurrency / load testing, use [vLLM benchmark scripts](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) instead of this single-request tool.

Token counts prefer server-reported values (OpenAI `usage`, Ollama `eval_count`). Use `--estimate-tokens` (default on) when the server omits counts.

### Ollama timing

Native Ollama (`--api ollama`) uses the server root URL **without** `/v1` (e.g. `http://gpu-server:11434`).

Some Ollama setups buffer the stream and only report `eval_count` / `eval_duration` on the final NDJSON chunk. In that case throughput uses Ollama's nanosecond timings.

## Troubleshooting

### `pip install -e ".[dev]"` fails

Upgrade pip inside your venv first — macOS system Python ships with pip 21.x, which cannot install hatchling projects in editable mode:

```bash
python -m pip install --upgrade pip
```

Ensure the venv was created with Python 3.11+ (`python --version` after `source .venv/bin/activate`). If you used pyenv without shims, recreate the venv with the explicit interpreter:

```bash
rm -rf .venv
"$(pyenv root)/versions/3.11.11/bin/python" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```


### Cannot reach host

1. Confirm network access: `ping gpu-server` and `curl http://gpu-server:11434/api/tags`
2. On macOS: **System Settings → Privacy & Security → Local Network** — allow your terminal app
3. Some IDE sandboxes block private LAN addresses; use a native terminal for LAN benchmarks
4. Use correct port: Ollama `:11434` / `:11436`, OpenAI proxy `:8000`

### Auto-detect failed

Pass API explicitly:

```bash
--api openai   # for ...:8000/v1
--api ollama   # for ...:11434
```

### Wrong API error

Using `--api ollama` against `http://host:8000/v1` sends requests to Ollama paths on the wrong port. Match API to how the server is actually exposed.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check llm_bench tests
```

## Module layout

```text
llm_bench/
  cli.py              # CLI entry point
  backends.py         # OpenAI + Ollama streaming adapters
  detect.py           # API auto-detect
  metrics.py          # tok/s aggregation
  output.py           # file write + human format
  runner.py           # orchestration
  models.py           # dataclasses
  prompts/            # built-in prompt profiles
```

## License

MIT — see [LICENSE](LICENSE).
