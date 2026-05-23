"""Format and persist benchmark output."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from llm_bench.models import BenchmarkReport, RunMetrics

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "llm-benchmarks"
OUTPUT_DIR_ENV = "LLM_BENCH_OUTPUT_DIR"


def resolve_output_dir(path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    env = os.environ.get(OUTPUT_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_OUTPUT_DIR


def model_slug(model: str) -> str:
    slug = re.sub(r"[/:\\s]+", "-", model.strip())
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug)
    return slug.strip("-") or "model"


def host_slug(base_url: str) -> str:
    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    host = parsed.hostname or "host"
    port = parsed.port
    label = f"{host}-{port}" if port is not None else host
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", label)
    return slug.strip("-") or "host"


def build_output_paths(
    *,
    output_dir: Path,
    output_file: Path | None,
    api: str,
    model: str,
    base_url: str,
    timestamp: datetime,
) -> tuple[Path, Path]:
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    slug = model_slug(model)
    host = host_slug(base_url)

    if output_file is not None:
        json_path = output_file.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path = json_path.with_suffix(".txt")
        return json_path, txt_path

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stamp}_{host}_{api}_{slug}"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.txt"


def format_human_report(report: BenchmarkReport) -> str:
    target = report.target
    lines = [
        f"LLM benchmark — {target.api} @ {target.base_url}",
        f"Model: {target.model} | rounds: {len(report.runs)} | "
        f"max_tokens: {report.config.get('max_tokens', '?')}",
        "",
    ]

    for run in report.runs:
        lines.append(_format_run_line(run))

    lines.append("")
    summary = report.summary
    if summary.tps_mean is not None:
        range_part = ""
        if summary.tok_per_sec_min is not None and summary.tok_per_sec_max is not None:
            range_part = f" ({summary.tok_per_sec_min:.1f}–{summary.tok_per_sec_max:.1f})"
        ttft_part = ""
        if summary.ttft_ms_mean is not None:
            ttft_part = f" | mean ttft {summary.ttft_ms_mean:.0f}ms"
        tbt_part = ""
        if summary.tbt_ms_mean is not None:
            tbt_part = f" | mean tbt {summary.tbt_ms_mean:.0f}ms"
        lines.append(f"Summary: mean {summary.tps_mean:.1f} tps{range_part}{ttft_part}{tbt_part}")
    else:
        lines.append("Summary: no throughput data")

    return "\n".join(lines)


def _format_run_line(run: RunMetrics) -> str:
    ttft = f"{run.ttft_ms:.0f}ms" if run.ttft_ms is not None else "n/a"
    if run.tbt_ms_mean is not None:
        tbt_label = "est" if run.tbt_source == "estimated" else "mean"
        tbt = f"{run.tbt_ms_mean:.0f}ms ({tbt_label})"
    else:
        tbt = "n/a"
    tps = f"{run.tps:.1f}" if run.tps is not None else "n/a"
    timing = f" [{run.timing_source}]" if run.timing_source != "stream" else ""
    return (
        f"Run {run.run}: ttft={ttft}  tbt={tbt}  tps={tps}  t_lat={run.t_lat_sec:.1f}s  "
        f"tokens={run.completion_tokens} ({run.token_source}){timing}"
    )


def save_report(
    report: BenchmarkReport,
    *,
    output_dir: Path,
    output_file: Path | None,
    no_save: bool,
) -> tuple[Path | None, Path | None]:
    if no_save:
        return None, None

    timestamp = datetime.fromisoformat(report.timestamp_utc.replace("Z", "+00:00"))
    json_path, txt_path = build_output_paths(
        output_dir=output_dir,
        output_file=output_file,
        api=report.target.api,
        model=report.target.model,
        base_url=report.target.base_url,
        timestamp=timestamp,
    )

    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    txt_path.write_text(format_human_report(report) + "\n", encoding="utf-8")
    return json_path, txt_path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
