"""Aggregate benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llm_bench.models import BenchmarkSummary, RunMetrics, StreamResult


@dataclass
class TbtStats:
    tbt_ms_mean: float | None
    tbt_ms_median: float | None
    tbt_ms_p95: float | None
    tbt_sample_count: int | None
    tbt_source: Literal["stream", "estimated", "n/a"]


def compute_inter_chunk_ms(times: list[float]) -> list[float]:
    return [(t2 - t1) * 1000 for t1, t2 in zip(times, times[1:])]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if pct == 50:
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[idx]


def compute_tbt_stats(
    chunk_arrival_times: list[float],
    *,
    decode_sec: float,
    completion_tokens: int,
    timing_source: str,
) -> TbtStats:
    deltas = compute_inter_chunk_ms(chunk_arrival_times)
    if len(deltas) >= 1 and timing_source == "stream":
        return TbtStats(
            tbt_ms_mean=sum(deltas) / len(deltas),
            tbt_ms_median=_percentile(deltas, 50),
            tbt_ms_p95=_percentile(deltas, 95),
            tbt_sample_count=len(deltas),
            tbt_source="stream",
        )

    if (
        timing_source in {"ollama_duration", "estimated"}
        and decode_sec > 0
        and completion_tokens > 1
    ):
        mean_ms = (decode_sec / max(1, completion_tokens - 1)) * 1000
        return TbtStats(
            tbt_ms_mean=mean_ms,
            tbt_ms_median=mean_ms,
            tbt_ms_p95=mean_ms,
            tbt_sample_count=completion_tokens - 1,
            tbt_source="estimated",
        )

    return TbtStats(None, None, None, None, "n/a")


def stream_to_run_metrics(
    run: int,
    result: StreamResult,
    *,
    estimate_tokens: bool,
) -> RunMetrics:
    completion_tokens = result.completion_tokens
    token_source = result.token_source
    timing_source = result.timing_source
    generate_sec = result.generate_sec
    ttft_sec = result.ttft_sec

    if completion_tokens is None and estimate_tokens and result.text:
        completion_tokens = max(1, len(result.text) // 4)
        token_source = "estimated"
    elif completion_tokens is None:
        completion_tokens = 0
        token_source = "unknown"

    if generate_sec <= 0 and result.total_sec > 0 and completion_tokens > 0:
        if ttft_sec is not None:
            fallback = max(0.0, result.total_sec - ttft_sec)
            if fallback > 0:
                generate_sec = fallback
                if timing_source == "stream":
                    timing_source = "estimated"
        elif result.total_sec > 0:
            generate_sec = result.total_sec
            timing_source = "estimated"

    tok_per_sec: float | None = None
    if generate_sec > 0 and completion_tokens > 0:
        tok_per_sec = completion_tokens / generate_sec

    ttft_ms = ttft_sec * 1000 if ttft_sec is not None else None
    tbt = compute_tbt_stats(
        result.chunk_arrival_times,
        decode_sec=generate_sec,
        completion_tokens=completion_tokens,
        timing_source=timing_source,
    )

    return RunMetrics(
        run=run,
        ttft_ms=ttft_ms,
        generate_sec=generate_sec,
        total_sec=result.total_sec,
        completion_tokens=completion_tokens,
        prefill_tokens=result.prefill_tokens,
        token_source=token_source,
        tok_per_sec=tok_per_sec,
        timing_source=timing_source,
        prefill_sec=ttft_sec,
        decode_sec=generate_sec,
        t_lat_sec=result.total_sec,
        tps=tok_per_sec,
        tbt_ms_mean=tbt.tbt_ms_mean,
        tbt_ms_median=tbt.tbt_ms_median,
        tbt_ms_p95=tbt.tbt_ms_p95,
        tbt_sample_count=tbt.tbt_sample_count,
        tbt_source=tbt.tbt_source,
    )


def summarize_runs(runs: list[RunMetrics]) -> BenchmarkSummary:
    tok_rates = [r.tok_per_sec for r in runs if r.tok_per_sec is not None]
    ttfts = [r.ttft_ms for r in runs if r.ttft_ms is not None]
    tbts = [r.tbt_ms_mean for r in runs if r.tbt_ms_mean is not None]

    tok_mean = sum(tok_rates) / len(tok_rates) if tok_rates else None

    return BenchmarkSummary(
        tok_per_sec_mean=tok_mean,
        tok_per_sec_min=min(tok_rates) if tok_rates else None,
        tok_per_sec_max=max(tok_rates) if tok_rates else None,
        ttft_ms_mean=sum(ttfts) / len(ttfts) if ttfts else None,
        tps_mean=tok_mean,
        tbt_ms_mean=sum(tbts) / len(tbts) if tbts else None,
    )
