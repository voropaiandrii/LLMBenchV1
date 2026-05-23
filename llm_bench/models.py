"""Data models for LLM benchmark results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float | None = None
    top_p: float | None = None
    system: str | None = None
    prompt: str = ""
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetConfig:
    base_url: str
    api: Literal["openai", "ollama"]
    model: str
    endpoint: str | None = None


@dataclass
class StreamResult:
    text: str
    ttft_sec: float | None
    generate_sec: float
    total_sec: float
    completion_tokens: int | None
    prefill_tokens: int | None
    token_source: Literal["reported", "estimated", "unknown"]
    timing_source: Literal["stream", "ollama_duration", "estimated"] = "stream"
    chunk_arrival_times: list[float] = field(default_factory=list)


@dataclass
class RunMetrics:
    run: int
    ttft_ms: float | None
    generate_sec: float
    total_sec: float
    completion_tokens: int
    prefill_tokens: int | None
    token_source: Literal["reported", "estimated", "unknown"]
    tok_per_sec: float | None
    timing_source: Literal["stream", "ollama_duration", "estimated"] = "stream"
    prefill_sec: float | None = None
    decode_sec: float = 0.0
    t_lat_sec: float = 0.0
    tps: float | None = None
    tbt_ms_mean: float | None = None
    tbt_ms_median: float | None = None
    tbt_ms_p95: float | None = None
    tbt_sample_count: int | None = None
    tbt_source: Literal["stream", "estimated", "n/a"] = "n/a"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "ttft_ms": round(self.ttft_ms, 1) if self.ttft_ms is not None else None,
            "generate_sec": round(self.generate_sec, 3),
            "total_sec": round(self.total_sec, 3),
            "completion_tokens": self.completion_tokens,
            "prefill_tokens": self.prefill_tokens,
            "token_source": self.token_source,
            "tok_per_sec": round(self.tok_per_sec, 2) if self.tok_per_sec is not None else None,
            "timing_source": self.timing_source,
            "prefill_sec": round(self.prefill_sec, 3) if self.prefill_sec is not None else None,
            "decode_sec": round(self.decode_sec, 3),
            "t_lat_sec": round(self.t_lat_sec, 3),
            "tps": round(self.tps, 2) if self.tps is not None else None,
            "tbt_ms_mean": round(self.tbt_ms_mean, 1) if self.tbt_ms_mean is not None else None,
            "tbt_ms_median": round(self.tbt_ms_median, 1) if self.tbt_ms_median is not None else None,
            "tbt_ms_p95": round(self.tbt_ms_p95, 1) if self.tbt_ms_p95 is not None else None,
            "tbt_sample_count": self.tbt_sample_count,
            "tbt_source": self.tbt_source,
        }


@dataclass
class BenchmarkSummary:
    tok_per_sec_mean: float | None
    tok_per_sec_min: float | None
    tok_per_sec_max: float | None
    ttft_ms_mean: float | None
    tps_mean: float | None = None
    tbt_ms_mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tok_per_sec_mean": round(self.tok_per_sec_mean, 2)
            if self.tok_per_sec_mean is not None
            else None,
            "tok_per_sec_min": round(self.tok_per_sec_min, 2)
            if self.tok_per_sec_min is not None
            else None,
            "tok_per_sec_max": round(self.tok_per_sec_max, 2)
            if self.tok_per_sec_max is not None
            else None,
            "ttft_ms_mean": round(self.ttft_ms_mean, 1) if self.ttft_ms_mean is not None else None,
            "tps_mean": round(self.tps_mean, 2) if self.tps_mean is not None else None,
            "tbt_ms_mean": round(self.tbt_ms_mean, 1) if self.tbt_ms_mean is not None else None,
        }


@dataclass
class BenchmarkReport:
    schema_version: int
    timestamp_utc: str
    target: TargetConfig
    config: dict[str, Any]
    runs: list[RunMetrics]
    summary: BenchmarkSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp_utc": self.timestamp_utc,
            "target": {
                "base_url": self.target.base_url,
                "api": self.target.api,
                "model": self.target.model,
                "endpoint": self.target.endpoint,
            },
            "config": self.config,
            "runs": [run.to_dict() for run in self.runs],
            "summary": self.summary.to_dict(),
        }
