"""Unit tests for LLM benchmark helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from llm_bench.backends import OllamaAdapter, OpenAIAdapter
from llm_bench.detect import detect_api, host_root_url, infer_api_from_url, normalize_root_url
from llm_bench.metrics import (
    compute_inter_chunk_ms,
    compute_tbt_stats,
    stream_to_run_metrics,
    summarize_runs,
)
from llm_bench.models import BenchmarkReport, GenerationConfig, StreamResult, TargetConfig
from llm_bench.output import (
    DEFAULT_OUTPUT_DIR,
    build_output_paths,
    format_human_report,
    host_slug,
    model_slug,
    resolve_output_dir,
    save_report,
)
from llm_bench.runner import BenchmarkRunner, ResolvedTarget, load_prompt, resolve_target

OPENAI_SSE = (
    'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":" world"}}],'
    '"usage":{"completion_tokens":2,"prompt_tokens":5}}\n\n'
    "data: [DONE]\n\n"
)

OLLAMA_NDJSON = (
    json.dumps({"message": {"content": "Hi"}, "done": False})
    + "\n"
    + json.dumps(
        {
            "message": {"content": "!"},
            "done": True,
            "eval_count": 2,
            "prompt_eval_count": 4,
        }
    )
    + "\n"
)


def test_model_slug():
    assert model_slug("llama3.2:3b") == "llama3.2-3b"
    assert model_slug("org/model/name") == "org-model-name"


def test_host_slug():
    assert host_slug("http://192.168.1.100:11434") == "192.168.1.100-11434"
    assert host_slug("http://gpu-server:8000/v1") == "gpu-server-8000"
    assert host_slug("gpu-server") == "gpu-server"


def test_default_output_dir_under_repo():
    repo_root = Path(__file__).resolve().parents[2]
    assert DEFAULT_OUTPUT_DIR == repo_root / "output" / "llm-benchmarks"


def test_resolve_output_dir_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LLM_BENCH_OUTPUT_DIR", str(tmp_path / "from-env"))
    assert resolve_output_dir() == (tmp_path / "from-env").resolve()
    assert resolve_output_dir(tmp_path / "explicit") == (tmp_path / "explicit").resolve()


def test_normalize_root_url():
    assert normalize_root_url("http://host:8000", "openai") == "http://host:8000/v1"
    assert normalize_root_url("http://host:8000/v1", "openai") == "http://host:8000/v1"
    assert normalize_root_url("http://host:11434", "ollama") == "http://host:11434"
    assert normalize_root_url("http://host:11434/v1", "ollama") == "http://host:11434"


def test_resolve_target_ollama_host():
    client = httpx.Client()
    try:
        target = resolve_target(
            host="192.168.1.100",
            port=11434,
            base_url=None,
            api="ollama",
            model="llama3.2:3b",
            endpoint=None,
            client=client,
        )
    finally:
        client.close()

    assert target.base_url == "http://192.168.1.100:11434"
    assert target.api == "ollama"


def test_host_root_url():
    assert host_root_url("gpu-server", 8000, "openai") == "http://gpu-server:8000/v1"


def test_load_prompt_structured_extract():
    prompt = load_prompt(prompt=None, prompt_file=None, prompt_profile="structured_extract")
    assert "Product candidate:" in prompt
    assert "Widget Pro" in prompt
    assert "Return JSON" in prompt


def test_stream_to_run_metrics_reported():
    result = StreamResult(
        text="hello world",
        ttft_sec=0.5,
        generate_sec=2.0,
        total_sec=2.5,
        completion_tokens=100,
        prefill_tokens=20,
        token_source="reported",
    )
    metrics = stream_to_run_metrics(1, result, estimate_tokens=True)
    assert metrics.completion_tokens == 100
    assert metrics.tok_per_sec == 50.0
    assert metrics.ttft_ms == 500.0


def test_stream_to_run_metrics_estimated():
    result = StreamResult(
        text="abcd" * 10,
        ttft_sec=0.1,
        generate_sec=1.0,
        total_sec=1.1,
        completion_tokens=None,
        prefill_tokens=None,
        token_source="unknown",
    )
    metrics = stream_to_run_metrics(1, result, estimate_tokens=True)
    assert metrics.token_source == "estimated"
    assert metrics.completion_tokens == 10


def test_stream_to_run_metrics_generate_fallback():
    result = StreamResult(
        text="",
        ttft_sec=0.5,
        generate_sec=0.0,
        total_sec=8.5,
        completion_tokens=512,
        prefill_tokens=20,
        token_source="reported",
        timing_source="stream",
    )
    metrics = stream_to_run_metrics(1, result, estimate_tokens=False)
    assert metrics.generate_sec == 8.0
    assert metrics.tok_per_sec == 64.0
    assert metrics.timing_source == "estimated"
    assert metrics.tbt_source == "estimated"
    assert metrics.tbt_ms_mean == pytest.approx(8000 / 511, rel=1e-3)


def test_compute_tbt_from_chunks():
    times = [0.0, 0.01, 0.03, 0.06, 0.10]
    deltas = compute_inter_chunk_ms(times)
    assert deltas == pytest.approx([10.0, 20.0, 30.0, 40.0])

    stats = compute_tbt_stats(times, decode_sec=0.1, completion_tokens=10, timing_source="stream")
    assert stats.tbt_source == "stream"
    assert stats.tbt_sample_count == 4
    assert stats.tbt_ms_mean == 25.0
    assert stats.tbt_ms_median == 25.0
    assert stats.tbt_ms_p95 == pytest.approx(40.0)


def test_compute_tbt_estimated_from_decode():
    stats = compute_tbt_stats([], decode_sec=8.0, completion_tokens=512, timing_source="ollama_duration")
    assert stats.tbt_source == "estimated"
    assert stats.tbt_ms_mean == pytest.approx(8000 / 511, rel=1e-3)
    assert stats.tbt_sample_count == 511


def test_summarize_runs():
    runs = [
        stream_to_run_metrics(
            1,
            StreamResult("x", 0.1, 1.0, 1.1, 50, None, "reported"),
            estimate_tokens=False,
        ),
        stream_to_run_metrics(
            2,
            StreamResult("x", 0.2, 1.0, 1.2, 100, None, "reported"),
            estimate_tokens=False,
        ),
    ]
    summary = summarize_runs(runs)
    assert summary.tok_per_sec_mean == 75.0
    assert summary.tok_per_sec_min == 50.0
    assert summary.tok_per_sec_max == 100.0


def test_build_output_paths_auto():
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    json_path, txt_path = build_output_paths(
        output_dir=Path("/tmp/out"),
        output_file=None,
        api="ollama",
        model="llama3.2:3b",
        base_url="http://gpu-server:11434",
        timestamp=ts,
    )
    assert json_path.name == "20260101_120000_gpu-server-11434_ollama_llama3.2-3b.json"
    assert txt_path.name == "20260101_120000_gpu-server-11434_ollama_llama3.2-3b.txt"


def test_build_output_paths_explicit_file(tmp_path: Path):
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    json_path, txt_path = build_output_paths(
        output_dir=tmp_path,
        output_file=tmp_path / "custom" / "run.json",
        api="openai",
        model="m",
        base_url="http://host/v1",
        timestamp=ts,
    )
    assert json_path == tmp_path / "custom" / "run.json"
    assert txt_path == tmp_path / "custom" / "run.txt"


def test_save_report_writes_files(tmp_path: Path):
    run = stream_to_run_metrics(
        1,
        StreamResult("ok", 0.4, 2.0, 2.4, 80, 10, "reported"),
        estimate_tokens=False,
    )
    report = BenchmarkReport(
        schema_version=1,
        timestamp_utc="2026-01-01T12:00:00Z",
        target=TargetConfig("http://host/v1", "openai", "llama3.2:3b"),
        config={"max_tokens": 128, "rounds": 1},
        runs=[run],
        summary=summarize_runs([run]),
    )
    json_path, txt_path = save_report(
        report,
        output_dir=tmp_path,
        output_file=None,
        no_save=False,
    )
    assert json_path is not None and json_path.exists()
    assert txt_path is not None and txt_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["target"]["model"] == "llama3.2:3b"
    assert "api_key" not in json_path.read_text()


def test_format_human_report_contains_summary():
    run = stream_to_run_metrics(
        1,
        StreamResult("ok", 0.4, 2.0, 2.4, 80, None, "reported"),
        estimate_tokens=False,
    )
    report = BenchmarkReport(
        schema_version=1,
        timestamp_utc="2026-01-01T12:00:00Z",
        target=TargetConfig("http://host/v1", "openai", "m"),
        config={"max_tokens": 512, "rounds": 1},
        runs=[run],
        summary=summarize_runs([run]),
    )
    text = format_human_report(report)
    assert "LLM benchmark" in text
    assert "Summary:" in text
    assert "tps=" in text
    assert "tbt=" in text


def _openai_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": []})
    if request.url.path == "/v1/chat/completions":
        return httpx.Response(200, content=OPENAI_SSE.encode())
    return httpx.Response(404)


def _ollama_detect_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(404)
    if request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": []})
    return httpx.Response(404)


def test_detect_api_openai():
    client = httpx.Client(transport=httpx.MockTransport(_openai_handler))
    assert detect_api(client, "http://host:8000", port=8000) == "openai"


def test_detect_api_ollama():
    client = httpx.Client(transport=httpx.MockTransport(_ollama_detect_handler))
    assert detect_api(client, "http://host:8000", port=8000) == "ollama"


def test_infer_api_from_v1_url():
    assert infer_api_from_url("http://gpu-server:8000/v1") == "openai"
    assert infer_api_from_url("http://host:11434") == "ollama"


def test_openai_adapter_stream():
    client = httpx.Client(transport=httpx.MockTransport(_openai_handler))
    adapter = OpenAIAdapter()
    target = TargetConfig("http://host/v1", "openai", "test-model")
    config = GenerationConfig(prompt="hi", max_tokens=32)

    result = adapter.stream_generate(
        client,
        target,
        config,
        api_key="local",
        estimate_tokens=False,
    )

    assert result.text == "Hello world"
    assert result.completion_tokens == 2
    assert result.prefill_tokens == 5
    assert result.token_source == "reported"
    assert len(result.chunk_arrival_times) == 2

    metrics = stream_to_run_metrics(1, result, estimate_tokens=False)
    assert metrics.tbt_source == "stream"
    assert metrics.tbt_sample_count == 1
    assert metrics.tps == metrics.tok_per_sec


def test_ollama_adapter_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            return httpx.Response(200, content=OLLAMA_NDJSON.encode())
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OllamaAdapter()
    target = TargetConfig("http://host:11434", "ollama", "test-model")
    config = GenerationConfig(prompt="hi", max_tokens=32)

    result = adapter.stream_generate(
        client,
        target,
        config,
        api_key="local",
        estimate_tokens=False,
    )

    assert result.text == "Hi!"
    assert result.completion_tokens == 2
    assert result.prefill_tokens == 4
    assert result.timing_source == "stream"


def test_ollama_adapter_final_only_chunk():
    payload = json.dumps(
        {
            "done": True,
            "eval_count": 512,
            "eval_duration": 8_000_000_000,
            "prompt_eval_duration": 420_000_000,
            "message": {"content": ""},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            return httpx.Response(200, content=(payload + "\n").encode())
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OllamaAdapter()
    target = TargetConfig("http://host:11434", "ollama", "test-model")
    config = GenerationConfig(prompt="hi", max_tokens=512)

    result = adapter.stream_generate(
        client,
        target,
        config,
        api_key="local",
        estimate_tokens=False,
    )

    assert result.text == ""
    assert result.completion_tokens == 512
    assert result.generate_sec == 8.0
    assert result.ttft_sec == 0.42
    assert result.timing_source == "ollama_duration"

    metrics = stream_to_run_metrics(1, result, estimate_tokens=False)
    assert metrics.tok_per_sec == 64.0
    assert metrics.ttft_ms == 420.0
    assert metrics.timing_source == "ollama_duration"
    assert metrics.tbt_source == "estimated"
    assert metrics.tbt_ms_mean == pytest.approx(8000 / 511, rel=1e-3)


def test_benchmark_runner_integration():
    client = httpx.Client(transport=httpx.MockTransport(_openai_handler))
    runner = BenchmarkRunner(client, estimate_tokens=False)
    target = ResolvedTarget("http://host/v1", "openai", "m", None)
    report = runner.run(
        target,
        GenerationConfig(prompt="test", max_tokens=16),
        rounds=2,
    )

    assert len(report.runs) == 2
    assert report.schema_version == 2
    assert report.runs[0].completion_tokens == 2
    assert report.runs[0].tps == report.runs[0].tok_per_sec
    assert report.summary.tps_mean is not None


def test_run_metrics_schema_v2():
    run = stream_to_run_metrics(
        1,
        StreamResult(
            "ok",
            0.4,
            2.0,
            2.4,
            80,
            10,
            "reported",
            chunk_arrival_times=[1.0, 1.01, 1.03],
        ),
        estimate_tokens=False,
    )
    report = BenchmarkReport(
        schema_version=2,
        timestamp_utc="2026-01-01T12:00:00Z",
        target=TargetConfig("http://host/v1", "openai", "m"),
        config={"max_tokens": 512, "rounds": 1},
        runs=[run],
        summary=summarize_runs([run]),
    )
    payload = report.to_dict()
    run_payload = payload["runs"][0]
    assert payload["schema_version"] == 2
    assert run_payload["tps"] == run_payload["tok_per_sec"]
    assert run_payload["t_lat_sec"] == 2.4
    assert run_payload["decode_sec"] == 2.0
    assert run_payload["tbt_ms_mean"] is not None
    assert run_payload["tbt_source"] == "stream"
    assert payload["summary"]["tps_mean"] == payload["summary"]["tok_per_sec_mean"]
