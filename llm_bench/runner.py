"""Benchmark orchestration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from llm_bench.backends import get_backend
from llm_bench.detect import detect_api, host_root_url, normalize_root_url, port_from_url
from llm_bench.metrics import stream_to_run_metrics, summarize_runs
from llm_bench.models import BenchmarkReport, GenerationConfig, TargetConfig
from llm_bench.output import utc_now_iso

DEFAULT_PROMPT = (
    "Write a concise technical summary of how a homelab LLM inference server works, "
    "including model loading, token streaming, batching, and typical bottlenecks on "
    "GPU and CPU. Use clear paragraphs and keep the answer informative."
)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

STRUCTURED_EXTRACT_FIXTURE = (
    "Product candidate:\n"
    "Source: catalog.example.com\n"
    "URL: https://example.com/products/widget-pro-w123\n"
    "Name: Widget Pro\n"
    "Price: USD 49.99 | SKU: W-123 | Category: Electronics\n"
    "Notes: Compact form factor, USB-C, 2-year warranty.\n"
)


@dataclass
class ResolvedTarget:
    base_url: str
    api: Literal["openai", "ollama"]
    model: str
    endpoint: str | None


def resolve_target(
    *,
    host: str | None,
    port: int | None,
    base_url: str | None,
    api: str,
    model: str,
    endpoint: str | None,
    client: httpx.Client,
) -> ResolvedTarget:
    env_base = os.environ.get("LLM_BASE_URL")

    if base_url:
        root = base_url.rstrip("/")
        chosen_port = port if port is not None else port_from_url(root)
    elif host:
        api_hint = None if api == "auto" else api  # type: ignore[arg-type]
        if port is None:
            if api_hint == "ollama":
                port = 11434
            elif api_hint == "openai":
                port = 8000
            else:
                port = 8000
        root = host_root_url(host, port, api_hint if api_hint else "openai")  # type: ignore[arg-type]
        chosen_port = port
    elif env_base:
        root = env_base.rstrip("/")
        chosen_port = port
    else:
        raise ValueError("Provide --host, --base-url, or set LLM_BASE_URL")

    if api == "auto":
        detected = detect_api(client, root, port=chosen_port)
        normalized = normalize_root_url(root, detected)
        return ResolvedTarget(
            base_url=normalized,
            api=detected,
            model=model,
            endpoint=endpoint,
        )

    normalized = normalize_root_url(root, api)  # type: ignore[arg-type]
    return ResolvedTarget(
        base_url=normalized,
        api=api,  # type: ignore[arg-type]
        model=model,
        endpoint=endpoint,
    )


def load_prompt(
    *,
    prompt: str | None,
    prompt_file: Path | None,
    prompt_profile: str | None,
) -> str:
    if prompt:
        return prompt
    if prompt_file is not None:
        return prompt_file.read_text(encoding="utf-8")
    if prompt_profile == "structured_extract":
        template_path = PROMPTS_DIR / "structured_extract.md"
        if template_path.exists():
            template = template_path.read_text(encoding="utf-8")
            return template.replace("{{document}}", STRUCTURED_EXTRACT_FIXTURE)
        return STRUCTURED_EXTRACT_FIXTURE
    return DEFAULT_PROMPT


class BenchmarkRunner:
    def __init__(
        self,
        client: httpx.Client,
        *,
        api_key: str = "local",
        estimate_tokens: bool = True,
    ) -> None:
        self.client = client
        self.api_key = api_key
        self.estimate_tokens = estimate_tokens

    def run(
        self,
        target: ResolvedTarget,
        gen_config: GenerationConfig,
        *,
        rounds: int = 1,
    ) -> BenchmarkReport:
        backend = get_backend(target.api)
        target_cfg = TargetConfig(
            base_url=target.base_url,
            api=target.api,
            model=target.model,
            endpoint=target.endpoint,
        )

        run_metrics = []
        for run_idx in range(1, rounds + 1):
            stream_result = backend.stream_generate(
                self.client,
                target_cfg,
                gen_config,
                api_key=self.api_key,
                estimate_tokens=self.estimate_tokens,
            )
            run_metrics.append(
                stream_to_run_metrics(
                    run_idx,
                    stream_result,
                    estimate_tokens=self.estimate_tokens,
                )
            )

        config_dict: dict[str, Any] = {
            "max_tokens": gen_config.max_tokens,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "rounds": rounds,
            "prompt_profile": gen_config.extra_params.get("_prompt_profile"),
        }
        if gen_config.system:
            config_dict["system"] = gen_config.system

        return BenchmarkReport(
            schema_version=2,
            timestamp_utc=utc_now_iso(),
            target=target_cfg,
            config={k: v for k, v in config_dict.items() if v is not None},
            runs=run_metrics,
            summary=summarize_runs(run_metrics),
        )
