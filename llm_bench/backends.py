"""LLM backend adapters."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

import httpx

from llm_bench.models import GenerationConfig, StreamResult, TargetConfig


class LlmBackend(Protocol):
    name: str

    def stream_generate(
        self,
        client: httpx.Client,
        target: TargetConfig,
        config: GenerationConfig,
        *,
        api_key: str,
        estimate_tokens: bool,
    ) -> StreamResult: ...


class OpenAIAdapter:
    name = "openai"

    def stream_generate(
        self,
        client: httpx.Client,
        target: TargetConfig,
        config: GenerationConfig,
        *,
        api_key: str,
        estimate_tokens: bool,
    ) -> StreamResult:
        endpoint = target.endpoint or "/chat/completions"
        url = f"{target.base_url.rstrip('/')}{endpoint}"

        messages: list[dict[str, str]] = []
        if config.system:
            messages.append({"role": "system", "content": config.system})
        messages.append({"role": "user", "content": config.prompt})

        body: dict[str, Any] = {
            "model": target.model,
            "messages": messages,
            "stream": True,
            "max_tokens": config.max_tokens,
        }
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.top_p is not None:
            body["top_p"] = config.top_p
        body.update({k: v for k, v in config.extra_params.items() if not k.startswith("_")})

        headers = {"Authorization": f"Bearer {api_key}"}
        start = time.perf_counter()
        first_token_at: float | None = None
        last_token_at = start
        chunk_arrival_times: list[float] = []
        text_parts: list[str] = []
        completion_tokens: int | None = None
        prefill_tokens: int | None = None

        with client.stream("POST", url, json=body, headers=headers) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                usage = chunk.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens")
                    prefill_tokens = usage.get("prompt_tokens")

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    chunk_arrival_times.append(now)
                    text_parts.append(content)

        end = time.perf_counter()
        text = "".join(text_parts)
        token_source: str = "reported" if completion_tokens is not None else "unknown"
        if completion_tokens is None and estimate_tokens and text:
            completion_tokens = max(1, len(text) // 4)
            token_source = "estimated"

        generate_sec = (last_token_at - first_token_at) if first_token_at else 0.0
        ttft_sec = (first_token_at - start) if first_token_at else None

        return StreamResult(
            text=text,
            ttft_sec=ttft_sec,
            generate_sec=generate_sec,
            total_sec=end - start,
            completion_tokens=completion_tokens,
            prefill_tokens=prefill_tokens,
            token_source=token_source,  # type: ignore[arg-type]
            timing_source="stream",
            chunk_arrival_times=chunk_arrival_times,
        )


def _ns_to_sec(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000


class OllamaAdapter:
    name = "ollama"

    def stream_generate(
        self,
        client: httpx.Client,
        target: TargetConfig,
        config: GenerationConfig,
        *,
        api_key: str,
        estimate_tokens: bool,
    ) -> StreamResult:
        del api_key
        endpoint = target.endpoint or "/api/chat"
        base = target.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        url = f"{base}{endpoint}"

        messages: list[dict[str, str]] = []
        if config.system:
            messages.append({"role": "system", "content": config.system})
        messages.append({"role": "user", "content": config.prompt})

        options: dict[str, Any] = {"num_predict": config.max_tokens}
        if config.temperature is not None:
            options["temperature"] = config.temperature
        if config.top_p is not None:
            options["top_p"] = config.top_p
        options.update({k: v for k, v in config.extra_params.items() if not k.startswith("_")})

        body = {
            "model": target.model,
            "messages": messages,
            "stream": True,
            "options": options,
        }

        start = time.perf_counter()
        first_token_at: float | None = None
        last_token_at: float | None = None
        first_byte_at: float | None = None
        chunk_arrival_times: list[float] = []
        text_parts: list[str] = []
        completion_tokens: int | None = None
        prefill_tokens: int | None = None
        eval_duration_sec: float | None = None
        prompt_eval_duration_sec: float | None = None

        with client.stream("POST", url, json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                chunk = json.loads(line)

                if chunk.get("eval_count") is not None:
                    completion_tokens = chunk.get("eval_count")
                if chunk.get("prompt_eval_count") is not None:
                    prefill_tokens = chunk.get("prompt_eval_count")

                eval_duration_sec = _ns_to_sec(chunk.get("eval_duration")) or eval_duration_sec
                prompt_eval_duration_sec = (
                    _ns_to_sec(chunk.get("prompt_eval_duration")) or prompt_eval_duration_sec
                )

                message = chunk.get("message") or {}
                content = message.get("content")
                if not content:
                    content = chunk.get("response")
                if content:
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    chunk_arrival_times.append(now)
                    text_parts.append(content)

        end = time.perf_counter()
        text = "".join(text_parts)
        token_source: str = "reported" if completion_tokens is not None else "unknown"
        if completion_tokens is None and estimate_tokens and text:
            completion_tokens = max(1, len(text) // 4)
            token_source = "estimated"

        stream_generate_sec = (
            (last_token_at - first_token_at) if first_token_at and last_token_at else 0.0
        )
        timing_source: str = "stream"

        if stream_generate_sec > 0:
            generate_sec = stream_generate_sec
            ttft_sec = (first_token_at - start) if first_token_at else None
        elif eval_duration_sec is not None and eval_duration_sec > 0:
            generate_sec = eval_duration_sec
            ttft_sec = prompt_eval_duration_sec
            timing_source = "ollama_duration"
        else:
            generate_sec = 0.0
            ttft_sec = None
            if prompt_eval_duration_sec is not None:
                ttft_sec = prompt_eval_duration_sec
                timing_source = "ollama_duration"
            elif first_byte_at is not None:
                ttft_sec = first_byte_at - start

        total_sec = end - start
        if generate_sec <= 0 and total_sec > 0 and ttft_sec is not None:
            generate_sec = max(0.0, total_sec - ttft_sec)
            if generate_sec > 0:
                timing_source = "estimated"
        elif generate_sec <= 0 and total_sec > 0 and completion_tokens:
            generate_sec = total_sec
            timing_source = "estimated"

        return StreamResult(
            text=text,
            ttft_sec=ttft_sec,
            generate_sec=generate_sec,
            total_sec=total_sec,
            completion_tokens=completion_tokens,
            prefill_tokens=prefill_tokens,
            token_source=token_source,  # type: ignore[arg-type]
            timing_source=timing_source,  # type: ignore[arg-type]
            chunk_arrival_times=chunk_arrival_times,
        )


BACKENDS: dict[str, LlmBackend] = {
    "openai": OpenAIAdapter(),
    "ollama": OllamaAdapter(),
}


def get_backend(api: str) -> LlmBackend:
    backend = BACKENDS.get(api)
    if backend is None:
        raise ValueError(f"Unknown API backend: {api}")
    return backend
