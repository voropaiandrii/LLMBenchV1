#!/usr/bin/env python3
"""Universal LLM throughput benchmark — tokens/sec and latency metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

from llm_bench.models import GenerationConfig
from llm_bench.output import (
    DEFAULT_OUTPUT_DIR,
    OUTPUT_DIR_ENV,
    format_human_report,
    resolve_output_dir,
    save_report,
)
from llm_bench.runner import BenchmarkRunner, load_prompt, resolve_target


def _parse_extra_params(values: list[str]) -> dict:
    params: dict = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--param expects key=value, got: {item}")
        key, val = item.split("=", 1)
        params[key.strip()] = _coerce_value(val.strip())
    return params


def _coerce_value(raw: str):
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark LLM output throughput (tokens/sec) via streaming APIs.",
    )
    target = parser.add_argument_group("target")
    target.add_argument("--host", help="Host/IP/hostname (e.g. gpu-server or 192.168.1.100)")
    target.add_argument("--port", type=int, help="Port (default: 8000 openai, 11434 ollama)")
    target.add_argument("--base-url", help="Full API root (e.g. http://gpu-server:8000/v1)")
    target.add_argument(
        "--api",
        choices=["auto", "openai", "ollama"],
        default="auto",
        help="API backend (default: auto-detect)",
    )
    target.add_argument("--endpoint", help="Override stream endpoint path")
    target.add_argument("--model", required=True, help="Model id/name")
    target.add_argument("--api-key", default="local", help="Bearer API key (default: local)")

    gen = parser.add_argument_group("generation")
    gen.add_argument("--max-tokens", type=int, default=512)
    gen.add_argument("--temperature", type=float)
    gen.add_argument("--top-p", type=float)
    gen.add_argument("--system", help="Optional system message")
    gen.add_argument("--prompt", help="User prompt text")
    gen.add_argument("--prompt-file", type=Path, help="Load user prompt from file")
    gen.add_argument(
        "--prompt-profile",
        choices=["structured_extract"],
        help="Built-in prompt profile for realistic prefill",
    )
    gen.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra JSON body params (repeatable)",
    )

    run = parser.add_argument_group("run")
    run.add_argument("--rounds", type=int, default=1)
    run.add_argument("--timeout", type=float, default=120.0)
    run.add_argument(
        "--estimate-tokens",
        action="store_true",
        default=True,
        help="Estimate tokens from output length if server omits counts (default: on)",
    )
    run.add_argument(
        "--no-estimate-tokens",
        action="store_false",
        dest="estimate_tokens",
        help="Do not estimate missing token counts",
    )

    out = parser.add_argument_group("output")
    out.add_argument("--json", action="store_true", help="Also print JSON summary to stdout")
    out.add_argument("--no-save", action="store_true", help="Skip writing result files")
    out.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            f"Output folder (default: {DEFAULT_OUTPUT_DIR}, "
            f"or ${OUTPUT_DIR_ENV} if set)"
        ),
    )
    out.add_argument("--output-file", type=Path, help="Exact JSON output path")

    return parser


def _friendly_http_error(exc: Exception, target_url: str = "") -> str:
    if isinstance(exc, httpx.ConnectError):
        host_hint = f" ({target_url})" if target_url else ""
        return (
            f"Cannot reach LLM server{host_hint}: {exc}\n"
            "Check: same LAN/VPN, correct IP/hostname, service running, firewall.\n"
            "Examples:\n"
            "  OpenAI-compatible: --base-url http://gpu-server:8000/v1 --api openai\n"
            "  Ollama direct:     --host gpu-server --port 11434 --api ollama"
        )
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code} from LLM server: {exc.request.url}"
    return str(exc)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.host and not args.base_url and not os.environ.get("LLM_BASE_URL"):
        parser.error("Provide --host, --base-url, or set LLM_BASE_URL")

    extra = _parse_extra_params(args.param)
    if args.prompt_profile:
        extra["_prompt_profile"] = args.prompt_profile

    prompt = load_prompt(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        prompt_profile=args.prompt_profile,
    )

    gen_config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system=args.system,
        prompt=prompt,
        extra_params=extra,
    )

    timeout = httpx.Timeout(args.timeout, connect=min(10.0, args.timeout))
    target_url = args.base_url or args.host or os.environ.get("LLM_BASE_URL", "")
    try:
        with httpx.Client(timeout=timeout) as client:
            target = resolve_target(
                host=args.host,
                port=args.port,
                base_url=args.base_url,
                api=args.api,
                model=args.model,
                endpoint=args.endpoint,
                client=client,
            )
            target_url = target.base_url
            runner = BenchmarkRunner(
                client,
                api_key=args.api_key,
                estimate_tokens=args.estimate_tokens,
            )
            report = runner.run(target, gen_config, rounds=args.rounds)
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        print(f"error: {_friendly_http_error(exc, target_url)}", file=sys.stderr)
        return 1

    human = format_human_report(report)
    print(human)

    output_dir = resolve_output_dir(args.output_dir)
    try:
        json_path, txt_path = save_report(
            report,
            output_dir=output_dir,
            output_file=args.output_file,
            no_save=args.no_save,
        )
    except OSError as exc:
        print(f"error: failed to save benchmark results to {output_dir}: {exc}", file=sys.stderr)
        return 1

    if json_path and txt_path:
        print(f"\nResults directory: {json_path.parent.resolve()}")
        print(f"Saved: {json_path.resolve()}")
        print(f"Saved: {txt_path.resolve()}")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
