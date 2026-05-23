"""API auto-detection for LLM backends."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

import httpx

ApiKind = Literal["openai", "ollama"]


def normalize_root_url(base_url: str, api: ApiKind) -> str:
    url = base_url.rstrip("/")
    if api == "openai":
        if url.endswith("/v1"):
            return url
        if "/v1/" in url:
            return url.split("/v1/")[0] + "/v1"
        return url + "/v1"
    if url.endswith("/v1"):
        return url[:-3]
    return url


def host_root_url(host: str, port: int, api: ApiKind) -> str:
    host = host.strip()
    if host.startswith("http://") or host.startswith("https://"):
        return normalize_root_url(host, api)
    scheme = "http"
    root = f"{scheme}://{host}:{port}"
    return normalize_root_url(root, api)


def probe_openai(client: httpx.Client, root: str) -> bool:
    try:
        response = client.get(f"{root.rstrip('/')}/models")
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def probe_ollama(client: httpx.Client, root: str) -> bool:
    base = root.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        response = client.get(f"{base}/api/tags")
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def port_from_url(root: str) -> int | None:
    if "://" not in root:
        return None
    return urlparse(root).port


def infer_api_from_url(root: str, port: int | None = None) -> ApiKind | None:
    """Guess API from URL shape when live probes are unavailable (offline / firewalled)."""
    url = root.rstrip("/")
    if url.endswith("/v1") or "/v1/" in url:
        return "openai"

    effective_port = port if port is not None else port_from_url(root)
    if effective_port in (11434, 11436):
        return "ollama"
    if effective_port == 8000:
        return "openai"
    if "/api/" in url:
        return "ollama"
    return None


def detect_api(
    client: httpx.Client,
    root: str,
    *,
    port: int | None = None,
    preferred: ApiKind | None = None,
) -> ApiKind:
    if preferred == "openai":
        openai_root = normalize_root_url(root, "openai")
        if probe_openai(client, openai_root):
            return "openai"
        inferred = infer_api_from_url(root, port)
        if inferred == "openai":
            return "openai"
        raise RuntimeError(
            f"OpenAI-compatible probe failed for {openai_root}. "
            "Check host reachability and try --api openai if the server is up."
        )

    if preferred == "ollama":
        ollama_root = normalize_root_url(root, "ollama")
        if probe_ollama(client, ollama_root):
            return "ollama"
        inferred = infer_api_from_url(root, port)
        if inferred == "ollama":
            return "ollama"
        raise RuntimeError(
            f"Ollama probe failed for {ollama_root}. "
            "Check host reachability and port (11434/11436)."
        )

    openai_root = normalize_root_url(root, "openai")
    if probe_openai(client, openai_root):
        return "openai"

    ollama_root = normalize_root_url(root, "ollama")
    if probe_ollama(client, ollama_root):
        return "ollama"

    inferred = infer_api_from_url(root, port)
    if inferred is not None:
        return inferred

    if port == 8000:
        return "openai"
    if port in (11434, 11436):
        return "ollama"

    raise RuntimeError(
        "Could not auto-detect API for "
        f"{root}. Try --api openai (port 8000, URL ending in /v1) "
        "or --api ollama (port 11434/11436). "
        "If the host is correct, verify network route/DNS (ping/curl)."
    )
