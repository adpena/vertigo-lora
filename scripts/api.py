#!/usr/bin/env python3
from __future__ import annotations

"""Shared LLM API client — single source of truth for OpenAI-compatible calls."""

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_API_BASE = "http://127.0.0.1:1234/v1"
DEFAULT_TIMEOUT = 120


def call_llm(
    messages: list[dict[str, Any]],
    *,
    model: str = "",
    api_base: str = DEFAULT_API_BASE,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Send a chat completion request to an OpenAI-compatible API."""
    body: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        body["model"] = model

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def detect_model(api_base: str = DEFAULT_API_BASE) -> str:
    """Auto-detect the first available model from the API."""
    url = f"{api_base.rstrip('/')}/models"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
    models = data.get("data", [])
    if not models:
        raise RuntimeError(f"No models at {api_base}")
    return models[0]["id"]


def extract_content(response: dict[str, Any]) -> str:
    """Extract the assistant message content from a chat completion response."""
    return response.get("choices", [{}])[0].get("message", {}).get("content", "")
