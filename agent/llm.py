"""The LLM plug: one adapter for any OpenAI-compatible chat endpoint.

Configured only by LLM_BASE_URL, LLM_MODEL and LLM_API_KEY, so a local llama-server
and a hosted provider are interchangeable.
"""

import os

import httpx

LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "300"))


class LLMError(Exception):
    """The LLM could not be reached or returned an unusable reply."""


def call_llm(system: str, messages: list[dict], tools: list[dict]) -> dict:
    """Send one chat turn and return the assistant message.

    The system prompt is sent once, first: some local chat templates drop later system messages.
    """
    base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    if not base_url:
        raise LLMError("LLM_BASE_URL is not set")
    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"},
            json={
                "model": os.environ.get("LLM_MODEL", ""),
                "messages": [{"role": "system", "content": system}, *messages],
                "tools": tools,
                "temperature": 0.2,
            },
            timeout=LLM_TIMEOUT_S,
        )
    except httpx.TransportError as exc:
        raise LLMError(f"LLM unavailable at {base_url}: {type(exc).__name__}") from exc
    if resp.status_code != 200:
        raise LLMError(f"LLM returned HTTP {resp.status_code}: {resp.text[:300]}")

    choice = resp.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        # A tool call may have been cut in half; acting on it would be unsafe.
        raise LLMError("LLM reply was cut off (finish_reason=length)")
    return choice["message"]
