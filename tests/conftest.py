"""Shared test setup: no real service is ever contacted."""

from pathlib import Path

import pytest

from agent.config import settings

CONFIG = Path(__file__).parent / "agent-test.yaml"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    # Closed local ports: a missed mock fails fast instead of reaching a real service.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("AIRFLOW_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("MLOPS_AGENT_CONFIG", str(CONFIG))
    settings.cache_clear()
    yield
    settings.cache_clear()


class FakeLLM:
    """Returns scripted replies in order, and records what the agent sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, messages, tools):
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


def tool_call(name, arguments="{}", call_id=None):
    return {"id": call_id or f"call-{name}", "function": {"name": name, "arguments": arguments}}


def calls(*tool_calls):
    return {"content": "", "tool_calls": list(tool_calls)}


def answer(text):
    return {"content": text}
