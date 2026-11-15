"""Tool registry and execute_tool: validation, confirmation, timeout, one envelope, no raise."""

import json
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

import httpx
import requests
from pydantic import BaseModel, ConfigDict, ValidationError
from urllib3.exceptions import HTTPError as Urllib3Error

log = logging.getLogger("agent.tools")

TOOL_TIMEOUT_S = float(os.environ.get("TOOL_TIMEOUT_S", "30"))
# ponytail: one shared pool; a timed-out call keeps its thread until the client timeout ends it.
_pool = ThreadPoolExecutor(max_workers=8)


class Args(BaseModel):
    """Base for tool arguments: unknown fields are rejected, not ignored."""

    model_config = ConfigDict(extra="forbid")


class NoArgs(Args):
    pass


class ServiceUnavailable(Exception):
    """A backing service (MLflow, Airflow, Kubernetes, Git/DVC) could not be reached."""


@dataclass(frozen=True)
class Tool:
    name: str
    fn: Callable[[Args], dict]
    args_model: type[Args]
    description: str
    mutates: bool


REGISTRY: dict[str, Tool] = {}


def tool(args_model: type[Args] = NoArgs, *, mutates: bool = False):
    """Register a function as a tool. Its docstring is the description the LLM sees."""

    def register(fn):
        doc = " ".join((fn.__doc__ or "").split())
        REGISTRY[fn.__name__] = Tool(fn.__name__, fn, args_model, doc, mutates)
        return fn

    return register


_UNAVAILABLE = (
    ConnectionError,
    ServiceUnavailable,
    httpx.TransportError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    Urllib3Error,
)


def _is_unavailable(exc: BaseException | None) -> bool:
    """True if the exception, or anything it wraps, is a connection failure."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, _UNAVAILABLE):
            return True
        text = str(exc)
        if "Max retries exceeded" in text or "Connection refused" in text:
            return True  # MLflow wraps connection errors in MlflowException
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def execute_tool(
    name: str, raw_args: dict | None, request_id: str, confirmed: bool = False
) -> dict:
    """Run one tool and return its envelope. Never raises."""
    start = time.monotonic()
    data, error = None, None
    entry = REGISTRY.get(name)
    try:
        if entry is None:
            status, error = "error", f"unknown tool {name!r}"
        else:
            args = entry.args_model.model_validate(raw_args or {})
            if entry.mutates and not confirmed:
                status = "confirmation_required"
                data = {"action": name, "arguments": args.model_dump()}
                error = "this action changes the system and needs the user's confirmation"
            else:
                data = _pool.submit(entry.fn, args).result(timeout=TOOL_TIMEOUT_S)
                status = "ok"
    except ValidationError as exc:
        status = "error"
        error = "invalid arguments: " + "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors()
        )
    except PermissionError as exc:
        status, error = "error", f"rejected: {exc}"
    except FutureTimeout:
        status, error = "timeout", f"no result within {TOOL_TIMEOUT_S:.0f}s"
    except Exception as exc:  # noqa: BLE001 - the envelope must capture every failure
        if _is_unavailable(exc):
            status, error = "unavailable", f"service unreachable: {type(exc).__name__}"
        else:
            status, error = "error", f"{type(exc).__name__}: {exc}"[:500]

    duration_ms = int((time.monotonic() - start) * 1000)
    # Fixed fields only: arguments and results are never logged, so secrets can't leak.
    log.info(
        json.dumps(
            {
                "event": "tool_call",
                "tool": name,
                "status": status,
                "duration_ms": duration_ms,
                "request_id": request_id,
            }
        )
    )
    return {
        "tool": name,
        "status": status,
        "data": data,
        "error": error,
        "duration_ms": duration_ms,
        "request_id": request_id,
    }


def tool_schemas() -> list[dict]:
    """OpenAI-style function definitions for every registered tool."""
    schemas = []
    for t in REGISTRY.values():
        params = t.args_model.model_json_schema()
        params.pop("title", None)
        desc = t.description + (
            " Changes the system: needs user confirmation." if t.mutates else ""
        )
        schemas.append(
            {
                "type": "function",
                "function": {"name": t.name, "description": desc, "parameters": params},
            }
        )
    return schemas


# Importing the modules registers their tools.
from agent.tools import (  # noqa: F401
    airflow_tools,
    dvc_tools,
    kubernetes_tools,
    mlflow_tools,
    monitoring_tools,
)
