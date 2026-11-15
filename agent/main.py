"""HTTP interface for the agent, plus a chat page at /.

Changes are confirmed through /actions/{id}/confirm, which only a person can call;
the LLM has no tool for it.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent.config import settings
from agent.orchestrator import confirm_action, run_agent, take_pending

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = FastAPI(title="Grounded MLOps Agent")
CHAT_PAGE = Path(__file__).parent / "static" / "chat.html"


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=10)


class Evidence(BaseModel):
    tool: str
    status: str
    data: Any = None
    error: str | None = None
    duration_ms: int
    request_id: str


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    evidence: list[Evidence]
    pending_action_id: str | None = None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict:
    if not req.message.strip():
        raise HTTPException(422, "message is empty")
    history = [t.model_dump() for t in req.history]
    return run_agent(req.message, uuid.uuid4().hex[:12], history=history)


@app.post("/actions/{action_id}/confirm", response_model=ChatResponse)
def confirm(action_id: str) -> dict:
    result = confirm_action(action_id)
    if result is None:
        raise HTTPException(404, "no such pending action, or it has expired")
    return result


@app.post("/actions/{action_id}/cancel")
def cancel(action_id: str) -> dict:
    if take_pending(action_id) is None:
        raise HTTPException(404, "no such pending action, or it has expired")
    return {"cancelled": action_id}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "project": settings().project}


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(CHAT_PAGE)
