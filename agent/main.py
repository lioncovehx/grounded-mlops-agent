"""HTTP interface for the agent.

Changes proposed by the agent are confirmed through /actions/{id}/confirm, which only a
person calling this API can reach; the LLM has no tool for it.
"""

import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.orchestrator import confirm_action, run_agent, take_pending

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = FastAPI(title="Grounded MLOps Agent")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


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
    return run_agent(req.message, uuid.uuid4().hex[:12])


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
    return {"status": "ok"}
