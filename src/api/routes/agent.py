from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.agent import orchestrator
from src.agent.logging_store import append_log

router = APIRouter()


class ConversationTurn(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    message: str
    page_context: Optional[str] = None
    conversation_history: List[ConversationTurn] = []


@router.post("/chat")
def chat(request: ChatRequest):
    """The explainability agent's entry point. Every number in `response`
    has been verified by groundedness.check_groundedness() before this
    returns -- see src/agent/orchestrator.py for the full pipeline. This
    endpoint itself adds no computation of its own: it calls the
    orchestrator and persists its log (principle 4)."""
    history = [turn.model_dump() for turn in request.conversation_history]
    result = orchestrator.handle_question(request.message, request.page_context, history)
    append_log(result["log"])
    return {
        "response": result["response"],
        "tool_calls_made": result["tool_calls_made"],
        "template_used": result["template_used"],
        "sources": result["sources"],
    }
