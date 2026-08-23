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
    """The explainability agent's entry point. Any specific project number
    in `response` has been verified against a real, executed SQL query's
    result set before this returns -- see src/agent/orchestrator.py for
    the full routing/gathering/synthesis/groundedness pipeline. This
    endpoint itself adds no computation of its own: it calls the
    orchestrator and persists its log."""
    history = [turn.model_dump() for turn in request.conversation_history]
    result = orchestrator.handle_question(request.message, request.page_context, history)
    append_log(result["log"])
    return {
        "response": result["response"],
        "sql_executed": result["sql_executed"],
        "doc_chunks_used": result["doc_chunks_used"],
        "mechanisms_used": result["mechanisms_used"],
        "sources": result["sources"],
        "tool_calls_made": result["tool_calls_made"],
    }
