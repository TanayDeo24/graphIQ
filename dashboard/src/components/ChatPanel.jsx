import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { api } from "../api";

const PAGE_CONTEXT_BY_PATH = {
  "/": "HomePage",
  "/attrition": "AttritionPage",
  "/spend": "SpendPage",
  "/cross-component": "CrossComponentPage",
};

// Matches src/agent/eval/run_eval.py's HEDGE_TEMPLATES set: any response
// the agent doesn't have a confident answer for, not just the generic
// refusal_* family -- segment_calibration_zero_events/_low_confidence are
// equally "no confident answer" outcomes (see README's Explainability
// agent section for why they're separate templates from a generic
// refusal in the first place). An earlier version of this check only
// matched the refusal_ prefix, which rendered the calibration hedge
// templates with normal-answer styling -- caught via a live screenshot.
const HEDGE_TEMPLATES = new Set([
  "refusal_not_found",
  "refusal_out_of_scope",
  "refusal_insufficient_data",
  "refusal_groundedness_failed",
  "segment_calibration_zero_events",
  "segment_calibration_low_confidence",
]);

function isRefusal(templateUsed) {
  return HEDGE_TEMPLATES.has(templateUsed);
}

export default function ChatPanel() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const listRef = useRef(null);
  const location = useLocation();
  const pageContext = PAGE_CONTEXT_BY_PATH[location.pathname] || null;

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, loading]);

  async function handleSend(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    const conversationHistory = messages.map((m) => ({ role: m.role, text: m.text }));
    const userMessage = { role: "user", text };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const result = await api.agentChat({
        message: text,
        page_context: pageContext,
        conversation_history: conversationHistory,
      });
      setMessages((prev) => [
        ...prev,
        {
          role: "model",
          text: result.response,
          sources: result.sources || [],
          templateUsed: result.template_used,
          toolCallsMade: result.tool_calls_made || [],
        },
      ]);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "The agent request failed. Check that the API server is running and GEMINI_API_KEY is configured."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`chat-panel-root ${open ? "open" : ""}`}>
      <button
        className="chat-panel-toggle"
        onClick={() => setOpen(true)}
        aria-label="Open explainability agent"
        style={open ? { display: "none" } : undefined}
      >
        💬
      </button>

      {open && (
        <div className="chat-panel-drawer">
          <div className="chat-panel-header">
            <div>
              <div className="chat-panel-title">Explainability agent</div>
              <div className="chat-panel-subtitle">
                Grounded in this project's own result tables{pageContext ? ` · ${pageContext}` : ""}
              </div>
            </div>
            <button className="chat-panel-close" onClick={() => setOpen(false)} aria-label="Close explainability agent">
              ✕
            </button>
          </div>

          <div className="chat-panel-messages" ref={listRef}>
            {messages.length === 0 && (
              <div className="chat-panel-empty">
                Ask about attrition risk scores, calibration, spend-anomaly detectors, transaction
                explanations, lead time, or the cross-component relationship. Every number in a response
                traces back to an actual tool result -- if the agent can't ground an answer, it says so
                instead of guessing.
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`chat-message ${m.role} ${m.role === "model" && isRefusal(m.templateUsed) ? "refusal" : ""}`}>
                <div className="chat-message-text">{m.text}</div>
                {m.role === "model" && m.sources && m.sources.length > 0 && (
                  <div className="chat-message-sources">
                    {isRefusal(m.templateUsed) && <span className="chat-refusal-tag">no confident answer</span>}
                    {m.sources.map((s, j) => (
                      <span key={j} className="chat-source-tag">
                        based on: {s}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {loading && <div className="chat-message model loading-message">Thinking…</div>}
            {error && <div className="chat-message model refusal">{error}</div>}
          </div>

          <form className="chat-panel-input-row" onSubmit={handleSend}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about the data…"
              disabled={loading}
            />
            <button type="submit" disabled={loading || !input.trim()}>
              Send
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
