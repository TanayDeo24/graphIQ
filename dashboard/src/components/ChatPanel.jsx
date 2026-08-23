import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { api } from "../api";

const PAGE_CONTEXT_BY_PATH = {
  "/": "HomePage",
  "/attrition": "AttritionPage",
  "/spend": "SpendPage",
  "/cross-component": "CrossComponentPage",
};

// The orchestrator returns an empty mechanisms_used list exactly when it
// fell back to its natural-language "I don't have a confident answer"
// text -- either nothing could be gathered, or a generated data claim
// failed the groundedness check against the real SQL results. That's the
// single, reliable signal for "this is a refusal/hedge," not a string
// match on response text (the whole point of the natural-language
// refusal is that it isn't a fixed, detectable template).
function isRefusal(mechanismsUsed) {
  return !mechanismsUsed || mechanismsUsed.length === 0;
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
          mechanismsUsed: result.mechanisms_used || [],
        },
      ]);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "The agent request failed. Check that the API server is running and the LLM backend (LLM_BACKEND) is configured."
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
                Ask a general concept question ("what is a SHAP value"), a specific project data question
                ("what's employee 42's risk score"), or a project rationale question ("why did the CUSUM
                ranking change") -- or combine them. Any specific project number is checked against a real,
                live query before being shown to you; if that check fails, the agent says so instead of
                guessing.
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`chat-message ${m.role} ${m.role === "model" && isRefusal(m.mechanismsUsed) ? "refusal" : ""}`}>
                <div className="chat-message-text">{m.text}</div>
                {m.role === "model" && m.sources && m.sources.length > 0 && (
                  <div className="chat-message-sources">
                    {isRefusal(m.mechanismsUsed) && <span className="chat-refusal-tag">no confident answer</span>}
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
