import React, { useEffect, useMemo, useRef, useState } from "react";

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(
  /\/+$/,
  ""
);

function generateSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `s-${crypto.randomUUID().slice(0, 12)}`;
  }
  return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function formatErrorDetail(detail, status) {
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((item) => item?.msg || JSON.stringify(item))
      .join("; ");
  }
  return `Request failed with status ${status}`;
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => generateSessionId());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulateFailure, setSimulateFailure] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [showOptions, setShowOptions] = useState(false);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    let active = true;
    fetch(`${API_URL}/health`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (active && data) {
          setHealth(data);
        }
      })
      .catch(() => {
        if (active) {
          setHealth(null);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const sessionStats = useMemo(() => {
    const assistantTurns = messages.filter((m) => m.role === "assistant");
    const cacheHits = assistantTurns.filter(
      (m) => m.metadata?.resolution_source === "LLM_CACHE"
    ).length;
    return {
      turns: assistantTurns.length,
      cacheHits,
    };
  }, [messages]);

  function handleNewSession() {
    setSessionId(generateSessionId());
    setMessages([]);
    setInput("");
  }

  async function handleSend(event) {
    if (event) {
      event.preventDefault();
    }
    const trimmed = input.trim();
    if (!trimmed || loading) {
      return;
    }

    setInput("");
    setMessages((prev) => [
      ...prev,
      { id: `u-${Date.now()}`, role: "user", content: trimmed },
    ]);
    setLoading(true);

    const payload = {
      session_id: sessionId,
      message: trimmed,
    };
    if (systemPrompt.trim()) {
      payload.system_prompt = systemPrompt.trim();
    }

    const headers = {
      "Content-Type": "application/json",
    };
    if (simulateFailure) {
      headers["X-Simulate-Failure"] = simulateFailure;
    }

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });

      const contentType = response.headers.get("content-type") || "";
      const body = contentType.includes("application/json")
        ? await response.json()
        : { detail: await response.text() };

      if (!response.ok) {
        const errorText = formatErrorDetail(body?.detail, response.status);
        setMessages((prev) => [
          ...prev,
          {
            id: `err-${Date.now()}`,
            role: "error",
            content: `Error (${response.status}): ${errorText}`,
          },
        ]);
        return;
      }

      setMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: body.reply,
          metadata: body.metadata,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: "error",
          content:
            err?.message
              ? `Network error: ${err.message}. Verify backend at ${API_URL}.`
              : `Network error: Unable to reach backend at ${API_URL}.`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-dot" aria-hidden="true" />
          <div>
            <h1>Lyftr LLM Gateway</h1>
            <p className="subtitle">
              Key-pool failover · LiteLLM response caching · SQLite memory
            </p>
          </div>
        </div>

        <div className="header-controls">
          <span className="status-badge" title={`Session: ${sessionId}`}>
            <span
              className={`status-indicator ${health ? "online" : "offline"}`}
            />
            {health
              ? `${health.keys_configured} keys ready`
              : "Gateway connecting"}
          </span>
          <span className="session-pill" title="Current ephemeral session ID">
            {sessionId}
          </span>
          <button
            type="button"
            className="secondary-btn"
            onClick={() => setShowOptions((v) => !v)}
          >
            {showOptions ? "Hide Controls" : "Controls"}
          </button>
          <button
            type="button"
            className="secondary-btn"
            onClick={handleNewSession}
            disabled={loading}
          >
            New Session
          </button>
        </div>
      </header>

      <section className={`controls-panel ${showOptions ? "open" : ""}`}>
        <div className="control-group control-grow">
          <label htmlFor="system-prompt">System Prompt (optional)</label>
          <input
            id="system-prompt"
            data-testid="system-prompt-input"
            type="text"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="Overrides the default assistant system prompt for this session..."
          />
        </div>

        <div className="control-group">
          <label htmlFor="simulate-failure">
            Failover Hook (<code>X-Simulate-Failure</code>)
          </label>
          <select
            id="simulate-failure"
            value={simulateFailure}
            onChange={(e) => setSimulateFailure(e.target.value)}
          >
            <option value="">Off (Normal)</option>
            <option value="key_1=429">key_1=429 (Rate limit → key_2)</option>
            <option value="key_1=500">key_1=500 (Server error → key_2)</option>
            <option value="*=429">*=429 (All keys fail → 503)</option>
          </select>
        </div>
      </section>

      <main className="chat-thread" aria-live="polite">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>
              Messages are stored per session in SQLite. Repeat an identical
              prompt in a new session to see <code>LLM_CACHE</code> hits, or
              open Controls to test API key failover.
            </p>
            <div className="prompt-chips">
              {[
                "What is the capital of France?",
                "Explain circuit breakers in 2 sentences.",
                "Remember my favorite color is cobalt blue.",
              ].map((sample) => (
                <button
                  key={sample}
                  type="button"
                  className="chip-btn"
                  onClick={() => setInput(sample)}
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => {
            if (msg.role === "user") {
              return (
                <div key={msg.id} className="message-row user-row">
                  <div className="bubble user-bubble">
                    <div className="role-label">You</div>
                    <div className="message-content">{msg.content}</div>
                  </div>
                </div>
              );
            }

            if (msg.role === "error") {
              return (
                <div key={msg.id} className="message-row assistant-row">
                  <div className="bubble error-bubble" role="alert">
                    <div className="role-label">Gateway Error</div>
                    <div className="message-content">{msg.content}</div>
                  </div>
                </div>
              );
            }

            const meta = msg.metadata || {};
            const isCache = meta.resolution_source === "LLM_CACHE";
            return (
              <div key={msg.id} className="message-row assistant-row">
                <div
                  className="bubble assistant-bubble"
                  data-testid="assistant-message"
                >
                  <div className="role-label">Assistant</div>
                  <div className="message-content">{msg.content}</div>

                  <div className="meta-bar">
                    <span
                      className={`source-badge ${isCache ? "cache" : "live"}`}
                      data-testid="meta-source"
                    >
                      {meta.resolution_source}
                    </span>

                    <span className="meta-pill" data-testid="meta-model">
                      {meta.model}
                    </span>

                    <span className="meta-pill" data-testid="meta-latency">
                      {`${meta.latency_ms} ms`}
                    </span>

                    <span className="meta-pill">
                      In:{" "}
                      <strong data-testid="meta-input-tokens">
                        {meta.input_tokens}
                      </strong>
                    </span>

                    <span className="meta-pill">
                      Out:{" "}
                      <strong data-testid="meta-output-tokens">
                        {meta.output_tokens}
                      </strong>
                    </span>

                    {meta.key_id && (
                      <span className="meta-pill key-pill">{meta.key_id}</span>
                    )}

                    {typeof meta.fallback_attempts === "number" &&
                      meta.fallback_attempts > 0 && (
                        <span className="meta-pill fallback-pill">
                          {meta.fallback_attempts} failover
                        </span>
                      )}
                  </div>
                </div>
              </div>
            );
          })
        )}

        {loading && (
          <div className="message-row assistant-row">
            <div className="bubble assistant-bubble loading-bubble">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="loading-text">Waiting for response...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </main>

      <footer className="composer-wrap">
        <div className="composer-meta">
          <span>
            Turns: {sessionStats.turns} · Cache hits: {sessionStats.cacheHits}
          </span>
          {simulateFailure && (
            <span className="sim-warning">
              Simulating: <code>{simulateFailure}</code>
            </span>
          )}
        </div>

        <form className="composer-form" onSubmit={handleSend}>
          <textarea
            data-testid="chat-input"
            className="chat-textarea"
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question... (Enter to send, Shift+Enter for new line)"
            maxLength={4000}
            disabled={loading}
          />
          <button
            data-testid="send-button"
            className="send-btn"
            type="submit"
            disabled={loading || !input.trim()}
          >
            {loading ? "Sending..." : "Send"}
          </button>
        </form>
      </footer>
    </div>
  );
}
