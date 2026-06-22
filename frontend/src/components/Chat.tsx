import { useState } from "react";
import { api, type QueryResult } from "../api";

interface Props {
  onResult: (result: QueryResult) => void;
}

interface Message {
  role: "user" | "assistant";
  text: string;
  result?: QueryResult;
}

export default function Chat({ onResult }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [strategy, setStrategy] = useState<"local" | "global" | "hybrid">("hybrid");

  const send = async () => {
    if (!input.trim() || loading) return;
    const q = input.trim();
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setLoading(true);
    try {
      const result = await api.query({ question: q, strategy });
      setMessages((m) => [...m, { role: "assistant", text: result.answer, result }]);
      onResult(result);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setMessages((m) => [...m, { role: "assistant", text: `Error: ${msg}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Strategy selector */}
      <div style={{
        padding: "8px 16px",
        borderBottom: "1px solid #1e293b",
        display: "flex",
        gap: "8px",
        alignItems: "center",
      }}>
        <span style={{ fontSize: "12px", color: "#64748b" }}>Strategy:</span>
        {(["local", "global", "hybrid"] as const).map((s) => (
          <button
            key={s}
            onClick={() => setStrategy(s)}
            style={{
              padding: "4px 10px",
              borderRadius: "4px",
              border: "1px solid #334155",
              background: strategy === s ? "#0ea5e9" : "transparent",
              color: strategy === s ? "#fff" : "#94a3b8",
              cursor: "pointer",
              fontSize: "12px",
            }}
          >
            {s}
          </button>
        ))}
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "16px" }}>
        {messages.length === 0 && (
          <div style={{ color: "#475569", textAlign: "center", marginTop: "40%" }}>
            Ask a question — multi-hop queries work best with Graph RAG.
            <br />
            <span style={{ fontSize: "13px" }}>
              e.g. "Which companies founded by ex-Apple employees compete with Apple?"
            </span>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              marginBottom: "16px",
              display: "flex",
              justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
            }}
          >
            <div style={{
              maxWidth: "75%",
              padding: "10px 14px",
              borderRadius: "12px",
              lineHeight: "1.6",
              fontSize: "14px",
              whiteSpace: "pre-wrap",
              background: msg.role === "user" ? "#0ea5e9" : "#1e293b",
              color: msg.role === "user" ? "#fff" : "#e2e8f0",
            }}>
              {msg.text}
              {msg.result && msg.result.citations.length > 0 && (
                <div style={{ marginTop: "8px", fontSize: "11px", color: "#64748b" }}>
                  Citations: {msg.result.citations.join(", ")}
                </div>
              )}
              {msg.result?.insufficient_information && (
                <div style={{ marginTop: "4px", fontSize: "11px", color: "#f59e0b" }}>
                  ⚠ Insufficient information in the knowledge graph
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ color: "#64748b", fontSize: "13px", padding: "8px" }}>
            Thinking... (traversing knowledge graph)
          </div>
        )}
      </div>

      {/* Input */}
      <div style={{
        padding: "12px 16px",
        borderTop: "1px solid #1e293b",
        display: "flex",
        gap: "8px",
      }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask a multi-hop question..."
          style={{
            flex: 1,
            padding: "10px 14px",
            borderRadius: "8px",
            border: "1px solid #334155",
            background: "#0f172a",
            color: "#e2e8f0",
            fontSize: "14px",
            outline: "none",
          }}
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          style={{
            padding: "10px 20px",
            borderRadius: "8px",
            border: "none",
            background: input.trim() && !loading ? "#0ea5e9" : "#334155",
            color: "#fff",
            cursor: input.trim() && !loading ? "pointer" : "not-allowed",
            fontSize: "14px",
            fontWeight: 600,
          }}
        >
          {loading ? "..." : "Ask"}
        </button>
      </div>
    </div>
  );
}
