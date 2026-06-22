import { useState, useEffect } from "react";
import { api, type HealthResponse, type QueryResult } from "./api";
import Chat from "./components/Chat";
import GraphExplorer from "./components/GraphExplorer";
import CitationPanel from "./components/CitationPanel";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [activeTab, setActiveTab] = useState<"chat" | "graph">("chat");

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    const interval = setInterval(() => {
      api.health().then(setHealth).catch(() => {});
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleQuery = (result: QueryResult) => {
    setQueryResult(result);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Header */}
      <header style={{
        padding: "12px 24px",
        borderBottom: "1px solid #1e293b",
        display: "flex",
        alignItems: "center",
        gap: "16px",
        background: "#1e293b",
      }}>
        <h1 style={{ fontSize: "20px", fontWeight: 700, color: "#38bdf8" }}>
          Graph RAG
        </h1>
        {/* Status indicators */}
        {health && (
          <div style={{ display: "flex", gap: "12px", fontSize: "12px" }}>
            <StatusDot label="Neo4j" ok={health.neo4j} />
            <StatusDot label="Qdrant" ok={health.qdrant} />
            <StatusDot label="LLM" ok={health.llm} />
            {health.llm_provider !== "none" && (
              <span style={{ color: "#94a3b8" }}>({health.llm_provider})</span>
            )}
          </div>
        )}
        {/* Tab switcher */}
        <div style={{ marginLeft: "auto", display: "flex", gap: "4px" }}>
          <TabButton
            active={activeTab === "chat"}
            onClick={() => setActiveTab("chat")}
            label="Chat"
          />
          <TabButton
            active={activeTab === "graph"}
            onClick={() => setActiveTab("graph")}
            label="Graph Explorer"
          />
        </div>
      </header>

      {/* Main content */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <div style={{ flex: 1, display: activeTab === "chat" ? "flex" : "none", flexDirection: "column" }}>
          <Chat onResult={handleQuery} />
        </div>
        <div style={{ flex: 1, display: activeTab === "graph" ? "flex" : "none", flexDirection: "column" }}>
          <GraphExplorer />
        </div>
        {/* Citation panel */}
        <div style={{
          width: "380px",
          borderLeft: "1px solid #1e293b",
          overflowY: "auto",
          padding: "16px",
        }}>
          <CitationPanel result={queryResult} />
        </div>
      </div>
    </div>
  );
}

function StatusDot({ label, ok }: { label: string; ok: boolean }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
      <span style={{
        width: "8px", height: "8px", borderRadius: "50%",
        background: ok ? "#22c55e" : "#ef4444",
      }} />
      <span style={{ color: "#94a3b8" }}>{label}</span>
    </span>
  );
}

function TabButton({ active, onClick, label }: { active: boolean; onClick: () => void; label: string }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "6px 16px",
        borderRadius: "6px",
        border: "none",
        cursor: "pointer",
        fontSize: "13px",
        fontWeight: active ? 600 : 400,
        background: active ? "#0ea5e9" : "transparent",
        color: active ? "#fff" : "#94a3b8",
        transition: "background 0.2s",
      }}
    >
      {label}
    </button>
  );
}
