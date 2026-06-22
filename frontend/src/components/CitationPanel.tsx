import type { QueryResult } from "../api";

interface Props {
  result: QueryResult | null;
}

export default function CitationPanel({ result }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "12px", color: "#94a3b8" }}>
        Provenance
      </h2>

      {!result && (
        <div style={{ color: "#475569", fontSize: "13px", textAlign: "center", marginTop: "40px" }}>
          Query the system to see citations and source chunks here.
        </div>
      )}

      {result && (
        <>
          {/* Strategy badge */}
          <div style={{
            marginBottom: "12px",
            padding: "6px 10px",
            borderRadius: "6px",
            background: "#1e293b",
            fontSize: "12px",
            color: "#64748b",
          }}>
            Strategy: <span style={{ color: "#38bdf8", fontWeight: 600 }}>{result.strategy}</span>
            {" · "}
            {result.chunks.length} chunks
            {" · "}
            {result.entities.length} entities
            {" · "}
            {result.communities.length} communities
          </div>

          {/* Answer */}
          <div style={{
            marginBottom: "16px",
            padding: "12px",
            borderRadius: "8px",
            background: "#1e293b",
            fontSize: "13px",
            lineHeight: "1.6",
            whiteSpace: "pre-wrap",
            color: "#e2e8f0",
          }}>
            {result.answer}
          </div>

          {/* Citations */}
          {result.citations.length > 0 && (
            <div style={{ marginBottom: "16px" }}>
              <h3 style={{ fontSize: "12px", fontWeight: 600, color: "#64748b", marginBottom: "8px" }}>
                Citations
              </h3>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                {result.citations.map((c) => (
                  <span key={c} style={{
                    padding: "2px 8px",
                    borderRadius: "4px",
                    background: "#7c3aed33",
                    color: "#a78bfa",
                    fontSize: "11px",
                    fontFamily: "monospace",
                  }}>
                    {c}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Entities */}
          {result.entities.length > 0 && (
            <div style={{ marginBottom: "16px" }}>
              <h3 style={{ fontSize: "12px", fontWeight: 600, color: "#64748b", marginBottom: "8px" }}>
                Entities
              </h3>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                {result.entities.map((e) => (
                  <span key={e.name} style={{
                    padding: "3px 8px",
                    borderRadius: "4px",
                    background: "#0ea5e922",
                    border: "1px solid #0ea5e944",
                    color: "#38bdf8",
                    fontSize: "11px",
                  }}>
                    {e.name} <span style={{ color: "#64748b" }}>({e.type})</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Relationships */}
          {result.relationships.length > 0 && (
            <div style={{ marginBottom: "16px" }}>
              <h3 style={{ fontSize: "12px", fontWeight: 600, color: "#64748b", marginBottom: "8px" }}>
                Relationships
              </h3>
              {result.relationships.map((r, i) => (
                <div key={i} style={{
                  padding: "6px 8px",
                  marginBottom: "4px",
                  borderRadius: "4px",
                  background: "#1e293b",
                  fontSize: "12px",
                  color: "#cbd5e1",
                }}>
                  <span style={{ color: "#38bdf8" }}>{r.source}</span>
                  {" → "}
                  <span style={{ color: "#a78bfa", fontWeight: 600 }}>{r.type}</span>
                  {" → "}
                  <span style={{ color: "#38bdf8" }}>{r.target}</span>
                  {r.confidence > 0 && (
                    <span style={{ color: "#64748b", marginLeft: "8px", fontSize: "11px" }}>
                      ({(r.confidence * 100).toFixed(0)}%)
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Source chunks */}
          {result.chunks.length > 0 && (
            <div>
              <h3 style={{ fontSize: "12px", fontWeight: 600, color: "#64748b", marginBottom: "8px" }}>
                Source Chunks
              </h3>
              {result.chunks.map((chunk) => (
                <div key={chunk.chunk_id} style={{
                  padding: "10px",
                  marginBottom: "8px",
                  borderRadius: "6px",
                  border: "1px solid #334155",
                  background: "#0f172a",
                  fontSize: "12px",
                  lineHeight: "1.5",
                }}>
                  <div style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: "6px",
                    fontSize: "11px",
                    color: "#64748b",
                  }}>
                    <span style={{ fontFamily: "monospace", color: "#a78bfa" }}>
                      {chunk.chunk_id}
                    </span>
                    <span>
                      {chunk.source_doc}
                      {chunk.page_num != null && ` · p${chunk.page_num}`}
                    </span>
                    <span style={{ color: "#34d399" }}>
                      score: {chunk.score.toFixed(3)}
                    </span>
                  </div>
                  {chunk.section_header && (
                    <div style={{ color: "#94a3b8", marginBottom: "4px", fontWeight: 600 }}>
                      {chunk.section_header}
                    </div>
                  )}
                  <div style={{ color: "#cbd5e1", whiteSpace: "pre-wrap" }}>
                    {chunk.text.length > 500
                      ? chunk.text.slice(0, 500) + "..."
                      : chunk.text}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Communities */}
          {result.communities.length > 0 && (
            <div style={{ marginTop: "16px" }}>
              <h3 style={{ fontSize: "12px", fontWeight: 600, color: "#64748b", marginBottom: "8px" }}>
                Relevant Communities
              </h3>
              {result.communities.map((c) => (
                <div key={c.community_id} style={{
                  padding: "8px 10px",
                  marginBottom: "6px",
                  borderRadius: "6px",
                  background: "#1e293b",
                  fontSize: "12px",
                }}>
                  <div style={{ color: "#f59e0b", fontWeight: 600, marginBottom: "4px" }}>
                    {c.community_id}
                  </div>
                  <div style={{ color: "#cbd5e1", lineHeight: 1.4 }}>{c.summary}</div>
                  {c.entities.length > 0 && (
                    <div style={{ marginTop: "4px", color: "#64748b" }}>
                      Entities: {c.entities.slice(0, 10).join(", ")}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
