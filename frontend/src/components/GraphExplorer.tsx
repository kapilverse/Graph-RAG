import { useState, useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import { api, type GraphNode, type GraphLink } from "../api";

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  type: string;
  description: string;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  type: string;
}

export default function GraphExplorer() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [searchEntity, setSearchEntity] = useState("");
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const [nodes, setNodes] = useState<SimNode[]>([]);
  const [links, setLinks] = useState<SimLink[]>([]);
  const [error, setError] = useState("");

  const explore = useCallback(async (name: string) => {
    if (!name.trim()) return;
    setError("");
    try {
      const data = await api.exploreGraph(name.trim());
      const n: SimNode[] = data.nodes.map((nd) => ({
        ...nd,
        x: undefined as unknown as number,
        y: undefined as unknown as number,
      }));
      const l: SimLink[] = data.links.map((lk) => ({
        source: lk.source,
        target: lk.target,
        type: lk.type,
      }));
      setNodes(n);
      setLinks(l);
      setSelectedEntity(name.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Entity not found");
      setNodes([]);
      setLinks([]);
    }
  }, []);

  // D3 force simulation
  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight || 500;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Build lookup maps
    const nodeById = new Map<string, SimNode>();
    nodes.forEach((n) => nodeById.set(n.name, n));

    // Resolve link source/target to actual node objects
    const resolvedLinks: SimLink[] = links
      .map((l) => ({
        ...l,
        source: nodeById.get(String(l.source)) || l.source,
        target: nodeById.get(String(l.target)) || l.target,
      }))
      .filter((l) => l.source && l.target);

    const simulation = d3
      .forceSimulation<SimNode>(nodes)
      .force("link", d3.forceLink<SimNode, SimLink>(resolvedLinks).id((d) => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const linkGroup = svg.append("g").attr("class", "links");
    const nodeGroup = svg.append("g").attr("class", "nodes");
    const labelGroup = svg.append("g").attr("class", "labels");

    const linkSel = linkGroup
      .selectAll<SVGLineElement, SimLink>("line")
      .data(resolvedLinks)
      .join("line")
      .attr("stroke", "#334155")
      .attr("stroke-width", 1.5);

    const nodeSel = nodeGroup
      .selectAll<SVGCircleElement, SimNode>("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d) => (d.name === selectedEntity ? 10 : 6))
      .attr("fill", nodeColor)
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("click", (_, d) => explore(d.name));

    const labelSel = labelGroup
      .selectAll<SVGTextElement, SimNode>("text")
      .data(nodes)
      .join("text")
      .text((d) => d.name)
      .attr("font-size", "11px")
      .attr("fill", "#cbd5e1")
      .attr("text-anchor", "middle")
      .attr("dy", -12);

    // Tooltip on hover
    const tooltip = svg
      .append("text")
      .attr("class", "tooltip")
      .attr("font-size", "12px")
      .attr("fill", "#f59e0b")
      .attr("x", 10)
      .attr("y", height - 10);

    nodeSel
      .on("mouseenter", (_, d) => tooltip.text(`${d.name} (${d.type})${d.description ? ": " + d.description.slice(0, 80) : ""}`))
      .on("mouseleave", () => tooltip.text(""));

    simulation.on("tick", () => {
      linkSel
        .attr("x1", (d) => (d.source as SimNode).x!)
        .attr("y1", (d) => (d.source as SimNode).y!)
        .attr("x2", (d) => (d.target as SimNode).x!)
        .attr("y2", (d) => (d.target as SimNode).y!);

      nodeSel.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);
      labelSel.attr("x", (d) => d.x!).attr("y", (d) => d.y!);
    });

    // Drag
    const drag = d3
      .drag<SVGCircleElement, SimNode>()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = undefined as unknown as number;
        d.fy = undefined as unknown as number;
      });
    nodeSel.call(drag);

    return () => {
      simulation.stop();
    };
  }, [nodes, links, selectedEntity, explore]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Search bar */}
      <div style={{
        padding: "12px 16px",
        borderBottom: "1px solid #1e293b",
        display: "flex",
        gap: "8px",
      }}>
        <input
          value={searchEntity}
          onChange={(e) => setSearchEntity(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && explore(searchEntity)}
          placeholder="Enter entity name (e.g. Apple, Tim Cook)..."
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
          onClick={() => explore(searchEntity)}
          style={{
            padding: "10px 20px",
            borderRadius: "8px",
            border: "none",
            background: "#7c3aed",
            color: "#fff",
            cursor: "pointer",
            fontSize: "14px",
            fontWeight: 600,
          }}
        >
          Explore
        </button>
      </div>

      {error && (
        <div style={{ padding: "8px 16px", color: "#f59e0b", fontSize: "13px" }}>{error}</div>
      )}

      {/* D3 SVG canvas */}
      <div style={{ flex: 1, position: "relative" }}>
        {nodes.length === 0 && !error && (
          <div style={{
            position: "absolute", top: "40%", left: "50%", transform: "translate(-50%, -50%)",
            color: "#475569", textAlign: "center",
          }}>
            Enter an entity name above to explore its neighborhood in the knowledge graph.
          </div>
        )}
        <svg
          ref={svgRef}
          style={{ width: "100%", height: "100%" }}
        />
      </div>

      {/* Legend */}
      <div style={{
        padding: "8px 16px",
        borderTop: "1px solid #1e293b",
        display: "flex",
        gap: "16px",
        fontSize: "11px",
        color: "#64748b",
      }}>
        {[
          { type: "Person", color: "#38bdf8" },
          { type: "Organization", color: "#a78bfa" },
          { type: "Technology", color: "#34d399" },
          { type: "Location", color: "#fb923c" },
          { type: "Concept", color: "#f472b6" },
        ].map(({ type, color }) => (
          <span key={type} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
            <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: color }} />
            {type}
          </span>
        ))}
      </div>
    </div>
  );
}

function nodeColor(d: SimNode): string {
  const colors: Record<string, string> = {
    Person: "#38bdf8",
    Organization: "#a78bfa",
    Technology: "#34d399",
    Location: "#fb923c",
    Concept: "#f472b6",
    Event: "#fbbf24",
    Product: "#2dd4bf",
  };
  return colors[d.type] || "#94a3b8";
}
