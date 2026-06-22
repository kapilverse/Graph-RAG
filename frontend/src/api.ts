/**
 * API client — talks to the FastAPI backend.
 */

const API_BASE = "/api";

export interface HealthResponse {
  status: string;
  neo4j: boolean;
  qdrant: boolean;
  llm: boolean;
  llm_provider: string;
}

export interface StatsResponse {
  neo4j: Record<string, number>;
  qdrant: Record<string, number>;
  llm_provider: string;
}

export interface QueryRequest {
  question: string;
  strategy: "local" | "global" | "hybrid";
  top_k?: number;
}

export interface ChunkHit {
  chunk_id: string;
  text: string;
  score: number;
  source_doc: string;
  page_num: number | null;
  section_header: string;
}

export interface EntityHit {
  name: string;
  type: string;
  description: string;
}

export interface RelationshipHit {
  source: string;
  target: string;
  type: string;
  confidence: number;
  evidence_chunk_ids: string[];
}

export interface CommunityHit {
  community_id: string;
  summary: string;
  entities: string[];
  score?: number;
}

export interface QueryResult {
  answer: string;
  strategy: string;
  chunks: ChunkHit[];
  entities: EntityHit[];
  relationships: RelationshipHit[];
  communities: CommunityHit[];
  citations: string[];
  insufficient_information: boolean;
}

export interface GraphNode {
  id: string;
  name: string;
  type: string;
  description: string;
}

export interface GraphLink {
  source: string;
  target: string;
  type: string;
}

export interface GraphExploreResponse {
  nodes: GraphNode[];
  links: GraphLink[];
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  stats: () => request<StatsResponse>("/stats"),

  ingest: (path: string, reset = false) =>
    request<{ files: unknown[] }>("/ingest", {
      method: "POST",
      body: JSON.stringify({ path, reset }),
    }),

  query: (req: QueryRequest) => request<QueryResult>("/query", {
    method: "POST",
    body: JSON.stringify(req),
  }),

  exploreGraph: (entityName: string, limit = 20) =>
    request<GraphExploreResponse>("/graph/explore", {
      method: "POST",
      body: JSON.stringify({ entity_name: entityName, limit }),
    }),

  listCommunities: (topK = 20) =>
    request<CommunityHit[]>(`/communities?top_k=${topK}`),
};
