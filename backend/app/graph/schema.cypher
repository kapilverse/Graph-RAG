// ============================================================================
// Graph RAG — Neo4j schema (constraints + indexes)
// Run once at startup via Neo4jStore.ensure_schema().
// Spec §3 Stage 3: design the graph schema before coding.
// ============================================================================

// --- Uniqueness constraints (also create indexes) ---
CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT community_id IF NOT EXISTS
FOR (cm:Community) REQUIRE cm.id IS UNIQUE;

// --- Lookup indexes for fast traversal ---
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) REQUIRE e.type IS NOT NULL;
CREATE INDEX community_id_lookup IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS NOT NULL;
