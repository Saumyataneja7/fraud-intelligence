import { useState, type FormEvent } from "react";
import { Search, Network, ArrowRight, Loader2 } from "lucide-react";

import { getGraphInvestigation } from "../api/client";
import type { GraphInvestigation, GraphNode } from "../types/graph";

const NODE_TYPES = [
  "customer",
  "account",
  "card",
  "transaction",
  "merchant",
  "device",
  "ip",
];

function formatNodeType(nodeType: string): string {
  return nodeType.replace(/_/g, " ");
}

function nodeKey(node: GraphNode): string {
  return `${node.node_type}:${node.node_id}`;
}

function GraphPage() {
  const [nodeType, setNodeType] = useState("customer");
  const [nodeId, setNodeId] = useState("");
  const [graph, setGraph] = useState<GraphInvestigation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSearch(event: FormEvent) {
    event.preventDefault();

    const trimmedId = nodeId.trim();

    if (!trimmedId) {
      setError("Enter a numeric node ID.");
      setGraph(null);
      return;
    }

    if (!/^\d+$/.test(trimmedId)) {
      setError("Node ID must be a non-negative integer.");
      setGraph(null);
      return;
    }

    setLoading(true);
    setError("");
    setGraph(null);

    try {
      const result = await getGraphInvestigation(
        nodeType,
        Number(trimmedId),
      );

      setGraph(result);
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "Unable to load graph investigation.";

      setError(message);
    } finally {
      setLoading(false);
    }
  }

  const targetKey = graph
    ? `${graph.target_node_type}:${graph.target_node_id}`
    : null;

  const neighbors =
    graph?.nodes.filter((node) => nodeKey(node) !== targetKey) ?? [];

  return (
    <div className="page-stack">
      <section className="page-heading">
        <div>
          <p className="eyebrow">GRAPH INVESTIGATION</p>
          <h1>Entity Network</h1>
          <p className="page-description">
            Investigate the first-hop relationships surrounding a graph node.
          </p>
        </div>

        <div className="page-heading-icon">
          <Network size={22} />
        </div>
      </section>

      <section className="graph-search-card">
        <form className="graph-search-form" onSubmit={handleSearch}>
          <label>
            <span>Node type</span>
            <select
              value={nodeType}
              onChange={(event) => setNodeType(event.target.value)}
            >
              {NODE_TYPES.map((type) => (
                <option key={type} value={type}>
                  {formatNodeType(type)}
                </option>
              ))}
            </select>
          </label>

          <label className="graph-node-id-field">
            <span>Node ID</span>
            <input
              value={nodeId}
              onChange={(event) => setNodeId(event.target.value)}
              placeholder="e.g. 0"
              inputMode="numeric"
            />
          </label>

          <button
            type="submit"
            className="primary-button"
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader2 size={16} className="spin" />
                Investigating
              </>
            ) : (
              <>
                <Search size={16} />
                Investigate
              </>
            )}
          </button>
        </form>
      </section>

      {error && (
        <section className="api-error-card">
          <strong>Graph investigation unavailable</strong>
          <span>{error}</span>
        </section>
      )}

      {!graph && !loading && !error && (
        <section className="graph-empty-state">
          <div className="empty-state-icon">
            <Network size={28} />
          </div>

          <h2>Start a graph investigation</h2>

          <p>
            Select an entity type and enter its numeric graph node ID to inspect
            its connected first-hop relationships.
          </p>

          <div className="graph-example">
            <span>Example</span>
            <code>customer:0</code>
          </div>
        </section>
      )}

      {graph && (
        <>
          <section className="graph-hero">
            <div>
              <p className="eyebrow">TARGET NODE</p>

              <div className="graph-target">
                <span className="node-type-badge">
                  {formatNodeType(graph.target_node_type)}
                </span>

                <code>{graph.target_node_id}</code>
              </div>

              <p>
                The graph index returned the target node and its direct
                first-hop relationships.
              </p>
            </div>

            <div className="graph-hero-stat">
              <strong>{graph.nodes.length}</strong>
              <span>nodes</span>
            </div>

            <div className="graph-hero-stat">
              <strong>{graph.edges.length}</strong>
              <span>relationships</span>
            </div>
          </section>

          <section className="graph-stat-grid">
            <div className="metric-card">
              <span>Target</span>
              <strong>
                {formatNodeType(graph.target_node_type)}
              </strong>
              <small>Investigated node type</small>
            </div>

            <div className="metric-card">
              <span>Neighbors</span>
              <strong>{neighbors.length}</strong>
              <small>Direct connected nodes</small>
            </div>

            <div className="metric-card">
              <span>Relationships</span>
              <strong>{graph.edges.length}</strong>
              <small>Returned graph edges</small>
            </div>

            <div className="metric-card">
              <span>Scope</span>
              <strong>1-hop</strong>
              <small>API investigation scope</small>
            </div>
          </section>

          <section className="graph-network-card">
            <div className="section-header">
              <div>
                <p className="eyebrow">NETWORK</p>
                <h2>Connected entities</h2>
              </div>

              <span className="section-count">
                {neighbors.length} neighbors
              </span>
            </div>

            <div className="graph-network">
              <div className="graph-target-node">
                <div className="graph-node-icon">
                  <Network size={18} />
                </div>

                <div>
                  <span>{formatNodeType(graph.target_node_type)}</span>
                  <strong>{graph.target_node_id}</strong>
                </div>
              </div>

              <div className="graph-connections">
                {neighbors.length === 0 ? (
                  <div className="graph-no-neighbors">
                    No direct neighbors were returned.
                  </div>
                ) : (
                  neighbors.map((node) => {
                    const connectedEdges = graph.edges.filter(
                      (edge) =>
                        (
                          edge.source_node_type === node.node_type &&
                          edge.source_node_id === node.node_id
                        ) ||
                        (
                          edge.target_node_type === node.node_type &&
                          edge.target_node_id === node.node_id
                        ),
                    );

                    return (
                      <div
                        className="graph-neighbor-card"
                        key={nodeKey(node)}
                      >
                        <div className="graph-neighbor-main">
                          <span className="node-type-badge">
                            {formatNodeType(node.node_type)}
                          </span>

                          <code>{node.node_id}</code>
                        </div>

                        <div className="graph-edge-list">
                          {connectedEdges.map((edge) => (
                            <div
                              className="graph-edge-row"
                              key={`${edge.relationship_type}-${edge.source_node_type}-${edge.source_node_id}-${edge.target_node_type}-${edge.target_node_id}`}
                            >
                              <span>{edge.relationship_type}</span>
                              <ArrowRight size={14} />
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </section>

          <section className="graph-relationships-card">
            <div className="section-header">
              <div>
                <p className="eyebrow">RELATIONSHIPS</p>
                <h2>Graph edges</h2>
              </div>

              <span className="section-count">
                {graph.edges.length} edges
              </span>
            </div>

            <div className="graph-edge-table">
              {graph.edges.map((edge, index) => (
                <div className="graph-edge-table-row" key={`${edge.relationship_type}-${index}`}>
                  <div>
                    <span className="edge-label">
                      {edge.relationship_type}
                    </span>
                  </div>

                  <div className="edge-endpoint">
                    <span>{edge.source_node_type}</span>
                    <code>{edge.source_node_id}</code>
                  </div>

                  <ArrowRight size={15} />

                  <div className="edge-endpoint">
                    <span>{edge.target_node_type}</span>
                    <code>{edge.target_node_id}</code>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <div className="graph-disclaimer">
            <strong>Investigation scope</strong>
            <span>
              This view represents structural relationships returned by the
              graph API. A relationship does not by itself establish fraud.
            </span>
          </div>
        </>
      )}
    </div>
  );
}

export default GraphPage;