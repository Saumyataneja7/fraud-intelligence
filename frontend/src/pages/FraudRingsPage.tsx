import { useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Network,
  Search,
  ShieldAlert,
} from "lucide-react";

import { getTransactionFraudRing } from "../api/client";
import type {
  FraudRingEntity,
  FraudRingInvestigation,
} from "../types/fraudRing";

function formatEntityType(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) =>
    char.toUpperCase(),
  );
}

function entityIcon(entityType: string): string {
  const icons: Record<string, string> = {
    customer: "CU",
    account: "AC",
    card: "CA",
    device: "DV",
    ip: "IP",
    merchant: "ME",
    transaction: "TX",
  };

  return icons[entityType] ?? "EN";
}

function groupEntities(
  entities: FraudRingEntity[],
): Record<string, FraudRingEntity[]> {
  return entities.reduce<Record<string, FraudRingEntity[]>>(
    (groups, entity) => {
      if (!groups[entity.entity_type]) {
        groups[entity.entity_type] = [];
      }

      groups[entity.entity_type].push(entity);
      return groups;
    },
    {},
  );
}

export default function FraudRingsPage() {
  const [transactionId, setTransactionId] = useState("");
  const [investigation, setInvestigation] =
    useState<FraudRingInvestigation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSearch(event: React.FormEvent) {
    event.preventDefault();

    const normalizedId = transactionId.trim();

    if (!normalizedId) {
      setError("Enter a transaction ID.");
      return;
    }

    setLoading(true);
    setError("");
    setInvestigation(null);

    try {
      const result = await getTransactionFraudRing(
        normalizedId,
      );

      setInvestigation(result);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unable to load fraud-ring investigation.";

      setError(message);
    } finally {
      setLoading(false);
    }
  }

  const groupedEntities = investigation
    ? groupEntities(investigation.entities)
    : {};

  return (
    <div className="page-stack">
      <section className="page-heading">
        <div>
          <p className="eyebrow">Network Investigation</p>
          <h1>Fraud Rings</h1>
          <p>
            Investigate the structural network surrounding a
            transaction using the frozen Phase 9 intelligence layer.
          </p>
        </div>
      </section>

      <section className="investigation-search-panel">
        <form
          className="investigation-search-form"
          onSubmit={handleSearch}
        >
          <div className="search-input-wrapper">
            <Search size={18} />
            <input
              value={transactionId}
              onChange={(event) =>
                setTransactionId(event.target.value)
              }
              placeholder="Enter transaction ID, e.g. TXN_000000000"
              aria-label="Transaction ID"
            />
          </div>

          <button
            type="submit"
            className="primary-button"
            disabled={loading}
          >
            {loading ? "Investigating..." : "Investigate Network"}
            {!loading && <ArrowRight size={17} />}
          </button>
        </form>

        <p className="search-helper">
          Start from a transaction to inspect its candidate
          fraud network. This investigation is structural and does
          not itself establish that fraud occurred.
        </p>
      </section>

      {error && (
        <section className="api-error-panel">
          <AlertTriangle size={20} />
          <div>
            <strong>Investigation unavailable</strong>
            <p>{error}</p>
          </div>
        </section>
      )}

      {!investigation && !loading && !error && (
        <section className="empty-investigation">
          <div className="empty-investigation-icon">
            <Network size={28} />
          </div>
          <h2>Ready for network investigation</h2>
          <p>
            Enter a transaction ID to inspect connected entities,
            relationships, structural scoring, and evidence.
          </p>
        </section>
      )}

      {investigation && (
        <>
          <section className="ring-hero-card">
            <div>
              <p className="eyebrow">Candidate Network</p>
              <h2>Structural Fraud-Ring Investigation</h2>
              <p className="ring-summary">
                {investigation.investigation_summary}
              </p>
            </div>

            <div className="ring-score">
              <span>Structural score</span>
              <strong>
                {investigation.network_score.structural_score.toFixed(
                  2,
                )}
              </strong>
              <small>/ 100</small>
            </div>
          </section>

          <section className="metric-grid">
            <div className="metric-card">
              <span>Entities</span>
              <strong>
                {investigation.network_score.entity_count}
              </strong>
            </div>

            <div className="metric-card">
              <span>Relationships</span>
              <strong>
                {investigation.network_score.relationship_count}
              </strong>
            </div>

            <div className="metric-card">
              <span>Entity types</span>
              <strong>
                {investigation.network_score.entity_type_count}
              </strong>
            </div>

            <div className="metric-card">
              <span>Evidence items</span>
              <strong>{investigation.evidence.length}</strong>
            </div>
          </section>

          <div className="investigation-grid">
            <section className="investigation-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Network Composition</p>
                  <h2>Connected Entities</h2>
                </div>
                <Network size={20} />
              </div>

              <div className="entity-groups">
                {Object.entries(groupedEntities).map(
                  ([entityType, entities]) => (
                    <div
                      className="entity-group"
                      key={entityType}
                    >
                      <div className="entity-group-header">
                        <span className="entity-type-badge">
                          {entityIcon(entityType)}
                        </span>
                        <div>
                          <strong>
                            {formatEntityType(entityType)}
                          </strong>
                          <span>
                            {entities.length} connected
                          </span>
                        </div>
                      </div>

                      {entities.map((entity) => (
                        <div
                          className="entity-row"
                          key={`${entity.entity_type}:${entity.entity_id}`}
                        >
                          <code>{entity.entity_id}</code>
                        </div>
                      ))}
                    </div>
                  ),
                )}
              </div>
            </section>

            <section className="investigation-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Structural Evidence</p>
                  <h2>Evidence</h2>
                </div>
                <ShieldAlert size={20} />
              </div>

              <div className="evidence-list">
                {investigation.evidence.map((item, index) => (
                  <article
                    className="evidence-card"
                    key={`${item.evidence_type}-${index}`}
                  >
                    <div className="evidence-card-header">
                      <strong>{item.evidence_type}</strong>
                      <span
                        className={`evidence-strength ${item.strength}`}
                      >
                        {item.strength}
                      </span>
                    </div>

                    <p>{item.description}</p>

                    <div className="evidence-tags">
                      {item.entity_types.map((type) => (
                        <span key={type}>
                          {formatEntityType(type)}
                        </span>
                      ))}
                    </div>
                  </article>
                ))}
              </div>
            </section>
          </div>

          <section className="investigation-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Relationships</p>
                <h2>Network Connections</h2>
              </div>
              <span className="panel-count">
                {investigation.relationships.length}
              </span>
            </div>

            <div className="relationship-list">
              {investigation.relationships.map(
                (relationship, index) => (
                  <div
                    className="relationship-row"
                    key={`${relationship.relationship_type}-${index}`}
                  >
                    <span>
                      {formatEntityType(
                        relationship.source.entity_type,
                      )}
                      <code>
                        {relationship.source.entity_id}
                      </code>
                    </span>

                    <ArrowRight size={16} />

                    <span>
                      {formatEntityType(
                        relationship.target.entity_type,
                      )}
                      <code>
                        {relationship.target.entity_id}
                      </code>
                    </span>

                    <small>
                      {formatEntityType(
                        relationship.relationship_type,
                      )}
                    </small>
                  </div>
                ),
              )}
            </div>
          </section>

          <section className="investigation-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Temporal Context</p>
                <h2>Investigation Rule</h2>
              </div>
            </div>

            <p className="temporal-rule">
              {investigation.temporal_rule}
            </p>
          </section>
        </>
      )}
    </div>
  );
}