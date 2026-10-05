import { useState } from "react";
import type { FormEvent } from "react";
import { Search, Users, Network, Link2, ShieldAlert } from "lucide-react";

import { getCustomerInvestigation } from "../api/client";
import type {
  CustomerInvestigation,
  EntityNetworkScore,
  RelatedEntity,
} from "../types/customer";

function formatEntityType(value: string): string {
  return value.replaceAll("_", " ");
}

function formatTimestamp(timestamp: string | null): string {
  if (!timestamp) {
    return "No timestamp";
  }

  return new Date(timestamp).toLocaleString();
}

function formatCandidateNetworkId(candidateId: string): string {
  if (candidateId.length > 200) {
    return "Global connected component";
  }

  if (candidateId === "ring-") {
    return "Global connected component";
  }

  return candidateId;
}

function NetworkScoreCard({
  score,
}: {
  score: EntityNetworkScore;
}) {
  return (
    <article className="customer-score-card">
      <div className="customer-score-header">
        <div>
          <span className="metric-label">Candidate network</span>
          <h3>{formatCandidateNetworkId(score.candidate_id)}</h3>
        </div>

        <div className="customer-score-value">
          {score.structural_score.toFixed(1)}
        </div>
      </div>

      <div className="customer-score-bar">
        <div
          className="customer-score-bar-fill"
          style={{
            width: `${Math.min(score.structural_score, 100)}%`,
          }}
        />
      </div>

      <div className="customer-score-grid">
        <div>
          <span>Entities</span>
          <strong>{score.entity_count}</strong>
        </div>

        <div>
          <span>Relationships</span>
          <strong>{score.relationship_count}</strong>
        </div>

        <div>
          <span>Entity types</span>
          <strong>{score.entity_type_count}</strong>
        </div>

        <div>
          <span>Relationship types</span>
          <strong>{score.relationship_type_count}</strong>
        </div>

        <div>
          <span>Non-transaction entities</span>
          <strong>{score.non_transaction_entity_count}</strong>
        </div>

        <div>
          <span>Connectivity</span>
          <strong>
            {score.non_transaction_connectivity.toFixed(3)}
          </strong>
        </div>
      </div>
    </article>
  );
}

function RelatedEntityCard({
  relationship,
}: {
  relationship: RelatedEntity;
}) {
  return (
    <article className="customer-entity-card">
      <div className="customer-entity-icon">
        <Link2 size={17} />
      </div>

      <div className="customer-entity-content">
        <div className="customer-entity-title">
          <strong>{relationship.entity.entity_id}</strong>

          <span className="entity-type-badge">
            {formatEntityType(relationship.entity.entity_type)}
          </span>
        </div>

        <p>
          {relationship.relationship_type.replaceAll("_", " ")}
          {" · "}
          {relationship.direction}
        </p>

        <span className="customer-entity-time">
          {formatTimestamp(relationship.timestamp)}
        </span>
      </div>
    </article>
  );
}

export function CustomersPage() {
  const [customerId, setCustomerId] = useState("");
  const [investigation, setInvestigation] =
    useState<CustomerInvestigation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedId = customerId.trim();

    if (!trimmedId) {
      setError("Enter a customer ID.");
      setInvestigation(null);
      return;
    }

    setLoading(true);
    setError("");
    setInvestigation(null);

    try {
      const result = await getCustomerInvestigation(trimmedId);
      setInvestigation(result);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unable to investigate customer.";

      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="page">
      <div className="page-header">
        <div>
          <span className="eyebrow">Entities</span>
          <h1>Customer Investigation</h1>
          <p>
            Explore customer-level relationships, transaction activity,
            and candidate fraud networks.
          </p>
        </div>
      </div>

      <form
        className="customer-search-panel"
        onSubmit={handleSubmit}
      >
        <div className="customer-search-copy">
          <div className="customer-search-icon">
            <Users size={19} />
          </div>

          <div>
            <span className="metric-label">Investigate customer</span>
            <strong>Search the entity graph</strong>
          </div>
        </div>

        <div className="customer-search-controls">
          <input
            type="text"
            value={customerId}
            onChange={(event) => setCustomerId(event.target.value)}
            placeholder="e.g. CUST_000000000"
            aria-label="Customer ID"
          />

          <button
            type="submit"
            className="primary-button"
            disabled={loading}
          >
            <Search size={17} />
            {loading ? "Investigating..." : "Investigate"}
          </button>
        </div>
      </form>

      {error && (
        <div className="customer-error-panel">
          <ShieldAlert size={19} />

          <div>
            <strong>Unable to investigate customer</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {investigation && (
        <div className="customer-investigation">
          <div className="customer-investigation-hero">
            <div>
              <span className="eyebrow">Investigation target</span>

              <div className="customer-target">
                <h2>{investigation.entity.entity_id}</h2>

                <span className="entity-type-badge large">
                  {formatEntityType(
                    investigation.entity.entity_type,
                  )}
                </span>
              </div>
            </div>

            <div className="customer-hero-stat">
              <span>Related transactions</span>
              <strong>
                {investigation.related_transactions.length}
              </strong>
            </div>
          </div>

          <section className="investigation-summary-panel">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Investigation summary</span>
                <h2>Entity context</h2>
              </div>
            </div>

            <p>{investigation.investigation_summary}</p>

            <div className="temporal-rule">
              <span>Temporal rule</span>
              <p>{investigation.temporal_rule}</p>
            </div>
          </section>

          <div className="customer-investigation-grid">
            <section className="investigation-section">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">Relationships</span>
                  <h2>Related entities</h2>
                </div>

                <span className="section-count">
                  {investigation.related_entities.length}
                </span>
              </div>

              {investigation.related_entities.length > 0 ? (
                <div className="customer-entity-list">
                  {investigation.related_entities.map(
                    (relationship, index) => (
                      <RelatedEntityCard
                        key={`${relationship.entity.entity_id}-${index}`}
                        relationship={relationship}
                      />
                    ),
                  )}
                </div>
              ) : (
                <div className="empty-investigation">
                  No related entities were returned.
                </div>
              )}
            </section>

            <section className="investigation-section">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">Transactions</span>
                  <h2>Related transactions</h2>
                </div>

                <span className="section-count">
                  {investigation.related_transactions.length}
                </span>
              </div>

              {investigation.related_transactions.length > 0 ? (
                <div className="transaction-id-list">
                  {investigation.related_transactions.map(
                    (transactionId) => (
                      <div
                        className="transaction-id-row"
                        key={transactionId}
                      >
                        <span>{transactionId}</span>
                      </div>
                    ),
                  )}
                </div>
              ) : (
                <div className="empty-investigation">
                  No related transactions were returned.
                </div>
              )}
            </section>
          </div>

          <section className="investigation-section">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Fraud intelligence</span>
                <h2>Candidate fraud rings</h2>
              </div>

              <span className="section-count">
                {investigation.candidate_ring_ids.length}
              </span>
            </div>

            {investigation.candidate_ring_ids.length > 0 ? (
              <div className="ring-id-grid">
                {investigation.candidate_ring_ids.map((ringId) => (
                  <div className="ring-id-card" key={ringId}>
                    <Network size={17} />
                    <span>{formatCandidateNetworkId(ringId)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-investigation">
                No candidate fraud rings were identified by the
                investigation service.
              </div>
            )}
          </section>

          <section className="investigation-section">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Network analysis</span>
                <h2>Structural network scores</h2>
              </div>

              <span className="section-count">
                {investigation.network_scores.length}
              </span>
            </div>

            {investigation.network_scores.length > 0 ? (
              <div className="customer-score-list">
                {investigation.network_scores.map((score) => (
                  <NetworkScoreCard
                    key={score.candidate_id}
                    score={score}
                  />
                ))}
              </div>
            ) : (
              <div className="empty-investigation">
                No network scores were returned.
              </div>
            )}
          </section>
        </div>
      )}
    </section>
  );
}