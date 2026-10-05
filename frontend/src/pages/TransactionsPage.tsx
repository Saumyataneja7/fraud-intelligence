import { useState } from "react";
import type { FormEvent } from "react";
import {
  AlertTriangle,
  ArrowRight,
  BrainCircuit,
  Clock3,
  Search,
  ShieldCheck,
} from "lucide-react";

import {
  getTransactionExplanation,
  getTransactionInvestigation,
} from "../api/client";
import type { Explanation } from "../types/explanation";
import type { TransactionInvestigation } from "../types/transaction";

function formatProbability(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`;
}

function formatDate(timestamp: string): string {
  return new Date(timestamp).toLocaleString();
}

function formatEntityType(entityType: string): string {
  return entityType
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function getRiskClass(label: number): string {
  return label === 1 ? "risk-high" : "risk-normal";
}

export function TransactionsPage() {
  const [transactionId, setTransactionId] = useState("");
  const [investigation, setInvestigation] =
    useState<TransactionInvestigation | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{
    status: number | null;
    message: string;
  } | null>(null);

  async function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const normalizedId = transactionId.trim();

    if (!normalizedId) {
      setError({
        status: null,
        message: "Enter a transaction ID to begin an investigation.",
      });
      return;
    }

    setLoading(true);
    setError(null);
    setInvestigation(null);
    setExplanation(null);

    try {
      const investigationResult =
        await getTransactionInvestigation(normalizedId);

      setInvestigation(investigationResult);

      try {
        const explanationResult =
          await getTransactionExplanation(normalizedId);

        setExplanation(explanationResult);
      } catch {
        // Explanation is supplementary. The main investigation remains visible.
      }
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unable to retrieve the transaction investigation.";

      const statusMatch = message.match(/^API request failed: (\d+)/);
      const status = statusMatch ? Number(statusMatch[1]) : null;

      setError({
        status,
        message,
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="page">
      <div className="page-header">
        <div>
          <span className="eyebrow">Investigations</span>
          <h1>Transaction Investigation</h1>
          <p>
            Search a transaction and inspect its fraud intelligence signals.
          </p>
        </div>
      </div>

      <form className="transaction-search" onSubmit={handleSearch}>
        <div className="transaction-search-input">
          <Search size={17} />
          <input
            type="text"
            value={transactionId}
            onChange={(event) => setTransactionId(event.target.value)}
            placeholder="Enter transaction ID, e.g. TXN_000000000"
            aria-label="Transaction ID"
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Investigating..." : "Investigate"}
          {!loading && <ArrowRight size={15} />}
        </button>
      </form>

      {error && (
        <div className="investigation-error">
          <AlertTriangle size={18} />

          <div>
            <strong>
              {error.status === 404
                ? "Transaction not found"
                : error.status === 503
                  ? "Investigation service unavailable"
                  : "Unable to investigate transaction"}
            </strong>

            <span>{error.message}</span>

            {error.status === 503 && (
              <small>
                The backend currently requires a model artifact that is not
                available in this repository.
              </small>
            )}
          </div>
        </div>
      )}

      {investigation && (
        <div className="investigation-content">
          <div className="investigation-hero">
            <div>
              <span className="eyebrow">Transaction</span>
              <h2>{investigation.transaction_id}</h2>
              <span className="transaction-timestamp">
                <Clock3 size={13} />
                {formatDate(investigation.timestamp)}
              </span>
            </div>

            <div
              className={`prediction-badge ${getRiskClass(
                investigation.prediction_label,
              )}`}
            >
              {investigation.prediction_label === 1 ? (
                <AlertTriangle size={16} />
              ) : (
                <ShieldCheck size={16} />
              )}

              <div>
                <strong>
                  {investigation.prediction_label === 1
                    ? "Suspicious"
                    : "Not Suspicious"}
                </strong>

                <span>
                  Probability{" "}
                  {formatProbability(investigation.prediction_probability)}
                </span>
              </div>
            </div>
          </div>

          <div className="investigation-summary">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Assessment</span>
                <h2>Investigation summary</h2>
              </div>
            </div>

            <div className="summary-body">
              <p>{investigation.investigation_summary}</p>

              <div className="temporal-rule">
                <Clock3 size={14} />
                <div>
                  <span>Temporal rule</span>
                  <strong>{investigation.temporal_rule}</strong>
                </div>
              </div>
            </div>
          </div>

          <div className="investigation-section-grid">
            <section className="investigation-card">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">Detection Signals</span>
                  <h2>Suspicion signals</h2>
                </div>

                <span className="card-count">
                  {investigation.suspicion_signals.length}
                </span>
              </div>

              {investigation.suspicion_signals.length === 0 ? (
                <div className="empty-investigation">
                  No suspicion signals were returned.
                </div>
              ) : (
                <div className="signal-list">
                  {investigation.suspicion_signals.map((signal) => (
                    <div className="signal-item" key={signal.name}>
                      <div className="signal-title">
                        <strong>{signal.name}</strong>
                        <span>{String(signal.value)}</span>
                      </div>

                      <p>{signal.description}</p>

                      <small>Source: {signal.source}</small>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <section className="investigation-card">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">Entity Context</span>
                  <h2>Related entities</h2>
                </div>

                <span className="card-count">
                  {investigation.related_entities.length}
                </span>
              </div>

              {investigation.related_entities.length === 0 ? (
                <div className="empty-investigation">
                  No related entities were returned.
                </div>
              ) : (
                <div className="entity-list">
                  {investigation.related_entities.map((entity) => (
                    <div
                      className="entity-item"
                      key={`${entity.entity_type}-${entity.entity_id}`}
                    >
                      <span>{formatEntityType(entity.entity_type)}</span>
                      <strong>{entity.entity_id}</strong>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>

          <section className="investigation-card">
            <div className="card-heading">
              <div>
                <span className="eyebrow">Relationships</span>
                <h2>Transaction relationships</h2>
              </div>

              <span className="card-count">
                {investigation.relationships.length}
              </span>
            </div>

            {investigation.relationships.length === 0 ? (
              <div className="empty-investigation">
                No relationships were returned.
              </div>
            ) : (
              <div className="relationship-list">
                {investigation.relationships.map((relationship, index) => (
                  <div
                    className="relationship-item"
                    key={`${relationship.relationship_type}-${index}`}
                  >
                    <div className="relationship-type">
                      {formatEntityType(relationship.relationship_type)}
                    </div>

                    <div className="relationship-node">
                      <span>
                        {formatEntityType(relationship.source.entity_type)}
                      </span>
                      <strong>{relationship.source.entity_id}</strong>
                    </div>

                    <ArrowRight size={15} />

                    <div className="relationship-node">
                      <span>
                        {formatEntityType(relationship.target.entity_type)}
                      </span>
                      <strong>{relationship.target.entity_id}</strong>
                    </div>

                    {relationship.timestamp && (
                      <time>{formatDate(relationship.timestamp)}</time>
                    )}
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="investigation-card">
            <div className="card-heading">
              <div>
                <span className="eyebrow">Fraud Intelligence</span>
                <h2>Candidate fraud rings</h2>
              </div>

              <span className="card-count">
                {investigation.candidate_ring_ids.length}
              </span>
            </div>

            {investigation.candidate_ring_ids.length === 0 ? (
              <div className="empty-investigation">
                No candidate fraud rings were associated with this
                transaction.
              </div>
            ) : (
              <div className="ring-list">
                {investigation.candidate_ring_ids.map((ringId) => (
                  <div className="ring-item" key={ringId}>
                    <span>Candidate ring</span>
                    <strong>{ringId}</strong>
                  </div>
                ))}
              </div>
            )}
          </section>

          {explanation && (
            <section className="investigation-card">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">Explainability</span>
                  <h2>Model explanation</h2>
                </div>

                <BrainCircuit size={17} />
              </div>

              <div className="explanation-summary">
                {explanation.summary}
              </div>

              {explanation.feature_attributions.length > 0 && (
                <div className="attribution-list">
                  {explanation.feature_attributions.map((item) => (
                    <div
                      className="attribution-item"
                      key={`${item.feature}-${item.rank}`}
                    >
                      <span className="attribution-rank">
                        #{item.rank}
                      </span>

                      <div className="attribution-feature">
                        <strong>{item.feature}</strong>

                        <div className="attribution-bar">
                          <span
                            style={{
                              width: `${Math.min(
                                item.absolute_attribution * 100,
                                100,
                              )}%`,
                            }}
                          />
                        </div>
                      </div>

                      <strong className="attribution-value">
                        {item.attribution >= 0 ? "+" : ""}
                        {item.attribution.toFixed(4)}
                      </strong>
                    </div>
                  ))}
                </div>
              )}

              {explanation.graph_findings.length > 0 && (
                <div className="graph-findings">
                  <span className="eyebrow">Graph findings</span>

                  {explanation.graph_findings.map((finding, index) => (
                    <div
                      className="graph-finding"
                      key={`${finding.finding_type}-${index}`}
                    >
                      <strong>{finding.finding_type}</strong>
                      <span>{finding.description}</span>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}
        </div>
      )}
    </section>
  );
}