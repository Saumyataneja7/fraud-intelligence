import { useEffect, useMemo, useState } from "react";
import { Activity, Database, Gauge, Layers3 } from "lucide-react";

import { getModelMetrics } from "../api/client";
import type {
  ModelEvaluation,
  ModelMetricsResponse,
} from "../types/modelMetrics";

function formatMetric(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatModelName(name: string): string {
  return name
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function getEvaluation(
  data: ModelMetricsResponse,
  modelName: string,
  splitName: string,
): ModelEvaluation | undefined {
  return data.results.find(
    (result) =>
      result.model_name === modelName && result.split_name === splitName,
  );
}

export function DashboardPage() {
  const [data, setData] = useState<ModelMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadMetrics() {
      try {
        setLoading(true);
        setError(null);

        const metrics = await getModelMetrics();

        if (!cancelled) {
          setData(metrics);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load model metrics.",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadMetrics();

    return () => {
      cancelled = true;
    };
  }, []);

  const testEvaluations = useMemo(() => {
    if (!data) {
      return [];
    }

    return data.models
      .map((modelName) => getEvaluation(data, modelName, "test"))
      .filter((evaluation): evaluation is ModelEvaluation => Boolean(evaluation));
  }, [data]);

  const aggregateTestCounts = useMemo(() => {
    if (testEvaluations.length === 0) {
      return null;
    }

    const first = testEvaluations[0].metrics;

    return {
      actualPositive: first.actual_positive_count,
      support: first.support,
    };
  }, [testEvaluations]);

  const xgboostTest = data
    ? getEvaluation(data, "xgboost", "test")
    : undefined;

  return (
    <section className="page">
      <div className="page-header">
        <div>
          <span className="eyebrow">Command Center</span>
          <h1>Fraud Overview</h1>
          <p>
            Model performance and detection signals from the validated
            evaluation dataset.
          </p>
        </div>

        <div className="dashboard-source">
          <span className="status-dot" />
          Live API data
        </div>
      </div>

      {loading && (
        <div className="dashboard-state">
          <Activity size={18} />
          Loading model intelligence...
        </div>
      )}

      {error && !loading && (
        <div className="dashboard-state dashboard-error">
          <Activity size={18} />
          <div>
            <strong>Unable to load dashboard data</strong>
            <span>{error}</span>
          </div>
        </div>
      )}

      {data && !loading && !error && (
        <>
          <div className="dashboard-grid">
            <div className="stat-card">
              <div className="stat-icon">
                <Database size={17} />
              </div>
              <span className="stat-label">Feature dataset</span>
              <strong>{data.feature_count}</strong>
              <small>approved features</small>
            </div>

            <div className="stat-card">
              <div className="stat-icon">
                <Layers3 size={17} />
              </div>
              <span className="stat-label">Models evaluated</span>
              <strong>{data.models.length}</strong>
              <small>classical models</small>
            </div>

            <div className="stat-card">
              <div className="stat-icon">
                <Gauge size={17} />
              </div>
              <span className="stat-label">Evaluation splits</span>
              <strong>{data.splits.length}</strong>
              <small>validation + test</small>
            </div>

            <div className="stat-card">
              <div className="stat-icon">
                <Activity size={17} />
              </div>
              <span className="stat-label">Test observations</span>
              <strong>
                {aggregateTestCounts?.support.toLocaleString() ?? "—"}
              </strong>
              <small>
                {aggregateTestCounts
                  ? `${aggregateTestCounts.actualPositive.toLocaleString()} actual positives`
                  : "evaluation records"}
              </small>
            </div>
          </div>

          <div className="dashboard-section">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Evaluation</span>
                <h2>Test-set model performance</h2>
              </div>
              <span className="section-meta">Baseline threshold: 0.5</span>
            </div>

            <div className="model-table-wrapper">
              <table className="model-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>PR-AUC</th>
                    <th>ROC-AUC</th>
                  </tr>
                </thead>

                <tbody>
                  {testEvaluations.map((evaluation) => {
                    const metrics = evaluation.metrics;

                    return (
                      <tr key={evaluation.model_name}>
                        <td>
                          <span className="model-name">
                            {formatModelName(evaluation.model_name)}
                          </span>
                        </td>
                        <td>{formatMetric(metrics.precision)}</td>
                        <td>{formatMetric(metrics.recall)}</td>
                        <td>{formatMetric(metrics.f1)}</td>
                        <td>{formatMetric(metrics.pr_auc)}</td>
                        <td>{formatMetric(metrics.roc_auc)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {xgboostTest && (
            <div className="dashboard-section">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">Detection Summary</span>
                  <h2>XGBoost test-set outcomes</h2>
                </div>
                <span className="section-meta">Persisted evaluation result</span>
              </div>

              <div className="outcome-grid">
                <div className="outcome-card">
                  <span>True positives</span>
                  <strong>
                    {xgboostTest.metrics.true_positives.toLocaleString()}
                  </strong>
                  <small>correctly identified positives</small>
                </div>

                <div className="outcome-card">
                  <span>False positives</span>
                  <strong>
                    {xgboostTest.metrics.false_positives.toLocaleString()}
                  </strong>
                  <small>negative records flagged positive</small>
                </div>

                <div className="outcome-card">
                  <span>False negatives</span>
                  <strong>
                    {xgboostTest.metrics.false_negatives.toLocaleString()}
                  </strong>
                  <small>positive records missed</small>
                </div>

                <div className="outcome-card">
                  <span>True negatives</span>
                  <strong>
                    {xgboostTest.metrics.true_negatives.toLocaleString()}
                  </strong>
                  <small>correctly identified negatives</small>
                </div>
              </div>
            </div>
          )}

          <div className="dataset-context">
            <span>Evaluation source</span>
            <code>{data.feature_dataset}</code>
          </div>
        </>
      )}
    </section>
  );
}