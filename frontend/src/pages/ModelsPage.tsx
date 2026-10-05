import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Target,
  TrendingUp,
} from "lucide-react";

import { getModelMetrics } from "../api/client";
import type {
  ClassificationMetrics,
  ModelEvaluation,
  ModelMetricsResponse,
} from "../types/modelMetrics";

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDecimal(value: number): string {
  return value.toFixed(3);
}

function getModelLabel(modelName: string): string {
  return modelName
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function getModelResult(
  results: ModelEvaluation[],
  modelName: string,
  splitName: string,
): ModelEvaluation | undefined {
  return results.find(
    (result) =>
      result.model_name === modelName &&
      result.split_name === splitName,
  );
}

function getBestModel(
  results: ModelEvaluation[],
  splitName: string,
): ModelEvaluation | undefined {
  const candidates = results.filter(
    (result) => result.split_name === splitName,
  );

  return candidates.reduce<ModelEvaluation | undefined>(
    (best, current) => {
      if (!best || current.metrics.pr_auc > best.metrics.pr_auc) {
        return current;
      }

      return best;
    },
    undefined,
  );
}

function MetricCard({
  label,
  value,
  description,
  icon,
}: {
  label: string;
  value: string;
  description: string;
  icon: ReactNode;
}) {
  return (
    <div className="model-metric-card">
      <div className="model-metric-card-top">
        <span>{label}</span>
        {icon}
      </div>

      <strong>{value}</strong>
      <small>{description}</small>
    </div>
  );
}

function ConfusionMatrix({
  metrics,
}: {
  metrics: ClassificationMetrics;
}) {
  return (
    <div className="confusion-matrix">
      <div className="confusion-corner" />

      <div className="confusion-axis-label">Predicted negative</div>
      <div className="confusion-axis-label">Predicted positive</div>

      <div className="confusion-axis-side">Actual negative</div>

      <div className="confusion-cell">
        <span>TN</span>
        <strong>{metrics.true_negatives.toLocaleString()}</strong>
        <small>True negative</small>
      </div>

      <div className="confusion-cell warning">
        <span>FP</span>
        <strong>{metrics.false_positives.toLocaleString()}</strong>
        <small>False positive</small>
      </div>

      <div className="confusion-axis-side">Actual positive</div>

      <div className="confusion-cell warning">
        <span>FN</span>
        <strong>{metrics.false_negatives.toLocaleString()}</strong>
        <small>False negative</small>
      </div>

      <div className="confusion-cell positive">
        <span>TP</span>
        <strong>{metrics.true_positives.toLocaleString()}</strong>
        <small>True positive</small>
      </div>
    </div>
  );
}

export function ModelsPage() {
  const [data, setData] = useState<ModelMetricsResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedSplit, setSelectedSplit] = useState("test");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadMetrics() {
      setLoading(true);
      setError("");

      try {
        const result = await getModelMetrics();

        if (!active) {
          return;
        }

        setData(result);

        if (result.models.length > 0) {
          setSelectedModel(result.models.includes("xgboost")
            ? "xgboost"
            : result.models[0]);
        }
      } catch (requestError) {
        if (!active) {
          return;
        }

        setError(
          requestError instanceof Error
            ? requestError.message
            : "Unable to load model metrics.",
        );
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadMetrics();

    return () => {
      active = false;
    };
  }, []);

  const selectedResult = useMemo(
    () =>
      data
        ? getModelResult(
            data.results,
            selectedModel,
            selectedSplit,
          )
        : undefined,
    [data, selectedModel, selectedSplit],
  );

  const bestTestModel = useMemo(
    () => (data ? getBestModel(data.results, "test") : undefined),
    [data],
  );

  const bestValidationModel = useMemo(
    () => (data ? getBestModel(data.results, "validation") : undefined),
    [data],
  );

  if (loading) {
    return (
      <section className="page">
        <div className="page-header">
          <div>
            <span className="eyebrow">Machine Learning</span>
            <h1>Models</h1>
            <p>Review fraud detection model performance.</p>
          </div>
        </div>

        <div className="model-loading">
          <BarChart3 size={24} />
          <span>Loading model evaluation...</span>
        </div>
      </section>
    );
  }

  if (error || !data) {
    return (
      <section className="page">
        <div className="page-header">
          <div>
            <span className="eyebrow">Machine Learning</span>
            <h1>Models</h1>
            <p>Review fraud detection model performance.</p>
          </div>
        </div>

        <div className="model-error">
          <AlertTriangle size={20} />
          <div>
            <strong>Unable to load model metrics</strong>
            <span>{error || "No model evaluation data was returned."}</span>
          </div>
        </div>
      </section>
    );
  }

  const metrics = selectedResult?.metrics;

  return (
    <section className="page">
      <div className="page-header model-page-heading">
        <div>
          <span className="eyebrow">Machine Learning</span>
          <h1>Model Performance</h1>
          <p>
            Compare the frozen classical fraud detection evaluations across
            validation and isolated test data.
          </p>
        </div>

        <div className="model-dataset-badge">
          <span>{data.feature_count}</span>
          <small>approved features</small>
        </div>
      </div>

      <div className="model-overview-grid">
        <div className="model-overview-card">
          <span>Models evaluated</span>
          <strong>{data.models.length}</strong>
          <small>{data.models.map(getModelLabel).join(" · ")}</small>
        </div>

        <div className="model-overview-card">
          <span>Evaluation splits</span>
          <strong>{data.splits.length}</strong>
          <small>Validation + isolated test</small>
        </div>

        <div className="model-overview-card">
          <span>Test observations</span>
          <strong>
            {data.results
              .find((result) => result.split_name === "test")
              ?.metrics.support.toLocaleString() ?? "—"}
          </strong>
          <small>Chronological holdout</small>
        </div>

        <div className="model-overview-card">
          <span>Best test PR-AUC</span>
          <strong>
            {bestTestModel
              ? formatDecimal(bestTestModel.metrics.pr_auc)
              : "—"}
          </strong>
          <small>
            {bestTestModel
              ? getModelLabel(bestTestModel.model_name)
              : "Unavailable"}
          </small>
        </div>
      </div>

      <section className="model-performance-panel">
        <div className="model-panel-header">
          <div>
            <span className="eyebrow">MODEL COMPARISON</span>
            <h2>Performance by split</h2>
          </div>

          <div className="model-controls">
            <select
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
            >
              {data.models.map((model) => (
                <option key={model} value={model}>
                  {getModelLabel(model)}
                </option>
              ))}
            </select>

            <div className="split-toggle">
              {data.splits.map((split) => (
                <button
                  key={split}
                  type="button"
                  className={selectedSplit === split ? "active" : ""}
                  onClick={() => setSelectedSplit(split)}
                >
                  {split}
                </button>
              ))}
            </div>
          </div>
        </div>

        {metrics && (
          <>
            <div className="model-metric-grid">
              <MetricCard
                label="PR-AUC"
                value={formatDecimal(metrics.pr_auc)}
                description="Primary imbalance-aware ranking metric"
                icon={<Target size={17} />}
              />

              <MetricCard
                label="Precision"
                value={formatPercent(metrics.precision)}
                description="Share of flagged transactions that were fraud"
                icon={<CheckCircle2 size={17} />}
              />

              <MetricCard
                label="Recall"
                value={formatPercent(metrics.recall)}
                description="Share of actual fraud detected"
                icon={<TrendingUp size={17} />}
              />

              <MetricCard
                label="F1"
                value={formatDecimal(metrics.f1)}
                description="Balance between precision and recall"
                icon={<BarChart3 size={17} />}
              />

              <MetricCard
                label="ROC-AUC"
                value={formatDecimal(metrics.roc_auc)}
                description="Overall ranking discrimination"
                icon={<TrendingUp size={17} />}
              />
            </div>

            <div className="model-detail-grid">
              <div className="model-detail-panel">
                <div className="model-section-heading">
                  <div>
                    <span className="eyebrow">ERROR ANALYSIS</span>
                    <h3>Confusion matrix</h3>
                  </div>

                  <span className="model-support">
                    {metrics.support.toLocaleString()} observations
                  </span>
                </div>

                <ConfusionMatrix metrics={metrics} />

                <div className="model-count-grid">
                  <div>
                    <span>Predicted positive</span>
                    <strong>
                      {metrics.predicted_positive_count.toLocaleString()}
                    </strong>
                  </div>

                  <div>
                    <span>Actual positive</span>
                    <strong>
                      {metrics.actual_positive_count.toLocaleString()}
                    </strong>
                  </div>
                </div>
              </div>

              <div className="model-detail-panel model-interpretation">
                <span className="eyebrow">INTERPRETATION</span>
                <h3>What this means</h3>

                <p>
                  {selectedModel === "logistic_regression"
                    ? "Logistic Regression prioritizes recall, but its low precision means a large number of legitimate transactions are flagged."
                    : selectedModel === "random_forest"
                      ? "Random Forest produces very few false positives, but its lower recall means many fraudulent transactions remain undetected."
                      : "XGBoost provides a middle ground between precision and recall, with a stronger test-set balance than the other evaluated approaches."}
                </p>

                <div className="model-note">
                  <strong>Why PR-AUC matters</strong>
                  <span>
                    Fraud detection is highly imbalanced. PR-AUC focuses on
                    performance for the positive fraud class and is therefore
                    more informative than accuracy alone.
                  </span>
                </div>

                <div className="model-note">
                  <strong>Test isolation</strong>
                  <span>
                    The test split is chronological and remains isolated from
                    model-selection decisions.
                  </span>
                </div>
              </div>
            </div>
          </>
        )}
      </section>

      <section className="model-comparison-panel">
        <div className="model-panel-header">
          <div>
            <span className="eyebrow">COMPARISON</span>
            <h2>All evaluated models</h2>
          </div>

          <span className="model-support">
            Ranked by test PR-AUC
          </span>
        </div>

        <div className="model-comparison-table">
          <div className="model-table-header">
            <span>Model</span>
            <span>Test PR-AUC</span>
            <span>Precision</span>
            <span>Recall</span>
            <span>F1</span>
            <span>ROC-AUC</span>
          </div>

          {data.models
            .map((model) => getModelResult(data.results, model, "test"))
            .filter(
              (result): result is ModelEvaluation => Boolean(result),
            )
            .sort(
              (a, b) => b.metrics.pr_auc - a.metrics.pr_auc,
            )
            .map((result) => (
              <div className="model-table-row" key={result.model_name}>
                <strong>{getModelLabel(result.model_name)}</strong>
                <span>{formatDecimal(result.metrics.pr_auc)}</span>
                <span>{formatPercent(result.metrics.precision)}</span>
                <span>{formatPercent(result.metrics.recall)}</span>
                <span>{formatDecimal(result.metrics.f1)}</span>
                <span>{formatDecimal(result.metrics.roc_auc)}</span>
              </div>
            ))}
        </div>
      </section>

      <section className="model-evaluation-notes">
        <div>
          <span className="eyebrow">EVALUATION CONTEXT</span>
          <h3>Frozen Phase 5 evaluation</h3>
        </div>

        <div className="evaluation-note-grid">
          <div>
            <strong>Dataset</strong>
            <span>{data.feature_dataset}</span>
          </div>

          <div>
            <strong>Features</strong>
            <span>{data.feature_count} approved features</span>
          </div>

          <div>
            <strong>Validation winner</strong>
            <span>
              {bestValidationModel
                ? `${getModelLabel(bestValidationModel.model_name)} · PR-AUC ${formatDecimal(bestValidationModel.metrics.pr_auc)}`
                : "Unavailable"}
            </span>
          </div>

          <div>
            <strong>Test winner</strong>
            <span>
              {bestTestModel
                ? `${getModelLabel(bestTestModel.model_name)} · PR-AUC ${formatDecimal(bestTestModel.metrics.pr_auc)}`
                : "Unavailable"}
            </span>
          </div>
        </div>
      </section>
    </section>
  );
}