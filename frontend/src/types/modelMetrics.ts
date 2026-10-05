export interface ClassificationMetrics {
    precision: number;
    recall: number;
    f1: number;
    pr_auc: number;
    roc_auc: number;

    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;

    support: number;
    predicted_positive_count: number;
    actual_positive_count: number;
  }

  export interface ModelEvaluation {
    model_name: string;
    split_name: string;
    metrics: ClassificationMetrics;
  }

  export interface ModelMetricsResponse {
    feature_dataset: string;
    feature_count: number;
    models: string[];
    splits: string[];
    results: ModelEvaluation[];
  }