export interface FeatureAttribution {
    feature: string;
    attribution: number;
    absolute_attribution: number;
    rank: number;
  }

  export interface GraphFinding {
    finding_type: string;
    description: string;
  }

  export interface Explanation {
    transaction_id: string;
    prediction_probability: number;
    prediction_label: number;
    feature_attributions: FeatureAttribution[];
    graph_findings: GraphFinding[];
    summary: string;
  }