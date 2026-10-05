export interface EntityReference {
    entity_type: string;
    entity_id: string;
  }

  export interface SuspicionSignal {
    name: string;
    value: number | boolean | string;
    description: string;
    source: string;
  }

  export interface TransactionRelationship {
    relationship_type: string;
    source: EntityReference;
    target: EntityReference;
    timestamp: string | null;
  }

  export interface TransactionInvestigation {
    transaction_id: string;
    timestamp: string;
    prediction_probability: number;
    prediction_label: number;
    suspicion_signals: SuspicionSignal[];
    related_entities: EntityReference[];
    relationships: TransactionRelationship[];
    candidate_ring_ids: string[];
    investigation_summary: string;
    temporal_rule: string;
  }