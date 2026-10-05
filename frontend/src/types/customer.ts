export interface EntityReference {
    entity_type: string;
    entity_id: string;
  }

  export interface RelatedEntity {
    entity: EntityReference;
    relationship_type: string;
    direction: string;
    timestamp: string | null;
  }

  export interface EntityNetworkScore {
    candidate_id: string;
    entity_count: number;
    relationship_count: number;
    relationship_density: number;
    entity_type_count: number;
    relationship_type_count: number;
    non_transaction_entity_count: number;
    non_transaction_connectivity: number;
    structural_score: number;
  }

  export interface CustomerInvestigation {
    entity: EntityReference;
    related_entities: RelatedEntity[];
    related_transactions: string[];
    candidate_ring_ids: string[];
    network_scores: EntityNetworkScore[];
    investigation_summary: string;
    temporal_rule: string;
  }