export interface FraudRingEntity {
  entity_type: string;
  entity_id: string;
}

export interface FraudRingRelationship {
  relationship_type: string;
  source: FraudRingEntity;
  target: FraudRingEntity;
  timestamp: string | null;
}

export interface RingTransaction {
  transaction_id: string;
  timestamp: string | null;
}

export interface RingNetworkScore {
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

export interface RingEvidence {
  evidence_type: string;
  description: string;
  entity_types: string[];
  relationship_types: string[];
  strength: string;
}

export interface FraudRingInvestigation {
  candidate_id: string;
  entities: FraudRingEntity[];
  relationships: FraudRingRelationship[];
  transactions: RingTransaction[];
  network_score: RingNetworkScore;
  evidence: RingEvidence[];
  entity_type_counts: Record<string, number>;
  relationship_type_counts: Record<string, number>;
  investigation_summary: string;
  temporal_rule: string;
}
