export interface GraphNode {
    node_type: string;
    node_id: number;
  }

  export interface GraphEdge {
    relationship_type: string;
    source_node_type: string;
    source_node_id: number;
    target_node_type: string;
    target_node_id: number;
  }

  export interface GraphInvestigation {
    target_node_type: string;
    target_node_id: number;
    nodes: GraphNode[];
    edges: GraphEdge[];
  }