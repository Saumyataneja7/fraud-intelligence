export interface HealthResponse {
    status: string;
    service: string;
    version: string;
  }

  export interface SystemMetadata {
    service: string;
    version: string;
    project: string;
    phase: string;
    status: string;
    dataset: {
      name: string;
      version: string;
    };
    architecture: Record<string, string>;
    frozen_phases: string[];
  }