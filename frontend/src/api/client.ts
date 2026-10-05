import type { CustomerInvestigation } from "../types/customer";
import type { Explanation } from "../types/explanation";
import type { ModelMetricsResponse } from "../types/modelMetrics";
import type { TransactionInvestigation } from "../types/transaction";
import type { FraudRingInvestigation } from "../types/fraudRing";
import type { GraphInvestigation } from "../types/graph";
import type { HealthResponse, SystemMetadata } from "../types/system";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);

  if (!response.ok) {
    let detail = `API request failed: ${response.status}`;

    try {
      const body = (await response.json()) as { detail?: string };

      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      // Keep the HTTP status message when the response isn't JSON.
    }

    const error = new Error(detail);
    error.name = `HTTP_${response.status}`;
    throw error;
  }

  return response.json() as Promise<T>;
}


export function getModelMetrics(): Promise<ModelMetricsResponse> {
  return request<ModelMetricsResponse>("/model-metrics");
}

export function getTransactionInvestigation(
  transactionId: string,
): Promise<TransactionInvestigation> {
  return request<TransactionInvestigation>(
    `/transaction/${encodeURIComponent(transactionId)}`,
  );
}

export function getTransactionExplanation(
  transactionId: string,
): Promise<Explanation> {
  return request<Explanation>(
    `/explanation/${encodeURIComponent(transactionId)}`,
  );
}

export function getCustomerInvestigation(
    customerId: string,
  ): Promise<CustomerInvestigation> {
    return request<CustomerInvestigation>(
      `/customer/${encodeURIComponent(customerId)}`,
    );
  }

  export function getTransactionFraudRing(
    transactionId: string,
  ): Promise<FraudRingInvestigation> {
    return request<FraudRingInvestigation>(
      `/transaction/${encodeURIComponent(transactionId)}/fraud-ring`,
    );
  }

  export function getGraphInvestigation(
    nodeType: string,
    nodeId: number,
  ): Promise<GraphInvestigation> {
    return request<GraphInvestigation>(
      `/graph/${encodeURIComponent(nodeId)}?node_type=${encodeURIComponent(nodeType)}`,
    );
  }

  export function getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>("/health");
  }

  export function getSystemMetadata(): Promise<SystemMetadata> {
    return request<SystemMetadata>("/metadata");
  }