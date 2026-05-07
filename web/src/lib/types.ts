/**
 * TypeScript mirror of app/schemas.py.
 * Update both files together if the API contract changes.
 */

export interface Transaction {
  Time: number;
  Amount: number;
  V1: number; V2: number; V3: number; V4: number; V5: number; V6: number; V7: number;
  V8: number; V9: number; V10: number; V11: number; V12: number; V13: number; V14: number;
  V15: number; V16: number; V17: number; V18: number; V19: number; V20: number; V21: number;
  V22: number; V23: number; V24: number; V25: number; V26: number; V27: number; V28: number;
}

export interface PredictionResponse {
  fraud_probability: number;
  is_fraud: boolean;
  threshold: number;
  model_name: string;
}

export interface BatchTransactionRequest {
  transactions: Transaction[];
}

export interface BatchPredictionResponse {
  predictions: PredictionResponse[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export interface ModelInfoResponse {
  model_name: string;
  model_type: string;
  trained_at_utc: string;
  threshold: number;
  feature_columns: string[];
  training_info: Record<string, unknown>;
  test_metrics: Record<string, unknown>;
}
