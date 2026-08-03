/**
 * API response types, hand-written to match the checked-in OpenAPI snapshot
 * (frontend/src/types/openapi.json). tests/test_openapi_snapshot.py fails
 * when the backend schema drifts from that snapshot; when it does, update
 * the snapshot AND these types together.
 */

export interface SessionSummary {
  session_id: string;
  agent_id: string;
  task_description: string | null;
  failure_type: string | null;
  cause_node_id: string | null;
  causal_responsibility_score: number;
  evidence: string[];
  created_at: string | null;
}

export interface SessionListResponse {
  items: SessionSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface Attribution {
  failure_type: string;
  cause_node_id: string | null;
  crs: number;
  evidence: string[];
}

export interface AttributionResponse {
  session_id: string;
  attribution: Attribution | null;
}

export interface GraphNodeOTAR {
  observation: string;
  thought: string;
  action: string;
  result: string;
}

export interface GraphNode {
  id: string;
  label: string;
  agent: string;
  step_index: number;
  action: string;
  is_root_cause: boolean;
  error: boolean;
  subtask: string | null;
  parent_id: string | null;
  execution_time_ms: number | null;
  otar: GraphNodeOTAR;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
}

export interface GraphResponse {
  session_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export type HealthFlag = "healthy" | "degraded" | "critical" | "unknown";
export type TrendDirection = "up" | "down" | "flat";

export interface AgentStatus {
  agent_id: string;
  score: number | null;
  n_sessions: number;
  trend: TrendDirection;
  trend_pct: number;
  health: HealthFlag;
  history: number[];
}

export interface HeatmapCell {
  failure_type: string;
  date: string;
  count: number;
}

export interface RecoveryBreakdownRow {
  playbook: string;
  selector: string;
  total: number;
  successes: number;
  success_rate: number;
}

export interface RecoveryStats {
  adaptive_success_rate: number;
  static_success_rate: number;
  adaptive_trials: number;
  static_trials: number;
  total_trials: number;
}

export interface EvalMethodSummary {
  mode_accuracy_pct: number;
  step_accuracy_pct: number;
}

export interface EvalResultsResponse {
  available: boolean;
  file?: string;
  total_scenarios?: number;
  oracle?: EvalMethodSummary;
  naive_baseline?: EvalMethodSummary;
}

export interface RecoveryFeedbackInput {
  session_id: string;
  agent_id?: string;
  task_description?: string;
  playbook_name: string;
  success: boolean;
  partial?: boolean;
}
