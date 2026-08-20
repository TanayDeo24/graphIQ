import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const client = axios.create({ baseURL: BASE_URL });

export const api = {
  riskScores: (params) => client.get("/api/attrition/risk-scores", { params }).then((r) => r.data),
  calibration: () => client.get("/api/attrition/calibration").then((r) => r.data),
  leadTime: () => client.get("/api/attrition/lead-time").then((r) => r.data),
  shap: (employeeId) => client.get(`/api/attrition/shap/${employeeId}`).then((r) => r.data),
  sensitivity: () => client.get("/api/attrition/sensitivity").then((r) => r.data),

  interactionHeatmap: () => client.get("/api/attrition/interaction-heatmap").then((r) => r.data),
  riskMigration: () => client.get("/api/attrition/risk-migration").then((r) => r.data),

  anomalies: (params) => client.get("/api/spend/anomalies", { params }).then((r) => r.data),
  gainsCurve: () => client.get("/api/spend/gains-curve").then((r) => r.data),
  drift: (params) => client.get("/api/spend/drift", { params }).then((r) => r.data),
  alertFatigue: () => client.get("/api/spend/alert-fatigue").then((r) => r.data),
  detectorComparison: () => client.get("/api/spend/detector-comparison").then((r) => r.data),
  explainTransaction: (transactionId) =>
    client.get(`/api/spend/transaction/${transactionId}/explain`).then((r) => r.data),
  detectorOverlap: () => client.get("/api/spend/detector-overlap").then((r) => r.data),
  dollarTreemap: () => client.get("/api/spend/dollar-treemap").then((r) => r.data),
  cusumAnnotatedTrajectories: () => client.get("/api/spend/cusum-annotated-trajectories").then((r) => r.data),

  crossComponentQuadrant: () => client.get("/api/cross-component/quadrant").then((r) => r.data),

  headlineStats: () => client.get("/api/headline/stats").then((r) => r.data),
};
