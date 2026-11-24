/**
 * Centralized API client module for Interactive ML Model Studio.
 * Connects frontend UI to backend core services.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

/**
 * Core request helper with error handling and response normalization.
 */
async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
  const headers = { ...options.headers };

  // Set Content-Type only if not sending FormData
  if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });

    const contentType = response.headers.get("content-type");
    let data;
    if (contentType && contentType.includes("application/json")) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    if (!response.ok) {
      const detail = data && typeof data === "object" ? data.detail : null;
      const errorMessage =
        (typeof detail === "string" && detail) ||
        (detail && typeof detail === "object" && detail.message) ||
        (data && typeof data === "object" && data.message) ||
        `HTTP ${response.status}: ${response.statusText}`;
      const error = new Error(errorMessage);
      error.status = response.status;
      error.detail = detail;
      throw error;
    }

    return data;
  } catch (error) {
    console.error(`API Error [${options.method || "GET"} ${endpoint}]:`, error);
    throw error;
  }
}

export const api = {
  // Phase 0: System Health
  healthCheck: () => request("/"),

  // Phase 1: Data Layer
  uploadDataset: (file) => {
    const formData = new FormData();
    formData.append("file", file);
    return request("/upload", {
      method: "POST",
      body: formData,
    });
  },
  getDatasets: () => request("/dataset"),
  getDiagnostics: (datasetId) => request(`/diagnostics?dataset_id=${datasetId}`),

  // Phase 2: Experiment Engine
  runExperiment: (payload) =>
    request("/experiment/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getExperiments: () => request("/experiment/list"),
  getExperimentDetail: (experimentId) => request(`/experiment/${experimentId}`),

  // Phase 3: Visual Dashboard
  compareExperiments: (experimentIds) =>
    request("/compare", {
      method: "POST",
      body: JSON.stringify({ experiment_ids: experimentIds }),
    }),
  restoreExperiment: (experimentId) => request(`/restore/${experimentId}`),

  // Phase 4: Model Export
  exportONNX: (experimentId) => request(`/export/${experimentId}`),
  exportCode: (experimentId) => request(`/export/${experimentId}`),

  // Visual pipeline editor (node graph -> code -> training)
  compileGraph: (graph) => request("/api/graph/compile", { method: "POST", body: JSON.stringify(graph) }),
  runGraph: (graph) => request("/api/graph/run", { method: "POST", body: JSON.stringify(graph) }),
  getRunGraph: (runId) => request(`/api/graph/runs/${runId}`),

  // Edge & browser deployment
  convertForBrowser: (runId, { quantize = false } = {}) =>
    request(`/api/runs/${runId}/onnx?quantize=${quantize}`, { method: "POST" }),
  getBrowserExport: (runId) => request(`/api/runs/${runId}/onnx`),
  onnxModelUrl: (runId, variant = "auto") => `${API_BASE_URL}/api/runs/${runId}/onnx/model?variant=${variant}`,
  onnxSpecUrl: (runId) => `${API_BASE_URL}/api/runs/${runId}/onnx/spec`,
  fetchOnnxModel: async (runId, variant = "auto") => {
    const res = await fetch(`${API_BASE_URL}/api/runs/${runId}/onnx/model?variant=${variant}`);
    if (!res.ok) throw new Error(`Could not download the model (HTTP ${res.status}).`);
    return res.arrayBuffer();
  },

  // Data Doctor
  getDatasetReport: (datasetId, target) =>
    request(`/api/datasets/${datasetId}/report${target ? `?target=${encodeURIComponent(target)}` : ""}`),

  // Active learning
  getActiveLearning: (runId) => request(`/api/runs/${runId}/active-learning`),
  scanRun: (runId, { threshold = 0.6, limit = 25 } = {}) =>
    request(`/api/runs/${runId}/scan?threshold=${threshold}&limit=${limit}`, { method: "POST" }),
  listEdgeCases: (runId, status) =>
    request(`/api/edge-cases?run_id=${runId}${status ? `&status=${status}` : ""}`),
  addEdgeCase: (payload) => request("/api/edge-cases", { method: "POST", body: JSON.stringify(payload) }),
  labelEdgeCase: (caseId, label) =>
    request(`/api/edge-cases/${caseId}/label`, { method: "POST", body: JSON.stringify({ label }) }),
  dismissEdgeCase: (caseId) => request(`/api/edge-cases/${caseId}/dismiss`, { method: "POST" }),
  retrainFromLabels: (runId) => request(`/api/runs/${runId}/retrain`, { method: "POST" }),
};

export default api;
