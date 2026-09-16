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
      const errorMessage =
        (data && typeof data === "object" && data.detail) ||
        (data && typeof data === "object" && data.message) ||
        `HTTP ${response.status}: ${response.statusText}`;
      throw new Error(errorMessage);
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
};

export default api;
