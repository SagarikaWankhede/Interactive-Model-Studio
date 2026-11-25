/**
 * Centralized API client module for Interactive ML Model Studio.
 * Connects frontend UI to backend core services.
 */

// Determine backend API URL:
// 1. Explicit environment variable (set in Render dashboard)
// 2. If running on Render (*.onrender.com), connect directly to the deployed backend API: https://model-studio-api.onrender.com
// 3. Otherwise (local development), default to http://127.0.0.1:8000
const isRenderDeployment = typeof window !== "undefined" && window.location.hostname.includes("onrender.com");
const API_BASE_URL = 
  import.meta.env.VITE_API_BASE_URL ||
  (isRenderDeployment ? "https://model-studio-api.onrender.com" : "http://127.0.0.1:8000");

/**
 * Core request helper with auth header injection, error handling, and response normalization.
 */
async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
  const headers = { ...options.headers };

  // Set Content-Type only if not sending FormData
  if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  // Inject Authorization Bearer token if user is signed in
  const token = typeof localStorage !== "undefined" ? localStorage.getItem("ims_token") : null;
  if (token && !headers["Authorization"]) {
    headers["Authorization"] = `Bearer ${token}`;
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
  // Base URL helper
  getBaseUrl: () => API_BASE_URL,

  // System Health
  healthCheck: () => request("/api/health").catch(() => request("/health")),

  // User Authentication
  login: (credentials) =>
    request("/api/auth/login", {
      method: "POST",
      body: JSON.stringify(credentials),
    }),
  register: (userData) =>
    request("/api/auth/register", {
      method: "POST",
      body: JSON.stringify(userData),
    }),
  getMe: () => request("/api/auth/me"),

  // Data Layer (Supports CSV, TSV, Parquet, JSON)
  uploadDataset: (file) => {
    const formData = new FormData();
    formData.append("file", file);
    return request("/api/datasets/upload", {
      method: "POST",
      body: formData,
    });
  },
  getDatasets: () => request("/api/datasets"),
  getDataset: (datasetId) => request(`/api/datasets/${datasetId}`),
  getDatasetPreview: (datasetId, limit = 10) =>
    request(`/api/datasets/${datasetId}/preview?limit=${limit}`),
  getDiagnostics: (datasetId) => request(`/diagnostics?dataset_id=${datasetId}`),

  // Pipeline Preview
  previewRecipe: (payload) =>
    request("/api/pipelines/preview", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // Experiment & Run Engine
  runExperiment: (payload) =>
    request("/api/runs/execute", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getExperiments: () => request("/api/runs"),
  getExperimentDetail: (experimentId) => request(`/api/runs/${experimentId}`),
  compareExperiments: (experimentIds) => {
    const idsStr = Array.isArray(experimentIds) ? experimentIds.join(",") : experimentIds;
    return request(`/api/runs/compare?ids=${encodeURIComponent(idsStr)}`);
  },
  restoreExperiment: (experimentId) =>
    request(`/api/runs/${experimentId}/verify-reproducibility`, {
      method: "POST",
    }),

  // Edge Lab & In-Browser Inference
  getEdgeModels: () => request("/api/runs/edge/models"),
  exportONNX: (experimentId) =>
    request(`/api/runs/${experimentId}/export-onnx`, {
      method: "POST",
    }),
  benchmarkEdge: (experimentId, iterations = 100) =>
    request(`/api/runs/${experimentId}/edge-benchmark?iterations=${iterations}`, {
      method: "POST",
    }),
  predictSample: (runId, features) =>
    request(`/api/runs/${runId}/predict`, {
      method: "POST",
      body: JSON.stringify({ features }),
    }),
  getOnnxDownloadUrl: (runId) => `${API_BASE_URL}/api/runs/${runId}/download-onnx`,
};

export default api;
