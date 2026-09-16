import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from "recharts";
import { 
  Award, 
  Zap, 
  Target, 
  CheckSquare, 
  Square, 
  RotateCcw, 
  Layers, 
  Sliders, 
  BarChart3,
  Sparkles,
  Info
} from "lucide-react";
import api from "../api/client";
import { mockExperiments } from "../api/mockData";
import BestModelBadge from "../components/BestModelBadge";
import ConfusionMatrixChart from "../components/ConfusionMatrixChart";
import FeatureImportanceChart from "../components/FeatureImportanceChart";

export default function Dashboard() {
  const [allExperiments, setAllExperiments] = useState(mockExperiments);
  // Default to selecting the first 2 or 3 experiments for comparison
  const [selectedIds, setSelectedIds] = useState(() => 
    mockExperiments.slice(0, 3).map((e) => e.id)
  );
  const [activeTab, setActiveTab] = useState("overview"); // 'overview' | 'confusion_matrix' | 'features'

  useEffect(() => {
    api.getExperiments()
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setAllExperiments(data);
          if (selectedIds.length === 0) {
            setSelectedIds(data.slice(0, 3).map((e) => e.id));
          }
        }
      })
      .catch(() => {
        setAllExperiments(mockExperiments);
      });
  }, []);

  const toggleSelect = (id) => {
    setSelectedIds((prev) => {
      if (prev.includes(id)) {
        if (prev.length <= 1) return prev; // Keep at least 1
        return prev.filter((item) => item !== id);
      }
      return [...prev, id];
    });
  };

  const selectedExperiments = allExperiments.filter((e) => selectedIds.includes(e.id));

  // Determine winners for each metric
  const bestAccuracy = Math.max(...selectedExperiments.map((e) => e.metrics?.accuracy || 0));
  const bestF1 = Math.max(...selectedExperiments.map((e) => e.metrics?.f1_score || 0));
  const bestSpeed = Math.min(...selectedExperiments.map((e) => e.train_time || 999));

  // Chart data for metric comparison
  const comparisonChartData = selectedExperiments.map((exp) => ({
    name: exp.model_name,
    Accuracy: parseFloat(((exp.metrics?.accuracy || 0) * 100).toFixed(1)),
    "F1 Score": parseFloat(((exp.metrics?.f1_score || 0) * 100).toFixed(1)),
    Precision: parseFloat(((exp.metrics?.precision || 0) * 100).toFixed(1)),
    Recall: parseFloat(((exp.metrics?.recall || 0) * 100).toFixed(1)),
  }));

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header flex justify-between items-start flex-wrap gap-4">
        <div>
          <h1 className="page-title">Visual Comparison Dashboard</h1>
          <p className="page-subtitle">
            Benchmark 2 or more trained models side-by-side across classification metrics, confusion matrix heatmaps, and feature importances.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Link to="/models" className="btn btn-primary btn-sm flex items-center gap-1.5">
            <Sparkles size={14} />
            <span>Train Another Model</span>
          </Link>
        </div>
      </div>

      {/* Model Selection Pills Bar */}
      <div className="studio-card mb-6">
        <div className="flex justify-between items-center mb-3 flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <CheckSquare size={16} className="text-primary" />
            <strong className="text-sm">Select Models to Benchmark (2+ recommended):</strong>
          </div>
          <span className="text-xs text-muted">
            {selectedExperiments.length} of {allExperiments.length} models selected
          </span>
        </div>

        <div className="flex flex-wrap gap-2">
          {allExperiments.map((exp) => {
            const isChecked = selectedIds.includes(exp.id);
            return (
              <button
                key={exp.id}
                onClick={() => toggleSelect(exp.id)}
                className={`compare-select-pill ${isChecked ? "active" : ""}`}
              >
                {isChecked ? <CheckSquare size={15} className="text-primary" /> : <Square size={15} />}
                <span className="font-semibold">{exp.model_name}</span>
                <span className="text-xs text-muted">({((exp.metrics?.accuracy || 0) * 100).toFixed(1)}%)</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* View Switcher Tabs */}
      <div className="flex items-center gap-2 mb-6 border-b border-slate-200 pb-2">
        <button
          className={`tab-btn ${activeTab === "overview" ? "active" : ""}`}
          onClick={() => setActiveTab("overview")}
        >
          <BarChart3 size={15} />
          <span>Metric Comparison</span>
        </button>
        <button
          className={`tab-btn ${activeTab === "confusion_matrix" ? "active" : ""}`}
          onClick={() => setActiveTab("confusion_matrix")}
        >
          <Layers size={15} />
          <span>Confusion Matrix Heatmaps</span>
        </button>
        <button
          className={`tab-btn ${activeTab === "features" ? "active" : ""}`}
          onClick={() => setActiveTab("features")}
        >
          <Sliders size={15} />
          <span>Feature Importances</span>
        </button>
      </div>

      {/* TAB 1: Metric Comparison Table & Bar Chart */}
      {activeTab === "overview" && (
        <>
          {/* Side-by-Side Comparison Table */}
          <div className="studio-card mb-6">
            <h2 className="card-heading mb-4">Side-by-Side Performance Matrix</h2>
            <div className="table-responsive">
              <table className="studio-table">
                <thead>
                  <tr>
                    <th>Evaluated Model</th>
                    <th>Accuracy</th>
                    <th>F1 Score</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>ROC-AUC</th>
                    <th>Train Time</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedExperiments.map((exp) => {
                    const isBestAcc = exp.metrics?.accuracy === bestAccuracy;
                    const isBestF1 = exp.metrics?.f1_score === bestF1;
                    const isFastest = exp.train_time === bestSpeed;

                    const accDelta = ((exp.metrics?.accuracy || 0) - bestAccuracy) * 100;

                    return (
                      <tr key={exp.id}>
                        <td>
                          <div>
                            <strong className="block text-sm">{exp.model_name}</strong>
                            <code className="text-xs text-muted">{exp.id}</code>
                          </div>
                        </td>
                        <td>
                          <div className="flex items-center gap-2">
                            <strong className="text-base text-primary">
                              {((exp.metrics?.accuracy || 0) * 100).toFixed(1)}%
                            </strong>
                            {isBestAcc && <BestModelBadge type="accuracy" />}
                            {!isBestAcc && (
                              <span className="text-xs text-danger font-semibold">
                                {accDelta.toFixed(1)}%
                              </span>
                            )}
                          </div>
                        </td>
                        <td>
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-indigo-700">
                              {((exp.metrics?.f1_score || 0) * 100).toFixed(1)}%
                            </span>
                            {isBestF1 && <BestModelBadge type="f1_score" />}
                          </div>
                        </td>
                        <td>{((exp.metrics?.precision || 0) * 100).toFixed(1)}%</td>
                        <td>{((exp.metrics?.recall || 0) * 100).toFixed(1)}%</td>
                        <td>{((exp.metrics?.roc_auc || 0) * 100).toFixed(1)}%</td>
                        <td>
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs bg-slate-100 px-2 py-1 rounded">
                              {exp.train_time}s
                            </span>
                            {isFastest && <BestModelBadge type="train_time" />}
                          </div>
                        </td>
                        <td>
                          <Link
                            to={`/restore/${exp.id}`}
                            className="btn btn-secondary btn-sm flex items-center gap-1"
                            title="Restore recipe and hyperparams"
                          >
                            <RotateCcw size={13} />
                            <span>Restore</span>
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Grouped Comparison Bar Chart */}
          <div className="studio-card">
            <h2 className="card-heading mb-4">Metric Benchmark Visualizer (%)</h2>
            <div style={{ width: "100%", height: 350 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={comparisonChartData}
                  margin={{ top: 20, right: 30, left: 10, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 12 }} />
                  <YAxis domain={[70, 100]} stroke="#64748b" unit="%" />
                  <Tooltip
                    formatter={(v) => [`${v}%`]}
                    contentStyle={{ backgroundColor: "#0f172a", borderRadius: "8px", color: "#fff", border: "none" }}
                  />
                  <Legend />
                  <Bar dataKey="Accuracy" fill="#4f46e5" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="F1 Score" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Precision" fill="#10b981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Recall" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* TAB 2: Confusion Matrix Heatmaps */}
      {activeTab === "confusion_matrix" && (
        <div className="studio-card">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h2 className="card-heading m-0">Confusion Matrix Heatmaps</h2>
              <span className="text-xs text-muted">
                Inspect raw classification counts and error distributions (False Positives vs False Negatives).
              </span>
            </div>
          </div>
          <ConfusionMatrixChart experiments={selectedExperiments} />
        </div>
      )}

      {/* TAB 3: Feature Importances */}
      {activeTab === "features" && (
        <div className="studio-card">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h2 className="card-heading m-0">Comparative Feature Importance Weights</h2>
              <span className="text-xs text-muted">
                Understand which columns contribute most heavily to each model's decisions.
              </span>
            </div>
          </div>
          <FeatureImportanceChart experiments={selectedExperiments} />
        </div>
      )}
    </div>
  );
}
