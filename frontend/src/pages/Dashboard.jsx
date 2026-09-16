import { useState } from "react";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  LineChart,
  Line
} from "recharts";
import { Award, Zap, Activity } from "lucide-react";

// Placeholder benchmarking data confirming Recharts functionality
const sampleExperimentData = [
  { model: "Logistic Regression", accuracy: 0.82, f1: 0.81, trainTime: 0.4 },
  { model: "Random Forest", accuracy: 0.94, f1: 0.93, trainTime: 1.8 },
  { model: "XGBoost", accuracy: 0.96, f1: 0.95, trainTime: 2.3 },
  { model: "Decision Tree", accuracy: 0.87, f1: 0.86, trainTime: 0.6 },
];

export default function Dashboard() {
  const [activeMetric, setActiveMetric] = useState("accuracy");

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Experiment Comparison & Visual Analytics</h1>
        <p className="page-subtitle">
          Benchmarking models across accuracy, F1-score, and execution latency. (Phase 0 Recharts verification)
        </p>
      </div>

      {/* KPI Cards */}
      <div className="metrics-grid mb-6">
        <div className="metric-card">
          <div className="metric-icon bg-emerald-100 text-emerald-600">
            <Award size={22} />
          </div>
          <div className="metric-info">
            <span className="metric-label">Best Performing Model</span>
            <span className="metric-value text-emerald-600">XGBoost (96.0%)</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon bg-blue-100 text-blue-600">
            <Zap size={22} />
          </div>
          <div className="metric-info">
            <span className="metric-label">Fastest Training Run</span>
            <span className="metric-value">Logistic Regression (0.4s)</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon bg-purple-100 text-purple-600">
            <Activity size={22} />
          </div>
          <div className="metric-info">
            <span className="metric-label">Active Experiments</span>
            <span className="metric-value">4 Evaluated</span>
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="card-grid">
        <div className="studio-card">
          <div className="flex justify-between items-center mb-4">
            <h2 className="card-heading">Model Accuracy & F1 Comparison</h2>
            <div className="button-toggle-group">
              <button 
                className={`toggle-btn ${activeMetric === "accuracy" ? "active" : ""}`}
                onClick={() => setActiveMetric("accuracy")}
              >
                Accuracy
              </button>
              <button 
                className={`toggle-btn ${activeMetric === "f1" ? "active" : ""}`}
                onClick={() => setActiveMetric("f1")}
              >
                F1 Score
              </button>
            </div>
          </div>
          
          <div style={{ width: "100%", height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={sampleExperimentData}
                margin={{ top: 20, right: 30, left: 10, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="model" stroke="#6b7280" tick={{ fontSize: 12 }} />
                <YAxis domain={[0.7, 1.0]} stroke="#6b7280" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip 
                  formatter={(value) => [`${(value * 100).toFixed(1)}%`, activeMetric.toUpperCase()]}
                  contentStyle={{ backgroundColor: "#1f2937", borderRadius: "8px", color: "#fff", border: "none" }}
                />
                <Legend />
                <Bar 
                  dataKey={activeMetric} 
                  fill="#4f46e5" 
                  radius={[6, 6, 0, 0]} 
                  name={activeMetric === "accuracy" ? "Accuracy" : "F1 Score"} 
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="studio-card">
          <h2 className="card-heading mb-4">Training Time Latency (Seconds)</h2>
          <div style={{ width: "100%", height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={sampleExperimentData}
                margin={{ top: 20, right: 30, left: 10, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="model" stroke="#6b7280" tick={{ fontSize: 12 }} />
                <YAxis stroke="#6b7280" unit="s" />
                <Tooltip 
                  formatter={(value) => [`${value}s`, "Training Time"]}
                  contentStyle={{ backgroundColor: "#1f2937", borderRadius: "8px", color: "#fff", border: "none" }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="trainTime" 
                  stroke="#06b6d4" 
                  strokeWidth={3}
                  dot={{ r: 5, fill: "#06b6d4" }}
                  name="Training Time (s)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
