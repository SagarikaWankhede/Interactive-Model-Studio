import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { 
  Play, 
  BarChart3, 
  ArrowUpDown, 
  Search, 
  Filter, 
  Clock, 
  Award, 
  Layers, 
  Cpu, 
  ArrowRight,
  Sparkles
} from "lucide-react";
import api from "../api/client";
import { mockExperiments } from "../api/mockData";

export default function ExperimentList() {
  const navigate = useNavigate();
  const [experiments, setExperiments] = useState(mockExperiments);
  const [searchTerm, setSearchTerm] = useState("");
  const [algoFilter, setAlgoFilter] = useState("all");
  const [sortBy, setSortBy] = useState("accuracy"); // 'accuracy' | 'f1_score' | 'train_time' | 'created_at'
  const [sortOrder, setSortOrder] = useState("desc");

  useEffect(() => {
    // Attempt live fetch from backend
    api.getExperiments()
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setExperiments(data);
        } else {
          setExperiments(mockExperiments);
        }
      })
      .catch(() => {
        setExperiments(mockExperiments);
      });
  }, []);

  const handleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortBy(field);
      setSortOrder("desc");
    }
  };

  const filteredExperiments = experiments
    .filter((exp) => {
      const matchesSearch =
        exp.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        exp.model_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        exp.dataset_name.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesAlgo = algoFilter === "all" || exp.model_type === algoFilter;

      return matchesSearch && matchesAlgo;
    })
    .sort((a, b) => {
      let valA, valB;
      if (sortBy === "train_time") {
        valA = a.train_time || 0;
        valB = b.train_time || 0;
      } else if (sortBy === "created_at") {
        valA = new Date(a.created_at).getTime();
        valB = new Date(b.created_at).getTime();
      } else {
        valA = a.metrics?.[sortBy] || 0;
        valB = b.metrics?.[sortBy] || 0;
      }

      if (sortOrder === "asc") return valA > valB ? 1 : -1;
      return valA < valB ? 1 : -1;
    });

  const bestExp = [...experiments].sort((a, b) => (b.metrics?.accuracy || 0) - (a.metrics?.accuracy || 0))[0];

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header flex justify-between items-start flex-wrap gap-4">
        <div>
          <h1 className="page-title">Experiment Run History</h1>
          <p className="page-subtitle">
            Track, sort, and inspect past model training runs, hyperparameters, and evaluation metrics.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Link to="/models" className="btn btn-primary flex items-center gap-2">
            <Play size={15} fill="white" />
            <span>Launch New Run</span>
          </Link>
          <Link to="/dashboard" className="btn btn-secondary flex items-center gap-2">
            <BarChart3 size={15} />
            <span>Open Dashboard</span>
          </Link>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="metrics-grid mb-6">
        <div className="metric-card">
          <div className="metric-icon bg-indigo-100 text-primary">
            <Layers size={22} />
          </div>
          <div className="metric-info">
            <span className="metric-label">Total Experiments Run</span>
            <span className="metric-value">{experiments.length} Runs</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon bg-emerald-100 text-emerald-600">
            <Award size={22} />
          </div>
          <div className="metric-info">
            <span className="metric-label">Top Performer (Accuracy)</span>
            <span className="metric-value text-emerald-600">
              {bestExp ? `${bestExp.model_name} (${(bestExp.metrics.accuracy * 100).toFixed(1)}%)` : "—"}
            </span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon bg-cyan-100 text-cyan-600">
            <Clock size={22} />
          </div>
          <div className="metric-info">
            <span className="metric-label">Fastest Training</span>
            <span className="metric-value">
              {Math.min(...experiments.map((e) => e.train_time || 999)).toFixed(2)}s
            </span>
          </div>
        </div>
      </div>

      {/* Search & Filter Controls */}
      <div className="studio-card mb-6">
        <div className="flex justify-between items-center flex-wrap gap-4">
          <div className="flex items-center gap-3 flex-1 min-w-[280px]">
            <div className="relative flex-1">
              <input
                type="text"
                placeholder="Search by Run ID, Model, or Dataset..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="search-input w-full pl-9"
              />
            </div>

            <select
              value={algoFilter}
              onChange={(e) => setAlgoFilter(e.target.value)}
              className="form-select-sm"
            >
              <option value="all">All Algorithms</option>
              <option value="RandomForest">Random Forest</option>
              <option value="XGBoost">XGBoost</option>
              <option value="LogisticRegression">Logistic Regression</option>
              <option value="DecisionTree">Decision Tree</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-muted font-medium">Sort by:</span>
            <button
              className={`toggle-btn ${sortBy === "accuracy" ? "active" : ""}`}
              onClick={() => handleSort("accuracy")}
            >
              Accuracy {sortBy === "accuracy" && (sortOrder === "desc" ? "↓" : "↑")}
            </button>
            <button
              className={`toggle-btn ${sortBy === "f1_score" ? "active" : ""}`}
              onClick={() => handleSort("f1_score")}
            >
              F1 {sortBy === "f1_score" && (sortOrder === "desc" ? "↓" : "↑")}
            </button>
            <button
              className={`toggle-btn ${sortBy === "train_time" ? "active" : ""}`}
              onClick={() => handleSort("train_time")}
            >
              Time {sortBy === "train_time" && (sortOrder === "desc" ? "↓" : "↑")}
            </button>
          </div>
        </div>
      </div>

      {/* Experiments Table */}
      <div className="studio-card">
        <div className="table-responsive">
          <table className="studio-table">
            <thead>
              <tr>
                <th>Run ID & Model</th>
                <th>Dataset</th>
                <th onClick={() => handleSort("accuracy")} className="cursor-pointer">
                  <div className="flex items-center gap-1">
                    <span>Accuracy</span>
                    <ArrowUpDown size={12} />
                  </div>
                </th>
                <th onClick={() => handleSort("f1_score")} className="cursor-pointer">
                  <div className="flex items-center gap-1">
                    <span>F1 Score</span>
                    <ArrowUpDown size={12} />
                  </div>
                </th>
                <th>Precision / Recall</th>
                <th onClick={() => handleSort("train_time")} className="cursor-pointer">
                  <div className="flex items-center gap-1">
                    <span>Train Time</span>
                    <ArrowUpDown size={12} />
                  </div>
                </th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredExperiments.map((exp) => {
                const isBest = bestExp && bestExp.id === exp.id;
                return (
                  <tr key={exp.id} className={isBest ? "target-row" : ""}>
                    <td>
                      <div className="flex items-center gap-2">
                        <Cpu size={16} className="text-primary" />
                        <div>
                          <strong className="block text-sm">{exp.model_name}</strong>
                          <code className="text-xs text-muted">{exp.id}</code>
                          {isBest && <span className="target-badge ml-1.5">Top Performer</span>}
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className="text-sm font-medium">{exp.dataset_name}</span>
                    </td>
                    <td>
                      <strong className="text-sm text-primary">
                        {(exp.metrics.accuracy * 100).toFixed(1)}%
                      </strong>
                    </td>
                    <td>
                      <span className="text-sm font-semibold text-indigo-700">
                        {(exp.metrics.f1_score * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td>
                      <span className="text-xs text-muted">
                        {(exp.metrics.precision * 100).toFixed(1)}% / {(exp.metrics.recall * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td>
                      <span className="text-xs font-mono font-bold bg-slate-100 px-2 py-1 rounded">
                        {exp.train_time}s
                      </span>
                    </td>
                    <td>
                      <span className="text-xs text-muted">
                        {new Date(exp.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <Link 
                          to="/dashboard" 
                          className="btn btn-secondary btn-sm flex items-center gap-1"
                        >
                          <BarChart3 size={13} />
                          <span>Compare</span>
                        </Link>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
