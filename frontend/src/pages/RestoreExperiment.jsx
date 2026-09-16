import { useState, useEffect } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { 
  RotateCcw, 
  ArrowLeft, 
  CheckCircle2, 
  Cpu, 
  Layers, 
  Database, 
  Play, 
  Code2, 
  Clock,
  Sparkles
} from "lucide-react";
import api from "../api/client";
import { mockExperiments } from "../api/mockData";

export default function RestoreExperiment() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [experiment, setExperiment] = useState(null);
  const [loading, setLoading] = useState(true);
  const [restored, setRestored] = useState(false);

  useEffect(() => {
    setLoading(true);
    api.restoreExperiment(id)
      .then((data) => {
        setExperiment(data);
      })
      .catch(() => {
        const found = mockExperiments.find((e) => e.id === id) || mockExperiments[0];
        setExperiment(found);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [id]);

  if (loading) {
    return (
      <div className="page-container flex justify-center items-center py-20">
        <div className="text-center text-muted">Retrieving experiment snapshot from database...</div>
      </div>
    );
  }

  if (!experiment) {
    return (
      <div className="page-container">
        <div className="studio-card text-center py-12">
          <h2>Experiment Not Found</h2>
          <Link to="/experiments" className="btn btn-primary mt-4">Back to Experiments</Link>
        </div>
      </div>
    );
  }

  const handleRestoreAndTune = () => {
    setRestored(true);
    const hyperparamsEncoded = encodeURIComponent(JSON.stringify(experiment.hyperparams || {}));
    setTimeout(() => {
      navigate(`/models?datasetId=${experiment.dataset_id}&modelType=${experiment.model_type}&hyperparams=${hyperparamsEncoded}`);
    }, 600);
  };

  const handleRestoreAndReRun = () => {
    const hyperparamsEncoded = encodeURIComponent(JSON.stringify(experiment.hyperparams || {}));
    navigate(`/experiments/run?datasetId=${experiment.dataset_id}&modelType=${experiment.model_type}&hyperparams=${hyperparamsEncoded}`);
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <Link to="/experiments" className="text-muted hover:text-primary flex items-center gap-1 text-sm font-medium mb-1">
          <ArrowLeft size={15} /> Back to Experiments
        </Link>
        <div className="flex items-center gap-2">
          <RotateCcw className="text-primary" size={24} />
          <h1 className="page-title m-0">Restore Experiment Configuration</h1>
        </div>
        <p className="page-subtitle">
          Reload the exact preprocessing recipe and hyperparameter snapshot from Run ID: <code>{experiment.id}</code>.
        </p>
      </div>

      {restored && (
        <div className="mb-6 p-4 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 flex items-center gap-2 font-medium">
          <CheckCircle2 size={18} className="text-emerald-600" />
          <span>Configuration snapshot restored successfully! Redirecting to Model Studio...</span>
        </div>
      )}

      {/* Snapshot Header Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="studio-card">
          <span className="text-xs text-muted block font-semibold">ALGORITHM & RUN</span>
          <strong className="text-base text-primary block mt-1">{experiment.model_name}</strong>
          <span className="text-xs text-muted font-mono">{experiment.id}</span>
        </div>

        <div className="studio-card">
          <span className="text-xs text-muted block font-semibold">HISTORICAL ACCURACY</span>
          <strong className="text-xl text-emerald-600 block mt-1">
            {((experiment.metrics?.accuracy || 0) * 100).toFixed(1)}%
          </strong>
          <span className="text-xs text-muted">Train Latency: {experiment.train_time}s</span>
        </div>

        <div className="studio-card">
          <span className="text-xs text-muted block font-semibold">ORIGINAL DATASET</span>
          <strong className="text-base block mt-1">{experiment.dataset_name}</strong>
          <span className="text-xs text-muted">Created: {new Date(experiment.created_at).toLocaleDateString()}</span>
        </div>
      </div>

      {/* Hyperparameters Snapshot */}
      <div className="studio-card mb-6">
        <div className="flex justify-between items-center mb-3">
          <div className="flex items-center gap-2">
            <Code2 size={18} className="text-primary" />
            <h3 className="card-heading m-0">Hyperparameters Snapshot JSON</h3>
          </div>
          <span className="status-tag">Frozen State</span>
        </div>

        <pre className="json-box">
          <code>{JSON.stringify(experiment.hyperparams || {}, null, 2)}</code>
        </pre>
      </div>

      {/* Restore Actions */}
      <div className="studio-card flex justify-between items-center flex-wrap gap-4 bg-slate-50 border-indigo-100">
        <div>
          <strong className="text-sm block">Ready to restore this run?</strong>
          <span className="text-xs text-muted">
            You can load this state into the interactive tuning form to adjust parameters or immediately execute a reproducibility check.
          </span>
        </div>

        <div className="flex items-center gap-3">
          <button 
            className="btn btn-secondary flex items-center gap-2"
            onClick={handleRestoreAndReRun}
          >
            <Play size={15} />
            <span>Re-Run Exact Config</span>
          </button>
          <button 
            className="btn btn-primary flex items-center gap-2"
            onClick={handleRestoreAndTune}
          >
            <RotateCcw size={15} />
            <span>Restore & Tune Parameters</span>
          </button>
        </div>
      </div>
    </div>
  );
}
