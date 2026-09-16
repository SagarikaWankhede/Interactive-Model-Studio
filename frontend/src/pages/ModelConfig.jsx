import { useState } from "react";
import { useSearchParams, useNavigate, Link } from "react-router-dom";
import { 
  Cpu, 
  Settings2, 
  ArrowRight, 
  ArrowLeft, 
  Sparkles, 
  RotateCcw, 
  Play,
  Zap,
  ShieldAlert,
  Info
} from "lucide-react";
import { supportedAlgorithms, mockDatasets } from "../api/mockData";

export default function ModelConfig() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const datasetId = searchParams.get("datasetId") || mockDatasets[0].id;

  const [selectedAlgoId, setSelectedAlgoId] = useState("RandomForest");
  const selectedAlgo = supportedAlgorithms.find((a) => a.id === selectedAlgoId) || supportedAlgorithms[0];

  // Map of hyperparams state: { [algoId]: { ...params } }
  const [hyperparamsByAlgo, setHyperparamsByAlgo] = useState(() => {
    const initial = {};
    supportedAlgorithms.forEach((a) => {
      initial[a.id] = { ...a.defaultHyperparams };
    });
    return initial;
  });

  const currentParams = hyperparamsByAlgo[selectedAlgo.id] || selectedAlgo.defaultHyperparams;

  const handleParamChange = (paramName, value) => {
    setHyperparamsByAlgo((prev) => ({
      ...prev,
      [selectedAlgo.id]: {
        ...prev[selectedAlgo.id],
        [paramName]: value,
      },
    }));
  };

  const resetParams = () => {
    setHyperparamsByAlgo((prev) => ({
      ...prev,
      [selectedAlgo.id]: { ...selectedAlgo.defaultHyperparams },
    }));
  };

  const handleProceedToRun = () => {
    // Navigate to /experiments/run passing algorithm and hyperparams state
    const paramsPayload = encodeURIComponent(JSON.stringify(currentParams));
    navigate(
      `/experiments/run?datasetId=${datasetId}&modelType=${selectedAlgo.id}&hyperparams=${paramsPayload}`
    );
  };

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header flex justify-between items-start flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Link to={`/preprocessing?datasetId=${datasetId}`} className="text-muted hover:text-primary flex items-center gap-1 text-sm font-medium">
              <ArrowLeft size={15} /> Back to Recipe
            </Link>
          </div>
          <h1 className="page-title">Model Selection & Hyperparameters</h1>
          <p className="page-subtitle">
            Choose an ML algorithm and configure dynamic hyperparameters tailored to your dataset before triggering training.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button 
            className="btn btn-secondary flex items-center gap-1.5"
            onClick={resetParams}
          >
            <RotateCcw size={15} />
            <span>Reset Hyperparams</span>
          </button>
        </div>
      </div>

      {/* Algorithm Selection Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {supportedAlgorithms.map((algo) => {
          const isSelected = algo.id === selectedAlgoId;
          return (
            <div
              key={algo.id}
              className={`algo-card ${isSelected ? "selected" : ""}`}
              onClick={() => setSelectedAlgoId(algo.id)}
            >
              <div className="flex justify-between items-start mb-2">
                <span className="algo-cat-tag">{algo.category}</span>
                {isSelected && <span className="algo-check-badge">Selected</span>}
              </div>
              <h3 className="algo-title">{algo.name}</h3>
              <p className="algo-desc">{algo.description}</p>
            </div>
          );
        })}
      </div>

      {/* Dynamic Hyperparameter Form */}
      <div className="studio-card mb-6">
        <div className="flex justify-between items-center mb-4 flex-wrap gap-2">
          <div className="flex items-center gap-2.5">
            <Settings2 className="text-primary" size={20} />
            <div>
              <h2 className="card-heading m-0">
                {selectedAlgo.name} — Hyperparameter Tuning
              </h2>
              <span className="text-xs text-muted">
                Fields adapt dynamically to {selectedAlgo.id} parameter specifications.
              </span>
            </div>
          </div>
          <span className="status-tag">Dynamic Form</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
          {selectedAlgo.paramSchema.map((field) => {
            const val = currentParams[field.name];

            return (
              <div key={field.name} className="hyperparam-field-box">
                <div className="flex justify-between items-center mb-1.5">
                  <label className="form-label mb-0">{field.label}</label>
                  <code className="text-xs text-primary font-bold">{field.name}</code>
                </div>

                {field.type === "number" ? (
                  <div className="flex items-center gap-3">
                    <input
                      type="number"
                      min={field.min}
                      max={field.max}
                      step={field.step || 1}
                      value={val !== undefined ? val : ""}
                      onChange={(e) => handleParamChange(field.name, parseFloat(e.target.value) || 0)}
                      className="form-input flex-1"
                    />
                    <span className="text-xs text-muted w-24">
                      [{field.min} – {field.max}]
                    </span>
                  </div>
                ) : field.type === "select" ? (
                  <select
                    value={val || field.options[0]}
                    onChange={(e) => handleParamChange(field.name, e.target.value)}
                    className="form-select"
                  >
                    {field.options.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt}
                      </option>
                    ))}
                  </select>
                ) : null}
              </div>
            );
          })}
        </div>

        {/* Action footer */}
        <div className="mt-8 pt-4 border-t border-slate-100 flex justify-between items-center flex-wrap gap-4">
          <div className="flex items-center gap-2 text-xs text-muted">
            <Info size={14} className="text-primary" />
            <span>Parameters will be passed to <code>ml_engine/training.py</code> via <code>/experiment/run</code>.</span>
          </div>

          <button
            className="btn btn-primary flex items-center gap-2 px-6 py-2.5"
            onClick={handleProceedToRun}
          >
            <Play size={16} fill="white" />
            <span>Launch Training Run</span>
            <ArrowRight size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
