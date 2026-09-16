import { useState, useEffect } from "react";
import { useSearchParams, useNavigate, Link } from "react-router-dom";
import { 
  Sliders, 
  Code2, 
  Copy, 
  Check, 
  ArrowRight, 
  ArrowLeft, 
  Wand2, 
  RotateCcw,
  Sparkles,
  HelpCircle,
  Trash2
} from "lucide-react";
import api from "../api/client";
import { getMockDatasetById, mockDatasets } from "../api/mockData";

export default function Preprocessing() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const datasetId = searchParams.get("datasetId") || mockDatasets[0].id;
  const initialTarget = searchParams.get("target");

  const [dataset, setDataset] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [testSize, setTestSize] = useState(0.2);
  const [randomState, setRandomState] = useState(42);
  const [copied, setCopied] = useState(false);

  // Column configurations: { [colName]: { action: 'keep'|'drop', imputer: string, transform: string } }
  const [columnConfigs, setColumnConfigs] = useState({});

  useEffect(() => {
    // Fetch dataset or load mock
    api.getDatasets()
      .then((data) => {
        const found = Array.isArray(data) ? data.find((d) => d.id === datasetId) : null;
        setupDataset(found || getMockDatasetById(datasetId));
      })
      .catch(() => {
        setupDataset(getMockDatasetById(datasetId));
      });
  }, [datasetId]);

  const setupDataset = (data) => {
    setDataset(data);
    const target = initialTarget || data.target_column || data.schema[data.schema.length - 1]?.name;
    setTargetColumn(target);

    // Initialize default configs based on inferred column types
    const initial = {};
    (data.schema || []).forEach((col) => {
      if (col.name === target) return;

      const isNumeric = col.type === "numerical";
      const isIdOrProxy = col.name.toLowerCase().includes("id") || col.name.toLowerCase().includes("proxy");

      initial[col.name] = {
        action: isIdOrProxy ? "drop" : "keep",
        imputer: isNumeric ? (col.null_pct > 0 ? "median" : "none") : (col.null_pct > 0 ? "most_frequent" : "none"),
        transform: isNumeric ? "StandardScaler" : "OneHotEncoder",
      };
    });
    setColumnConfigs(initial);
  };

  const updateColConfig = (colName, key, value) => {
    setColumnConfigs((prev) => ({
      ...prev,
      [colName]: {
        ...prev[colName],
        [key]: value,
      },
    }));
  };

  const applyPreset = (presetName) => {
    if (!dataset) return;
    const updated = {};
    dataset.schema.forEach((col) => {
      if (col.name === targetColumn) return;
      const isNumeric = col.type === "numerical";
      const isIdOrProxy = col.name.toLowerCase().includes("id") || col.name.toLowerCase().includes("proxy");

      if (presetName === "recommended") {
        updated[col.name] = {
          action: isIdOrProxy ? "drop" : "keep",
          imputer: isNumeric ? "median" : "most_frequent",
          transform: isNumeric ? "StandardScaler" : (col.cardinality > 20 ? "OrdinalEncoder" : "OneHotEncoder"),
        };
      } else if (presetName === "minimal") {
        updated[col.name] = {
          action: "keep",
          imputer: isNumeric ? "mean" : "most_frequent",
          transform: isNumeric ? "none" : "OneHotEncoder",
        };
      } else {
        updated[col.name] = {
          action: "keep",
          imputer: "none",
          transform: "none",
        };
      }
    });
    setColumnConfigs(updated);
  };

  // Generate recipe JSON matching Person B's pipeline step schema
  const generateRecipeJson = () => {
    const dropCols = [];
    const imputeGroups = {};
    const scaleGroups = {};
    const encodeGroups = {};

    Object.entries(columnConfigs).forEach(([colName, cfg]) => {
      if (cfg.action === "drop") {
        dropCols.push(colName);
        return;
      }

      if (cfg.imputer && cfg.imputer !== "none") {
        if (!imputeGroups[cfg.imputer]) imputeGroups[cfg.imputer] = [];
        imputeGroups[cfg.imputer].push(colName);
      }

      if (cfg.transform && cfg.transform !== "none") {
        if (cfg.transform.includes("Scaler")) {
          if (!scaleGroups[cfg.transform]) scaleGroups[cfg.transform] = [];
          scaleGroups[cfg.transform].push(colName);
        } else if (cfg.transform.includes("Encoder")) {
          if (!encodeGroups[cfg.transform]) encodeGroups[cfg.transform] = [];
          encodeGroups[cfg.transform].push(colName);
        }
      }
    });

    const steps = [];

    // Drop step
    if (dropCols.length > 0) {
      steps.push({
        step: "drop_columns",
        params: { columns: dropCols },
      });
    }

    // Impute steps
    Object.entries(imputeGroups).forEach(([strategy, cols]) => {
      steps.push({
        step: "impute",
        params: { strategy, columns: cols },
      });
    });

    // Scale steps
    Object.entries(scaleGroups).forEach(([method, cols]) => {
      steps.push({
        step: "scale",
        params: { method, columns: cols },
      });
    });

    // Encode steps
    Object.entries(encodeGroups).forEach(([method, cols]) => {
      steps.push({
        step: "encode",
        params: { method, columns: cols },
      });
    });

    return {
      dataset_id: dataset?.id || datasetId,
      target_column: targetColumn,
      split: {
        test_size: Number(testSize),
        random_state: Number(randomState),
      },
      steps,
    };
  };

  const recipe = generateRecipeJson();
  const recipeJsonString = JSON.stringify(recipe, null, 2);

  const copyRecipe = () => {
    navigator.clipboard.writeText(recipeJsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!dataset) {
    return (
      <div className="page-container flex justify-center items-center py-20">
        <div className="text-center text-muted">Loading dataset preprocessing setup...</div>
      </div>
    );
  }

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header flex justify-between items-start flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Link to={`/dataset/${dataset.id}`} className="text-muted hover:text-primary flex items-center gap-1 text-sm font-medium">
              <ArrowLeft size={15} /> Back to Schema
            </Link>
          </div>
          <h1 className="page-title">Preprocessing Recipe Builder</h1>
          <p className="page-subtitle">
            Configure reproducible data cleaning, imputations, feature scalers, and train/test splits. Outputs standard schema for the ML engine.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button 
            className="btn btn-secondary flex items-center gap-1.5"
            onClick={() => applyPreset("recommended")}
          >
            <Wand2 size={15} className="text-primary" />
            <span>Auto-Clean Preset</span>
          </button>
          <button 
            className="btn btn-secondary flex items-center gap-1.5"
            onClick={() => applyPreset("reset")}
          >
            <RotateCcw size={15} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Train / Test & Target Settings Bar */}
      <div className="studio-card mb-6 bg-slate-50">
        <h3 className="text-sm font-bold uppercase tracking-wider text-muted mb-4">
          Global Split & Target Setup
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="form-label">Target Column (y)</label>
            <select 
              value={targetColumn} 
              onChange={(e) => setTargetColumn(e.target.value)}
              className="form-select"
            >
              {dataset.schema.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name} ({col.type})
                </option>
              ))}
            </select>
          </div>

          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="form-label">Test Split Ratio</label>
              <span className="text-xs font-bold text-primary">{(testSize * 100).toFixed(0)}% Test / {((1 - testSize) * 100).toFixed(0)}% Train</span>
            </div>
            <input 
              type="range" 
              min="0.1" 
              max="0.4" 
              step="0.05"
              value={testSize} 
              onChange={(e) => setTestSize(parseFloat(e.target.value))}
              className="w-full accent-indigo-600"
            />
          </div>

          <div>
            <label className="form-label">Random Seed</label>
            <input 
              type="number" 
              value={randomState} 
              onChange={(e) => setRandomState(parseInt(e.target.value) || 42)}
              className="form-input"
            />
          </div>
        </div>
      </div>

      {/* Main Grid: Transformations Table + Real-Time JSON Preview */}
      <div className="recipe-grid">
        {/* Transformations Table */}
        <div className="studio-card">
          <h2 className="card-heading mb-3">Feature Transformations</h2>
          <div className="table-responsive">
            <table className="studio-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Type</th>
                  <th>Action</th>
                  <th>Missing Imputer</th>
                  <th>Scaling / Encoding</th>
                </tr>
              </thead>
              <tbody>
                {dataset.schema
                  .filter((col) => col.name !== targetColumn)
                  .map((col) => {
                    const cfg = columnConfigs[col.name] || { action: "keep", imputer: "none", transform: "none" };
                    const isNumeric = col.type === "numerical";
                    const isDropped = cfg.action === "drop";

                    return (
                      <tr key={col.name} className={isDropped ? "opacity-50" : ""}>
                        <td>
                          <strong>{col.name}</strong>
                          {col.null_pct > 0 && (
                            <span className="text-xs block text-muted">
                              {col.null_pct.toFixed(0)}% nulls
                            </span>
                          )}
                        </td>
                        <td>
                          <span className={`type-tag type-${col.type}`}>{col.type}</span>
                        </td>
                        <td>
                          <button
                            className={`btn-action-drop ${isDropped ? "dropped" : ""}`}
                            onClick={() => updateColConfig(col.name, "action", isDropped ? "keep" : "drop")}
                            title={isDropped ? "Restore feature" : "Drop feature"}
                          >
                            {isDropped ? "Dropped" : "Keep"}
                          </button>
                        </td>
                        <td>
                          {isDropped ? (
                            <span className="text-muted text-xs">—</span>
                          ) : isNumeric ? (
                            <select
                              value={cfg.imputer}
                              onChange={(e) => updateColConfig(col.name, "imputer", e.target.value)}
                              className="form-select-sm"
                            >
                              <option value="none">None (No Nulls)</option>
                              <option value="median">Median Impute</option>
                              <option value="mean">Mean Impute</option>
                              <option value="constant">Constant (0)</option>
                            </select>
                          ) : (
                            <select
                              value={cfg.imputer}
                              onChange={(e) => updateColConfig(col.name, "imputer", e.target.value)}
                              className="form-select-sm"
                            >
                              <option value="none">None (No Nulls)</option>
                              <option value="most_frequent">Most Frequent (Mode)</option>
                              <option value="constant">Constant ("Missing")</option>
                            </select>
                          )}
                        </td>
                        <td>
                          {isDropped ? (
                            <span className="text-muted text-xs">—</span>
                          ) : isNumeric ? (
                            <select
                              value={cfg.transform}
                              onChange={(e) => updateColConfig(col.name, "transform", e.target.value)}
                              className="form-select-sm"
                            >
                              <option value="none">None (Raw)</option>
                              <option value="StandardScaler">StandardScaler</option>
                              <option value="MinMaxScaler">MinMaxScaler</option>
                              <option value="RobustScaler">RobustScaler</option>
                            </select>
                          ) : (
                            <select
                              value={cfg.transform}
                              onChange={(e) => updateColConfig(col.name, "transform", e.target.value)}
                              className="form-select-sm"
                            >
                              <option value="none">None (Raw)</option>
                              <option value="OneHotEncoder">OneHotEncoder</option>
                              <option value="OrdinalEncoder">OrdinalEncoder</option>
                            </select>
                          )}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Real-Time Recipe JSON Panel */}
        <div className="studio-card recipe-preview-card">
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center gap-2">
              <Code2 size={18} className="text-primary" />
              <h3 className="font-bold text-sm">Pipeline Recipe JSON</h3>
            </div>
            <button 
              className="btn btn-secondary btn-sm flex items-center gap-1"
              onClick={copyRecipe}
            >
              {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
              <span>{copied ? "Copied!" : "Copy JSON"}</span>
            </button>
          </div>
          <p className="text-xs text-muted mb-3">
            This standardized specification is passed to <code>recipe_runner.py</code> to transform data deterministically.
          </p>

          <pre className="json-box">
            <code>{recipeJsonString}</code>
          </pre>

          <div className="mt-4 flex justify-between items-center">
            <span className="text-xs text-muted">
              {recipe.steps.length} transformation steps configured
            </span>
            <Link 
              to="/pipeline"
              className="btn btn-primary btn-sm flex items-center gap-1.5"
            >
              <span>Save & Proceed</span>
              <ArrowRight size={14} />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
