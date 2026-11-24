import { useState, useEffect } from "react";
import { useSearchParams, Link } from "react-router-dom";
import {
  Sliders,
  Cpu,
  Code2,
  Copy,
  Check,
  ArrowRight,
  Wand2,
  RotateCcw,
  Plus,
  Trash2,
  Download,
  Play,
  Database,
  ChevronDown,
  Workflow,
} from "lucide-react";
import { getMockDatasetById, mockDatasets, supportedAlgorithms } from "../api/mockData";
import { generatePythonCode, summarizeRecipe } from "../services/codegen";
import PipelineCanvas from "../components/PipelineCanvas";

const STEP_TYPES = [
  { value: "impute", label: "Impute (Missing Values)" },
  { value: "scale", label: "Scale (Numerical)" },
  { value: "encode", label: "Encode (Categorical)" },
  { value: "drop_columns", label: "Drop Columns" },
];

const IMPUTE_STRATEGIES = ["mean", "median", "most_frequent", "constant"];
const SCALE_METHODS = ["standard", "minmax", "robust"];
const ENCODE_METHODS = ["onehot", "ordinal"];

function StepRow({ step, index, columns, onUpdate, onRemove }) {
  const featureCols = columns.filter((c) => c !== step._target);

  const updateParam = (key, value) => {
    onUpdate(index, { ...step, params: { ...step.params, [key]: value } });
  };

  const toggleColumn = (col) => {
    const current = step.columns || [];
    const next = current.includes(col)
      ? current.filter((c) => c !== col)
      : [...current, col];
    onUpdate(index, { ...step, columns: next });
  };

  return (
    <div className="step-row-card">
      <div className="step-row-header">
        <span className="step-index-badge">Step {index + 1}</span>

        <select
          value={step.step}
          onChange={(e) => onUpdate(index, { ...step, step: e.target.value, columns: [], params: {} })}
          className="form-select-sm"
        >
          {STEP_TYPES.map((t) => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>

        {step.step === "impute" && (
          <select
            value={step.params?.strategy || "mean"}
            onChange={(e) => updateParam("strategy", e.target.value)}
            className="form-select-sm"
          >
            {IMPUTE_STRATEGIES.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        )}

        {step.step === "scale" && (
          <select
            value={step.params?.method || "standard"}
            onChange={(e) => updateParam("method", e.target.value)}
            className="form-select-sm"
          >
            {SCALE_METHODS.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        )}

        {step.step === "encode" && (
          <select
            value={step.params?.method || "onehot"}
            onChange={(e) => updateParam("method", e.target.value)}
            className="form-select-sm"
          >
            {ENCODE_METHODS.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        )}

        <button
          className="btn-icon-danger ml-auto"
          onClick={() => onRemove(index)}
          title="Remove step"
        >
          <Trash2 size={15} />
        </button>
      </div>

      {step.step !== "split" && (
        <div className="step-columns-grid">
          {featureCols.map((col) => {
            const selected = (step.columns || []).includes(col);
            return (
              <button
                key={col}
                onClick={() => toggleColumn(col)}
                className={`col-toggle-chip ${selected ? "active" : ""}`}
              >
                {col}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function Pipeline() {
  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get("datasetId") || mockDatasets[0].id;

  const [dataset, setDataset] = useState(getMockDatasetById(datasetId));
  const [targetColumn, setTargetColumn] = useState("");
  const [testSize, setTestSize] = useState(0.2);
  const [randomState, setRandomState] = useState(42);
  const [selectedAlgoId, setSelectedAlgoId] = useState("RandomForest");
  const [hyperparams, setHyperparams] = useState({});
  const [steps, setSteps] = useState([]);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState("builder"); // 'builder' | 'dag' | 'code'

  // Initialize from dataset
  useEffect(() => {
    const ds = getMockDatasetById(datasetId);
    setDataset(ds);
    const target = ds.target_column || ds.schema?.[ds.schema.length - 1]?.name || "target";
    setTargetColumn(target);
    loadSmartDefaults(ds, target);
  }, [datasetId]);

  // Keep hyperparams in sync with selected algo
  useEffect(() => {
    const algo = supportedAlgorithms.find((a) => a.id === selectedAlgoId);
    if (algo) setHyperparams({ ...algo.defaultHyperparams });
  }, [selectedAlgoId]);

  const loadSmartDefaults = (ds, target) => {
    if (!ds?.schema) return;
    const numCols = ds.schema
      .filter((c) => c.name !== target && c.type === "numerical")
      .map((c) => c.name);
    const catCols = ds.schema
      .filter((c) => c.name !== target && c.type === "categorical")
      .map((c) => c.name);
    const idCols = ds.schema
      .filter((c) => /id|proxy/i.test(c.name))
      .map((c) => c.name);

    const defaultSteps = [];
    if (idCols.length > 0) {
      defaultSteps.push({ step: "drop_columns", columns: idCols, params: { columns: idCols } });
    }
    if (numCols.length > 0) {
      defaultSteps.push({ step: "impute", columns: numCols, params: { strategy: "median" } });
      defaultSteps.push({ step: "scale", columns: numCols, params: { method: "standard" } });
    }
    if (catCols.length > 0) {
      defaultSteps.push({ step: "impute", columns: catCols, params: { strategy: "most_frequent" } });
      defaultSteps.push({ step: "encode", columns: catCols, params: { method: "onehot" } });
    }
    setSteps(defaultSteps);
  };

  const addStep = () => {
    setSteps((prev) => [...prev, { step: "impute", columns: [], params: { strategy: "mean" } }]);
  };

  const updateStep = (index, updated) => {
    setSteps((prev) => prev.map((s, i) => (i === index ? updated : s)));
  };

  const removeStep = (index) => {
    setSteps((prev) => prev.filter((_, i) => i !== index));
  };

  const reset = () => loadSmartDefaults(dataset, targetColumn);

  const selectedAlgo = supportedAlgorithms.find((a) => a.id === selectedAlgoId) || supportedAlgorithms[0];

  // Build the full recipe for codegen
  const fullRecipe = [
    ...steps,
    {
      step: "split",
      params: { test_size: testSize, random_state: randomState },
    },
  ];

  const generatedCode = generatePythonCode({
    datasetName: dataset?.name || "dataset.csv",
    targetColumn,
    testSize,
    randomState,
    recipe: fullRecipe,
    modelType: selectedAlgoId,
    hyperparams,
  });

  const copyCode = () => {
    navigator.clipboard.writeText(generatedCode).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const downloadCode = () => {
    const blob = new Blob([generatedCode], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pipeline_${selectedAlgoId.toLowerCase()}.py`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const columns = dataset?.schema?.map((c) => c.name) || [];

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header flex justify-between items-start flex-wrap gap-4">
        <div>
          <h1 className="page-title">Pipeline Studio &amp; Code Generator</h1>
          <p className="page-subtitle">
            Build a reproducible preprocessing recipe, pick your model, and instantly
            export a clean, executable Python script — no manual coding needed.
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <button className="btn btn-secondary flex items-center gap-1.5" onClick={reset}>
            <RotateCcw size={15} />
            <span>Smart Reset</span>
          </button>
          <button
            className="btn btn-primary flex items-center gap-1.5"
            onClick={() => { setActiveTab("code"); }}
          >
            <Code2 size={15} />
            <span>Generate Code</span>
          </button>
        </div>
      </div>

      {/* Dataset + Target Bar */}
      <div className="studio-card mb-6 bg-slate-50">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
          <div>
            <label className="form-label">Dataset</label>
            <div className="flex items-center gap-2 p-2.5 bg-white border border-slate-200 rounded-lg text-sm">
              <Database size={15} className="text-primary shrink-0" />
              <span className="font-medium truncate">{dataset?.name || "No dataset"}</span>
            </div>
          </div>
          <div>
            <label className="form-label">Target Column (y)</label>
            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="form-select"
            >
              {columns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          <div>
            <div className="flex justify-between mb-1">
              <label className="form-label">Test Split Ratio</label>
              <span className="text-xs font-bold text-primary">
                {((1 - testSize) * 100).toFixed(0)}% Train / {(testSize * 100).toFixed(0)}% Test
              </span>
            </div>
            <input
              type="range" min="0.1" max="0.4" step="0.05"
              value={testSize}
              onChange={(e) => setTestSize(parseFloat(e.target.value))}
              className="w-full accent-indigo-600"
            />
          </div>
          <div>
            <label className="form-label">Random Seed</label>
            <input
              type="number" value={randomState}
              onChange={(e) => setRandomState(parseInt(e.target.value) || 42)}
              className="form-input"
            />
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-2 mb-6 border-b border-slate-200 pb-2">
        {[
          { key: "builder", icon: <Sliders size={15} />, label: "Recipe Builder" },
          { key: "dag", icon: <Workflow size={15} />, label: "DAG Canvas" },
          { key: "code", icon: <Code2 size={15} />, label: "Python Codegen" },
        ].map((tab) => (
          <button
            key={tab.key}
            className={`tab-btn ${activeTab === tab.key ? "active" : ""}`}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* TAB 1 — Recipe Builder */}
      {activeTab === "builder" && (
        <div className="recipe-grid">
          <div>
            {/* Step list */}
            <div className="studio-card mb-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="card-heading">Preprocessing Steps</h2>
                <button className="btn btn-secondary btn-sm flex items-center gap-1.5" onClick={addStep}>
                  <Plus size={14} />
                  <span>Add Step</span>
                </button>
              </div>

              {steps.length === 0 && (
                <div className="text-center text-muted py-8 text-sm">
                  No steps configured. Click "Add Step" or use Smart Reset.
                </div>
              )}

              <div className="space-y-3">
                {steps.map((step, i) => (
                  <StepRow
                    key={i}
                    step={step}
                    index={i}
                    columns={columns}
                    onUpdate={updateStep}
                    onRemove={removeStep}
                  />
                ))}
              </div>
            </div>

            {/* Model Selection */}
            <div className="studio-card">
              <h2 className="card-heading mb-3">Model Selection</h2>
              <div className="grid grid-cols-2 gap-3 mb-4">
                {supportedAlgorithms.map((algo) => (
                  <div
                    key={algo.id}
                    className={`algo-card ${selectedAlgoId === algo.id ? "selected" : ""}`}
                    onClick={() => setSelectedAlgoId(algo.id)}
                  >
                    <span className="algo-cat-tag">{algo.category}</span>
                    {selectedAlgoId === algo.id && (
                      <span className="algo-check-badge">Selected</span>
                    )}
                    <h3 className="algo-title">{algo.name}</h3>
                  </div>
                ))}
              </div>

              {/* Hyperparam quick-edit */}
              <div className="grid grid-cols-2 gap-3">
                {selectedAlgo.paramSchema.slice(0, 4).map((field) => (
                  <div key={field.name} className="hyperparam-field-box">
                    <div className="flex justify-between mb-1">
                      <label className="form-label mb-0 text-xs">{field.label}</label>
                      <code className="text-xs text-primary">{field.name}</code>
                    </div>
                    {field.type === "number" ? (
                      <input
                        type="number" min={field.min} max={field.max} step={field.step}
                        value={hyperparams[field.name] ?? field.min}
                        onChange={(e) =>
                          setHyperparams((prev) => ({
                            ...prev,
                            [field.name]: parseFloat(e.target.value) || 0,
                          }))
                        }
                        className="form-input text-sm"
                      />
                    ) : (
                      <select
                        value={hyperparams[field.name] ?? field.options[0]}
                        onChange={(e) =>
                          setHyperparams((prev) => ({ ...prev, [field.name]: e.target.value }))
                        }
                        className="form-select text-sm"
                      >
                        {field.options.map((opt) => (
                          <option key={opt} value={opt}>{opt}</option>
                        ))}
                      </select>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sidebar: Recipe JSON preview */}
          <div className="studio-card recipe-preview-card">
            <div className="flex justify-between items-center mb-3">
              <div className="flex items-center gap-2">
                <Code2 size={16} className="text-primary" />
                <h3 className="font-bold text-sm">Pipeline Recipe JSON</h3>
              </div>
              <div className="flex gap-2">
                <button
                  className="btn btn-secondary btn-sm flex items-center gap-1"
                  onClick={() => setActiveTab("code")}
                >
                  <Play size={13} />
                  <span>Codegen</span>
                </button>
              </div>
            </div>
            <p className="text-xs text-muted mb-3">
              This JSON is passed to <code>recipe_runner.py</code> for deterministic execution.
            </p>
            <pre className="json-box">
              <code>{JSON.stringify({ target_column: targetColumn, recipe: fullRecipe, model_config: { algorithm: selectedAlgoId, hyperparameters: hyperparams } }, null, 2)}</code>
            </pre>
            <div className="mt-4 flex justify-between items-center">
              <span className="text-xs text-muted">{steps.length} transform steps</span>
              <Link
                to={`/models?datasetId=${dataset?.id || datasetId}`}
                className="btn btn-primary btn-sm flex items-center gap-1.5"
              >
                <span>Proceed to Run</span>
                <ArrowRight size={14} />
              </Link>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2 — DAG Canvas */}
      {activeTab === "dag" && (
        <div className="studio-card">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h2 className="card-heading">Pipeline DAG Visualization</h2>
              <p className="text-xs text-muted">
                Each node represents a transformation stage. Data flows left → right.
              </p>
            </div>
            <span className="status-tag">{steps.length + 3} Nodes</span>
          </div>

          <PipelineCanvas
            recipe={fullRecipe}
            modelType={selectedAlgoId}
            datasetName={dataset?.name}
          />

          {/* Step summary list */}
          <div className="mt-6">
            <h3 className="text-xs font-bold uppercase tracking-wider text-muted mb-3">Step Summary</h3>
            <ol className="space-y-2">
              {summarizeRecipe(fullRecipe).map((desc, i) => (
                <li key={i} className="flex items-center gap-2 text-sm">
                  <span className="step-index-badge shrink-0">{i + 1}</span>
                  <span className="text-slate-700">{desc}</span>
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}

      {/* TAB 3 — Python Codegen */}
      {activeTab === "code" && (
        <div className="studio-card">
          <div className="flex justify-between items-center mb-4 flex-wrap gap-3">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Code2 size={20} className="text-primary" />
                <h2 className="card-heading m-0">Generated Python Pipeline</h2>
              </div>
              <p className="text-xs text-muted">
                Standalone, executable scikit-learn script — no Studio dependency needed.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                className="btn btn-secondary flex items-center gap-1.5"
                onClick={copyCode}
              >
                {copied ? <Check size={15} className="text-success" /> : <Copy size={15} />}
                <span>{copied ? "Copied!" : "Copy Code"}</span>
              </button>
              <button
                className="btn btn-primary flex items-center gap-1.5"
                onClick={downloadCode}
              >
                <Download size={15} />
                <span>Download .py</span>
              </button>
            </div>
          </div>

          {/* Generated code preview */}
          <div className="codegen-output-box">
            <div className="codegen-topbar">
              <div className="flex items-center gap-2">
                <div className="codegen-dot dot-red" />
                <div className="codegen-dot dot-yellow" />
                <div className="codegen-dot dot-green" />
                <span className="codegen-filename">
                  pipeline_{selectedAlgoId.toLowerCase()}.py
                </span>
              </div>
              <span className="text-xs text-slate-400">Python 3.8+</span>
            </div>
            <pre className="codegen-pre">
              <code>{generatedCode}</code>
            </pre>
          </div>

          <div className="mt-4 p-3 bg-indigo-50 border border-indigo-100 rounded-lg flex items-start gap-3">
            <div className="text-indigo-500 mt-0.5">💡</div>
            <p className="text-xs text-indigo-700">
              <strong>How to run:</strong> Save the file and execute{" "}
              <code>python pipeline_{selectedAlgoId.toLowerCase()}.py</code>. Make sure
              your CSV is in the same directory, or update the path at Line 1.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
