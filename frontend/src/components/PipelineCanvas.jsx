import { useMemo } from "react";
import { Database, Filter, Shuffle, Cpu, ArrowRight, CheckCircle2 } from "lucide-react";

/**
 * PipelineCanvas — Visual DAG-style pipeline step renderer.
 * Displays each stage of the ML pipeline as a connected node graph.
 */

const NODE_ICONS = {
  dataset: Database,
  impute: Filter,
  scale: Filter,
  encode: Filter,
  drop_columns: Filter,
  split: Shuffle,
  model: Cpu,
  output: CheckCircle2,
};

const NODE_COLORS = {
  dataset: { bg: "bg-indigo-50", border: "border-indigo-300", icon: "text-indigo-600", label: "bg-indigo-100 text-indigo-700" },
  impute: { bg: "bg-blue-50", border: "border-blue-300", icon: "text-blue-600", label: "bg-blue-100 text-blue-700" },
  scale: { bg: "bg-purple-50", border: "border-purple-300", icon: "text-purple-600", label: "bg-purple-100 text-purple-700" },
  encode: { bg: "bg-pink-50", border: "border-pink-300", icon: "text-pink-600", label: "bg-pink-100 text-pink-700" },
  drop_columns: { bg: "bg-red-50", border: "border-red-200", icon: "text-red-500", label: "bg-red-100 text-red-600" },
  split: { bg: "bg-amber-50", border: "border-amber-300", icon: "text-amber-600", label: "bg-amber-100 text-amber-700" },
  model: { bg: "bg-emerald-50", border: "border-emerald-300", icon: "text-emerald-600", label: "bg-emerald-100 text-emerald-700" },
  output: { bg: "bg-teal-50", border: "border-teal-300", icon: "text-teal-600", label: "bg-teal-100 text-teal-700" },
};

function stepToNode(step, index) {
  const type = step.step || "impute";
  const cols = step.columns || step.params?.columns || [];
  const methodKey =
    step.params?.method || step.params?.strategy || step.params?.transform || "";

  let label = "";
  let detail = "";

  switch (type) {
    case "impute":
      label = "Imputer";
      detail = `${methodKey || "mean"} · ${cols.length} cols`;
      break;
    case "scale":
      label = "Scaler";
      detail = `${methodKey || "standard"} · ${cols.length} cols`;
      break;
    case "encode":
      label = "Encoder";
      detail = `${methodKey || "onehot"} · ${cols.length} cols`;
      break;
    case "drop_columns":
      label = "Drop";
      detail = `${(step.params?.columns || []).length} cols excluded`;
      break;
    case "split":
      label = "Split";
      detail = `${((1 - (step.params?.test_size || 0.2)) * 100).toFixed(0)}% / ${((step.params?.test_size || 0.2) * 100).toFixed(0)}%`;
      break;
    default:
      label = type;
      detail = "";
  }

  return { type, label, detail, id: `step-${index}` };
}

function PipelineNode({ node, isLast }) {
  const colors = NODE_COLORS[node.type] || NODE_COLORS.impute;
  const Icon = NODE_ICONS[node.type] || Filter;

  return (
    <div className="pipeline-dag-node-wrapper">
      <div className={`pipeline-dag-node ${colors.bg} ${colors.border}`}>
        <div className={`pipeline-node-icon-wrap ${colors.icon}`}>
          <Icon size={18} />
        </div>
        <div className="pipeline-node-text">
          <span className={`pipeline-node-label ${colors.label}`}>{node.label}</span>
          {node.detail && (
            <span className="pipeline-node-detail">{node.detail}</span>
          )}
        </div>
      </div>
      {!isLast && (
        <div className="pipeline-dag-arrow">
          <ArrowRight size={18} className="text-slate-400" />
        </div>
      )}
    </div>
  );
}

export default function PipelineCanvas({ recipe = [], modelType = "", datasetName = "" }) {
  const nodes = useMemo(() => {
    const all = [];

    // Source node
    all.push({
      type: "dataset",
      label: "Dataset",
      detail: datasetName || "source.csv",
      id: "node-dataset",
    });

    // Recipe step nodes
    recipe.forEach((step, i) => {
      all.push(stepToNode(step, i));
    });

    // Model node
    if (modelType) {
      all.push({
        type: "model",
        label: "Estimator",
        detail: modelType.replace("Classifier", "").replace("Regressor", ""),
        id: "node-model",
      });
    }

    // Output node
    all.push({ type: "output", label: "Predictions", detail: "metrics + export", id: "node-output" });

    return all;
  }, [recipe, modelType, datasetName]);

  if (nodes.length <= 2) {
    return (
      <div className="pipeline-canvas-empty">
        <p className="text-muted text-sm text-center py-6">
          Configure preprocessing steps above to see the pipeline DAG visualize here.
        </p>
      </div>
    );
  }

  return (
    <div className="pipeline-canvas">
      <div className="pipeline-dag-track">
        {nodes.map((node, i) => (
          <PipelineNode key={node.id} node={node} isLast={i === nodes.length - 1} />
        ))}
      </div>
    </div>
  );
}
