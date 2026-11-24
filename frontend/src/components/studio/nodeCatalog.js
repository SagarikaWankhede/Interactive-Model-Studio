/**
 * Catalog of node types for the visual pipeline editor.
 * `sklearn` is shown on each node so students can connect the box to the code.
 */

export const NODE_TYPES = {
  dataset: {
    label: "CSV Dataset",
    group: "Data",
    color: "#4f46e5",
    sklearn: "pd.read_csv()",
    hint: "Where the data comes from, and which column to predict.",
    defaults: { dataset_id: "", filename: "", target_column: "", task_type: "auto" },
  },
  impute: {
    label: "Missing Value Imputer",
    group: "Preprocess",
    color: "#0284c7",
    sklearn: "SimpleImputer",
    hint: "Fills empty cells so the model can use every row.",
    defaults: { strategy: "median", columns: [] },
  },
  scale: {
    label: "Feature Scaling",
    group: "Preprocess",
    color: "#7c3aed",
    sklearn: "StandardScaler",
    hint: "Puts numeric features on a comparable scale.",
    defaults: { method: "standard", columns: [] },
  },
  encode: {
    label: "Categorical Encoder",
    group: "Preprocess",
    color: "#db2777",
    sklearn: "OneHotEncoder",
    hint: "Turns text categories into numbers.",
    defaults: { method: "onehot", columns: [] },
  },
  drop_columns: {
    label: "Drop Columns",
    group: "Preprocess",
    color: "#dc2626",
    sklearn: "df.drop()",
    hint: "Removes IDs, leaky or useless columns.",
    defaults: { columns: [] },
  },
  split: {
    label: "Train / Test Split",
    group: "Training",
    color: "#d97706",
    sklearn: "train_test_split",
    hint: "Holds out data to test the model honestly. It always runs first; every later step learns from training rows only.",
    defaults: { test_size: 0.2, random_state: 42, stratify: false },
  },
  model: {
    label: "Model",
    group: "Training",
    color: "#059669",
    sklearn: "RandomForestClassifier",
    hint: "The algorithm that learns from the prepared data.",
    defaults: { algorithm: "RandomForestClassifier", hyperparameters: { n_estimators: 100, random_state: 42 } },
  },
  evaluate: {
    label: "Evaluation",
    group: "Output",
    color: "#0f766e",
    sklearn: "accuracy_score, f1_score…",
    hint: "Scores the trained model on the held-out test rows.",
    defaults: {},
  },
};

export const PALETTE_GROUPS = ["Data", "Preprocess", "Training", "Output"];

export const SCALER_CLASS = { standard: "StandardScaler", minmax: "MinMaxScaler", robust: "RobustScaler" };
export const ENCODER_CLASS = { onehot: "OneHotEncoder", ordinal: "OrdinalEncoder" };

const num = (name, def, extra = {}) => ({ name, type: "number", default: def, ...extra });
const treeParams = [num("max_depth", "", { placeholder: "None (unlimited)", min: 1 }), num("random_state", 42)];

export const ALGORITHMS = [
  { id: "LogisticRegression", task: "classification", label: "Logistic Regression",
    params: [num("C", 1.0, { step: 0.1, min: 0.0001 }), num("max_iter", 1000, { min: 50 })] },
  { id: "DecisionTreeClassifier", task: "classification", label: "Decision Tree", params: [num("max_depth", 5, { min: 1 }), num("random_state", 42)] },
  { id: "RandomForestClassifier", task: "classification", label: "Random Forest", params: [num("n_estimators", 100, { min: 1 }), ...treeParams] },
  { id: "GradientBoostingClassifier", task: "classification", label: "Gradient Boosting",
    params: [num("n_estimators", 100, { min: 1 }), num("learning_rate", 0.1, { step: 0.01, min: 0.001 }), num("max_depth", 3, { min: 1 }), num("random_state", 42)] },
  { id: "XGBClassifier", task: "classification", label: "XGBoost", note: "Needs numeric class labels (0, 1, …). Not exportable to the browser yet.",
    params: [num("n_estimators", 100, { min: 1 }), num("learning_rate", 0.1, { step: 0.01 }), num("max_depth", 6, { min: 1 }), num("random_state", 42)] },
  { id: "LinearRegression", task: "regression", label: "Linear Regression", params: [] },
  { id: "DecisionTreeRegressor", task: "regression", label: "Decision Tree", params: [num("max_depth", 5, { min: 1 }), num("random_state", 42)] },
  { id: "RandomForestRegressor", task: "regression", label: "Random Forest", params: [num("n_estimators", 100, { min: 1 }), ...treeParams] },
  { id: "GradientBoostingRegressor", task: "regression", label: "Gradient Boosting",
    params: [num("n_estimators", 100, { min: 1 }), num("learning_rate", 0.1, { step: 0.01 }), num("max_depth", 3, { min: 1 }), num("random_state", 42)] },
  { id: "XGBRegressor", task: "regression", label: "XGBoost", note: "Not exportable to the browser yet.",
    params: [num("n_estimators", 100, { min: 1 }), num("learning_rate", 0.1, { step: 0.01 }), num("max_depth", 6, { min: 1 }), num("random_state", 42)] },
];

export const algorithmById = (id) => ALGORITHMS.find((a) => a.id === id);

export function defaultHyperparams(algorithmId) {
  const algo = algorithmById(algorithmId);
  return Object.fromEntries((algo?.params || []).map((p) => [p.name, p.default]));
}

/** One-line summary shown under the node title. */
export function describeNode(kind, params = {}) {
  const cols = params.columns?.length ? `${params.columns.length} col${params.columns.length === 1 ? "" : "s"}` : "no columns";
  switch (kind) {
    case "dataset":
      return params.filename ? `${params.filename} → ${params.target_column || "?"}` : "choose a dataset";
    case "impute":
      return `${params.strategy || "mean"} · ${cols}`;
    case "scale":
      return `${params.method || "standard"} · ${cols}`;
    case "encode":
      return `${params.method || "onehot"} · ${cols}`;
    case "drop_columns":
      return cols;
    case "split": {
      const t = Number(params.test_size ?? 0.2);
      return `${Math.round((1 - t) * 100)}% train / ${Math.round(t * 100)}% test${params.stratify ? " · stratified" : ""}`;
    }
    case "model":
      return algorithmById(params.algorithm)?.label || params.algorithm || "choose an algorithm";
    case "evaluate":
      return "metrics on test set";
    default:
      return "";
  }
}

/** The scikit-learn class this node becomes in the generated code. */
export function sklearnName(kind, params = {}) {
  if (kind === "scale") return SCALER_CLASS[params.method] || "StandardScaler";
  if (kind === "encode") return ENCODER_CLASS[params.method] || "OneHotEncoder";
  if (kind === "model") return params.algorithm || "Model";
  return NODE_TYPES[kind]?.sklearn || "";
}

let counter = 0;
export function newNodeId(kind) {
  counter += 1;
  return `${kind}_${Date.now().toString(36)}_${counter}`;
}

export function makeNode(kind, position, params) {
  return {
    id: newNodeId(kind),
    type: "mlNode",
    position,
    data: { kind, params: structuredClone(params ?? NODE_TYPES[kind].defaults) },
  };
}

/**
 * Build a sensible starter chain from a dataset's schema:
 * CSV → Split → Drop IDs → Impute numeric → Impute categorical → Encode → Scale → Model → Evaluate
 */
export function starterPipeline(dataset, target) {
  const schema = dataset?.schema || [];
  const features = schema.filter((c) => c.name !== target);
  const idCols = features.filter((c) => /(^id$|_id$|^id_|uuid)/i.test(c.name) || (c.type === "categorical" && c.cardinality === dataset.row_count)).map((c) => c.name);
  const usable = features.filter((c) => !idCols.includes(c.name));
  const numeric = usable.filter((c) => c.type === "numerical").map((c) => c.name);
  const categorical = usable.filter((c) => c.type === "categorical").map((c) => c.name);
  const targetInfo = schema.find((c) => c.name === target);
  const isClassification = !targetInfo || targetInfo.type === "categorical" || (targetInfo.cardinality ?? 99) <= 10;

  const steps = [["dataset", { dataset_id: dataset.id, filename: dataset.filename || dataset.name, target_column: target, task_type: "auto" }]];
  // The engine splits first, so the graph shows it first too.
  steps.push(["split", { test_size: 0.2, random_state: 42, stratify: isClassification }]);
  if (idCols.length) steps.push(["drop_columns", { columns: idCols }]);
  if (numeric.some((c) => schema.find((s) => s.name === c)?.null_count > 0)) steps.push(["impute", { strategy: "median", columns: numeric }]);
  if (categorical.length) {
    if (categorical.some((c) => schema.find((s) => s.name === c)?.null_count > 0)) steps.push(["impute", { strategy: "most_frequent", columns: categorical }]);
    steps.push(["encode", { method: "onehot", columns: categorical }]);
  }
  if (numeric.length) steps.push(["scale", { method: "standard", columns: numeric }]);
  const algorithm = isClassification ? "RandomForestClassifier" : "RandomForestRegressor";
  steps.push(["model", { algorithm, hyperparameters: defaultHyperparams(algorithm) }]);
  steps.push(["evaluate", {}]);

  const nodes = steps.map(([kind, params], i) => makeNode(kind, { x: 40 + (i % 2) * 30, y: 20 + i * 105 }, params));
  const edges = nodes.slice(1).map((n, i) => makeEdge(nodes[i].id, n.id));
  return { nodes, edges };
}

export function makeEdge(source, target) {
  return { id: `e_${source}_${target}`, source, target, type: "smoothstep", animated: false };
}
