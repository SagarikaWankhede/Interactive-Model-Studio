/**
 * Python Codegen Service
 * Generates standalone, executable Python scripts from a visual pipeline configuration.
 * This is "Differentiator #3" – the flagship feature of Interactive Model Studio.
 */

/**
 * Map frontend step/transform names to scikit-learn import paths.
 */
const SKLEARN_IMPORTS = {
  StandardScaler: "from sklearn.preprocessing import StandardScaler",
  MinMaxScaler: "from sklearn.preprocessing import MinMaxScaler",
  RobustScaler: "from sklearn.preprocessing import RobustScaler",
  OneHotEncoder: "from sklearn.preprocessing import OneHotEncoder",
  OrdinalEncoder: "from sklearn.preprocessing import OrdinalEncoder",
  SimpleImputer: "from sklearn.impute import SimpleImputer",
  RandomForestClassifier:
    "from sklearn.ensemble import RandomForestClassifier",
  RandomForestRegressor: "from sklearn.ensemble import RandomForestRegressor",
  GradientBoostingClassifier:
    "from sklearn.ensemble import GradientBoostingClassifier",
  GradientBoostingRegressor:
    "from sklearn.ensemble import GradientBoostingRegressor",
  LogisticRegression: "from sklearn.linear_model import LogisticRegression",
  LinearRegression: "from sklearn.linear_model import LinearRegression",
  DecisionTreeClassifier: "from sklearn.tree import DecisionTreeClassifier",
  XGBClassifier: "from xgboost import XGBClassifier",
  XGBRegressor: "from xgboost import XGBRegressor",
};

const ALGO_MAP = {
  RandomForest: "RandomForestClassifier",
  XGBoost: "XGBClassifier",
  LogisticRegression: "LogisticRegression",
  DecisionTree: "DecisionTreeClassifier",
  RandomForestClassifier: "RandomForestClassifier",
  RandomForestRegressor: "RandomForestRegressor",
  XGBClassifier: "XGBClassifier",
  XGBRegressor: "XGBRegressor",
};

/**
 * Generate standalone Python code from a pipeline configuration.
 *
 * @param {Object} config  Pipeline config object:
 *   {
 *     datasetName: string,       // e.g. "customer_churn.csv"
 *     targetColumn: string,
 *     testSize: number,          // e.g. 0.2
 *     randomState: number,       // e.g. 42
 *     recipe: Array<{            // preprocessing steps
 *       step: string,
 *       columns: string[],
 *       params: Object
 *     }>,
 *     modelType: string,         // e.g. "RandomForest"
 *     hyperparams: Object,       // e.g. { n_estimators: 100 }
 *   }
 *
 * @returns {string} Executable Python script.
 */
export function generatePythonCode(config = {}) {
  const {
    datasetName = "dataset.csv",
    targetColumn = "target",
    testSize = 0.2,
    randomState = 42,
    recipe = [],
    modelType = "RandomForest",
    hyperparams = {},
  } = config;

  const algoClass = ALGO_MAP[modelType] || modelType;

  // ── Collect required imports ────────────────────────────────────────────────
  const importsSet = new Set([
    "import pandas as pd",
    "import numpy as np",
    "from sklearn.model_selection import train_test_split",
    "from sklearn.metrics import accuracy_score, f1_score, classification_report",
  ]);

  importsSet.add("from sklearn.impute import SimpleImputer");
  importsSet.add("from sklearn.pipeline import Pipeline");
  importsSet.add("from sklearn.compose import ColumnTransformer");

  if (SKLEARN_IMPORTS[algoClass]) {
    importsSet.add(SKLEARN_IMPORTS[algoClass]);
  }

  recipe.forEach((step) => {
    const transform = step.params?.method || step.params?.transform;
    if (transform && SKLEARN_IMPORTS[transform]) {
      importsSet.add(SKLEARN_IMPORTS[transform]);
    }
  });

  const imports = [...importsSet].sort().join("\n");

  // ── Build recipe code blocks ─────────────────────────────────────────────
  const dropCols = recipe
    .filter((s) => s.step === "drop_columns")
    .flatMap((s) => s.params?.columns || []);

  const imputeSteps = recipe.filter((s) => s.step === "impute");
  const scaleSteps = recipe.filter(
    (s) => s.step === "scale"
  );
  const encodeSteps = recipe.filter((s) => s.step === "encode");

  // Build column groups
  const numCols = [
    ...new Set([
      ...imputeSteps.flatMap((s) => s.params?.columns || s.columns || []),
      ...scaleSteps.flatMap((s) => s.params?.columns || s.columns || []),
    ]),
  ].filter((c) => c !== targetColumn && !dropCols.includes(c));

  const catCols = [
    ...new Set(
      encodeSteps.flatMap((s) => s.params?.columns || s.columns || [])
    ),
  ].filter((c) => c !== targetColumn && !dropCols.includes(c));

  const imputeStratNum =
    imputeSteps.find((s) => {
      const cols = s.params?.columns || s.columns || [];
      return numCols.some((c) => cols.includes(c));
    })?.params?.strategy || "median";

  const imputeStratCat =
    imputeSteps.find((s) => {
      const cols = s.params?.columns || s.columns || [];
      return catCols.some((c) => cols.includes(c));
    })?.params?.strategy || "most_frequent";

  const scaleMethod =
    scaleSteps[0]?.params?.method || scaleSteps[0]?.params?.transform || "StandardScaler";
  const scaleClass =
    scaleMethod === "minmax"
      ? "MinMaxScaler"
      : scaleMethod === "robust"
      ? "RobustScaler"
      : "StandardScaler";

  const encodeMethod =
    encodeSteps[0]?.params?.method || encodeSteps[0]?.params?.transform || "OneHotEncoder";
  const encodeClass =
    encodeMethod === "ordinal" || encodeMethod === "OrdinalEncoder"
      ? "OrdinalEncoder"
      : "OneHotEncoder";

  // Add required scalers/encoders
  if (SKLEARN_IMPORTS[scaleClass]) importsSet.add(SKLEARN_IMPORTS[scaleClass]);
  if (SKLEARN_IMPORTS[encodeClass])
    importsSet.add(SKLEARN_IMPORTS[encodeClass]);

  // ── Hyperparams string ───────────────────────────────────────────────────
  const hyperStr = Object.entries(hyperparams)
    .map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`)
    .join(", ");

  // ── Full script template ─────────────────────────────────────────────────
  const dropLine =
    dropCols.length > 0
      ? `df.drop(columns=${JSON.stringify(dropCols)}, errors="ignore", inplace=True)`
      : "# No columns dropped";

  const numColsLine =
    numCols.length > 0
      ? `numeric_features = ${JSON.stringify(numCols)}`
      : "numeric_features = []";
  const catColsLine =
    catCols.length > 0
      ? `categorical_features = ${JSON.stringify(catCols)}`
      : "categorical_features = []";

  const encodeExtra =
    encodeClass === "OneHotEncoder"
      ? `(${encodeClass}(handle_unknown="ignore", sparse_output=False))`
      : `(${encodeClass}())`;

  const script = `"""
Interactive Model Studio — Auto-Generated Pipeline
Dataset  : ${datasetName}
Target   : ${targetColumn}
Algorithm: ${algoClass}
Generated: ${new Date().toLocaleString()}

This script is fully self-contained and can be run with:
  pip install scikit-learn pandas numpy${algoClass.startsWith("XGB") ? " xgboost" : ""}
  python pipeline_${algoClass.toLowerCase()}.py
"""

${[...importsSet].sort().join("\n")}

# =============================================================================
# 1. Load Dataset
# =============================================================================
df = pd.read_csv("${datasetName}")
print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# =============================================================================
# 2. Drop Excluded Columns
# =============================================================================
${dropLine}

# =============================================================================
# 3. Define Feature Groups & Target
# =============================================================================
${numColsLine}
${catColsLine}
target = "${targetColumn}"

X = df.drop(columns=[target])
y = df[target]

# =============================================================================
# 4. Train / Test Split
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=${testSize},
    random_state=${randomState},
    stratify=y if y.nunique() <= 10 else None
)

# =============================================================================
# 5. Preprocessing Pipeline (ColumnTransformer)
# =============================================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="${imputeStratNum}")),
    ("scaler", ${scaleClass}()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="${imputeStratCat}")),
    ("encoder", ${encodeExtra}),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# =============================================================================
# 6. Fit Full Pipeline + Model
# =============================================================================
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", ${algoClass}(${hyperStr ? hyperStr + ", " : ""}random_state=${randomState})),
])

model_pipeline.fit(X_train, y_train)
print("Training complete.")

# =============================================================================
# 7. Evaluate
# =============================================================================
y_pred = model_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")

print(f"\\n{'='*50}")
print(f"  Accuracy : {acc * 100:.2f}%")
print(f"  F1 Score : {f1 * 100:.2f}%")
print(f"{'='*50}")
print(classification_report(y_test, y_pred))
`;

  return script.trim();
}

/**
 * Generate a minimal recipe summary string for display.
 * @param {Array} recipe  Array of step objects
 * @returns {string[]}    Human-readable step descriptions
 */
export function summarizeRecipe(recipe = []) {
  return recipe.map((step) => {
    const cols = (step.columns || step.params?.columns || []).join(", ");
    switch (step.step) {
      case "impute":
        return `Impute [${cols}] using ${step.params?.strategy || "mean"}`;
      case "scale":
        return `Scale [${cols}] with ${step.params?.method || "StandardScaler"}`;
      case "encode":
        return `Encode [${cols}] via ${step.params?.method || "OneHotEncoder"}`;
      case "drop_columns":
        return `Drop columns: ${(step.params?.columns || []).join(", ")}`;
      case "split":
        return `Train/Test Split: ${(1 - (step.params?.test_size || 0.2)) * 100}% / ${(step.params?.test_size || 0.2) * 100}%`;
      default:
        return `${step.step}`;
    }
  });
}
