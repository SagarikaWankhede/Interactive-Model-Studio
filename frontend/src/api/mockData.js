/**
 * Fallback / Mock Data for Interactive ML Model Studio.
 * Supports standalone testing of Phase 1 (Data Layer) and Phase 2 (Experiment Engine).
 */

export const mockDatasets = [
  {
    id: "ds-churn-001",
    name: "customer_churn.csv",
    version: 1,
    file_hash: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    upload_date: "2026-09-16T08:30:00Z",
    row_count: 7043,
    column_count: 11,
    target_column: "Churn",
    schema: [
      { name: "CustomerID", type: "categorical", null_count: 0, null_pct: 0.0, cardinality: 7043, sample: "7590-VHVEG" },
      { name: "TenureMonths", type: "numerical", null_count: 12, null_pct: 0.17, cardinality: 73, sample: "12" },
      { name: "MonthlyCharges", type: "numerical", null_count: 0, null_pct: 0.0, cardinality: 1585, sample: "29.85" },
      { name: "TotalCharges", type: "numerical", null_count: 2450, null_pct: 34.78, cardinality: 6530, sample: "840.50" },
      { name: "ContractType", type: "categorical", null_count: 0, null_pct: 0.0, cardinality: 3, sample: "Month-to-month" },
      { name: "PaymentMethod", type: "categorical", null_count: 150, null_pct: 2.13, cardinality: 4, sample: "Electronic check" },
      { name: "InternetService", type: "categorical", null_count: 0, null_pct: 0.0, cardinality: 3, sample: "Fiber optic" },
      { name: "ZipCode", type: "categorical", null_count: 0, null_pct: 0.0, cardinality: 840, sample: "90210" },
      { name: "CancellationReason", type: "categorical", null_count: 5174, null_pct: 73.46, cardinality: 21, sample: "Competitor offered higher download speed" },
      { name: "ChurnFlagProxy", type: "categorical", null_count: 0, null_pct: 0.0, cardinality: 2, sample: "Yes" },
      { name: "Churn", type: "categorical", null_count: 0, null_pct: 0.0, cardinality: 2, sample: "No" },
    ],
    diagnostics: [
      {
        id: "diag-leak-1",
        type: "leakage",
        severity: "red",
        title: "Potential Target Leakage Detected",
        column: "ChurnFlagProxy",
        description: "Column 'ChurnFlagProxy' has a 99.4% correlation with target column 'Churn'. Including this column will lead to unrealistically inflated test scores that fail in production.",
        suggested_fix: "Exclude 'ChurnFlagProxy' from the feature set before training."
      },
      {
        id: "diag-imbalance-1",
        type: "imbalance",
        severity: "red",
        title: "Severe Class Imbalance in Target",
        column: "Churn",
        description: "The target distribution is skewed: 73.5% 'No' vs 26.5% 'Yes'. Standard classifiers may bias heavily toward the majority class.",
        suggested_fix: "Consider using class weighting or SMOTE oversampling during training."
      },
      {
        id: "diag-nulls-1",
        type: "missing_values",
        severity: "yellow",
        title: "High Missing Value Rate (>30%)",
        column: "TotalCharges",
        description: "34.78% of values (2,450 rows) are null. Simple mean imputation may distort the distribution variance.",
        suggested_fix: "Use median imputation or iterative regression imputation."
      },
      {
        id: "diag-card-1",
        type: "high_cardinality",
        severity: "yellow",
        title: "High Cardinality Categorical Feature",
        column: "ZipCode",
        description: "Contains 840 distinct categorical levels. Standard One-Hot Encoding will create 840 sparse columns, drastically increasing memory and overfitting risk.",
        suggested_fix: "Use Target Encoding, frequency binning, or drop if unneeded."
      }
    ]
  },
  {
    id: "ds-housing-002",
    name: "housing_prices.csv",
    version: 1,
    file_hash: "a4f91b5c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a",
    upload_date: "2026-09-15T14:15:00Z",
    row_count: 1460,
    column_count: 8,
    target_column: "SalePrice",
    schema: [
      { name: "LotArea", type: "numerical", null_count: 0, null_pct: 0.0, cardinality: 1073, sample: "8450" },
      { name: "OverallQual", type: "numerical", null_count: 0, null_pct: 0.0, cardinality: 10, sample: "7" },
      { name: "YearBuilt", type: "numerical", null_count: 0, null_pct: 0.0, cardinality: 112, sample: "2003" },
      { name: "TotalBsmtSF", type: "numerical", null_count: 18, null_pct: 1.23, cardinality: 721, sample: "856" },
      { name: "Neighborhood", type: "categorical", null_count: 0, null_pct: 0.0, cardinality: 25, sample: "CollgCr" },
      { name: "PoolQC", type: "categorical", null_count: 1453, null_pct: 99.52, cardinality: 3, sample: "Gd" },
      { name: "CentralAir", type: "categorical", null_count: 0, null_pct: 0.0, cardinality: 2, sample: "Y" },
      { name: "SalePrice", type: "numerical", null_count: 0, null_pct: 0.0, cardinality: 663, sample: "208500" },
    ],
    diagnostics: [
      {
        id: "diag-nulls-pool",
        type: "missing_values",
        severity: "red",
        title: "Extreme Missing Value Rate (>90%)",
        column: "PoolQC",
        description: "99.52% of entries are missing. Imputing this feature will introduce meaningless noise.",
        suggested_fix: "Drop column 'PoolQC' entirely from the feature set."
      }
    ]
  }
];

export function getMockDatasetById(id) {
  return mockDatasets.find((d) => d.id === id) || mockDatasets[0];
}

/**
 * Supported ML Algorithms and their dynamic hyperparameter schemas.
 */
export const supportedAlgorithms = [
  {
    id: "RandomForest",
    name: "Random Forest Classifier",
    category: "Ensemble",
    description: "Robust ensemble of decision trees. Excellent generalist model resistant to overfitting.",
    defaultHyperparams: {
      n_estimators: 100,
      max_depth: 12,
      min_samples_split: 2,
      criterion: "gini",
    },
    paramSchema: [
      { name: "n_estimators", label: "Number of Trees", type: "number", min: 10, max: 500, step: 10 },
      { name: "max_depth", label: "Max Depth", type: "number", min: 1, max: 50, step: 1 },
      { name: "min_samples_split", label: "Min Samples to Split", type: "number", min: 2, max: 20, step: 1 },
      { name: "criterion", label: "Split Criterion", type: "select", options: ["gini", "entropy", "log_loss"] },
    ],
  },
  {
    id: "XGBoost",
    name: "XGBoost Classifier",
    category: "Gradient Boosting",
    description: "High-performance gradient boosted decision trees. Often yields top leaderboard accuracy.",
    defaultHyperparams: {
      n_estimators: 150,
      learning_rate: 0.08,
      max_depth: 6,
      subsample: 0.8,
    },
    paramSchema: [
      { name: "n_estimators", label: "Boosting Rounds", type: "number", min: 20, max: 1000, step: 10 },
      { name: "learning_rate", label: "Learning Rate (eta)", type: "number", min: 0.01, max: 0.5, step: 0.01 },
      { name: "max_depth", label: "Max Tree Depth", type: "number", min: 2, max: 15, step: 1 },
      { name: "subsample", label: "Subsample Ratio", type: "number", min: 0.5, max: 1.0, step: 0.05 },
    ],
  },
  {
    id: "LogisticRegression",
    name: "Logistic Regression",
    category: "Linear Models",
    description: "Fast, interpretable linear classification with L1/L2 regularization.",
    defaultHyperparams: {
      C: 1.0,
      penalty: "l2",
      max_iter: 200,
      solver: "lbfgs",
    },
    paramSchema: [
      { name: "C", label: "Inverse Regularization (C)", type: "number", min: 0.01, max: 100.0, step: 0.1 },
      { name: "penalty", label: "Penalty Regularization", type: "select", options: ["l2", "none"] },
      { name: "max_iter", label: "Maximum Iterations", type: "number", min: 50, max: 1000, step: 50 },
      { name: "solver", label: "Optimization Solver", type: "select", options: ["lbfgs", "saga", "newton-cg"] },
    ],
  },
  {
    id: "DecisionTree",
    name: "Decision Tree Classifier",
    category: "Trees",
    description: "Single decision tree offering maximum explainability and rapid training time.",
    defaultHyperparams: {
      max_depth: 8,
      min_samples_split: 5,
      criterion: "gini",
    },
    paramSchema: [
      { name: "max_depth", label: "Max Depth", type: "number", min: 2, max: 30, step: 1 },
      { name: "min_samples_split", label: "Min Samples Split", type: "number", min: 2, max: 20, step: 1 },
      { name: "criterion", label: "Split Criterion", type: "select", options: ["gini", "entropy"] },
    ],
  },
];

/**
 * Initial Experiment Runs History for Phase 2.
 */
export let mockExperiments = [
  {
    id: "exp-run-001",
    dataset_id: "ds-churn-001",
    dataset_name: "customer_churn.csv",
    model_name: "XGBoost Classifier",
    model_type: "XGBoost",
    status: "completed",
    train_time: 1.94,
    created_at: "2026-09-16T08:45:00Z",
    metrics: {
      accuracy: 0.958,
      f1_score: 0.949,
      precision: 0.952,
      recall: 0.946,
      roc_auc: 0.984,
    },
    hyperparams: {
      n_estimators: 150,
      learning_rate: 0.08,
      max_depth: 6,
      subsample: 0.8,
    },
    confusion_matrix: [
      [972, 45],
      [58, 334]
    ],
    feature_importances: [
      { feature: "ContractType", importance: 0.34 },
      { feature: "MonthlyCharges", importance: 0.26 },
      { feature: "TenureMonths", importance: 0.22 },
      { feature: "InternetService", importance: 0.12 },
      { feature: "PaymentMethod", importance: 0.06 },
    ],
  },
  {
    id: "exp-run-002",
    dataset_id: "ds-churn-001",
    dataset_name: "customer_churn.csv",
    model_name: "Random Forest Classifier",
    model_type: "RandomForest",
    status: "completed",
    train_time: 1.45,
    created_at: "2026-09-16T08:50:00Z",
    metrics: {
      accuracy: 0.938,
      f1_score: 0.929,
      precision: 0.931,
      recall: 0.927,
      roc_auc: 0.967,
    },
    hyperparams: {
      n_estimators: 100,
      max_depth: 12,
      min_samples_split: 2,
      criterion: "gini",
    },
    confusion_matrix: [
      [950, 67],
      [73, 319]
    ],
    feature_importances: [
      { feature: "TenureMonths", importance: 0.31 },
      { feature: "MonthlyCharges", importance: 0.29 },
      { feature: "ContractType", importance: 0.24 },
      { feature: "TotalCharges", importance: 0.11 },
      { feature: "PaymentMethod", importance: 0.05 },
    ],
  },
  {
    id: "exp-run-003",
    dataset_id: "ds-churn-001",
    dataset_name: "customer_churn.csv",
    model_name: "Logistic Regression",
    model_type: "LogisticRegression",
    status: "completed",
    train_time: 0.38,
    created_at: "2026-09-16T08:55:00Z",
    metrics: {
      accuracy: 0.825,
      f1_score: 0.812,
      precision: 0.819,
      recall: 0.806,
      roc_auc: 0.883,
    },
    hyperparams: {
      C: 1.0,
      penalty: "l2",
      max_iter: 200,
      solver: "lbfgs",
    },
    confusion_matrix: [
      [875, 142],
      [104, 288]
    ],
    feature_importances: [
      { feature: "MonthlyCharges", importance: 0.42 },
      { feature: "ContractType", importance: 0.35 },
      { feature: "TenureMonths", importance: 0.23 },
    ],
  },
];

export function addMockExperiment(exp) {
  mockExperiments = [exp, ...mockExperiments];
  return exp;
}
