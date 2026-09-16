/**
 * Fallback / Mock Datasets for Phase 1 Data Layer.
 * Allows Person C's UI to function standalone while Person A and Person B build backend services.
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
