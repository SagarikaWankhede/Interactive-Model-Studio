# Interactive Model Studio — Pipeline Schema Specification
**Owner: Person B (ML Engine & Tracking)**  
**Version:** 1.0.0

This document defines the formal JSON contract shared across:
- **ML Engine (`ml_engine/`)**: For executing preprocessing, training, and evaluation.
- **Backend Core (`backend/`)**: For persisting recipes, model configurations, and run metrics.
- **Frontend / Codegen (`frontend/`)**: For constructing recipes and generating standalone Python scripts.

---

## 1. Top-Level Pipeline Configuration Structure

A full pipeline configuration is a JSON object with four primary blocks:
1. `task_type`: `"classification"` or `"regression"`.
2. `target_column`: Name of the target variable column in the dataset.
3. `recipe`: An ordered array of preprocessing steps.
4. `model_config`: Model algorithm identifier and hyperparameters.

### Example Configuration
```json
{
  "dataset_id": "dataset_123",
  "task_type": "classification",
# Pipeline Schema Specification

This document defines the strict JSON contract between the Frontend/Backend and the ML Engine.

## 1. Pipeline Request Schema

```json
{
  "task_type": "classification", // "classification" | "regression"
  "target_column": "target",
  "recipe": [
    {
      "step": "impute",
<<<<<<< HEAD
      "params": {
        "columns": ["age", "fare"],
        "strategy": "median"
=======
      "columns": ["age", "income"],
      "params": {
        "strategy": "mean", // "mean" | "median" | "most_frequent" | "constant"
        "fill_value": null
>>>>>>> origin/main
      }
    },
    {
      "step": "encode",
<<<<<<< HEAD
      "params": {
        "columns": ["sex", "embarked"],
        "method": "onehot",
        "drop": "first"
=======
      "columns": ["gender", "city"],
      "params": {
        "method": "onehot" // "onehot" | "ordinal"
>>>>>>> origin/main
      }
    },
    {
      "step": "scale",
<<<<<<< HEAD
      "params": {
        "columns": ["age", "fare"],
        "method": "standard"
=======
      "columns": ["age", "income"],
      "params": {
        "method": "standard" // "standard" | "minmax" | "robust"
>>>>>>> origin/main
      }
    },
    {
      "step": "split",
      "params": {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": true
      }
    }
  ],
  "model_config": {
    "algorithm": "RandomForestClassifier",
    "hyperparameters": {
      "n_estimators": 100,
<<<<<<< HEAD
      "max_depth": 6,
=======
      "max_depth": 10,
>>>>>>> origin/main
      "random_state": 42
    }
  }
}
```

<<<<<<< HEAD
---

## 2. Preprocessing Steps Specification (`recipe`)

The `recipe` is an ordered list executed sequentially.

### 2.1 `impute` (Missing Value Imputation)
Replaces missing (`NaN`, `None`) values in specified columns.
```json
{
  "step": "impute",
  "params": {
    "columns": ["col_a", "col_b"],
    "strategy": "mean",
    "fill_value": null
  }
}
```
* **Parameters:**
  * `columns` (*list[str]*): Columns to impute.
  * `strategy` (*str*): One of `"mean"`, `"median"`, `"most_frequent"`, `"constant"`.
  * `fill_value` (*optional*): Used only when `strategy == "constant"`.

### 2.2 `encode` (Categorical Encoding)
Converts categorical / string columns to numerical features.
```json
{
  "step": "encode",
  "params": {
    "columns": ["col_c", "col_d"],
    "method": "onehot",
    "drop": "first"
  }
}
```
* **Parameters:**
  * `columns` (*list[str]*): Categorical columns to encode.
  * `method` (*str*): One of `"onehot"`, `"ordinal"`, or `"target"`.
  * `drop` (*optional str*): `"first"` or `null` (for one-hot encoding).

### 2.3 `scale` (Feature Scaling)
Scales numerical features.
```json
{
  "step": "scale",
  "params": {
    "columns": ["col_a", "col_b"],
    "method": "standard"
  }
}
```
* **Parameters:**
  * `columns` (*list[str]*): Numerical columns to scale.
  * `method` (*str*): One of `"standard"` (`StandardScaler`), `"minmax"` (`MinMaxScaler`), `"robust"` (`RobustScaler`).

### 2.4 `split` (Train / Test Splitting)
Splits dataset into training and testing partitions.
```json
{
  "step": "split",
  "params": {
    "test_size": 0.2,
    "random_state": 42,
    "stratify": true,
    "shuffle": true
  }
}
```
* **Parameters:**
  * `test_size` (*float*): Fraction of rows for test set (default `0.2`).
  * `random_state` (*int*): Seed for reproducible splits (default `42`).
  * `stratify` (*bool*): Whether to stratify by target (classification only, default `false`).
  * `shuffle` (*bool*): Whether to shuffle data before splitting (default `true`).

---

## 3. Model Configuration Specification (`model_config`)

Identifies the learning algorithm and hyperparameter arguments passed to the underlying estimator.

### 3.1 Supported Classification Algorithms
| Key | Scikit-Learn / XGBoost Estimator | Common Hyperparameters |
| :--- | :--- | :--- |
| `RandomForestClassifier` | `sklearn.ensemble.RandomForestClassifier` | `n_estimators`, `max_depth`, `random_state`, `min_samples_split` |
| `LogisticRegression` | `sklearn.linear_model.LogisticRegression` | `C`, `penalty`, `solver`, `max_iter`, `random_state` |
| `DecisionTreeClassifier` | `sklearn.tree.DecisionTreeClassifier` | `max_depth`, `criterion`, `min_samples_split`, `random_state` |
| `GradientBoostingClassifier`| `sklearn.ensemble.GradientBoostingClassifier` | `n_estimators`, `learning_rate`, `max_depth`, `random_state`|
| `XGBClassifier` | `xgboost.XGBClassifier` | `n_estimators`, `learning_rate`, `max_depth`, `random_state` |

### 3.2 Supported Regression Algorithms
| Key | Scikit-Learn / XGBoost Estimator | Common Hyperparameters |
| :--- | :--- | :--- |
| `RandomForestRegressor` | `sklearn.ensemble.RandomForestRegressor` | `n_estimators`, `max_depth`, `random_state`, `min_samples_split` |
| `LinearRegression` | `sklearn.linear_model.LinearRegression` | `fit_intercept` |
| `DecisionTreeRegressor` | `sklearn.tree.DecisionTreeRegressor` | `max_depth`, `criterion`, `min_samples_split`, `random_state` |
| `GradientBoostingRegressor` | `sklearn.ensemble.GradientBoostingRegressor` | `n_estimators`, `learning_rate`, `max_depth`, `random_state` |
| `XGBRegressor` | `xgboost.XGBRegressor` | `n_estimators`, `learning_rate`, `max_depth`, `random_state` |

---

## 4. Pipeline Execution Output Schema (`ResultDict`)

The output returned by `ml_engine.pipeline_executor.execute_pipeline()`:

```json
{
  "status": "SUCCESS",
  "task_type": "classification",
  "train_time_seconds": 0.142,
  "metrics": {
    "accuracy": 0.825,
    "f1_score": 0.812,
    "precision": 0.830,
    "recall": 0.795,
    "roc_auc": 0.887
  },
  "confusion_matrix": {
    "labels": ["Class 0", "Class 1"],
    "matrix": [
      [50, 10],
      [8, 32]
    ]
  },
  "feature_importances": [
    {"feature": "fare", "importance": 0.42},
    {"feature": "age", "importance": 0.31},
    {"feature": "sex_male", "importance": 0.27}
  ],
  "dataset_shape": {
    "train_rows": 400,
    "test_rows": 100,
    "features_count": 3
  },
  "recipe_applied": [ ... ],
  "model_config": { ... },
  "fitted_model": "<estimator_object>"
=======
## 2. Pipeline Execution Result Schema

```json
{
  "status": "success",
  "task_type": "classification",
  "metrics": {
    "accuracy": 0.95,
    "balanced_accuracy": 0.94,
    "precision_macro": 0.95,
    "precision_weighted": 0.95,
    "recall_macro": 0.94,
    "recall_weighted": 0.95,
    "f1_macro": 0.945,
    "f1_weighted": 0.95,
    "roc_auc": 0.98
  },
  "confusion_matrix": {
    "labels": [0, 1],
    "matrix": [[45, 5], [2, 48]]
  },
  "feature_importances": [
    {"feature": "income", "importance": 0.45},
    {"feature": "age", "importance": 0.35}
  ],
  "train_time_sec": 0.124,
  "reproducibility": {
    "is_reproducible": true,
    "diff": {}
  }
>>>>>>> origin/main
}
```
