# Pipeline Schema Specification

This document defines the strict JSON contract between the Frontend/Backend and the ML Engine.

## 1. Pipeline Request Schema

```json
{
  "task_type": "classification",
  "target_column": "target",
  "recipe": [
    {
      "step": "impute",
      "columns": ["age", "income"],
      "params": {
        "strategy": "mean",
        "fill_value": null
      }
    },
    {
      "step": "encode",
      "columns": ["gender", "city"],
      "params": {
        "method": "onehot"
      }
    },
    {
      "step": "scale",
      "columns": ["age", "income"],
      "params": {
        "method": "standard"
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
      "max_depth": 10,
      "random_state": 42
    }
  }
}
```

---

## 2. Preprocessing Steps Specification (`recipe`)

The `recipe` is an ordered list executed sequentially.

### 2.1 `impute` (Missing Value Imputation)
Replaces missing (`NaN`, `None`) values in specified columns.
```json
{
  "step": "impute",
  "columns": ["col_a", "col_b"],
  "params": {
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
  "columns": ["col_c", "col_d"],
  "params": {
    "method": "onehot"
  }
}
```
* **Parameters:**
  * `columns` (*list[str]*): Categorical columns to encode.
  * `method` (*str*): One of `"onehot"`, `"ordinal"`.

### 2.3 `scale` (Feature Scaling)
Scales numerical features.
```json
{
  "step": "scale",
  "columns": ["col_a", "col_b"],
  "params": {
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
    "stratify": true
  }
}
```
* **Parameters:**
  * `test_size` (*float*): Fraction of rows for test set (default `0.2`).
  * `random_state` (*int*): Seed for reproducible splits (default `42`).
  * `stratify` (*bool*): Whether to stratify by target (classification only).

---

## 3. Model Configuration Specification (`model_config`)

Identifies the learning algorithm and hyperparameter arguments passed to the underlying estimator.

### 3.1 Supported Classification Algorithms
| Key | Scikit-Learn / XGBoost Estimator | Common Hyperparameters |
| :--- | :--- | :--- |
| `RandomForestClassifier` | `sklearn.ensemble.RandomForestClassifier` | `n_estimators`, `max_depth`, `random_state` |
| `LogisticRegression` | `sklearn.linear_model.LogisticRegression` | `C`, `max_iter`, `random_state` |
| `GradientBoostingClassifier`| `sklearn.ensemble.GradientBoostingClassifier` | `n_estimators`, `learning_rate`, `max_depth`, `random_state`|
| `XGBoostClassifier` / `XGBClassifier` | `xgboost.XGBClassifier` | `n_estimators`, `learning_rate`, `max_depth`, `random_state` |

### 3.2 Supported Regression Algorithms
| Key | Scikit-Learn / XGBoost Estimator | Common Hyperparameters |
| :--- | :--- | :--- |
| `RandomForestRegressor` | `sklearn.ensemble.RandomForestRegressor` | `n_estimators`, `max_depth`, `random_state` |
| `LinearRegression` | `sklearn.linear_model.LinearRegression` | `fit_intercept` |
| `GradientBoostingRegressor` | `sklearn.ensemble.GradientBoostingRegressor` | `n_estimators`, `learning_rate`, `max_depth`, `random_state` |
| `XGBoostRegressor` / `XGBRegressor` | `xgboost.XGBRegressor` | `n_estimators`, `learning_rate`, `max_depth`, `random_state` |

---

## 4. Pipeline Execution Output Schema

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
}
```
