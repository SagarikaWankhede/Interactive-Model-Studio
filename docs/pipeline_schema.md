# Interactive Model Studio — Pipeline Schema Specification
**Owner: Person B (ML Engine & Tracking)**  
**Version:** 1.2.0 (60% Milestone: Advanced Transforms & ONNX Export)

This document defines the formal JSON contract shared across:
- **ML Engine (`ml_engine/`)**: For executing data ingestion, memory optimization, preprocessing, training, evaluation, and ONNX export.
- **Backend Core (`backend/`)**: For persisting recipes, model configurations, and run metrics.
- **Frontend / Codegen (`frontend/`)**: For constructing recipes and generating standalone Python scripts.

---

## 1. Top-Level Pipeline Configuration Structure

A full pipeline configuration is a JSON object with the following primary blocks:
1. `task_type`: `"classification"` or `"regression"`.
2. `target_column`: Name of the target variable column in the dataset.
3. `recipe`: An ordered array of preprocessing steps.
4. `model_config`: Model algorithm identifier and hyperparameters.
5. `export_onnx` *(optional bool)*: Whether to export the fitted model into `.onnx` binary format.
6. `onnx_output_path` *(optional str)*: Destination filepath for the exported `.onnx` file.

### Example Configuration
```json
{
  "dataset_id": "dataset_hr_attrition",
  "task_type": "classification",
  "target_column": "attrition",
  "export_onnx": true,
  "onnx_output_path": "storage/models/attrition_model.onnx",
  "recipe": [
    {
      "step": "split",
      "params": {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": true
      }
    },
    {
      "step": "datetime",
      "params": {
        "columns": ["hire_date"],
        "components": ["year", "month", "day", "dayofweek", "is_weekend"]
      }
    },
    {
      "step": "impute",
      "params": {
        "columns": ["age", "income"],
        "strategy": "median"
      }
    },
    {
      "step": "outlier_clip",
      "params": {
        "columns": ["income"],
        "method": "iqr",
        "factor": 1.5
      }
    },
    {
      "step": "power",
      "params": {
        "columns": ["income"],
        "method": "log1p"
      }
    },
    {
      "step": "encode",
      "params": {
        "columns": ["department"],
        "method": "onehot",
        "drop": "first"
      }
    },
    {
      "step": "scale",
      "params": {
        "columns": ["age", "income"],
        "method": "standard"
      }
    }
  ],
  "model_config": {
    "algorithm": "RandomForestClassifier",
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": 6,
      "random_state": 42
    }
  }
}
```

---

## 2. Preprocessing Steps Specification (`recipe`)

The `recipe` is an ordered list executed sequentially.

### 2.1 `split` (Train / Test Splitting)
Splits dataset into training and testing partitions.
* **Parameters:**
  * `test_size` (*float*): Fraction of rows for test set (default `0.2`).
  * `random_state` (*int*): Seed for reproducible splits (default `42`).
  * `stratify` (*bool*): Whether to stratify by target (classification only, default `false`).
  * `shuffle` (*bool*): Whether to shuffle data before splitting (default `true`).

### 2.2 `impute` (Missing Value Imputation)
Replaces missing (`NaN`, `None`) values in specified columns.
* **Parameters:**
  * `columns` (*list[str]*): Columns to impute.
  * `strategy` (*str*): One of `"mean"`, `"median"`, `"most_frequent"`, `"constant"`.
  * `fill_value` (*optional*): Used only when `strategy == "constant"`.

### 2.3 `encode` (Categorical Encoding)
Converts categorical / string columns to numerical features.
* **Parameters:**
  * `columns` (*list[str]*): Categorical columns to encode.
  * `method` (*str*): One of `"onehot"`, `"ordinal"`.
  * `drop` (*optional str*): `"first"` or `null`.

### 2.4 `scale` (Feature Scaling)
Scales numerical features.
* **Parameters:**
  * `columns` (*list[str]*): Numerical columns to scale.
  * `method` (*str*): One of `"standard"` (`StandardScaler`), `"minmax"` (`MinMaxScaler`), `"robust"` (`RobustScaler`).

### 2.5 `power` / `power_transform` (Mathematical & Power Transforms)
Applies non-linear transformations to normalize skewed distributions.
* **Parameters:**
  * `columns` (*list[str]*): Numerical columns to transform.
  * `method` (*str*): One of `"log1p"`, `"sqrt"`, `"yeo-johnson"`, `"box-cox"`.
  * `standardize` (*bool*): Whether to center and scale output (for Yeo-Johnson / Box-Cox, default `true`).

### 2.6 `bin` (Continuous Discretization)
Converts continuous numerical variables into discrete quantile or uniform intervals.
* **Parameters:**
  * `columns` (*list[str]*): Numerical columns to discretize.
  * `n_bins` (*int*): Number of bins (default `5`).
  * `strategy` (*str*): `"quantile"` or `"uniform"`.
  * `encode` (*str*): `"ordinal"` or `"onehot"`.

### 2.7 `polynomial` (Polynomial & Interaction Expansion)
Generates interaction terms ($x_1 x_2$) and squared terms ($x_1^2$).
* **Parameters:**
  * `columns` (*list[str]*): Numerical columns to expand.
  * `degree` (*int*): Polynomial degree (default `2`).
  * `interaction_only` (*bool*): Exclude squared terms (default `false`).

### 2.8 `pca` (Dimensionality Reduction)
Applies Principal Component Analysis to compress high-dimensional or collinear features.
* **Parameters:**
  * `columns` (*list[str]*): Numerical columns to compress.
  * `n_components` (*int*): Number of principal components to extract.

### 2.9 `outlier_clip` (Outlier Winsorization)
Clips extreme outlier values to statistically sound bounds calculated on the training set.
* **Parameters:**
  * `columns` (*list[str]*): Columns to clip.
  * `method` (*str*): `"iqr"` (Interquartile Range) or `"quantile"`.
  * `factor` (*float*): IQR multiplier (default `1.5`).
  * `lower_quantile` (*float*): Bottom percentile for quantile mode (default `0.01`).
  * `upper_quantile` (*float*): Top percentile for quantile mode (default `0.99`).

### 2.10 `datetime` (Timestamp Feature Extraction)
Decomposes date/time columns into integer components and drops raw strings.
* **Parameters:**
  * `columns` (*list[str]*): Datetime columns to decompose.
  * `components` (*list[str]*): List containing `"year"`, `"month"`, `"day"`, `"dayofweek"`, `"is_weekend"`.

---

## 3. Model Configuration Specification (`model_config`)

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

```json
{
  "status": "SUCCESS",
  "task_type": "classification",
  "algorithm": "RandomForestClassifier",
  "train_time_seconds": 0.142,
  "cached_preprocessing": true,
  "metrics": {
    "accuracy": 0.845,
    "f1_score": 0.832,
    "precision": 0.850,
    "recall": 0.815,
    "roc_auc": 0.892
  },
  "confusion_matrix": {
    "labels": ["0", "1"],
    "matrix": [
      [52, 8],
      [7, 33]
    ]
  },
  "feature_importances": [
    {"feature": "income", "importance": 0.38},
    {"feature": "age", "importance": 0.28},
    {"feature": "department_Sales", "importance": 0.21}
  ],
  "dataset_shape": {
    "train_rows": 400,
    "test_rows": 100,
    "features_count": 5
  },
  "onnx_export": {
    "model_type": "RandomForestClassifier",
    "input_tensor_name": "float_input",
    "feature_count": 5,
    "target_opset": 15,
    "file_size_bytes": 142560,
    "saved_path": "storage/models/attrition_model.onnx"
  }
}
```
