# ML Engine Documentation & Extension Guide
**Interactive Model Studio · Machine Learning Engine**

The ML Engine provides a modular, reproducible framework for executing tabular data transformations, training machine learning models, computing diagnostic evaluation metrics, and verifying execution reproducibility.

---

## Architecture Overview

```
ml_engine/
├── registry.py           # Central registry: step & model mappings (source of truth)
├── preprocessing.py      # Primitive transforms: imputation, encoding, scaling, splitting
├── recipe_runner.py      # Sequential recipe execution with leakage prevention
├── training.py           # Model initialization, fitting, and train_time measurement
├── evaluation.py         # Evaluation metrics, confusion matrix, feature importances
├── pipeline_executor.py  # Unified pipeline runner with dataset & recipe caching
└── reproduce.py          # Config-driven execution verification & metric comparison
```

---

## How to Add a New Preprocessing Step

Adding a new preprocessing step (e.g. `log_transform`, `binning`, or `pca`) requires 3 quick steps:

### Step 1: Register the Transformer in `registry.py`
In `ml_engine/registry.py`, add your step and strategy to `get_preprocessor_class`:

```python
# Example: Adding PowerTransformer for box-cox / yeo-johnson scaling
from sklearn.preprocessing import PowerTransformer

mapping = {
    ...
    ("transform", "yeo-johnson"): PowerTransformer,
}
```

### Step 2: Implement the Step Function in `preprocessing.py`
In `ml_engine/preprocessing.py`, add a function following the standard signature:
- Takes `(X_train, X_test, columns, **params)`
- Fits only on `X_train`
- Transforms both `X_train` and `X_test`
- Returns `(X_train_transformed, X_test_transformed, fitted_transformer)`

```python
def apply_power_transform(X_train, X_test, columns, method="yeo-johnson"):
    transformer = PowerTransformer(method=method)
    X_train_t = X_train.copy()
    X_test_t = X_test.copy()
    X_train_t[columns] = transformer.fit_transform(X_train[columns])
    X_test_t[columns] = transformer.transform(X_test[columns])
    return X_train_t, X_test_t, transformer
```

### Step 3: Wire into `recipe_runner.py`
Add the handler to the loop in `run_recipe()`:

```python
elif step_name == "transform":
    method = params.get("method", "yeo-johnson")
    X_train, X_test, transformer = apply_power_transform(
        X_train, X_test, columns=columns, method=method
    )
    fitted_transformers.append({
        "step": "transform",
        "columns": columns,
        "method": method,
        "transformer": transformer,
    })
```

---

## How to Add a New Model Algorithm

1. Open `ml_engine/registry.py`.
2. Add the algorithm class to `get_model_class()`:
   ```python
   "SupportVectorClassifier": "sklearn.svm.SVC"
   ```
3. Add metadata to `MODEL_METADATA`:
   ```python
   "SupportVectorClassifier": {
       "task_type": "classification",
       "library": "sklearn",
       "default_params": {"C": 1.0, "kernel": "rbf", "random_state": 42},
       "description": "Support Vector Machine classifier."
   }
   ```
4. Done! The training engine, evaluation engine, and API immediately support the new model.
# Machine Learning Engine (`ml_engine`)

The `ml_engine` package provides a deterministic, modular framework for preprocessing, training, evaluation, and reproducibility verification.

## Architecture

- **`registry.py`**: Global registry mapping names to scikit-learn / XGBoost estimators and transformers.
- **`preprocessing.py`**: Transformation functions for imputation, categorical encoding, scaling, and train-test splits.
- **`recipe_runner.py`**: Sequential recipe executor that transforms data step-by-step without data leakage.
- **`training.py`**: Estimator factory and fitting runner with wall-clock timing profiling.
- **`evaluation.py`**: Standardized metric evaluation (Accuracy, F1, ROC-AUC, RMSE, MAE, R²), confusion matrices, and feature importances.
- **`pipeline_executor.py`**: Main entry point `execute_pipeline()` with DataFrame content hashing and preprocessed data caching.
- **`reproduce.py`**: Reproducibility validator `verify_reproducibility()`.

## Usage Example

```python
import pandas as pd
from ml_engine import execute_pipeline

df = pd.read_csv("sample.csv")

config = {
    "task_type": "classification",
    "target_column": "target",
    "recipe": [
        {"step": "impute", "columns": ["age"], "params": {"strategy": "mean"}},
        {"step": "encode", "columns": ["gender"], "params": {"method": "onehot"}},
        {"step": "scale", "columns": ["age"], "params": {"method": "standard"}},
        {"step": "split", "params": {"test_size": 0.2, "random_state": 42}}
    ],
    "model_config": {
        "algorithm": "RandomForestClassifier",
        "hyperparameters": {"n_estimators": 50, "random_state": 42}
    }
}

result = execute_pipeline(df, config)
print("Accuracy:", result["metrics"]["accuracy"])
```

## Adding New Models or Transformers

To register a new algorithm or transformer, add it to `TRANSFORMER_REGISTRY` or `MODEL_REGISTRY` in `ml_engine/registry.py`.
