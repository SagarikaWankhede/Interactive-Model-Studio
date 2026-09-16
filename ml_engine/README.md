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
