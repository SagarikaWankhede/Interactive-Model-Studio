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
      "columns": ["age", "income"],
      "params": {
        "strategy": "mean", // "mean" | "median" | "most_frequent" | "constant"
        "fill_value": null
      }
    },
    {
      "step": "encode",
      "columns": ["gender", "city"],
      "params": {
        "method": "onehot" // "onehot" | "ordinal"
      }
    },
    {
      "step": "scale",
      "columns": ["age", "income"],
      "params": {
        "method": "standard" // "standard" | "minmax" | "robust"
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
}
```
