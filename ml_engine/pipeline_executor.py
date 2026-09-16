"""
Unified pipeline orchestrator with preprocessed data caching.
Chains recipe_runner -> training -> evaluation.
"""

import hashlib
import json
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from .recipe_runner import run_recipe
from .training import train_model
from .evaluation import evaluate_model


# In-memory LRU-style cache for preprocessed datasets
_PREPROCESSING_CACHE: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list]] = {}


def _generate_cache_key(df: pd.DataFrame, recipe: list, target_col: str) -> str:
    """Computes MD5 hash from DataFrame content summary and recipe steps."""
    df_sig = f"{df.shape}_{list(df.columns)}_{df.head(5).values.tobytes().hex()}"
    recipe_sig = json.dumps(recipe, sort_keys=True)
    combined = f"{df_sig}_{recipe_sig}_{target_col}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def execute_pipeline(
    df: pd.DataFrame,
    config_dict: Dict[str, Any],
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Main entry point for running complete ML pipeline.

    config_dict format:
    {
        "task_type": "classification",
        "target_column": "target",
        "recipe": [...],
        "model_config": {
            "algorithm": "RandomForestClassifier",
            "hyperparameters": {"n_estimators": 100}
        }
    }
    """
    task_type = config_dict.get("task_type", "classification")
    target_col = config_dict.get("target_column")
    recipe = config_dict.get("recipe", [])
    model_config = config_dict.get("model_config", {})

    if not target_col:
        raise ValueError("Missing 'target_column' in config_dict")

    model_name = model_config.get("algorithm")
    if not model_name:
        raise ValueError("Missing 'algorithm' in model_config")
    
    hyperparams = model_config.get("hyperparameters", {})

    # Check caching
    cache_key = _generate_cache_key(df, recipe, target_col)
    if use_cache and cache_key in _PREPROCESSING_CACHE:
        X_train, X_test, y_train, y_test, fitted_transformers = _PREPROCESSING_CACHE[cache_key]
    else:
        X_train, X_test, y_train, y_test, fitted_transformers = run_recipe(df, recipe, target_col)
        if use_cache:
            _PREPROCESSING_CACHE[cache_key] = (X_train, X_test, y_train, y_test, fitted_transformers)

    # Train model
    fitted_model, train_time_sec = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name=model_name,
        task_type=task_type,
        hyperparameters=hyperparams
    )

    # Evaluate model
    eval_results = evaluate_model(
        fitted_model=fitted_model,
        X_test=X_test,
        y_test=y_test,
        task_type=task_type,
        feature_names=list(X_train.columns)
    )

    return {
        "status": "success",
        "task_type": task_type,
        "target_column": target_col,
        "algorithm": model_name,
        "hyperparameters": hyperparams,
        "metrics": eval_results.get("metrics"),
        "confusion_matrix": eval_results.get("confusion_matrix"),
        "feature_importances": eval_results.get("feature_importances"),
        "train_time_sec": train_time_sec,
        "fitted_model": fitted_model,
        "fitted_transformers": fitted_transformers,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
    }
