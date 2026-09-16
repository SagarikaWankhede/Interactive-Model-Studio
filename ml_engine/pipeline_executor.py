"""
Pipeline Executor Module
The central orchestrator of the ML engine:
Chains Recipe Runner -> Model Training -> Evaluation.
Implements an in-memory transform cache to avoid recomputing preprocessing on repeat runs.
"""

import hashlib
import json
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from ml_engine.recipe_runner import run_recipe
from ml_engine.training import train_model
from ml_engine.evaluation import evaluate_model

# Global cache mapping cache_key -> (X_train, X_test, y_train, y_test, fitted_transformers)
_TRANSFORM_CACHE: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]] = {}


def _generate_cache_key(df: pd.DataFrame, target_column: str, recipe: list) -> str:
    """Generates a stable SHA-256 hash for dataset content + recipe configuration."""
    hasher = hashlib.sha256()
    # Hash row count, column names, and sample values
    meta_str = f"{len(df)}_{list(df.columns)}_{target_column}_"
    hasher.update(meta_str.encode("utf-8"))
    
    # Fast content sampling (head + tail)
    sample_bytes = pd.concat([df.head(10), df.tail(10)]).to_csv(index=False).encode("utf-8")
    hasher.update(sample_bytes)
    
    # Hash recipe
    recipe_str = json.dumps(recipe, sort_keys=True)
    hasher.update(recipe_str.encode("utf-8"))
    
    return hasher.hexdigest()


def clear_pipeline_cache():
    """Clears the preprocessing transform cache."""
    global _TRANSFORM_CACHE
    _TRANSFORM_CACHE.clear()


def execute_pipeline(
    df: pd.DataFrame,
    config: Dict[str, Any],
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Executes a complete end-to-end ML pipeline:
    1. Preprocessing recipe execution (with cache check)
    2. Model training and wall-clock time recording
    3. Performance evaluation and feature importance extraction

    Args:
        df: Input pandas DataFrame.
        config: Pipeline configuration dictionary adhering to docs/pipeline_schema.md.
        use_cache: Whether to use cached preprocessed data if available.

    Returns:
        Structured result dictionary.
    """
    target_column = config.get("target_column")
    if not target_column:
        raise ValueError("Missing required key 'target_column' in pipeline config.")

    task_type = config.get("task_type", "classification").lower()
    recipe = config.get("recipe", [])
    model_config = config.get("model_config", {})
    algorithm = model_config.get("algorithm", "RandomForestClassifier")
    hyperparameters = model_config.get("hyperparameters", {})

    # Check or compute preprocessing transforms
    cache_key = _generate_cache_key(df, target_column, recipe)
    cached_hit = False

    if use_cache and cache_key in _TRANSFORM_CACHE:
        X_train, X_test, y_train, y_test, fitted_transformers = _TRANSFORM_CACHE[cache_key]
        cached_hit = True
    else:
        X_train, X_test, y_train, y_test, fitted_transformers = run_recipe(
            df=df,
            target_column=target_column,
            recipe=recipe,
        )
        if use_cache:
            _TRANSFORM_CACHE[cache_key] = (X_train, X_test, y_train, y_test, fitted_transformers)

    # Train model
    fitted_model, train_time = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name=algorithm,
        hyperparameters=hyperparameters,
    )

    # Evaluate model
    evaluation_results = evaluate_model(
        fitted_model=fitted_model,
        X_test=X_test,
        y_test=y_test,
        task_type=task_type,
    )

    return {
        "status": "SUCCESS",
        "task_type": task_type,
        "algorithm": algorithm,
        "train_time_seconds": train_time,
        "cached_preprocessing": cached_hit,
        "metrics": evaluation_results["metrics"],
        "confusion_matrix": evaluation_results.get("confusion_matrix"),
        "feature_importances": evaluation_results.get("feature_importances", []),
        "dataset_shape": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "features_count": X_train.shape[1],
        },
        "features": list(X_train.columns),
        "recipe_applied": recipe,
        "model_config": model_config,
        "fitted_model": fitted_model,
        "fitted_transformers": fitted_transformers,
    }
