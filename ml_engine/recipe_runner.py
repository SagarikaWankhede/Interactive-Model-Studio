"""
Recipe Runner Module
Executes an ordered list of preprocessing steps deterministically on a dataset.
Prevents data leakage by fitting transformers exclusively on the training split.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from ml_engine.preprocessing import (
    apply_split,
    apply_imputation,
    apply_scaling,
    apply_encoding,
)


def run_recipe(
    df: pd.DataFrame,
    target_column: str,
    recipe: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[Dict[str, Any]]]:
    """
    Executes a sequence of preprocessing steps on a raw DataFrame.

    Args:
        df: Raw pandas DataFrame.
        target_column: Name of the target variable.
        recipe: List of step dictionaries matching docs/pipeline_schema.md.

    Returns:
        X_train (pd.DataFrame): Transformed training features.
        X_test (pd.DataFrame): Transformed testing features.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Testing target labels.
        fitted_transformers (List[Dict]): List of fitted transformer objects and metadata.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the dataset.")

    # 1. Identify split configuration (default to 80/20 if not explicitly defined)
    split_step = None
    feature_steps = []

    for step_cfg in recipe:
        if step_cfg.get("step") == "split":
            split_step = step_cfg
        else:
            feature_steps.append(step_cfg)

    split_params = split_step.get("params", {}) if split_step else {}
    test_size = float(split_params.get("test_size", 0.2))
    random_state = int(split_params.get("random_state", 42))
    stratify = bool(split_params.get("stratify", False))
    shuffle = bool(split_params.get("shuffle", True))

    # Apply initial train/test split to prevent leakage
    X_train, X_test, y_train, y_test = apply_split(
        df=df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=shuffle,
    )

    fitted_transformers: List[Dict[str, Any]] = []

    # 2. Sequentially apply each feature preprocessing step
    for step_cfg in feature_steps:
        step_name = step_cfg.get("step", "").lower()
        params = step_cfg.get("params", {})
        columns = params.get("columns", [])

        if not columns:
            continue

        if step_name == "impute":
            strategy = params.get("strategy", "mean")
            fill_value = params.get("fill_value", None)
            X_train, X_test, transformer = apply_imputation(
                X_train, X_test, columns=columns, strategy=strategy, fill_value=fill_value
            )
            fitted_transformers.append({
                "step": "impute",
                "columns": columns,
                "strategy": strategy,
                "transformer": transformer,
            })

        elif step_name == "encode":
            method = params.get("method", "onehot")
            drop = params.get("drop", "first")
            X_train, X_test, transformer = apply_encoding(
                X_train, X_test, columns=columns, method=method, drop=drop
            )
            fitted_transformers.append({
                "step": "encode",
                "columns": columns,
                "method": method,
                "transformer": transformer,
            })

        elif step_name == "scale":
            method = params.get("method", "standard")
            X_train, X_test, transformer = apply_scaling(
                X_train, X_test, columns=columns, method=method
            )
            fitted_transformers.append({
                "step": "scale",
                "columns": columns,
                "method": method,
                "transformer": transformer,
            })

        else:
            raise ValueError(f"Unknown or unsupported preprocessing step: '{step_name}'")

    # Ensure all columns have string names and numerical indices are reset
    X_train.columns = [str(c) for c in X_train.columns]
    X_test.columns = [str(c) for c in X_test.columns]

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        fitted_transformers,
    )
