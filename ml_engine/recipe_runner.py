"""
Recipe Runner Module
Executes an ordered list of preprocessing steps deterministically on a dataset.
Prevents data leakage by fitting transformers exclusively on the training split.
Supports all core and advanced transforms (imputation, encoding, scaling,
power transforms, binning, polynomial features, PCA, outlier clipping, datetime decomposition).
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from ml_engine.preprocessing import (
    apply_split,
    apply_imputation,
    apply_scaling,
    apply_encoding,
    apply_power_transform,
    apply_binning,
    apply_polynomial_features,
    apply_pca,
    apply_outlier_clipping,
    apply_datetime_decomposition,
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
        # Support both 'columns' in params and top-level 'columns'
        columns = params.get("columns", step_cfg.get("columns", []))

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

        elif step_name in ["power", "power_transform"]:
            method = params.get("method", "yeo-johnson")
            standardize = params.get("standardize", True)
            X_train, X_test, transformer = apply_power_transform(
                X_train, X_test, columns=columns, method=method, standardize=standardize
            )
            fitted_transformers.append({
                "step": "power_transform",
                "columns": columns,
                "method": method,
                "transformer": transformer,
            })

        elif step_name in ["bin", "discretize"]:
            n_bins = int(params.get("n_bins", 5))
            strategy = params.get("strategy", "quantile")
            encode = params.get("encode", "ordinal")
            X_train, X_test, transformer = apply_binning(
                X_train, X_test, columns=columns, n_bins=n_bins, strategy=strategy, encode=encode
            )
            fitted_transformers.append({
                "step": "bin",
                "columns": columns,
                "n_bins": n_bins,
                "strategy": strategy,
                "transformer": transformer,
            })

        elif step_name in ["polynomial", "poly"]:
            degree = int(params.get("degree", 2))
            interaction_only = bool(params.get("interaction_only", False))
            include_bias = bool(params.get("include_bias", False))
            X_train, X_test, transformer = apply_polynomial_features(
                X_train, X_test, columns=columns, degree=degree,
                interaction_only=interaction_only, include_bias=include_bias
            )
            fitted_transformers.append({
                "step": "polynomial",
                "columns": columns,
                "degree": degree,
                "transformer": transformer,
            })

        elif step_name == "pca":
            n_components = int(params.get("n_components", 2))
            X_train, X_test, transformer = apply_pca(
                X_train, X_test, columns=columns, n_components=n_components
            )
            fitted_transformers.append({
                "step": "pca",
                "columns": columns,
                "n_components": n_components,
                "transformer": transformer,
            })

        elif step_name in ["outlier_clip", "clip"]:
            method = params.get("method", "iqr")
            factor = float(params.get("factor", 1.5))
            lower_q = float(params.get("lower_quantile", 0.01))
            upper_q = float(params.get("upper_quantile", 0.99))
            X_train, X_test, thresholds = apply_outlier_clipping(
                X_train, X_test, columns=columns, method=method,
                factor=factor, lower_quantile=lower_q, upper_quantile=upper_q
            )
            fitted_transformers.append({
                "step": "outlier_clip",
                "columns": columns,
                "method": method,
                "thresholds": thresholds,
            })

        elif step_name in ["datetime", "datetime_features"]:
            components = tuple(params.get("components", ["year", "month", "day", "dayofweek", "is_weekend"]))
            X_train, X_test, transformer = apply_datetime_decomposition(
                X_train, X_test, columns=columns, components=components
            )
            fitted_transformers.append({
                "step": "datetime",
                "columns": columns,
                "components": components,
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
