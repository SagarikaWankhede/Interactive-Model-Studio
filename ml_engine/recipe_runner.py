"""
Sequential deterministic preprocessing recipe executor.
Processes data transformations step-by-step without data leakage.
"""

import pandas as pd
from typing import List, Dict, Any, Tuple
from .preprocessing import (
    apply_imputation,
    apply_encoding,
    apply_scaling,
    apply_train_test_split,
)


def run_recipe(
    df: pd.DataFrame,
    recipe: List[Dict[str, Any]],
    target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[Any]]:
    """
    Executes a list of preprocessing recipe steps on input DataFrame.

    Example recipe format:
    [
        {"step": "impute", "columns": ["age"], "params": {"strategy": "mean"}},
        {"step": "encode", "columns": ["gender"], "params": {"method": "onehot"}},
        {"step": "scale", "columns": ["age"], "params": {"method": "standard"}},
        {"step": "split", "params": {"test_size": 0.2, "random_state": 42}}
    ]
    """
    processed_df = df.copy()
    fitted_transformers = []
    split_params = {"test_size": 0.2, "random_state": 42, "stratify": False}

    for step_config in recipe:
        step_type = step_config.get("step")
        cols = step_config.get("columns", [])
        params = step_config.get("params", {})

        # Exclude target column from transformations if included accidentally
        cols = [c for c in cols if c != target_col]

        if step_type == "impute":
            strategy = params.get("strategy", "mean")
            fill_val = params.get("fill_value", None)
            processed_df, imputer = apply_imputation(processed_df, cols, strategy, fill_val)
            if imputer:
                fitted_transformers.append({"step": "impute", "transformer": imputer, "columns": cols})

        elif step_type == "encode":
            method = params.get("method", "onehot")
            processed_df, encoder = apply_encoding(processed_df, cols, method)
            if encoder:
                fitted_transformers.append({"step": "encode", "transformer": encoder, "columns": cols})

        elif step_type == "scale":
            method = params.get("method", "standard")
            processed_df, scaler = apply_scaling(processed_df, cols, method)
            if scaler:
                fitted_transformers.append({"step": "scale", "transformer": scaler, "columns": cols})

        elif step_type == "split":
            split_params.update(params)

        else:
            raise ValueError(f"Unknown recipe step type: '{step_type}'")

    X_train, X_test, y_train, y_test = apply_train_test_split(
        processed_df,
        target_col=target_col,
        test_size=split_params.get("test_size", 0.2),
        random_state=split_params.get("random_state", 42),
        stratify=split_params.get("stratify", False)
    )

    return X_train, X_test, y_train, y_test, fitted_transformers
