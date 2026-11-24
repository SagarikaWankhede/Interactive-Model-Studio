"""
Sequential deterministic preprocessing recipe executor.

Steps run in recipe order. The position of the ``split`` step matters:

* Steps **before** ``split`` are fitted on the full dataset (this is how the
  classic recipes behave — split is usually the last step).
* Steps **after** ``split`` are fitted on the training partition only and then
  applied to the test partition, which is the leakage-free pattern.

If the recipe has no ``split`` step, the split happens after the last step.
"""

import pandas as pd
from typing import List, Dict, Any, Tuple
from .preprocessing import apply_train_test_split, fit_step, transform_with_entry


def run_recipe(
    df: pd.DataFrame,
    recipe: List[Dict[str, Any]],
    target_col: str = None,
    target_column: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[Any]]:
    """
    Executes a list of preprocessing recipe steps on input DataFrame.
    Returns X_train, X_test, y_train, y_test and the ordered fitted transformers.
    """
    target = target_col or target_column
    if not target:
        raise ValueError("Target column must be provided to run_recipe.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    processed_df = df.copy()
    fitted_transformers = []
    split_params = {"test_size": 0.2, "random_state": 42, "stratify": False}
    split_done = False
    X_train = X_test = y_train = y_test = None

    for step_config in recipe:
        step_type = step_config.get("step")
        params = step_config.get("params", {}) or {}

        # Support columns specified at top-level or inside params dictionary
        cols = step_config.get("columns") or params.get("columns", [])
        if isinstance(cols, str):
            cols = [cols]
        cols = [c for c in cols if c != target]

        if step_type == "split":
            if split_done:
                raise ValueError("Recipe contains more than one 'split' step.")
            split_params.update({k: v for k, v in params.items() if k != "columns"})
            X_train, X_test, y_train, y_test = _split(processed_df, target, split_params)
            split_done = True
            continue

        if step_type not in ("impute", "encode", "scale", "drop_columns"):
            raise ValueError(f"Unknown recipe step type: '{step_type}'")

        if not split_done:
            processed_df, entry = fit_step(processed_df, step_type, cols, params)
        else:
            # Fit on the training partition only, then replay on the test partition.
            X_train, entry = fit_step(X_train, step_type, cols, params)
            if entry is not None:
                X_test = transform_with_entry(X_test, entry)

        if entry is not None:
            fitted_transformers.append(entry)

    if not split_done:
        X_train, X_test, y_train, y_test = _split(processed_df, target, split_params)

    return X_train, X_test, y_train, y_test, fitted_transformers


def _split(df: pd.DataFrame, target: str, split_params: Dict[str, Any]):
    return apply_train_test_split(
        df,
        target_col=target,
        test_size=split_params.get("test_size", 0.2),
        random_state=split_params.get("random_state", 42),
        stratify=split_params.get("stratify", False),
    )
