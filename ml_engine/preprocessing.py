"""
Modular preprocessing transform functions.
Handles imputation, encoding, scaling, and train-test splitting cleanly on pandas DataFrames.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from .registry import get_transformer


def apply_imputation(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    fill_value: Optional[Any] = None
) -> Tuple[pd.DataFrame, Any]:
    """Applies missing value imputation on specified columns."""
    df_out = df.copy()
    valid_cols = [c for c in columns if c in df_out.columns]
    if not valid_cols:
        return df_out, None

    imputer_cls = get_transformer("impute", "simple")
    imputer = imputer_cls(strategy=strategy, fill_value=fill_value)
    
    transformed_vals = imputer.fit_transform(df_out[valid_cols])
    df_out[valid_cols] = transformed_vals
    return df_out, imputer


def apply_encoding(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "onehot"
) -> Tuple[pd.DataFrame, Any]:
    """Encodes categorical columns using OneHotEncoder or OrdinalEncoder."""
    df_out = df.copy()
    valid_cols = [c for c in columns if c in df_out.columns]
    if not valid_cols:
        return df_out, None

    encoder_cls = get_transformer("encode", method)
    
    if method == "onehot":
        encoder = encoder_cls(sparse_output=False, handle_unknown="ignore")
        encoded_array = encoder.fit_transform(df_out[valid_cols])
        feature_names = encoder.get_feature_names_out(valid_cols)
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df_out.index)
        df_out = df_out.drop(columns=valid_cols).join(encoded_df)
    elif method == "ordinal":
        encoder = encoder_cls(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded_array = encoder.fit_transform(df_out[valid_cols])
        df_out[valid_cols] = encoded_array
    else:
        raise ValueError(f"Unsupported encoding method: {method}")

    return df_out, encoder


def apply_scaling(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard"
) -> Tuple[pd.DataFrame, Any]:
    """Applies numeric scaling on specified columns."""
    df_out = df.copy()
    valid_cols = [c for c in columns if c in df_out.columns]
    if not valid_cols:
        return df_out, None

    scaler_cls = get_transformer("scale", method)
    scaler = scaler_cls()
    transformed_vals = scaler.fit_transform(df_out[valid_cols])
    df_out[valid_cols] = transformed_vals
    return df_out, scaler


def apply_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataset into X_train, X_test, y_train, y_test."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe columns: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify_target = y if (stratify and len(np.unique(y)) > 1) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Fit-on-one-frame / transform-another helpers
# Used by the recipe runner for steps placed *after* the train/test split
# (fit on train only, transform test) and by inference code that must replay
# an already-fitted recipe on new raw rows.
# ---------------------------------------------------------------------------

def fit_step(
    df: pd.DataFrame,
    step_type: str,
    columns: List[str],
    params: Dict[str, Any],
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Fits one recipe step on ``df`` and returns (transformed_df, fitted_entry)."""
    if step_type == "drop_columns":
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df.copy(), None
        return df.drop(columns=valid_cols), {"step": "drop_columns", "transformer": None, "columns": valid_cols}

    if step_type == "impute":
        out, t = apply_imputation(df, columns, params.get("strategy", "mean"), params.get("fill_value"))
        entry = {"step": "impute", "transformer": t, "columns": columns}
    elif step_type == "encode":
        method = params.get("method", "onehot")
        out, t = apply_encoding(df, columns, method)
        entry = {"step": "encode", "transformer": t, "columns": columns, "method": method}
    elif step_type == "scale":
        method = params.get("method", "standard")
        out, t = apply_scaling(df, columns, method)
        entry = {"step": "scale", "transformer": t, "columns": columns, "method": method}
    else:
        raise ValueError(f"Unknown recipe step type: '{step_type}'")

    return out, (entry if t is not None else None)


def transform_with_entry(df: pd.DataFrame, entry: Dict[str, Any]) -> pd.DataFrame:
    """Applies an already-fitted recipe entry (from ``fit_step``) to new data."""
    step_type = entry.get("step")
    cols = [c for c in entry.get("columns", []) if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    transformer = entry.get("transformer")

    if step_type == "drop_columns":
        return out.drop(columns=cols)
    if transformer is None:
        return out

    if step_type in ("impute", "scale"):
        out[cols] = transformer.transform(out[cols])
    elif step_type == "encode":
        method = entry.get("method") or ("onehot" if type(transformer).__name__ == "OneHotEncoder" else "ordinal")
        if method == "onehot":
            encoded = transformer.transform(out[cols])
            names = transformer.get_feature_names_out(cols)
            out = out.drop(columns=cols).join(pd.DataFrame(encoded, columns=names, index=out.index))
        else:
            out[cols] = transformer.transform(out[cols])
    return out


def apply_fitted_transformers(
    df: pd.DataFrame,
    fitted_transformers: List[Dict[str, Any]],
    feature_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Replays a fitted recipe on raw rows, optionally aligning to the model's feature order."""
    out = df.copy()
    for entry in fitted_transformers or []:
        if isinstance(entry, dict):
            out = transform_with_entry(out, entry)
    if feature_order:
        out = out.reindex(columns=feature_order, fill_value=0)
    return out
