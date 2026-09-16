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
