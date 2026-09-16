"""
Preprocessing Module
Provides modular, reproducible data transformation operations:
- Imputation (mean, median, most_frequent, constant)
- Encoding (one-hot, ordinal)
- Scaling (standard, min-max, robust)
- Splitting (train/test split with stratification and reproducible seed)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_engine.registry import get_preprocessor_class


def apply_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits a DataFrame into X_train, X_test, y_train, y_test.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame columns: {list(df.columns)}")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    stratify_target = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_target,
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def apply_imputation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    fill_value: Optional[Any] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Fits an imputer on X_train[columns] and transforms both X_train and X_test.
    """
    valid_cols = [col for col in columns if col in X_train.columns]
    if not valid_cols:
        return X_train, X_test, None

    imputer_cls = get_preprocessor_class("impute", strategy)
    imputer = imputer_cls(strategy=strategy, fill_value=fill_value)

    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()

    # Fit on train, transform both
    X_train_transformed[valid_cols] = imputer.fit_transform(X_train[valid_cols])
    X_test_transformed[valid_cols] = imputer.transform(X_test[valid_cols])

    return X_train_transformed, X_test_transformed, imputer


def apply_scaling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Fits a scaler on X_train[columns] and transforms both X_train and X_test.
    """
    valid_cols = [col for col in columns if col in X_train.columns]
    if not valid_cols:
        return X_train, X_test, None

    scaler_cls = get_preprocessor_class("scale", method)
    scaler = scaler_cls()

    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()

    X_train_transformed[valid_cols] = scaler.fit_transform(X_train[valid_cols])
    X_test_transformed[valid_cols] = scaler.transform(X_test[valid_cols])

    return X_train_transformed, X_test_transformed, scaler


def apply_encoding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    method: str = "onehot",
    drop: Optional[str] = "first",
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Encodes categorical features using OneHotEncoder or OrdinalEncoder.
    Returns transformed train & test dataframes with proper column headers.
    """
    valid_cols = [col for col in columns if col in X_train.columns]
    if not valid_cols:
        return X_train, X_test, None

    encoder_cls = get_preprocessor_class("encode", method)

    if method == "onehot":
        encoder = encoder_cls(
            drop=drop if drop in ["first", "if_binary"] else None,
            sparse_output=False,
            handle_unknown="ignore" if drop is None else "error",
        )
        
        # Fit on train
        encoded_train_arr = encoder.fit_transform(X_train[valid_cols].astype(str))
        encoded_test_arr = encoder.transform(X_test[valid_cols].astype(str))
        
        feature_names = encoder.get_feature_names_out(valid_cols)
        
        df_encoded_train = pd.DataFrame(encoded_train_arr, columns=feature_names, index=X_train.index)
        df_encoded_test = pd.DataFrame(encoded_test_arr, columns=feature_names, index=X_test.index)
        
        # Drop original columns and concatenate new encoded columns
        X_train_transformed = pd.concat([X_train.drop(columns=valid_cols), df_encoded_train], axis=1)
        X_test_transformed = pd.concat([X_test.drop(columns=valid_cols), df_encoded_test], axis=1)
        
    elif method == "ordinal":
        encoder = encoder_cls(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()
        
        X_train_transformed[valid_cols] = encoder.fit_transform(X_train[valid_cols].astype(str))
        X_test_transformed[valid_cols] = encoder.transform(X_test[valid_cols].astype(str))
    else:
        raise ValueError(f"Unsupported encoding method: {method}")

    return X_train_transformed, X_test_transformed, encoder
