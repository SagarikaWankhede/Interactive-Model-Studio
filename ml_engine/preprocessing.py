"""
Preprocessing Module
Provides modular, reproducible data transformation operations:
- Imputation (mean, median, most_frequent, constant)
- Encoding (one-hot, ordinal)
- Scaling (standard, min-max, robust)
- Splitting (train/test split with stratification and reproducible seed)
- Power & Log transforms (yeo-johnson, box-cox, log1p, sqrt)
- Binning & Discretization (quantile, uniform)
- Polynomial & Interaction features
- Dimensionality reduction (PCA)
- Outlier clipping (IQR & quantile winsorization)
- Datetime feature decomposition
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

        encoded_train_arr = encoder.fit_transform(X_train[valid_cols].astype(str))
        encoded_test_arr = encoder.transform(X_test[valid_cols].astype(str))

        feature_names = encoder.get_feature_names_out(valid_cols)

        df_encoded_train = pd.DataFrame(encoded_train_arr, columns=feature_names, index=X_train.index)
        df_encoded_test = pd.DataFrame(encoded_test_arr, columns=feature_names, index=X_test.index)

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


def apply_power_transform(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    method: str = "yeo-johnson",
    standardize: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Applies non-linear power transformations (log1p, sqrt, yeo-johnson, box-cox).
    Helps normalize skewed distributions for linear models and distance-based algorithms.
    """
    valid_cols = [col for col in columns if col in X_train.columns]
    if not valid_cols:
        return X_train, X_test, None

    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()

    if method == "log1p":
        # log(1 + x) for non-negative values
        for c in valid_cols:
            min_val = min(X_train[c].min(), X_test[c].min())
            shift = abs(min_val) if min_val < 0 else 0
            X_train_transformed[c] = np.log1p(np.maximum(0, X_train[c] + shift))
            X_test_transformed[c] = np.log1p(np.maximum(0, X_test[c] + shift))
        return X_train_transformed, X_test_transformed, {"method": "log1p"}

    elif method == "sqrt":
        for c in valid_cols:
            min_val = min(X_train[c].min(), X_test[c].min())
            shift = abs(min_val) if min_val < 0 else 0
            X_train_transformed[c] = np.sqrt(np.maximum(0, X_train[c] + shift))
            X_test_transformed[c] = np.sqrt(np.maximum(0, X_test[c] + shift))
        return X_train_transformed, X_test_transformed, {"method": "sqrt"}

    else:
        # yeo-johnson or box-cox via PowerTransformer
        transformer_cls = get_preprocessor_class("power", method)
        transformer = transformer_cls(method=method, standardize=standardize)
        X_train_transformed[valid_cols] = transformer.fit_transform(X_train[valid_cols])
        X_test_transformed[valid_cols] = transformer.transform(X_test[valid_cols])
        return X_train_transformed, X_test_transformed, transformer


def apply_binning(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    n_bins: int = 5,
    strategy: str = "quantile",
    encode: str = "ordinal",
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Discretizes continuous features into intervals using KBinsDiscretizer.
    """
    valid_cols = [col for col in columns if col in X_train.columns]
    if not valid_cols:
        return X_train, X_test, None

    discretizer_cls = get_preprocessor_class("bin", strategy)
    discretizer = discretizer_cls(n_bins=n_bins, encode=encode, strategy=strategy, subsample=None)

    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()

    X_train_transformed[valid_cols] = discretizer.fit_transform(X_train[valid_cols])
    X_test_transformed[valid_cols] = discretizer.transform(X_test[valid_cols])

    return X_train_transformed, X_test_transformed, discretizer


def apply_polynomial_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Generates polynomial and interaction features (x1^2, x1*x2, etc.).
    """
    valid_cols = [col for col in columns if col in X_train.columns]
    if not valid_cols:
        return X_train, X_test, None

    poly_cls = get_preprocessor_class("polynomial", "features")
    poly = poly_cls(degree=degree, interaction_only=interaction_only, include_bias=include_bias)

    train_poly_arr = poly.fit_transform(X_train[valid_cols])
    test_poly_arr = poly.transform(X_test[valid_cols])

    feature_names = poly.get_feature_names_out(valid_cols)
    poly_train_df = pd.DataFrame(train_poly_arr, columns=feature_names, index=X_train.index)
    poly_test_df = pd.DataFrame(test_poly_arr, columns=feature_names, index=X_test.index)

    # Drop original columns and concatenate polynomial expansion
    X_train_transformed = pd.concat([X_train.drop(columns=valid_cols), poly_train_df], axis=1)
    X_test_transformed = pd.concat([X_test.drop(columns=valid_cols), poly_test_df], axis=1)

    return X_train_transformed, X_test_transformed, poly


def apply_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    n_components: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Performs Principal Component Analysis (PCA) for linear dimensionality reduction.
    """
    valid_cols = [col for col in columns if col in X_train.columns]
    if not valid_cols or len(valid_cols) < n_components:
        return X_train, X_test, None

    pca_cls = get_preprocessor_class("pca", "decomposition")
    pca = pca_cls(n_components=n_components, random_state=42)

    train_pca = pca.fit_transform(X_train[valid_cols])
    test_pca = pca.transform(X_test[valid_cols])

    pca_cols = [f"pca_{i+1}" for i in range(n_components)]
    df_pca_train = pd.DataFrame(train_pca, columns=pca_cols, index=X_train.index)
    df_pca_test = pd.DataFrame(test_pca, columns=pca_cols, index=X_test.index)

    X_train_transformed = pd.concat([X_train.drop(columns=valid_cols), df_pca_train], axis=1)
    X_test_transformed = pd.concat([X_test.drop(columns=valid_cols), df_pca_test], axis=1)

    return X_train_transformed, X_test_transformed, pca


def apply_outlier_clipping(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    factor: float = 1.5,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Clips extreme outliers based on IQR bounds or fixed quantiles calculated on X_train.
    Prevents outlier leverage from skewing model weights.
    """
    valid_cols = [col for col in columns if col in X_train.columns]
    if not valid_cols:
        return X_train, X_test, None

    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()
    thresholds = {}

    for c in valid_cols:
        series = X_train[c].dropna()
        if len(series) == 0:
            continue

        if method == "iqr":
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - factor * iqr
            upper_bound = q75 + factor * iqr
        else:  # quantile
            lower_bound = float(series.quantile(lower_quantile))
            upper_bound = float(series.quantile(upper_quantile))

        thresholds[c] = (lower_bound, upper_bound)
        X_train_transformed[c] = X_train_transformed[c].clip(lower=lower_bound, upper=upper_bound)
        X_test_transformed[c] = X_test_transformed[c].clip(lower=lower_bound, upper=upper_bound)

    return X_train_transformed, X_test_transformed, thresholds


def apply_datetime_decomposition(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    components: Tuple[str, ...] = ("year", "month", "day", "dayofweek", "is_weekend"),
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Extracts numerical features from datetime columns and drops original timestamp strings.
    """
    valid_cols = [col for col in columns if col in X_train.columns]
    if not valid_cols:
        return X_train, X_test, None

    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()

    for col in valid_cols:
        # Convert to datetime
        train_dt = pd.to_datetime(X_train[col], errors="coerce")
        test_dt = pd.to_datetime(X_test[col], errors="coerce")

        if "year" in components:
            X_train_transformed[f"{col}_year"] = train_dt.dt.year.fillna(-1).astype(int)
            X_test_transformed[f"{col}_year"] = test_dt.dt.year.fillna(-1).astype(int)
        if "month" in components:
            X_train_transformed[f"{col}_month"] = train_dt.dt.month.fillna(-1).astype(int)
            X_test_transformed[f"{col}_month"] = test_dt.dt.month.fillna(-1).astype(int)
        if "day" in components:
            X_train_transformed[f"{col}_day"] = train_dt.dt.day.fillna(-1).astype(int)
            X_test_transformed[f"{col}_day"] = test_dt.dt.day.fillna(-1).astype(int)
        if "dayofweek" in components:
            X_train_transformed[f"{col}_dayofweek"] = train_dt.dt.dayofweek.fillna(-1).astype(int)
            X_test_transformed[f"{col}_dayofweek"] = test_dt.dt.dayofweek.fillna(-1).astype(int)
        if "is_weekend" in components:
            X_train_transformed[f"{col}_is_weekend"] = (train_dt.dt.dayofweek >= 5).astype(int)
            X_test_transformed[f"{col}_is_weekend"] = (test_dt.dt.dayofweek >= 5).astype(int)

        # Drop original raw datetime column
        X_train_transformed = X_train_transformed.drop(columns=[col])
        X_test_transformed = X_test_transformed.drop(columns=[col])

    return X_train_transformed, X_test_transformed, {"components": components}
