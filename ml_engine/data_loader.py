"""
Multi-type Data Loader with Memory Optimization
Supports CSV, Parquet, Excel; automatically downcasts numeric columns
on large datasets to reduce RAM usage.
"""

import io
import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any, List


# Threshold: datasets with more than this many cells get auto-downcast
_DOWNCAST_THRESHOLD_CELLS = 100_000


def load_dataset(
    source: Union[str, bytes, io.BytesIO],
    file_format: Optional[str] = None,
    downcast: bool = True,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load a tabular dataset from file path or raw bytes.

    Args:
        source:      File path string, raw bytes, or BytesIO object.
        file_format: One of 'csv', 'parquet', 'excel'. Auto-detected from
                     file extension when source is a path and format is None.
        downcast:    If True, automatically downcast numeric dtypes on large
                     datasets to conserve memory.
        encoding:    Character encoding for CSV files.

    Returns:
        pandas DataFrame with inferred column types.
    """
    if file_format is None:
        file_format = _infer_format(source)

    if isinstance(source, bytes):
        source = io.BytesIO(source)

    df = _read_by_format(source, file_format, encoding)

    if downcast and df.size > _DOWNCAST_THRESHOLD_CELLS:
        df = downcast_dataframe(df)

    return df


def _infer_format(source: Union[str, bytes, io.BytesIO]) -> str:
    """Infer file format from source path extension."""
    if isinstance(source, str):
        lower = source.lower()
        if lower.endswith(".parquet"):
            return "parquet"
        if lower.endswith((".xls", ".xlsx")):
            return "excel"
    return "csv"


def _read_by_format(source, file_format: str, encoding: str) -> pd.DataFrame:
    """Dispatch read to the correct pandas reader."""
    if file_format == "parquet":
        return pd.read_parquet(source)
    if file_format == "excel":
        return pd.read_excel(source)
    # Default: CSV
    try:
        return pd.read_csv(source, encoding=encoding)
    except UnicodeDecodeError:
        if isinstance(source, io.BytesIO):
            source.seek(0)
        return pd.read_csv(source, encoding="latin-1")


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory footprint of a DataFrame by downcasting numeric columns.

    Integer columns → smallest int type that safely holds the range.
    Float columns   → float32 (preserves ~7 significant decimal digits).

    Returns a copy with reduced dtypes.
    """
    df_out = df.copy()

    for col in df_out.select_dtypes(include=["int64", "int32"]).columns:
        col_min = df_out[col].min()
        col_max = df_out[col].max()
        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
            df_out[col] = df_out[col].astype(np.int8)
        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
            df_out[col] = df_out[col].astype(np.int16)
        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df_out[col] = df_out[col].astype(np.int32)
        # else: keep int64

    for col in df_out.select_dtypes(include=["float64"]).columns:
        df_out[col] = df_out[col].astype(np.float32)

    return df_out


def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """Return per-column and total memory usage in MB."""
    per_col = (df.memory_usage(deep=True) / 1024 ** 2).to_dict()
    total_mb = sum(per_col.values())
    return {
        "total_mb": round(total_mb, 3),
        "per_column_mb": {k: round(v, 4) for k, v in per_col.items()},
    }


def infer_column_roles(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    id_patterns: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Heuristically classify each column as:
      'target', 'numeric', 'categorical', 'id', or 'text'.

    Args:
        df:          Input DataFrame.
        target_col:  Known target column name (marked as 'target').
        id_patterns: List of substrings that identify ID columns
                     (default: ['id', 'uuid', 'key', 'index']).

    Returns:
        Dict mapping column_name → role string.
    """
    if id_patterns is None:
        id_patterns = ["id", "uuid", "key", "index", "proxy"]

    roles: Dict[str, str] = {}
    for col in df.columns:
        if target_col and col == target_col:
            roles[col] = "target"
            continue

        lower = col.lower()
        if any(pat in lower for pat in id_patterns):
            roles[col] = "id"
            continue

        dtype = df[col].dtype
        nunique = df[col].nunique()

        if pd.api.types.is_numeric_dtype(dtype):
            roles[col] = "numeric"
        elif dtype == object:
            # Heuristic: very long average strings → 'text', else 'categorical'
            avg_len = df[col].dropna().astype(str).str.len().mean()
            roles[col] = "text" if avg_len > 50 else "categorical"
        else:
            roles[col] = "categorical"

    return roles
