"""
Data Loader & Large Dataset Optimization Module
Supports ingestion of all major tabular formats (CSV, TSV, Parquet, JSON),
handles all fundamental data types (numerical, categorical, boolean, datetime, text),
and provides automated memory downcasting to efficiently process large datasets.
"""

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Boolean string equivalents for robust coercion
BOOL_TRUE_VALUES = {"true", "1", "t", "yes", "y", "enable"}
BOOL_FALSE_VALUES = {"false", "0", "f", "no", "n", "disable"}


def detect_and_coerce_types(df: pd.DataFrame, max_cardinality_ratio: float = 0.5) -> pd.DataFrame:
    """
    Infers and coerces columns into appropriate semantic types:
    - Boolean: detects binary string columns representing boolean flags.
    - Datetime: detects and converts date/time string columns.
    - Categorical: converts low-cardinality string columns to 'category' dtype.
    - Numeric: ensures numeric strings are parsed to proper numeric dtypes.
    """
    df = df.copy()

    for col in df.columns:
        series = df[col]

        # 1. Skip already strongly typed columns
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        # 2. Check for Boolean values disguised as strings or mixed objects
        if series.dtype == "object":
            non_null = series.dropna().astype(str).str.strip().str.lower()
            if len(non_null) > 0:
                unique_vals = set(non_null.unique())
                if unique_vals.issubset(BOOL_TRUE_VALUES | BOOL_FALSE_VALUES):
                    # Coerce to nullable boolean
                    bool_map = {v: True for v in BOOL_TRUE_VALUES}
                    bool_map.update({v: False for v in BOOL_FALSE_VALUES})
                    df[col] = non_null.map(bool_map).astype("boolean")
                    continue

        # 3. Check for Datetime strings (e.g. '2024-01-15', '2023-08-01 14:30:00')
        if series.dtype == "object":
            sample = series.dropna().astype(str).head(50)
            if len(sample) > 0 and sample.str.contains(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").all():
                try:
                    df[col] = pd.to_datetime(series, errors="coerce")
                    continue
                except Exception:
                    pass

        # 4. Check for Numeric disguised as strings
        if series.dtype == "object":
            converted_num = pd.to_numeric(series, errors="coerce")
            # If at least 90% of non-null values converted cleanly to numbers
            if converted_num.notna().sum() > 0.9 * series.notna().sum():
                df[col] = converted_num
                continue

    return df


def optimize_memory(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Optimizes memory footprint of a DataFrame for large datasets:
    - Downcasts float64 -> float32
    - Downcasts integer types to smallest safe integer representation (int8/int16/int32/int64)
    - Converts repetitive string/object columns to category dtype
    
    Returns:
        optimized_df: Memory-efficient DataFrame
        report: Dict containing RAM usage before, after, and reduction percentage.
    """
    df = df.copy()
    start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

    for col in df.columns:
        col_type = df[col].dtype

        # Optimize floating point columns
        if pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")

        # Optimize integer columns
        elif pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")

        # Optimize object / string columns to categories if cardinality is reasonably low
        elif col_type == "object":
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_total > 0 and (num_unique / num_total < 0.5) and num_unique < 1000:
                df[col] = df[col].astype("category")

    end_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    percent_reduction = 100 * (start_mem - end_mem) / (start_mem if start_mem > 0 else 1.0)

    report = {
        "memory_before_mb": round(start_mem, 3),
        "memory_after_mb": round(end_mem, 3),
        "memory_saved_mb": round(start_mem - end_mem, 3),
        "percent_reduction": round(percent_reduction, 2),
    }

    if verbose:
        print(f"Memory optimization: {report['memory_before_mb']} MB -> {report['memory_after_mb']} MB "
              f"({report['percent_reduction']}% reduction)")

    return df, report


def load_dataset(
    file_path: Union[str, Path],
    optimize: bool = True,
    coerce_types: bool = True,
    **read_kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified loader for tabular datasets supporting all major file formats:
    - CSV (.csv), TSV (.tsv)
    - Parquet (.parquet, .pq)
    - JSON (.json)
    - Excel (.xlsx, .xls)

    Returns:
        df: Loaded pandas DataFrame
        metadata: Dict detailing rows, columns, memory usage, and column dtypes.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    ext = path.suffix.lower()

    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path, **read_kwargs)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", **read_kwargs)
    elif ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path, **read_kwargs)
    elif ext == ".json":
        df = pd.read_json(path, **read_kwargs)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, **read_kwargs)
    else:
        # Default fallback to CSV reader
        df = pd.read_csv(path, **read_kwargs)

    # 1. Type inference & coercion
    if coerce_types:
        df = detect_and_coerce_types(df)

    # 2. Large dataset memory downcasting
    mem_report = {}
    if optimize:
        df, mem_report = optimize_memory(df)

    # 3. Assemble metadata
    metadata = {
        "file_path": str(path),
        "format": ext,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "null_counts": df.isna().sum().to_dict(),
        "memory_optimization": mem_report,
    }

    return df, metadata


def load_dataset_chunked(
    file_path: Union[str, Path],
    chunksize: int = 10000,
    optimize: bool = True,
    **read_kwargs: Any,
) -> Generator[pd.DataFrame, None, None]:
    """
    Streams large CSV/TSV datasets in manageable chunks to handle multi-gigabyte files.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    chunks = pd.read_csv(path, sep=sep, chunksize=chunksize, **read_kwargs)

    for chunk in chunks:
        chunk = detect_and_coerce_types(chunk)
        if optimize:
            chunk, _ = optimize_memory(chunk)
        yield chunk


def get_dataset_type_summary(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorizes columns into semantic categories for pipeline configuration:
    - numeric: int, float
    - categorical: object, category, string
    - boolean: bool, boolean
    - datetime: datetime64
    """
    summary = {
        "numeric": [],
        "categorical": [],
        "boolean": [],
        "datetime": [],
        "other": [],
    }

    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            summary["boolean"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            summary["datetime"].append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            summary["numeric"].append(col)
        elif pd.api.types.is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            summary["categorical"].append(col)
        else:
            summary["other"].append(col)

    return summary
