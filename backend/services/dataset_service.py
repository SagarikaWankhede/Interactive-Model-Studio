import os
import uuid
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sqlalchemy.orm import Session
from ..models import DatasetDB

STORAGE_DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "datasets")
os.makedirs(STORAGE_DATASETS_DIR, exist_ok=True)


def get_dataset_schema(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Builds a rich schema dictionary array for frontend tables."""
    schema = []
    total_rows = len(df)
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        col_type = "numerical" if is_num else "categorical"
        null_count = int(s.isnull().sum())
        null_pct = round(float(null_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        cardinality = int(s.nunique(dropna=True))
        
        non_null_samples = s.dropna()
        sample = str(non_null_samples.iloc[0]) if not non_null_samples.empty else ""

        schema.append({
            "name": str(col),
            "type": col_type,
            "null_count": null_count,
            "null_pct": null_pct,
            "cardinality": cardinality,
            "sample": sample,
        })
    return schema


def compute_dataset_diagnostics(df: pd.DataFrame, target_column: Optional[str] = None) -> List[Dict[str, Any]]:
    """Computes automated data quality, imbalance, and leakage diagnostics."""
    diagnostics = []
    total_rows = len(df)
    if total_rows == 0:
        return diagnostics

    # Auto-detect target column if not provided
    if not target_column or target_column not in df.columns:
        possible_targets = [c for c in df.columns if c.lower() in ["target", "label", "churn", "class", "y"]]
        target_column = possible_targets[0] if possible_targets else df.columns[-1]

    # 1. High Missing Values Diagnostic
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = (null_count / total_rows) * 100
        if null_pct > 30.0:
            diagnostics.append({
                "id": f"diag-null-{col}",
                "type": "missing_values",
                "severity": "red" if null_pct > 60 else "yellow",
                "title": f"High Missing Values in '{col}' ({null_pct:.1f}%)",
                "column": col,
                "description": f"{null_count} out of {total_rows} rows ({null_pct:.1f}%) are null.",
                "suggested_fix": "Add an imputation step (median or constant) or drop column if non-informative."
            })

    # 2. High Cardinality Categorical Features
    for col in df.columns:
        if col == target_column:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            cardinality = int(df[col].nunique(dropna=True))
            if cardinality > 50 and cardinality < total_rows:
                diagnostics.append({
                    "id": f"diag-card-{col}",
                    "type": "high_cardinality",
                    "severity": "yellow",
                    "title": f"High Cardinality in '{col}' ({cardinality} categories)",
                    "column": col,
                    "description": f"Categorical column '{col}' has {cardinality} distinct string values.",
                    "suggested_fix": "Use target/ordinal encoding or group infrequent categories before one-hot encoding."
                })

    # 3. Class Imbalance on Target
    if target_column in df.columns:
        target_series = df[target_column].dropna()
        if not pd.api.types.is_numeric_dtype(target_series) or target_series.nunique() <= 10:
            value_counts = target_series.value_counts(normalize=True)
            if not value_counts.empty:
                max_prop = float(value_counts.iloc[0])
                if max_prop > 0.70:
                    diagnostics.append({
                        "id": "diag-imbalance-target",
                        "type": "imbalance",
                        "severity": "red" if max_prop > 0.85 else "yellow",
                        "title": f"Class Imbalance in Target '{target_column}' ({max_prop*100:.1f}%)",
                        "column": target_column,
                        "description": f"The dominant class represents {max_prop*100:.1f}% of observed records.",
                        "suggested_fix": "Enable stratification in train/test split or use balanced class weights."
                    })

    # 4. Target Leakage (High correlation with target)
    if target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target_column]
        for col in numeric_cols:
            try:
                corr = abs(df[col].corr(df[target_column]))
                if corr > 0.95:
                    diagnostics.append({
                        "id": f"diag-leak-{col}",
                        "type": "leakage",
                        "severity": "red",
                        "title": f"Potential Target Leakage in '{col}'",
                        "column": col,
                        "description": f"Feature '{col}' has an extreme {corr*100:.1f}% correlation with target '{target_column}'.",
                        "suggested_fix": f"Exclude '{col}' from the recipe features to prevent overfitting."
                    })
            except Exception:
                pass

    return diagnostics


def process_and_save_dataset(file_name: str, file_bytes: bytes, db: Session) -> DatasetDB:
    """Saves file to disk, computes schema statistics, and stores record in DB."""
    file_ext = os.path.splitext(file_name)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}_{file_name}"
    file_path = os.path.join(STORAGE_DATASETS_DIR, unique_filename)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    if file_ext == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    row_count, col_count = df.shape
    col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_counts = {col: int(count) for col, count in df.isnull().sum().items()}

    dataset_record = DatasetDB(
        filename=file_name,
        file_path=file_path,
        row_count=row_count,
        col_count=col_count,
        column_types=col_types,
        missing_counts=missing_counts,
    )

    db.add(dataset_record)
    db.commit()
    db.refresh(dataset_record)

    return dataset_record


def load_dataset_dataframe(dataset_id: int, db: Session) -> pd.DataFrame:
    """Loads pandas DataFrame for given dataset ID."""
    dataset_record = db.query(DatasetDB).filter(DatasetDB.id == dataset_id).first()
    if not dataset_record:
        raise ValueError(f"Dataset with ID {dataset_id} not found.")

    file_path = dataset_record.file_path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file at {file_path} missing.")

    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)
