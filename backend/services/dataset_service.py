import os
import uuid
import pandas as pd
from typing import Tuple, Dict, Any
from sqlalchemy.orm import Session
from ..models import DatasetDB

STORAGE_DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "datasets")
os.makedirs(STORAGE_DATASETS_DIR, exist_ok=True)


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
