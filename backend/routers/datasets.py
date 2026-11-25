from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import DatasetDB, UserDB
from ..schemas import DatasetResponse
from ..services.dataset_service import (
    process_and_save_dataset,
    load_dataset_dataframe,
    get_dataset_schema,
    compute_dataset_diagnostics,
)
from .auth import get_current_user_optional

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".parquet", ".json")


def _enrich_dataset_record(record: DatasetDB, db: Session) -> Dict[str, Any]:
    """Enriches dataset record with schema, diagnostics, and frontend-friendly keys."""
    data = {
        "id": record.id,
        "user_id": record.user_id,
        "filename": record.filename,
        "name": record.filename,
        "row_count": record.row_count,
        "col_count": record.col_count,
        "column_count": record.col_count,
        "column_types": record.column_types,
        "missing_counts": record.missing_counts,
        "created_at": record.created_at,
    }

    try:
        df = load_dataset_dataframe(record.id, db)
        data["schema"] = get_dataset_schema(df)
        data["diagnostics"] = compute_dataset_diagnostics(df)
        data["target_column"] = data["schema"][-1]["name"] if data["schema"] else None
    except Exception:
        data["schema"] = []
        data["diagnostics"] = []
        data["target_column"] = None

    return data


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """Upload a CSV, TSV, Parquet, or JSON dataset file and compute schema summary metadata."""
    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    content = await file.read()
    try:
        user_id = current_user.id if current_user else None
        record = process_and_save_dataset(file.filename, content, db, user_id=user_id)
        return _enrich_dataset_record(record, db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process dataset file: {str(e)}")


@router.get("", response_model=List[DatasetResponse])
def list_datasets(
    db: Session = Depends(get_db),
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """List uploaded datasets with schema and diagnostics. When authenticated, filters to user datasets + sample datasets."""
    if current_user:
        records = db.query(DatasetDB).filter(
            (DatasetDB.user_id == current_user.id) | (DatasetDB.user_id.is_(None))
        ).order_by(DatasetDB.created_at.desc()).all()
    else:
        records = db.query(DatasetDB).order_by(DatasetDB.created_at.desc()).all()
    return [_enrich_dataset_record(r, db) for r in records]


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Retrieve details for a specific dataset."""
    record = db.query(DatasetDB).filter(DatasetDB.id == dataset_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return _enrich_dataset_record(record, db)


@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: int, limit: int = 10, db: Session = Depends(get_db)):
    """Retrieve sample head rows for frontend data tables."""
    try:
        df = load_dataset_dataframe(dataset_id, db)
        return {
            "columns": list(df.columns),
            "total_rows": len(df),
            "sample_rows": df.head(limit).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{dataset_id}/diagnostics")
def get_diagnostics(dataset_id: int, db: Session = Depends(get_db)):
    """Retrieve data quality and leakage diagnostics for a dataset."""
    try:
        df = load_dataset_dataframe(dataset_id, db)
        diagnostics = compute_dataset_diagnostics(df)
        return {"diagnostics": diagnostics}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

