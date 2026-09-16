from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import RunDB
from ..schemas import ExecutionRequest, RunResponse
from ..services.ml_service import execute_pipeline_run, verify_run_reproducibility

router = APIRouter(prefix="/api/runs", tags=["Runs & Evaluation"])


@router.post("/execute", response_model=RunResponse)
def execute_run(payload: ExecutionRequest, db: Session = Depends(get_db)):
    """Triggers ML Engine pipeline execution, fits model, evaluates metrics, and logs run."""
    try:
        recipe_dicts = [step.model_dump() for step in payload.recipe]
        model_config_dict = payload.model_settings.model_dump()

        run_record = execute_pipeline_run(
            dataset_id=payload.dataset_id,
            task_type=payload.task_type,
            target_column=payload.target_column,
            recipe=recipe_dicts,
            model_config=model_config_dict,
            db=db
        )
        return run_record
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline execution failed: {str(e)}")


@router.get("", response_model=List[RunResponse])
def list_runs(db: Session = Depends(get_db)):
    """List execution run history."""
    return db.query(RunDB).order_by(RunDB.created_at.desc()).all()


@router.get("/compare")
def compare_runs(ids: str = Query(..., description="Comma-separated run IDs, e.g. '1,2,3'"), db: Session = Depends(get_db)):
    """Side-by-side comparison table of metrics across multiple runs."""
    try:
        run_ids = [int(i.strip()) for i in ids.split(",") if i.strip().isdigit()]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid run IDs format. Expected comma-separated integers.")

    runs = db.query(RunDB).filter(RunDB.id.in_(run_ids)).all()
    if not runs:
        raise HTTPException(status_code=404, detail="No runs found for provided IDs.")

    comparison_data = []
    for r in runs:
        comparison_data.append({
            "run_id": r.id,
            "algorithm": r.algorithm,
            "task_type": r.task_type,
            "train_time_sec": r.train_time_sec,
            "metrics": r.metrics_json,
            "is_reproducible": r.is_reproducible,
            "created_at": r.created_at.isoformat(),
        })

    return {"comparison": comparison_data}


@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: int, db: Session = Depends(get_db)):
    """Get single run execution details."""
    record = db.query(RunDB).filter(RunDB.id == run_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Run record not found.")
    return record


@router.post("/{run_id}/verify-reproducibility")
def verify_reproducibility_endpoint(run_id: int, db: Session = Depends(get_db)):
    """Verifies reproducibility by re-running the stored pipeline config."""
    try:
        res = verify_run_reproducibility(run_id, db)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reproducibility check failed: {str(e)}")
