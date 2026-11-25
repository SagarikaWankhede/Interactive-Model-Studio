import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import RunDB, UserDB
from ..schemas import ExecutionRequest, RunResponse
from ..services.ml_service import (
    execute_pipeline_run, 
    verify_run_reproducibility,
    export_run_onnx,
    benchmark_run_edge,
)
from .auth import get_current_user_optional

router = APIRouter(prefix="/api/runs", tags=["Runs & Evaluation"])


@router.post("/execute", response_model=RunResponse)
def execute_run(
    payload: ExecutionRequest, 
    db: Session = Depends(get_db),
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """Triggers ML Engine pipeline execution, fits model, evaluates metrics, and logs run with optional user ownership."""
    try:
        recipe_dicts = [step.model_dump() for step in payload.recipe]
        model_config_dict = payload.model_settings.model_dump()
        user_id = current_user.id if current_user else None

        run_record = execute_pipeline_run(
            dataset_id=payload.dataset_id,
            task_type=payload.task_type,
            target_column=payload.target_column,
            recipe=recipe_dicts,
            model_config=model_config_dict,
            db=db,
            user_id=user_id
        )
        return run_record
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline execution failed: {str(e)}")


@router.get("", response_model=List[RunResponse])
def list_runs(
    db: Session = Depends(get_db),
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """List execution run history. Filters by authenticated user if signed in."""
    if current_user:
        return db.query(RunDB).filter(
            (RunDB.user_id == current_user.id) | (RunDB.user_id.is_(None))
        ).order_by(RunDB.created_at.desc()).all()
    return db.query(RunDB).order_by(RunDB.created_at.desc()).all()


@router.get("/edge/models")
def list_edge_models(
    db: Session = Depends(get_db),
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """Lists all available trained models with their Edge deployment status and ONNX artifact availability."""
    query = db.query(RunDB)
    if current_user:
        query = query.filter((RunDB.user_id == current_user.id) | (RunDB.user_id.is_(None)))
    runs = query.order_by(RunDB.created_at.desc()).all()

    models_list = []
    for r in runs:
        has_onnx = bool(r.onnx_model_path and os.path.exists(r.onnx_model_path))
        models_list.append({
            "run_id": r.id,
            "algorithm": r.algorithm,
            "task_type": r.task_type,
            "train_time_sec": r.train_time_sec,
            "metrics": r.metrics_json,
            "has_onnx": has_onnx,
            "onnx_download_url": f"/api/runs/{r.id}/download-onnx" if has_onnx else None,
            "created_at": r.created_at.isoformat(),
        })

    return {"models": models_list}


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


@router.post("/{run_id}/export-onnx")
def export_onnx_endpoint(run_id: int, db: Session = Depends(get_db)):
    """Exports trained scikit-learn model to standard ONNX binary format for Edge Lab and in-browser deployment."""
    try:
        res = export_run_onnx(run_id, db)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ONNX export failed: {str(e)}")


@router.get("/{run_id}/download-onnx")
def download_onnx_endpoint(run_id: int, db: Session = Depends(get_db)):
    """Downloads the compiled .onnx model binary file."""
    run_record = db.query(RunDB).filter(RunDB.id == run_id).first()
    if not run_record:
        raise HTTPException(status_code=404, detail="Run not found.")

    # If ONNX has not been generated yet, generate it on demand!
    if not run_record.onnx_model_path or not os.path.exists(run_record.onnx_model_path):
        try:
            export_run_onnx(run_id, db)
            db.refresh(run_record)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to generate ONNX artifact: {str(e)}")

    if not run_record.onnx_model_path or not os.path.exists(run_record.onnx_model_path):
        raise HTTPException(status_code=404, detail="ONNX model file not found.")

    return FileResponse(
        path=run_record.onnx_model_path,
        media_type="application/octet-stream",
        filename=f"{run_record.algorithm}_run_{run_id}.onnx"
    )


@router.post("/{run_id}/edge-benchmark")
def edge_benchmark_endpoint(run_id: int, iterations: int = 100, db: Session = Depends(get_db)):
    """Runs simulated edge device benchmark measuring latency, throughput, and memory footprint."""
    try:
        res = benchmark_run_edge(run_id, db, iterations=iterations)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Edge benchmark failed: {str(e)}")

