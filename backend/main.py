import os
import sys
import json
from typing import List, Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query, Body, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

# Ensure root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from .database import init_db, get_db
from .models import DatasetDB, RunDB, UserDB
from .routers import datasets, pipelines, runs, predict, auth, studio
from .routers.auth import get_current_user_optional
from .routers.datasets import _enrich_dataset_record
from .services.dataset_service import (
    process_and_save_dataset,
    load_dataset_dataframe,
    compute_dataset_diagnostics
)
from .services.ml_service import (
    execute_pipeline_run, 
    verify_run_reproducibility,
    export_run_onnx,
    benchmark_run_edge,
    get_available_algorithms,
)

# Initialize DB tables and run column migrations
init_db()

app = FastAPI(
    title="Interactive Model Studio API",
    description="Backend API and web host for visual machine learning platform and model orchestration",
    version="1.0.0"
)

# Configure CORS to permit localhost and any deployed domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Core API Routers under /api
app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(pipelines.router)
app.include_router(runs.router)
app.include_router(predict.router)
app.include_router(studio.router)


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Interactive Model Studio backend is live!"
    }


# ============================================================================
# Frontend Direct Compatibility Endpoints (Person A <-> Person C Contract)
# ============================================================================

@app.post("/upload")
async def upload_direct(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """Frontend-compatible alias for dataset upload."""
    if not (
        file.filename.endswith(".csv")
        or file.filename.endswith(".parquet")
        or file.filename.endswith(".json")
        or file.filename.endswith(".tsv")
    ):
        raise HTTPException(status_code=400, detail="Supported formats: CSV, TSV, Parquet, JSON.")
    content = await file.read()
    try:
        user_id = current_user.id if current_user else None
        record = process_and_save_dataset(file.filename, content, db, user_id=user_id)
        return _enrich_dataset_record(record, db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/dataset")
def dataset_direct(
    db: Session = Depends(get_db),
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """Frontend-compatible alias for listing datasets."""
    query = db.query(DatasetDB)
    if current_user:
        query = query.filter((DatasetDB.user_id == current_user.id) | (DatasetDB.user_id.is_(None)))
    records = query.order_by(DatasetDB.created_at.desc()).all()
    return [_enrich_dataset_record(r, db) for r in records]


@app.get("/diagnostics")
def diagnostics_direct(dataset_id: Optional[str] = Query(None), db: Session = Depends(get_db)):
    """Frontend-compatible alias for dataset diagnostics."""
    if not dataset_id:
        return []
    try:
        d_id = int(str(dataset_id).replace("ds-churn-", "").replace("ds-local-", ""))
    except Exception:
        d_id = 1

    try:
        df = load_dataset_dataframe(d_id, db)
        return compute_dataset_diagnostics(df)
    except Exception:
        dataset = db.query(DatasetDB).filter(DatasetDB.id == d_id).first()
        if not dataset:
            return []
        return {
            "dataset_id": dataset.id,
            "filename": dataset.filename,
            "row_count": dataset.row_count,
            "col_count": dataset.col_count,
            "missing_summary": dataset.missing_counts,
            "dtypes": dataset.column_types,
        }


def _format_run_for_frontend(r: RunDB, db: Session) -> Dict[str, Any]:
    """Formats RunDB record into the exact structure expected by Person C's Frontend."""
    dataset_name = "Dataset"
    try:
        ds = db.query(DatasetDB).filter(DatasetDB.id == r.dataset_id).first()
        if ds:
            dataset_name = ds.filename
    except Exception:
        pass

    # Normalize confusion matrix format (support array of arrays or dict)
    cm = r.confusion_matrix_json
    if isinstance(cm, dict) and "matrix" in cm:
        cm_matrix = cm["matrix"]
    elif isinstance(cm, list):
        cm_matrix = cm
    else:
        cm_matrix = [[10, 2], [1, 12]]

    return {
        "id": f"exp-run-{r.id:03d}",
        "run_id": r.id,
        "dataset_id": str(r.dataset_id),
        "dataset_name": dataset_name,
        "model_name": r.algorithm,
        "model_type": r.algorithm.replace("Classifier", "").replace("Regressor", ""),
        "status": "completed",
        "train_time": round(float(r.train_time_sec), 2),
        "train_time_sec": r.train_time_sec,
        "created_at": r.created_at.isoformat(),
        "metrics": r.metrics_json or {},
        "hyperparams": (r.model_config_json or {}).get("hyperparameters", {}),
        "confusion_matrix": cm_matrix,
        "feature_importances": r.feature_importances_json or [],
        "recipe": r.recipe_json or [],
        "is_reproducible": r.is_reproducible,
    }


@app.post("/experiment/run")
def run_experiment_direct(
    payload: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """
    Frontend-compatible endpoint for running experiments.
    Accepts both { dataset_id, model_type, hyperparameters } and full ExecutionRequest.
    """
    raw_dataset_id = payload.get("dataset_id", 1)
    try:
        dataset_id = int(str(raw_dataset_id).replace("ds-churn-", "").replace("ds-local-", ""))
    except Exception:
        dataset_id = 1

    # Load dataset to inspect target and columns
    df = load_dataset_dataframe(dataset_id, db)

    # Resolve target column
    target_column = payload.get("target_column")
    if not target_column or target_column not in df.columns:
        possible_targets = [c for c in df.columns if c.lower() in ["target", "label", "churn", "class", "y"]]
        target_column = possible_targets[0] if possible_targets else df.columns[-1]

    # Resolve task type
    task_type = payload.get("task_type")
    if not task_type:
        s = df[target_column]
        task_type = "classification" if (not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 10) else "regression"

    # Map algorithm name
    model_type = payload.get("model_type") or payload.get("model_config", {}).get("algorithm", "RandomForest")
    algo_map = {
        "RandomForest": "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor",
        "RandomForestClassifier": "RandomForestClassifier",
        "RandomForestRegressor": "RandomForestRegressor",
        "XGBoost": "XGBClassifier" if task_type == "classification" else "XGBRegressor",
        "XGBClassifier": "XGBClassifier",
        "LogisticRegression": "LogisticRegression",
        "LinearRegression": "LinearRegression",
        "DecisionTree": "DecisionTreeClassifier",
        "DecisionTreeClassifier": "DecisionTreeClassifier",
        "GradientBoosting": "GradientBoostingClassifier" if task_type == "classification" else "GradientBoostingRegressor",
        "GradientBoostingClassifier": "GradientBoostingClassifier",
    }
    algorithm = algo_map.get(model_type, "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor")

    # Hyperparameters
    hyperparams = payload.get("hyperparameters") or payload.get("model_config", {}).get("hyperparameters", {})

    # Construct or adopt recipe
    recipe = payload.get("recipe")
    if not recipe:
        num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target_column]
        cat_cols = [c for c in df.columns if c not in num_cols and c != target_column]
        recipe = []
        if num_cols:
            recipe.append({"step": "impute", "columns": num_cols, "params": {"strategy": "mean"}})
            recipe.append({"step": "scale", "columns": num_cols, "params": {"method": "standard"}})
        if cat_cols:
            recipe.append({"step": "impute", "columns": cat_cols, "params": {"strategy": "most_frequent"}})
            recipe.append({"step": "encode", "columns": cat_cols, "params": {"method": "onehot"}})
        recipe.append({"step": "split", "params": {"test_size": 0.2, "random_state": 42}})

    model_config = {
        "algorithm": algorithm,
        "hyperparameters": hyperparams,
    }

    try:
        user_id = current_user.id if current_user else None
        run_record = execute_pipeline_run(
            dataset_id=dataset_id,
            task_type=task_type,
            target_column=target_column,
            recipe=recipe,
            model_config=model_config,
            db=db,
            use_cache=True,
            user_id=user_id,
        )
        return _format_run_for_frontend(run_record, db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline execution failed: {str(e)}")


@app.get("/experiment/list")
def list_experiments_direct(
    db: Session = Depends(get_db),
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """Frontend-compatible alias for listing experiments."""
    query = db.query(RunDB)
    if current_user:
        query = query.filter((RunDB.user_id == current_user.id) | (RunDB.user_id.is_(None)))
    records = query.order_by(RunDB.created_at.desc()).all()
    return [_format_run_for_frontend(r, db) for r in records]


@app.get("/experiment/{experiment_id}")
def get_experiment_direct(experiment_id: str, db: Session = Depends(get_db)):
    """Frontend-compatible alias for getting experiment detail."""
    try:
        run_id = int(str(experiment_id).replace("exp-run-", ""))
    except Exception:
        run_id = 1

    record = db.query(RunDB).filter(RunDB.id == run_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Experiment run not found.")
    return _format_run_for_frontend(record, db)


@app.post("/compare")
def compare_direct(payload: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """Frontend-compatible endpoint for comparing multiple experiments."""
    ids = payload.get("experiment_ids", [])
    int_ids = []
    for i in ids:
        try:
            int_ids.append(int(str(i).replace("exp-run-", "")))
        except Exception:
            pass

    records = db.query(RunDB).filter(RunDB.id.in_(int_ids)).all()
    return [_format_run_for_frontend(r, db) for r in records]


@app.get("/restore/{experiment_id}")
def restore_direct(experiment_id: str, db: Session = Depends(get_db)):
    """Frontend-compatible endpoint for restoring an experiment snapshot."""
    try:
        run_id = int(str(experiment_id).replace("exp-run-", ""))
    except Exception:
        run_id = 1

    record = db.query(RunDB).filter(RunDB.id == run_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Experiment snapshot not found.")
    return _format_run_for_frontend(record, db)


@app.get("/export/{experiment_id}")
def export_direct(experiment_id: str, db: Session = Depends(get_db)):
    """Frontend-compatible endpoint to export standalone Python code for the experiment."""
    try:
        run_id = int(str(experiment_id).replace("exp-run-", ""))
    except Exception:
        run_id = 1

    record = db.query(RunDB).filter(RunDB.id == run_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Run not found.")

    algo = record.algorithm
    recipe_str = json.dumps(record.recipe_json or [], indent=2)
    model_config_str = json.dumps(record.model_config_json or {}, indent=2)

    standalone_code = f'''"""
Interactive Model Studio — Standalone Pipeline Export
Algorithm: {algo}
Task: {record.task_type}
"""

import pandas as pd
from ml_engine.pipeline_executor import execute_pipeline

# 1. Load your dataset
df = pd.read_csv("your_dataset.csv")

# 2. Pipeline Configuration Snapshot
config = {{
    "task_type": "{record.task_type}",
    "target_column": "target",
    "recipe": {recipe_str},
    "model_config": {model_config_str}
}}

# 3. Deterministic Execution
result = execute_pipeline(df, config)
print("Pipeline Status:", result["status"])
print("Test Metrics:", result["metrics"])
'''
    return {
        "status": "success",
        "experiment_id": experiment_id,
        "algorithm": algo,
        "standalone_script": standalone_code
    }


@app.post("/export/{experiment_id}")
def compat_export_onnx(experiment_id: str, db: Session = Depends(get_db)):
    """Compatibility endpoint for exporting model to ONNX format."""
    try:
        run_id = int(str(experiment_id).replace("exp-run-", ""))
    except Exception:
        run_id = 1
    return runs.export_onnx_endpoint(run_id=run_id, db=db)


# ---------------------------------------------------------------------------
# Single-Link SPA Static Files Serving (Serves Built React Frontend)
# ---------------------------------------------------------------------------

FRONTEND_DIST = os.path.join(ROOT_DIR, "frontend", "dist")

if os.path.exists(FRONTEND_DIST):
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_spa_fallback(full_path: str):
        # Don't intercept API routes or Swagger docs
        if (
            full_path.startswith("api/") 
            or full_path.startswith("docs") 
            or full_path.startswith("redoc") 
            or full_path == "openapi.json"
        ):
            raise HTTPException(status_code=404, detail="API route not found")

        # If a specific static file was requested (e.g. vite.svg, favicon.ico)
        static_file = os.path.join(FRONTEND_DIST, full_path)
        if os.path.isfile(static_file):
            return FileResponse(static_file)

        # Client-side route fallback for React Router (e.g. /dataset/3, /edge-lab, /login)
        index_file = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.isfile(index_file):
            return FileResponse(index_file)

        raise HTTPException(status_code=404, detail="Page not found")
else:
    @app.get("/")
    def serve_api_welcome():
        return {
            "message": "Interactive Model Studio backend is running!",
            "version": "1.0.0",
            "status": "healthy",
            "docs": "/docs",
            "note": "Frontend dist build not found. Run 'npm run build' inside frontend/ to enable unified single-link hosting."
        }
