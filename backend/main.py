import os
import sys
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import init_db, get_db
from .models import DatasetDB, RunDB, UserDB
from .routers import datasets, pipelines, runs, predict, auth
from .routers.auth import get_current_user_optional
from .services.dataset_service import process_and_save_dataset, load_dataset_dataframe
from .services.ml_service import (
    execute_pipeline_run, 
    verify_run_reproducibility,
    export_run_onnx,
    benchmark_run_edge,
)

# Initialize DB tables
init_db()

app = FastAPI(
    title="Interactive Model Studio API",
    description="Backend API and web host for visual machine learning platform and model orchestration",
    version="1.0.0"
)

# Configure CORS to permit localhost and any deployed domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Main Routers under /api
app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(pipelines.router)
app.include_router(runs.router)
app.include_router(predict.router)


# ---------------------------------------------------------------------------
# Backward-Compatibility Aliases (Matches frontend mock client endpoints)
# ---------------------------------------------------------------------------

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Interactive Model Studio backend is live!"
    }


@app.post("/upload")
async def compat_upload(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user_optional)
):
    """Compatibility alias for POST /upload."""
    return await datasets.upload_dataset(file=file, db=db, current_user=current_user)


@app.get("/dataset")
def compat_list_datasets(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user_optional)
):
    """Compatibility alias for GET /dataset."""
    return datasets.list_datasets(db=db, current_user=current_user)


@app.get("/diagnostics")
def compat_diagnostics(dataset_id: int = Query(...), db: Session = Depends(get_db)):
    """Compatibility alias for GET /diagnostics."""
    dataset = db.query(DatasetDB).filter(DatasetDB.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "col_count": dataset.col_count,
        "missing_summary": dataset.missing_counts,
        "dtypes": dataset.column_types,
    }


@app.post("/experiment/run")
def compat_run_experiment(
    payload: dict, 
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user_optional)
):
    """Compatibility alias for running experiment from legacy frontend payloads."""
    from .schemas import ExecutionRequest, RecipeStep, ModelConfig
    try:
        req = ExecutionRequest(
            dataset_id=int(payload.get("dataset_id", 1)),
            task_type=payload.get("task_type", "classification"),
            target_column=payload.get("target_column", "target"),
            recipe=[RecipeStep(**s) for s in payload.get("recipe", [])],
            model_config=ModelConfig(
                algorithm=payload.get("algorithm", payload.get("model_name", "RandomForestClassifier")),
                hyperparameters=payload.get("hyperparameters", {})
            )
        )
        return runs.execute_run(payload=req, db=db, current_user=current_user)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/experiment/list")
def compat_list_experiments(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user_optional)
):
    """Compatibility alias for GET /experiment/list."""
    return runs.list_runs(db=db, current_user=current_user)


@app.get("/experiment/{experiment_id}")
def compat_get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """Compatibility alias for GET /experiment/{id}."""
    return runs.get_run(run_id=experiment_id, db=db)


@app.post("/compare")
def compat_compare(payload: dict, db: Session = Depends(get_db)):
    """Compatibility alias for POST /compare."""
    exp_ids = payload.get("experiment_ids", [])
    if isinstance(exp_ids, list):
        ids_str = ",".join(str(i) for i in exp_ids)
    else:
        ids_str = str(exp_ids)
    return runs.compare_runs(ids=ids_str, db=db)


@app.get("/restore/{experiment_id}")
def compat_restore(experiment_id: int, db: Session = Depends(get_db)):
    """Compatibility alias for GET /restore/{id}."""
    return runs.verify_reproducibility_endpoint(run_id=experiment_id, db=db)


@app.get("/export/{experiment_id}")
@app.post("/export/{experiment_id}")
def compat_export(experiment_id: int, db: Session = Depends(get_db)):
    """Compatibility alias for exporting model to ONNX format."""
    return runs.export_onnx_endpoint(run_id=experiment_id, db=db)


# ---------------------------------------------------------------------------
# Single-Link SPA Static Files Serving (Serves Built React Frontend)
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
            "docs": "/docs",
            "note": "Frontend dist build not found. Run 'npm run build' inside frontend/ to enable unified single-link hosting."
        }
