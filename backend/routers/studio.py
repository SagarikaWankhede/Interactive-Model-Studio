"""
REST endpoints for the visual pipeline editor, edge/browser deployment,
the Data Doctor and the active-learning edge-case queue.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..services import studio_service as svc
from ..services.studio_service import NotFound

router = APIRouter(prefix="/api", tags=["Studio: Graph, Edge Deploy, Data Doctor, Active Learning"])


def _handle(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---- 1. Visual graph --------------------------------------------------------

@router.post("/graph/compile")
def compile_graph(graph: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """Validate a node graph and return the engine config + generated Python."""
    return _handle(svc.compile_graph_request, graph, db)


@router.post("/graph/run")
def run_graph(graph: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """Compile the graph and train it. Returns 422 with the compile errors if invalid."""
    try:
        result = svc.run_graph(graph, db)
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training failed: {e}")
    if result["status"] == "invalid":
        raise HTTPException(status_code=422, detail={"message": "Fix the pipeline errors first.", "errors": result["compiled"]["errors"]})
    return result


@router.get("/graph/runs/{run_id}")
def get_run_graph(run_id: int, db: Session = Depends(get_db)):
    return _handle(svc.get_run_graph, run_id, db)


# ---- 2. Edge / browser deployment ------------------------------------------

@router.post("/runs/{run_id}/onnx")
def export_onnx(run_id: int, quantize: bool = Query(False), db: Session = Depends(get_db)):
    """Convert a trained run to ONNX (+ optional INT8 quantization) and verify parity."""
    result = _handle(svc.export_run_for_browser, run_id, quantize, db)
    if result.get("status") != "success":
        raise HTTPException(status_code=400, detail=result.get("message") or "ONNX export failed.")
    return result


@router.get("/runs/{run_id}/onnx")
def get_onnx_export(run_id: int):
    return _handle(svc.get_browser_export, run_id)


@router.get("/runs/{run_id}/onnx/model")
def download_onnx(run_id: int, variant: Optional[str] = Query(None, description="fp32 | int8 | auto")):
    path = _handle(svc.onnx_file_path, run_id, variant)
    suffix = ".int8.onnx" if path.endswith(".int8.onnx") else ".onnx"
    return FileResponse(path, media_type="application/octet-stream", filename=f"model_run_{run_id}{suffix}")


@router.get("/runs/{run_id}/onnx/spec")
def download_spec(run_id: int):
    path = _handle(svc.onnx_file_path, run_id, "spec")
    return FileResponse(path, media_type="application/json", filename=f"preprocessing_run_{run_id}.json")


# ---- 3a. Data Doctor --------------------------------------------------------

@router.get("/datasets/{dataset_id}/report")
def dataset_report(dataset_id: int, target: Optional[str] = Query(None), db: Session = Depends(get_db)):
    return _handle(svc.dataset_report, dataset_id, target, db)


# ---- 3b. Active learning ----------------------------------------------------

@router.post("/edge-cases")
def add_edge_case(payload: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    return _handle(svc.add_edge_case, payload, db)


@router.get("/edge-cases")
def list_edge_cases(run_id: Optional[int] = Query(None), status: Optional[str] = Query(None), db: Session = Depends(get_db)):
    return svc.list_edge_cases(db, run_id=run_id, status=status)


@router.post("/edge-cases/{case_id}/label")
def label_edge_case(case_id: int, payload: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    return _handle(svc.label_edge_case, case_id, payload.get("label"), db)


@router.post("/edge-cases/{case_id}/dismiss")
def dismiss_edge_case(case_id: int, db: Session = Depends(get_db)):
    return _handle(svc.label_edge_case, case_id, None, db, dismiss=True)


@router.get("/runs/{run_id}/active-learning")
def active_learning_summary(run_id: int, db: Session = Depends(get_db)):
    return _handle(svc.run_label_options, run_id, db)


@router.post("/runs/{run_id}/scan")
def scan_run(run_id: int, threshold: float = Query(0.6, ge=0.0, le=1.0), limit: int = Query(25, ge=1, le=500), db: Session = Depends(get_db)):
    return _handle(svc.scan_run, run_id, threshold, limit, db)


@router.post("/runs/{run_id}/retrain")
def retrain(run_id: int, db: Session = Depends(get_db)):
    try:
        return svc.retrain_from_labels(run_id, db)
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Retraining failed: {e}")
