"""
Services for the three "Studio" features:

1. Visual graph  -> compile / train / export Python          (compile_graph_request, run_graph)
2. Edge deploy   -> ONNX + preprocessing spec + quantization (export_run_for_browser)
3. Data Doctor + active learning                             (dataset_report, scan_run, retrain_from_labels, ...)
"""

import datetime
import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ml_engine.data_doctor import analyze_dataset
from ml_engine.export_onnx import export_browser_bundle
from ml_engine.graph_compiler import compile_graph
from ml_engine.preprocessing import apply_fitted_transformers

from ..models import DatasetDB, EdgeCaseDB, RunDB
from .dataset_service import load_dataset_dataframe, process_and_save_dataset
from .ml_service import execute_pipeline_run

STORAGE_ONNX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "onnx")
os.makedirs(STORAGE_ONNX_DIR, exist_ok=True)


class NotFound(Exception):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_id(raw: Any) -> Optional[int]:
    if raw in (None, ""):
        return None
    m = re.search(r"(\d+)$", str(raw))
    return int(m.group(1)) if m else None


def _json_value(v: Any) -> Any:
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def _get_run(run_id: int, db: Session) -> RunDB:
    run = db.query(RunDB).filter(RunDB.id == run_id).first()
    if not run:
        raise NotFound(f"Run {run_id} not found.")
    return run


def _load_artifact(run: RunDB) -> Dict[str, Any]:
    if not run.model_artifact_path or not os.path.exists(run.model_artifact_path):
        raise NotFound("The trained model file for this run is missing on disk. Re-train the pipeline.")
    return joblib.load(run.model_artifact_path)


def format_run(run: RunDB, db: Session) -> Dict[str, Any]:
    ds = db.query(DatasetDB).filter(DatasetDB.id == run.dataset_id).first()
    target = None
    try:
        target = (_load_artifact(run).get("target_column"))
    except Exception:
        pass
    return {
        "id": f"exp-run-{run.id:03d}",
        "run_id": run.id,
        "dataset_id": run.dataset_id,
        "dataset_name": ds.filename if ds else None,
        "algorithm": run.algorithm,
        "task_type": run.task_type,
        "target_column": target,
        "metrics": run.metrics_json or {},
        "confusion_matrix": run.confusion_matrix_json,
        "feature_importances": run.feature_importances_json or [],
        "train_time_sec": run.train_time_sec,
        "has_graph": bool(run.graph_json),
        "parent_run_id": run.parent_run_id,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


# ---------------------------------------------------------------------------
# 1. Visual graph
# ---------------------------------------------------------------------------

def _graph_dataset(graph: Dict[str, Any], db: Session):
    ds_nodes = [n for n in graph.get("nodes", []) if (n.get("data") or {}).get("kind") == "dataset"]
    if not ds_nodes:
        return None, None, None
    ds_id = _parse_id(((ds_nodes[0].get("data") or {}).get("params") or {}).get("dataset_id"))
    if ds_id is None:
        return None, None, None
    record = db.query(DatasetDB).filter(DatasetDB.id == ds_id).first()
    if not record:
        return None, None, None
    return ds_id, record, load_dataset_dataframe(ds_id, db)


def compile_graph_request(graph: Dict[str, Any], db: Session) -> Dict[str, Any]:
    ds_id, record, df = _graph_dataset(graph, db)
    result = compile_graph(graph, df=df, dataset_filename=record.filename if record else None)
    has_ds_node = any((n.get("data") or {}).get("kind") == "dataset" for n in graph.get("nodes", []))
    if has_ds_node and ds_id is None:
        result["ok"] = False
        if not any("Choose a dataset" in e["message"] for e in result["errors"]):
            result["errors"].insert(0, {"node_id": None, "message": "Choose a dataset that exists on the server."})
    result["dataset_id"] = ds_id
    return result


def run_graph(graph: Dict[str, Any], db: Session) -> Dict[str, Any]:
    compiled = compile_graph_request(graph, db)
    if not compiled["ok"]:
        return {"status": "invalid", "compiled": compiled}
    cfg = compiled["config"]
    run = execute_pipeline_run(
        dataset_id=compiled["dataset_id"],
        task_type=cfg["task_type"],
        target_column=cfg["target_column"],
        recipe=cfg["recipe"],
        model_config=cfg["model_config"],
        db=db,
        use_cache=True,
    )
    run.graph_json = graph
    run.generated_code = compiled["code"]
    db.commit()
    db.refresh(run)
    return {"status": "success", "run": format_run(run, db), "compiled": compiled}


def get_run_graph(run_id: int, db: Session) -> Dict[str, Any]:
    run = _get_run(run_id, db)
    if not run.graph_json:
        raise NotFound("This run was not created from the visual editor.")
    return {"run": format_run(run, db), "graph": run.graph_json, "code": run.generated_code}


# ---------------------------------------------------------------------------
# 2. Edge / browser deployment
# ---------------------------------------------------------------------------

def _onnx_paths(run_id: int) -> Dict[str, str]:
    base = os.path.join(STORAGE_ONNX_DIR, f"run_{run_id}")
    return {"fp32": base + ".onnx", "int8": base + ".int8.onnx", "spec": base + ".spec.json", "report": base + ".report.json"}


def export_run_for_browser(run_id: int, quantize: bool, db: Session) -> Dict[str, Any]:
    run = _get_run(run_id, db)
    artifact = _load_artifact(run)
    raw_df = load_dataset_dataframe(run.dataset_id, db)
    bundle = export_browser_bundle(artifact, raw_df, quantize=quantize)
    if bundle.get("status") != "success":
        return {"status": bundle.get("status", "error"), "message": bundle.get("message")}

    paths = _onnx_paths(run_id)
    with open(paths["fp32"], "wb") as f:
        f.write(bundle["onnx_bytes"])
    if bundle.get("quantized_bytes"):
        with open(paths["int8"], "wb") as f:
            f.write(bundle["quantized_bytes"])
    elif os.path.exists(paths["int8"]):
        os.remove(paths["int8"])
    with open(paths["spec"], "w", encoding="utf-8") as f:
        json.dump(bundle["spec"], f)

    report = {
        "status": "success",
        "run_id": run_id,
        "algorithm": run.algorithm,
        "size_bytes": bundle["size_bytes"],
        "gzip_bytes": bundle["gzip_bytes"],
        "quantization": bundle["quantization"],
        "parity": bundle.get("parity"),
        "serving": "int8" if bundle.get("quantized_bytes") else "fp32",
        "spec": bundle["spec"],
    }
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in report.items() if k != "spec"}, f)
    return report


def get_browser_export(run_id: int) -> Dict[str, Any]:
    paths = _onnx_paths(run_id)
    if not (os.path.exists(paths["spec"]) and os.path.exists(paths["report"])):
        raise NotFound("This run has not been converted to ONNX yet.")
    with open(paths["report"], encoding="utf-8") as f:
        report = json.load(f)
    with open(paths["spec"], encoding="utf-8") as f:
        report["spec"] = json.load(f)
    return report


def onnx_file_path(run_id: int, variant: Optional[str] = None) -> str:
    paths = _onnx_paths(run_id)
    if variant in (None, "", "auto"):
        variant = "int8" if os.path.exists(paths["int8"]) else "fp32"
    path = paths.get(variant)
    if not path or not os.path.exists(path):
        raise NotFound("ONNX file not found. Convert the run first.")
    return path


# ---------------------------------------------------------------------------
# 3a. Data Doctor
# ---------------------------------------------------------------------------

def dataset_report(dataset_id: int, target: Optional[str], db: Session) -> Dict[str, Any]:
    record = db.query(DatasetDB).filter(DatasetDB.id == dataset_id).first()
    if not record:
        raise NotFound("Dataset not found.")
    df = load_dataset_dataframe(dataset_id, db)
    report = analyze_dataset(df, target)
    report["dataset_id"] = dataset_id
    report["filename"] = record.filename
    report["all_columns"] = [str(c) for c in df.columns]
    return report


# ---------------------------------------------------------------------------
# 3b. Active learning
# ---------------------------------------------------------------------------

def _edge_case_dict(ec: EdgeCaseDB) -> Dict[str, Any]:
    return {
        "id": ec.id,
        "run_id": ec.run_id,
        "source": ec.source,
        "row_index": ec.row_index,
        "features": ec.features_json,
        "predicted_label": ec.predicted_label,
        "confidence": ec.confidence,
        "probabilities": ec.probabilities_json,
        "dataset_label": ec.dataset_label,
        "status": ec.status,
        "corrected_label": ec.corrected_label,
        "retrain_run_id": ec.retrain_run_id,
        "created_at": ec.created_at.isoformat() if ec.created_at else None,
        "labeled_at": ec.labeled_at.isoformat() if ec.labeled_at else None,
    }


def add_edge_case(payload: Dict[str, Any], db: Session) -> Dict[str, Any]:
    run_id = _parse_id(payload.get("run_id"))
    if run_id is None:
        raise ValueError("run_id is required.")
    _get_run(run_id, db)
    features = payload.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty object.")
    ec = EdgeCaseDB(
        run_id=run_id,
        source=str(payload.get("source") or "browser")[:32],
        features_json={str(k): _json_value(v) for k, v in features.items()},
        predicted_label=None if payload.get("predicted_label") is None else str(payload.get("predicted_label")),
        confidence=None if payload.get("confidence") is None else float(payload["confidence"]),
        probabilities_json=payload.get("probabilities"),
        status="pending",
    )
    db.add(ec)
    db.commit()
    db.refresh(ec)
    return _edge_case_dict(ec)


def list_edge_cases(db: Session, run_id: Optional[int] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    q = db.query(EdgeCaseDB)
    if run_id is not None:
        q = q.filter(EdgeCaseDB.run_id == run_id)
    if status:
        q = q.filter(EdgeCaseDB.status == status)
    rows = q.order_by(EdgeCaseDB.confidence.asc(), EdgeCaseDB.created_at.desc()).all()
    return [_edge_case_dict(r) for r in rows]


def label_edge_case(case_id: int, label: Optional[str], db: Session, dismiss: bool = False) -> Dict[str, Any]:
    ec = db.query(EdgeCaseDB).filter(EdgeCaseDB.id == case_id).first()
    if not ec:
        raise NotFound("Edge case not found.")
    if ec.status == "applied":
        raise ValueError("This example was already used for retraining.")
    if dismiss:
        ec.status, ec.corrected_label = "dismissed", None
    else:
        if label is None or str(label).strip() == "":
            raise ValueError("A label is required.")
        ec.status, ec.corrected_label = "labeled", str(label).strip()
    ec.labeled_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(ec)
    return _edge_case_dict(ec)


def run_label_options(run_id: int, db: Session) -> Dict[str, Any]:
    run = _get_run(run_id, db)
    artifact = _load_artifact(run)
    model = artifact["fitted_model"]
    classes = [str(_json_value(c)) for c in getattr(model, "classes_", [])]
    pending = db.query(EdgeCaseDB).filter(EdgeCaseDB.run_id == run_id)
    counts: Dict[str, int] = {}
    for ec in pending.all():
        counts[ec.status] = counts.get(ec.status, 0) + 1
    return {"run": format_run(run, db), "classes": classes, "counts": counts,
            "can_scan": hasattr(model, "predict_proba")}


def scan_run(run_id: int, threshold: float, limit: int, db: Session) -> Dict[str, Any]:
    """Find dataset rows the model is unsure about and queue them for review."""
    run = _get_run(run_id, db)
    artifact = _load_artifact(run)
    model = artifact["fitted_model"]
    if not hasattr(model, "predict_proba"):
        raise ValueError("Uncertainty scanning needs a classifier with predicted probabilities.")
    target = artifact["target_column"]
    df = load_dataset_dataframe(run.dataset_id, db).reset_index(drop=True)
    X_raw = df.drop(columns=[target])
    X = apply_fitted_transformers(X_raw, artifact.get("fitted_transformers", []), artifact.get("features"))
    probs = model.predict_proba(X)
    conf = probs.max(axis=1)
    classes = list(model.classes_)

    already = {r[0] for r in db.query(EdgeCaseDB.row_index).filter(EdgeCaseDB.run_id == run_id, EdgeCaseDB.row_index.isnot(None)).all()}
    candidates = [i for i in np.argsort(conf) if conf[i] < threshold and int(i) not in already][:limit]

    for i in candidates:
        i = int(i)
        row = X_raw.iloc[i]
        db.add(EdgeCaseDB(
            run_id=run_id,
            source="scan",
            row_index=i,
            features_json={str(k): _json_value(v) for k, v in row.items()},
            predicted_label=str(_json_value(classes[int(probs[i].argmax())])),
            confidence=round(float(conf[i]), 4),
            probabilities_json={str(_json_value(c)): round(float(p), 4) for c, p in zip(classes, probs[i])},
            dataset_label=str(_json_value(df.at[i, target])),
            status="pending",
        ))
    db.commit()
    below = int((conf < threshold).sum())
    return {
        "scanned_rows": int(len(df)),
        "below_threshold": below,
        "queued": len(candidates),
        "already_queued": len(already),
        "threshold": threshold,
        "confidence_histogram": np.histogram(conf, bins=10, range=(0, 1))[0].tolist(),
    }


def _coerce_label(label: str, series: pd.Series) -> Any:
    if pd.api.types.is_numeric_dtype(series):
        v = float(label)
        if pd.api.types.is_integer_dtype(series) or v.is_integer():
            return int(v)
        return v
    return label


def retrain_from_labels(run_id: int, db: Session) -> Dict[str, Any]:
    """Fold human-labelled edge cases into a new dataset version and retrain."""
    run = _get_run(run_id, db)
    artifact = _load_artifact(run)
    cfg = artifact.get("config") or {}
    target = artifact["target_column"]
    cases = db.query(EdgeCaseDB).filter(EdgeCaseDB.run_id == run_id, EdgeCaseDB.status == "labeled").all()
    if not cases:
        raise ValueError("Label at least one edge case before retraining.")

    df = load_dataset_dataframe(run.dataset_id, db).reset_index(drop=True)
    relabeled, appended, new_rows = 0, 0, []
    for ec in cases:
        label = _coerce_label(ec.corrected_label, df[target])
        if ec.source == "scan" and ec.row_index is not None and ec.row_index < len(df):
            if str(_json_value(df.at[ec.row_index, target])) != str(label):
                relabeled += 1
            df.at[ec.row_index, target] = label
        else:
            row = {}
            for c in df.columns:
                if c == target:
                    row[c] = label
                    continue
                v = (ec.features_json or {}).get(str(c))
                if pd.api.types.is_numeric_dtype(df[c]):
                    v = pd.to_numeric(v, errors="coerce")
                row[c] = v
            new_rows.append(row)
            appended += 1
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)

    parent_ds = db.query(DatasetDB).filter(DatasetDB.id == run.dataset_id).first()
    stem = os.path.splitext(parent_ds.filename if parent_ds else "dataset.csv")[0]
    stem = re.sub(r"_al\d+$", "", stem)
    version = db.query(RunDB).filter(RunDB.parent_run_id.isnot(None)).count() + 1
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    new_ds = process_and_save_dataset(f"{stem}_al{version}.csv", buf.getvalue(), db)

    new_run = execute_pipeline_run(
        dataset_id=new_ds.id,
        task_type=cfg.get("task_type", run.task_type),
        target_column=target,
        recipe=cfg.get("recipe", run.recipe_json or []),
        model_config=cfg.get("model_config", run.model_config_json or {}),
        db=db,
        use_cache=False,
    )
    new_run.parent_run_id = run.id
    if run.graph_json:
        graph = json.loads(json.dumps(run.graph_json))
        for n in graph.get("nodes", []):
            if (n.get("data") or {}).get("kind") == "dataset":
                n["data"].setdefault("params", {})["dataset_id"] = new_ds.id
                n["data"]["params"]["filename"] = new_ds.filename
        new_run.graph_json = graph
        new_run.generated_code = compile_graph(graph, df=df, dataset_filename=new_ds.filename)["code"]
    for ec in cases:
        ec.status = "applied"
        ec.retrain_run_id = new_run.id
    db.commit()
    db.refresh(new_run)

    return {
        "status": "success",
        "previous_run": format_run(run, db),
        "new_run": format_run(new_run, db),
        "new_dataset": {"id": new_ds.id, "filename": new_ds.filename, "rows": new_ds.row_count},
        "applied": len(cases),
        "relabeled_rows": relabeled,
        "appended_rows": appended,
    }
