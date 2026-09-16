import os
import sys
import joblib
import pandas as pd
from typing import Dict, Any, List
from sqlalchemy.orm import Session

# Ensure ml_engine is on Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_engine.pipeline_executor import execute_pipeline
from ml_engine.recipe_runner import run_recipe
from ml_engine.reproduce import verify_reproducibility

from .dataset_service import load_dataset_dataframe
from ..models import RunDB, ExperimentDB, DatasetDB

STORAGE_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "models")
os.makedirs(STORAGE_MODELS_DIR, exist_ok=True)


def execute_pipeline_run(
    dataset_id: int,
    task_type: str,
    target_column: str,
    recipe: List[Dict[str, Any]],
    model_config: Dict[str, Any],
    db: Session
) -> RunDB:
    """Loads dataset, executes ML engine pipeline, saves model artifact, and persists run metrics."""
    df = load_dataset_dataframe(dataset_id, db)

    config_dict = {
        "task_type": task_type,
        "target_column": target_column,
        "recipe": recipe,
        "model_config": model_config,
    }

    pipeline_result = execute_pipeline(df, config_dict)

    # Save fitted model & transformers artifact
    artifact_filename = f"run_{dataset_id}_{model_config.get('algorithm')}.joblib"
    artifact_path = os.path.join(STORAGE_MODELS_DIR, artifact_filename)
    
    artifact_payload = {
        "fitted_model": pipeline_result["fitted_model"],
        "fitted_transformers": pipeline_result["fitted_transformers"],
        "target_column": target_column,
        "task_type": task_type,
        "config": config_dict,
    }
    joblib.dump(artifact_payload, artifact_path)

    run_record = RunDB(
        dataset_id=dataset_id,
        task_type=task_type,
        algorithm=model_config.get("algorithm"),
        train_time_sec=pipeline_result["train_time_sec"],
        metrics_json=pipeline_result["metrics"],
        confusion_matrix_json=pipeline_result.get("confusion_matrix"),
        feature_importances_json=pipeline_result.get("feature_importances"),
        model_artifact_path=artifact_path,
        is_reproducible=True,
    )

    db.add(run_record)
    db.commit()
    db.refresh(run_record)

    return run_record


def preview_recipe(
    dataset_id: int,
    target_column: str,
    recipe: List[Dict[str, Any]],
    db: Session
) -> Dict[str, Any]:
    """Runs preprocessing recipe preview without model training."""
    df = load_dataset_dataframe(dataset_id, db)
    X_tr, X_te, y_tr, y_te, _ = run_recipe(df, recipe, target_column)
    
    preview_df = pd.concat([X_tr, y_tr], axis=1).head(10)
    return {
        "transformed_columns": list(preview_df.columns),
        "train_rows": X_tr.shape[0],
        "test_rows": X_te.shape[0],
        "sample_preview": preview_df.to_dict(orient="records")
    }


def verify_run_reproducibility(run_id: int, db: Session) -> Dict[str, Any]:
    """Verifies reproducibility of a historical run."""
    run_record = db.query(RunDB).filter(RunDB.id == run_id).first()
    if not run_record:
        raise ValueError(f"Run with ID {run_id} not found.")

    df = load_dataset_dataframe(run_record.dataset_id, db)
    
    # Load original config from saved artifact if available
    if run_record.model_artifact_path and os.path.exists(run_record.model_artifact_path):
        payload = joblib.load(run_record.model_artifact_path)
        original_config = payload.get("config")
    else:
        raise ValueError("Model artifact file missing, cannot reconstruct exact config.")

    verification_result = verify_reproducibility(
        df,
        original_config=original_config,
        original_metrics=run_record.metrics_json
    )

    # Update DB status if changed
    if run_record.is_reproducible != verification_result["is_reproducible"]:
        run_record.is_reproducible = verification_result["is_reproducible"]
        db.commit()

    return verification_result


def predict_sample(run_id: int, feature_dict: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """Uses stored fitted model artifact to perform inference on input features."""
    run_record = db.query(RunDB).filter(RunDB.id == run_id).first()
    if not run_record:
        raise ValueError(f"Run with ID {run_id} not found.")

    if not run_record.model_artifact_path or not os.path.exists(run_record.model_artifact_path):
        raise FileNotFoundError("Model artifact file not found on disk.")

    artifact = joblib.load(run_record.model_artifact_path)
    model = artifact["fitted_model"]
    
    input_df = pd.DataFrame([feature_dict])
    
    prediction = model.predict(input_df)[0]
    
    result = {"prediction": float(prediction) if isinstance(prediction, (int, float, float)) else str(prediction)}
    
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(input_df)[0].tolist()
            result["probabilities"] = probabilities
        except Exception:
            pass

    return result
