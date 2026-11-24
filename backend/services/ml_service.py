import os
import sys
import uuid
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

# Ensure ml_engine is on Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_engine.pipeline_executor import execute_pipeline, clear_pipeline_cache
from ml_engine.recipe_runner import run_recipe
from ml_engine.reproduce import verify_reproducibility
from ml_engine.registry import (
    MODEL_METADATA,
    list_supported_models,
    get_model_defaults,
)

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
    db: Session,
    use_cache: bool = True,
) -> RunDB:
    """Loads dataset, executes ML engine pipeline, saves model artifact, and persists run metrics."""
    df = load_dataset_dataframe(dataset_id, db)

    config_dict = {
        "task_type": task_type,
        "target_column": target_column,
        "recipe": recipe,
        "model_config": model_config,
    }

    pipeline_result = execute_pipeline(df, config_dict, use_cache=use_cache)

    algorithm = model_config.get("algorithm", "RandomForestClassifier")
    features_list = pipeline_result.get("features", [])
    if not features_list and hasattr(pipeline_result.get("fitted_model"), "feature_names_in_"):
        features_list = list(pipeline_result["fitted_model"].feature_names_in_)

    # 1. Create run record in DB to obtain unique primary key ID
    run_record = RunDB(
        dataset_id=dataset_id,
        task_type=task_type,
        algorithm=algorithm,
        train_time_sec=pipeline_result["train_time_sec"],
        metrics_json=pipeline_result["metrics"],
        confusion_matrix_json=pipeline_result.get("confusion_matrix"),
        feature_importances_json=pipeline_result.get("feature_importances"),
        recipe_json=recipe,
        model_config_json=model_config,
        is_reproducible=True,
    )
    db.add(run_record)
    db.commit()
    db.refresh(run_record)

    # 2. Key model artifact by unique run ID so repeat runs on the same dataset are never overwritten
    artifact_filename = f"run_{run_record.id}_{algorithm}.joblib"
    artifact_path = os.path.join(STORAGE_MODELS_DIR, artifact_filename)

    artifact_payload = {
        "run_id": run_record.id,
        "fitted_model": pipeline_result["fitted_model"],
        "fitted_transformers": pipeline_result.get("fitted_transformers", []),
        "features": features_list,
        "target_column": target_column,
        "task_type": task_type,
        "config": config_dict,
    }
    joblib.dump(artifact_payload, artifact_path)

    # 3. Update artifact path in DB
    run_record.model_artifact_path = artifact_path
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
        "transformed_columns": [str(c) for c in preview_df.columns],
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
    
    # Reconstruct original config from DB record or stored model artifact
    original_config = None
    if run_record.recipe_json and run_record.model_config_json:
        original_config = {
            "task_type": run_record.task_type,
            "target_column": (
                run_record.recipe_json[0].get("target_column")
                if run_record.recipe_json and "target_column" in run_record.recipe_json[0]
                else None
            ),
            "recipe": run_record.recipe_json,
            "model_config": run_record.model_config_json,
        }

    if not original_config or not original_config.get("target_column"):
        if run_record.model_artifact_path and os.path.exists(run_record.model_artifact_path):
            payload = joblib.load(run_record.model_artifact_path)
            original_config = payload.get("config")

    if not original_config:
        raise ValueError("Cannot reconstruct configuration for reproducibility verification.")

    verification_result = verify_reproducibility(
        df,
        original_config=original_config,
        original_metrics=run_record.metrics_json or {}
    )

    # Update DB status if changed
    if run_record.is_reproducible != verification_result["is_reproducible"]:
        run_record.is_reproducible = verification_result["is_reproducible"]
        db.commit()

    return verification_result


def predict_sample(run_id: int, feature_dict: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """Uses stored fitted model and fitted preprocessing transformers to perform inference."""
    run_record = db.query(RunDB).filter(RunDB.id == run_id).first()
    if not run_record:
        raise ValueError(f"Run with ID {run_id} not found.")

    if not run_record.model_artifact_path or not os.path.exists(run_record.model_artifact_path):
        raise FileNotFoundError("Model artifact file not found on disk.")

    artifact = joblib.load(run_record.model_artifact_path)
    model = artifact["fitted_model"]
    fitted_transformers = artifact.get("fitted_transformers", [])
    expected_features = artifact.get("features", [])

    input_df = pd.DataFrame([feature_dict])

    # Pass raw input features sequentially through fitted transformers
    for entry in fitted_transformers:
        if not isinstance(entry, dict):
            continue
        step_type = entry.get("step")
        cols = entry.get("columns", [])
        transformer = entry.get("transformer")
        if transformer is None:
            continue

        valid_cols = [c for c in cols if c in input_df.columns]
        if not valid_cols:
            continue

        try:
            if step_type == "impute":
                input_df[valid_cols] = transformer.transform(input_df[valid_cols])
            elif step_type == "encode":
                method = entry.get("method", "onehot")
                if method == "onehot":
                    encoded_arr = transformer.transform(input_df[valid_cols].astype(str))
                    feature_names = transformer.get_feature_names_out(valid_cols)
                    encoded_df = pd.DataFrame(encoded_arr, columns=feature_names, index=input_df.index)
                    input_df = input_df.drop(columns=valid_cols).join(encoded_df)
                elif method == "ordinal":
                    input_df[valid_cols] = transformer.transform(input_df[valid_cols].astype(str))
            elif step_type == "scale":
                input_df[valid_cols] = transformer.transform(input_df[valid_cols])
        except Exception:
            # Fall back safely if pre-encoded
            pass

    # Align columns to what the model was trained on if known
    if expected_features:
        input_df = input_df.reindex(columns=expected_features, fill_value=0)

    prediction = model.predict(input_df)[0]
    
    result = {
        "prediction": float(prediction) if isinstance(prediction, (np.integer, np.floating, int, float)) else str(prediction)
    }
    
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(input_df)[0].tolist()
            result["probabilities"] = [round(float(p), 4) for p in probabilities]
        except Exception:
            pass

    return result


def get_available_algorithms(task_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves supported algorithms and their default hyperparameters from ml_engine.registry."""
    algorithms = []
    for model_name, meta in MODEL_METADATA.items():
        if task_type and meta.get("task_type") != task_type:
            continue
        algorithms.append({
            "name": model_name,
            "task_type": meta.get("task_type"),
            "library": meta.get("library"),
            "description": meta.get("description"),
            "default_params": meta.get("default_params", {}),
        })
    return algorithms
