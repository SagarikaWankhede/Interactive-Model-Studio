"""
Comprehensive model evaluation metrics, confusion matrix, and feature importances.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def extract_feature_importances(
    fitted_model: Any,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """Extracts and ranks feature importances or coefficients from a fitted model."""
    importances = []
    try:
        if hasattr(fitted_model, "feature_importances_"):
            raw_importances = fitted_model.feature_importances_
            for name, val in zip(feature_names, raw_importances):
                importances.append({"feature": name, "importance": round(float(val), 5)})

        elif hasattr(fitted_model, "coef_"):
            raw_coefs = fitted_model.coef_
            if raw_coefs.ndim > 1:
                mean_coefs = np.mean(np.abs(raw_coefs), axis=0)
            else:
                mean_coefs = np.abs(raw_coefs)
            
            total = np.sum(mean_coefs)
            norm_coefs = mean_coefs / (total if total > 0 else 1.0)
            for name, val in zip(feature_names, norm_coefs):
                importances.append({"feature": name, "importance": round(float(val), 5)})

        importances.sort(key=lambda x: x["importance"], reverse=True)
    except Exception:
        importances = []

    return importances


def evaluate_model(
    fitted_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str = "classification",
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """Evaluates fitted model on test set and computes standardized metrics."""
    if feature_names is None:
        feature_names = list(X_test.columns)

    y_pred = fitted_model.predict(X_test)
    results = {}

    if task_type.lower() == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
            "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        }

        try:
            if hasattr(fitted_model, "predict_proba"):
                y_prob = fitted_model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
                else:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")), 4)
        except Exception:
            metrics["roc_auc"] = None

        cm = confusion_matrix(y_test, y_pred)
        unique_labels = sorted(list(set(y_test).union(set(y_pred))))
        cm_data = {
            "labels": [str(lbl) for lbl in unique_labels],
            "matrix": cm.tolist()
        }

        results["metrics"] = metrics
        results["confusion_matrix"] = cm_data

    elif task_type.lower() == "regression":
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "r2": round(float(r2), 4),
        }

        results["metrics"] = metrics
        results["confusion_matrix"] = None

    else:
        raise ValueError(f"Invalid task_type: {task_type}")

    results["feature_importances"] = extract_feature_importances(fitted_model, feature_names)
    return results
