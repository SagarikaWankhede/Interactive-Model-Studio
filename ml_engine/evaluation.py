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

    if task_type == "classification":
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

        # Calculate ROC-AUC if probability predictions available
        try:
            if hasattr(fitted_model, "predict_proba"):
                y_prob = fitted_model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
                else:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")), 4)
        except Exception:
            metrics["roc_auc"] = None

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        unique_labels = sorted(list(set(y_test).union(set(y_pred))))
        cm_data = {
            "labels": [str(lbl) for lbl in unique_labels],
            "matrix": cm.tolist()
        }

        results["metrics"] = metrics
        results["confusion_matrix"] = cm_data

    elif task_type == "regression":
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

    # Feature Importances Extraction
    feature_importances = []
    try:
        if hasattr(fitted_model, "feature_importances_"):
            importances = fitted_model.feature_importances_
        elif hasattr(fitted_model, "coef_"):
            coef = fitted_model.coef_
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
        else:
            importances = None

        if importances is not None and len(importances) == len(feature_names):
            paired = [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(feature_names, importances)]
            paired.sort(key=lambda x: x["importance"], reverse=True)
            feature_importances = paired
    except Exception:
        feature_importances = []

    results["feature_importances"] = feature_importances
    return results
