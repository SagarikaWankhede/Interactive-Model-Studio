"""
Evaluation Module
Calculates performance metrics, confusion matrices, and feature importances
for classification and regression tasks.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
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
    """
    Extracts and ranks feature importances or coefficients from a fitted model.
    """
    importances = []

    # Tree-based models (Random Forest, Decision Tree, Gradient Boosting, XGBoost)
    if hasattr(fitted_model, "feature_importances_"):
        raw_importances = fitted_model.feature_importances_
        for name, val in zip(feature_names, raw_importances):
            importances.append({"feature": name, "importance": round(float(val), 5)})

    # Linear models (Logistic Regression, Linear Regression)
    elif hasattr(fitted_model, "coef_"):
        raw_coefs = fitted_model.coef_
        if raw_coefs.ndim > 1:
            # Average across classes for multi-class linear models
            mean_coefs = np.mean(np.abs(raw_coefs), axis=0)
        else:
            mean_coefs = np.abs(raw_coefs)
        
        # Normalize sum to 1 for consistent comparison
        total = np.sum(mean_coefs)
        norm_coefs = mean_coefs / (total if total > 0 else 1.0)
        for name, val in zip(feature_names, norm_coefs):
            importances.append({"feature": name, "importance": round(float(val), 5)})

    # Sort descending by importance score
    importances.sort(key=lambda x: x["importance"], reverse=True)
    return importances


def evaluate_classification(
    fitted_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Computes classification metrics, ROC-AUC, and confusion matrix.
    """
    y_pred = fitted_model.predict(X_test)
    labels = np.unique(np.concatenate([y_test, y_pred]))

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    # ROC-AUC score (safe fallback)
    roc_auc = None
    if hasattr(fitted_model, "predict_proba"):
        try:
            y_proba = fitted_model.predict_proba(X_test)
            if len(labels) == 2:
                # Binary classification: pass probability of positive class
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                # Multi-class classification: ovr
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            roc_auc = None

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_dict = {
        "labels": [str(l) for l in labels],
        "matrix": cm.tolist(),
    }

    metrics = {
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
    }
    if roc_auc is not None:
        metrics["roc_auc"] = round(float(roc_auc), 4)

    return {
        "metrics": metrics,
        "confusion_matrix": cm_dict,
        "feature_importances": extract_feature_importances(fitted_model, list(X_test.columns)),
    }


def evaluate_regression(
    fitted_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Computes regression evaluation metrics (RMSE, MAE, R², MSE).
    """
    y_pred = fitted_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2_score": round(float(r2), 4),
        "mse": round(float(mse), 4),
    }

    return {
        "metrics": metrics,
        "confusion_matrix": None,
        "feature_importances": extract_feature_importances(fitted_model, list(X_test.columns)),
    }


def evaluate_model(
    fitted_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str = "classification",
) -> Dict[str, Any]:
    """
    Dispatches evaluation according to task_type ('classification' or 'regression').
    """
    if task_type.lower() == "classification":
        return evaluate_classification(fitted_model, X_test, y_test)
    elif task_type.lower() == "regression":
        return evaluate_regression(fitted_model, X_test, y_test)
    else:
        raise ValueError(f"Unknown task_type: '{task_type}'. Must be 'classification' or 'regression'.")
