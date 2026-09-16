"""
Training Module
Instantiates and fits machine learning models, tracking precise training execution times.
"""

import time
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from ml_engine.registry import get_model_class, get_model_defaults


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, float]:
    """
    Instantiates and trains a model using provided features and labels.

    Args:
        X_train: Preprocessed training features DataFrame.
        y_train: Training target Series.
        model_name: Algorithm key defined in registry (e.g. 'RandomForestClassifier').
        hyperparameters: Dictionary of hyperparameters to override defaults.

    Returns:
        fitted_model: Trained scikit-learn / XGBoost estimator object.
        train_time (float): Training duration in seconds.
    """
    model_cls = get_model_class(model_name)
    
    # Merge default hyperparameters with user-provided overrides
    params = get_model_defaults(model_name)
    if hyperparameters:
        params.update(hyperparameters)

    # Instantiate estimator
    try:
        estimator = model_cls(**params)
    except TypeError as e:
        raise ValueError(
            f"Invalid hyperparameter configuration for model '{model_name}': {str(e)}"
        )

    # Measure wall-clock training time
    start_time = time.perf_counter()
    estimator.fit(X_train, y_train)
    elapsed_time = time.perf_counter() - start_time

    return estimator, round(elapsed_time, 4)
