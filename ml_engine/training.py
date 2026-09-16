"""
Training Module
Instantiates and fits machine learning models, tracking precise training execution times.
"""

import time
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from ml_engine.registry import get_model_class, get_model_defaults
Model initialization, training, and execution time profiling.
"""

import time
import pandas as pd
from typing import Dict, Any, Tuple
from .registry import get_model_class


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
<<<<<<< HEAD
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
=======
    task_type: str = "classification",
    hyperparameters: Dict[str, Any] = None
) -> Tuple[Any, float]:
    """
    Instantiates and fits an estimator based on task_type and algorithm name.
    Tracks wall-clock training time in seconds.
    """
    if hyperparameters is None:
        hyperparameters = {}

    model_cls = get_model_class(task_type, model_name)
    
    # Sanitize integer hyperparameters if passed as float strings
    clean_params = {}
    for k, v in hyperparameters.items():
        clean_params[k] = v

    model_instance = model_cls(**clean_params)

    start_time = time.perf_counter()
    model_instance.fit(X_train, y_train)
    end_time = time.perf_counter()

    train_time_sec = round(end_time - start_time, 4)

    return model_instance, train_time_sec
>>>>>>> origin/main
