"""
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
