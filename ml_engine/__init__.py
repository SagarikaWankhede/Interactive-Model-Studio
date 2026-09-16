"""
Interactive Model Studio — Machine Learning Engine
Package initialization and core API exports.
"""

from ml_engine.registry import (
    get_model_class,
    get_preprocessor_class,
    list_supported_models,
    get_model_defaults,
    MODEL_METADATA,
)

__version__ = "1.0.0"

__all__ = [
    "get_model_class",
    "get_preprocessor_class",
    "list_supported_models",
    "get_model_defaults",
    "MODEL_METADATA",
Machine Learning Engine for Interactive Model Studio.
Provides deterministic preprocessing, training, evaluation, and reproducibility verification.
"""

from .pipeline_executor import execute_pipeline
from .reproduce import verify_reproducibility
from .registry import MODEL_REGISTRY, TRANSFORMER_REGISTRY

__all__ = [
    "execute_pipeline",
    "verify_reproducibility",
    "MODEL_REGISTRY",
    "TRANSFORMER_REGISTRY",
]
