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
]
