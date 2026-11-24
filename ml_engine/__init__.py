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
    MODEL_REGISTRY,
    TRANSFORMER_REGISTRY,
)
from ml_engine.pipeline_executor import execute_pipeline
from ml_engine.reproduce import verify_reproducibility
from ml_engine.data_loader import load_dataset, downcast_dataframe, infer_column_roles
from ml_engine.export_onnx import export_model_to_onnx, get_onnx_availability

__version__ = "1.1.0"

__all__ = [
    "get_model_class",
    "get_preprocessor_class",
    "list_supported_models",
    "get_model_defaults",
    "MODEL_METADATA",
    "MODEL_REGISTRY",
    "TRANSFORMER_REGISTRY",
    "execute_pipeline",
    "verify_reproducibility",
    "load_dataset",
    "downcast_dataframe",
    "infer_column_roles",
    "export_model_to_onnx",
    "get_onnx_availability",
]

