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
from ml_engine.data_loader import (
    load_dataset,
    optimize_memory,
    detect_and_coerce_types,
    load_dataset_chunked,
    get_dataset_type_summary,
)
from ml_engine.recipe_runner import run_recipe
from ml_engine.training import train_model
from ml_engine.evaluation import evaluate_model
from ml_engine.pipeline_executor import execute_pipeline, clear_pipeline_cache
from ml_engine.reproduce import verify_reproducibility
from ml_engine.export_onnx import export_model_to_onnx, verify_onnx_parity

__version__ = "1.1.0"

__all__ = [
    "get_model_class",
    "get_preprocessor_class",
    "list_supported_models",
    "get_model_defaults",
    "MODEL_METADATA",
    "load_dataset",
    "optimize_memory",
    "detect_and_coerce_types",
    "load_dataset_chunked",
    "get_dataset_type_summary",
    "run_recipe",
    "train_model",
    "evaluate_model",
    "execute_pipeline",
    "clear_pipeline_cache",
    "verify_reproducibility",
    "export_model_to_onnx",
    "verify_onnx_parity",
]
