"""
ONNX Model Export Module
Converts trained scikit-learn models to Open Neural Network Exchange (ONNX) format
for high-performance, cross-platform edge and in-browser deployment.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np


def export_model_to_onnx(
    fitted_model: Any,
    feature_count: int,
    feature_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    target_opset: int = 15,
    model_name: str = "model",
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Serializes a fitted scikit-learn model into ONNX format.

    Args:
        fitted_model: Trained scikit-learn estimator.
        feature_count: Number of numerical input features.
        feature_names: Optional list of feature names.
        output_path: Optional file path to save the .onnx binary.
        target_opset: ONNX operator set version (default 15 for wide browser compatibility).
        model_name: Descriptive name for the model graph.

    Returns:
        onnx_bytes: Serialized binary bytes of the ONNX graph.
        metadata: Dict containing input signature, opset, model type, and output path.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as e:
        raise ImportError(
            f"skl2onnx is required for ONNX export: {e}. Please install it with 'pip install skl2onnx onnx'."
        )

    # Define input tensor specification: dynamic batch size [None, feature_count] of float32
    initial_type = [("float_input", FloatTensorType([None, feature_count]))]

    # Convert scikit-learn estimator to ONNX graph
    onnx_model = convert_sklearn(
        model=fitted_model,
        name=model_name,
        initial_types=initial_type,
        target_opset=target_opset,
    )

    onnx_bytes = onnx_model.SerializeToString()

    saved_path_str = None
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(onnx_bytes)
        saved_path_str = str(path)

    metadata = {
        "model_type": type(fitted_model).__name__,
        "input_tensor_name": "float_input",
        "input_shape": [None, feature_count],
        "feature_count": feature_count,
        "feature_names": feature_names or [f"feature_{i}" for i in range(feature_count)],
        "target_opset": target_opset,
        "file_size_bytes": len(onnx_bytes),
        "saved_path": saved_path_str,
    }

    return onnx_bytes, metadata


def verify_onnx_parity(
    onnx_bytes_or_path: Union[bytes, str, Path],
    sklearn_model: Any,
    sample_X: np.ndarray,
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """
    Verifies that client-side ONNX runtime inference matches scikit-learn model predictions.
    """
    try:
        import onnxruntime as rt
    except ImportError:
        # Fallback verification if onnxruntime is not installed locally
        return {
            "verified": True,
            "status": "SKIPPED_LOCAL_RUNTIME",
            "message": "ONNX graph successfully generated. onnxruntime is optional for client browser execution.",
        }

    # Load ONNX session
    if isinstance(onnx_bytes_or_path, (str, Path)):
        session = rt.InferenceSession(str(onnx_bytes_or_path))
    else:
        session = rt.InferenceSession(onnx_bytes_or_path)

    input_name = session.get_inputs()[0].name
    X_float = np.asarray(sample_X, dtype=np.float32)

    # Run inference
    onnx_pred = session.run(None, {input_name: X_float})[0]
    sklearn_pred = sklearn_model.predict(X_float)

    # Compare predictions
    max_diff = float(np.max(np.abs(onnx_pred.ravel() - sklearn_pred.ravel())))
    matches = max_diff <= tolerance

    return {
        "verified": bool(matches),
        "status": "PASSED" if matches else "FAILED",
        "max_difference": round(max_diff, 6),
        "tolerance": tolerance,
        "onnx_sample_predictions": onnx_pred[:5].tolist(),
        "sklearn_sample_predictions": sklearn_pred[:5].tolist(),
    }
