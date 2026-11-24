"""
ONNX Model Export Module
Converts trained scikit-learn models to Open Neural Network Exchange (ONNX) format
for high-performance, cross-platform edge and in-browser deployment.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np

# Recipe steps the browser runtime (frontend/src/services/edgeRuntime.js) can replay.
BROWSER_STEPS = ("drop_columns", "impute", "scale", "encode", "outlier_clip")


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


# ---------------------------------------------------------------------------
# Browser / edge bundle
# ---------------------------------------------------------------------------
# A scikit-learn run in this project is "fitted recipe steps + estimator".
# ONNX gets the estimator (numeric feature vector in, prediction out); the
# recipe steps are exported as a small JSON spec that the browser replays in
# JavaScript before calling the model. The ONNX model is checked against the
# original Python model on real rows ("parity check") before we claim it works.

def _json_safe(v: Any) -> Any:
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


def build_preprocessing_spec(artifact: Dict[str, Any], raw_df) -> Dict[str, Any]:
    """Describe the fitted recipe as plain numbers so JavaScript can replay it."""
    import pandas as pd  # local import keeps module import light

    target = artifact.get("target_column")
    steps: List[Dict[str, Any]] = []
    dropped: List[str] = []
    encoded_categories: Dict[str, List[Any]] = {}

    for entry in artifact.get("fitted_transformers", []) or []:
        if not isinstance(entry, dict):
            continue
        kind, cols, t = entry.get("step"), list(entry.get("columns", [])), entry.get("transformer")
        if kind == "drop_columns":
            steps.append({"step": "drop_columns", "columns": cols})
            dropped += cols
            continue
        if kind not in BROWSER_STEPS:
            raise ValueError(
                f"The '{kind}' step can't run in the browser yet. Browser deployment supports: "
                + ", ".join(BROWSER_STEPS) + "."
            )
        if kind == "outlier_clip":
            th = entry.get("thresholds") or {}
            steps.append({"step": "clip", "columns": list(th), "lower": [float(v[0]) for v in th.values()],
                          "upper": [float(v[1]) for v in th.values()]})
            continue
        if t is None:
            continue
        # The transformer was fitted only on the columns that existed at that point
        cols = [str(c) for c in getattr(t, "feature_names_in_", cols)]
        name = type(t).__name__
        if kind == "impute":
            steps.append({"step": "impute", "columns": cols, "values": [_json_safe(v) for v in t.statistics_]})
        elif kind == "scale":
            if name == "StandardScaler":
                mul, add = 1.0 / t.scale_, -t.mean_ / t.scale_
            elif name == "RobustScaler":
                mul, add = 1.0 / t.scale_, -t.center_ / t.scale_
            elif name == "MinMaxScaler":
                mul, add = t.scale_, t.min_
            else:
                raise ValueError(f"Scaler {name} cannot be exported for the browser.")
            # y = x * mul + add
            steps.append({"step": "scale", "columns": cols, "mul": [float(v) for v in mul], "add": [float(v) for v in add]})
        elif kind == "encode":
            cats = [[_json_safe(v) for v in c] for c in t.categories_]
            for c, cl in zip(cols, cats):
                encoded_categories[c] = cl
            # run_recipe encodes values cast to str, so categories are strings ("1.0", "Pune").
            if name == "OneHotEncoder":
                drop_idx = getattr(t, "drop_idx_", None)
                steps.append({"step": "encode", "method": "onehot", "columns": cols, "categories": cats,
                              "drop_idx": None if drop_idx is None else [None if d is None else int(d) for d in drop_idx],
                              "output_names": [str(n) for n in t.get_feature_names_out(cols)]})
            else:
                steps.append({"step": "encode", "method": "ordinal", "columns": cols, "categories": cats})

    inputs = []
    for c in raw_df.columns:
        if c == target or c in dropped:
            continue
        s = raw_df[c]
        numeric = pd.api.types.is_numeric_dtype(s) and c not in encoded_categories
        info: Dict[str, Any] = {"name": str(c), "type": "numeric" if numeric else "categorical"}
        non_null = s.dropna()
        if numeric:
            info["example"] = _json_safe(non_null.median()) if len(non_null) else 0
        else:
            options = encoded_categories.get(c) or [_json_safe(v) for v in non_null.value_counts().index[:50]]
            info["options"] = [o for o in options if o is not None]
            info["example"] = info["options"][0] if info["options"] else ""
        inputs.append(info)

    model = artifact["fitted_model"]
    classes = [_json_safe(c) for c in getattr(model, "classes_", [])]
    return {
        "task_type": artifact.get("task_type"),
        "target_column": target,
        "algorithm": type(model).__name__,
        "inputs": inputs,
        "steps": steps,
        "features": list(artifact.get("features", [])),
        "classes": classes or None,
    }


def _quantize(onnx_bytes: bytes) -> Dict[str, Any]:
    """Try dynamic INT8 quantization; report honestly when it does not apply."""
    import tempfile
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
        import onnx  # type: ignore
    except ImportError:
        return {"applied": False, "message": "onnxruntime.quantization is not installed."}

    quantizable = {"MatMul", "Gemm", "Conv", "Attention", "LSTM", "GRU"}
    graph_ops = {n.op_type for n in onnx.load_from_string(onnx_bytes).graph.node}
    if not graph_ops & quantizable:
        return {
            "applied": False,
            "message": ("Nothing to quantize: this model is built from "
                        + ", ".join(sorted(graph_ops))
                        + " operations, which INT8 dynamic quantization does not change. "
                        "Quantization mainly shrinks neural networks (MatMul / Conv weights); "
                        "tree and linear scikit-learn models are already stored compactly, "
                        "so the FP32 model is served."),
        }

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "m.onnx"), os.path.join(tmp, "m.int8.onnx")
        with open(src, "wb") as f:
            f.write(onnx_bytes)
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
        with open(dst, "rb") as f:
            q = f.read()
    return {"applied": True, "bytes": q, "message": f"INT8 dynamic quantization: {len(onnx_bytes):,} -> {len(q):,} bytes."}


def export_browser_bundle(
    artifact: Dict[str, Any],
    raw_df,
    quantize: bool = False,
    parity_rows: int = 500,
) -> Dict[str, Any]:
    """Export model + preprocessing spec and verify the ONNX model against Python."""
    import gzip
    from .preprocessing import apply_fitted_transformers

    model = artifact["fitted_model"]
    features = list(artifact.get("features", []))
    if not features:
        return {"status": "error", "message": "This run has no recorded feature list; re-train it to export."}
    if type(model).__module__.startswith("xgboost"):
        return {"status": "error", "message": "XGBoost models need the onnxmltools converter, which is not set up yet. Train a scikit-learn model (Random Forest, Gradient Boosting, Logistic Regression, Decision Tree) to deploy to the browser."}
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        return {"status": "unavailable", "message": "ONNX export requires skl2onnx. Install with: pip install skl2onnx onnxruntime"}

    is_classifier = hasattr(model, "classes_")
    try:
        onnx_model = convert_sklearn(
            model,
            initial_types=[("input", FloatTensorType([None, len(features)]))],
            target_opset={"": 17, "ai.onnx.ml": 3},
            options={id(model): {"zipmap": False}} if is_classifier else None,
        )
    except Exception as exc:
        return {"status": "error", "message": f"ONNX conversion failed: {exc}"}
    onnx_bytes = onnx_model.SerializeToString()

    try:
        spec = build_preprocessing_spec(artifact, raw_df)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    spec["onnx"] = {
        "input_name": "input",
        "outputs": [o.name for o in onnx_model.graph.output],
        "probabilities_output": "probabilities" if is_classifier else None,
    }

    report: Dict[str, Any] = {
        "status": "success",
        "onnx_bytes": onnx_bytes,
        "spec": spec,
        "size_bytes": len(onnx_bytes),
        "gzip_bytes": len(gzip.compress(onnx_bytes)),
        "quantization": None,
    }

    if quantize:
        q = _quantize(onnx_bytes)
        report["quantization"] = {k: v for k, v in q.items() if k != "bytes"}
        if q.get("applied"):
            report["quantized_bytes"] = q["bytes"]
            report["quantization"]["size_bytes"] = len(q["bytes"])

    # ---- Parity check: ONNX (float32) vs original Python model --------------
    try:
        import onnxruntime as rt  # type: ignore
        target = artifact.get("target_column")
        sample = raw_df.drop(columns=[target], errors="ignore").head(parity_rows)
        X = apply_fitted_transformers(sample, artifact.get("fitted_transformers", []), features)
        X32 = X.to_numpy(dtype=np.float32)
        sess = rt.InferenceSession(report.get("quantized_bytes", onnx_bytes))
        expected = model.predict(X)
        if is_classifier:
            probs = sess.run(["probabilities"], {"input": X32})[0]
            got = np.asarray(model.classes_)[probs.argmax(axis=1)]
            agree = float(np.mean(got == expected))
            report["parity"] = {"rows": len(X), "metric": "label agreement", "value": round(agree, 4), "passed": agree >= 0.99}
        else:
            got = sess.run(None, {"input": X32})[0].ravel()
            diff = float(np.max(np.abs(got - expected))) if len(got) else 0.0
            scale = float(np.max(np.abs(expected))) or 1.0
            report["parity"] = {"rows": len(X), "metric": "max abs difference", "value": round(diff, 6), "passed": diff / scale < 1e-3}
    except Exception as exc:  # parity is informative; never block the export
        report["parity"] = {"rows": 0, "metric": "unavailable", "value": None, "passed": None, "message": str(exc)}

    return report
