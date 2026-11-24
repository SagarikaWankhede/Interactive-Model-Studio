"""
ONNX Model Export Utility
Exports fitted scikit-learn models to ONNX format for edge deployment.
Requires optional dependency: skl2onnx

Install with:  pip install skl2onnx onnxruntime
"""

import os
import io
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
    _HAS_SKL2ONNX = True
except ImportError:  # pragma: no cover
    _HAS_SKL2ONNX = False
    logger.warning(
        "skl2onnx is not installed. ONNX export will be unavailable. "
        "Install with: pip install skl2onnx onnxruntime"
    )


def export_model_to_onnx(
    fitted_model: Any,
    n_features: int,
    model_name: str = "ml_model",
    output_path: Optional[str] = None,
    use_float32: bool = True,
) -> Dict[str, Any]:
    """
    Export a fitted scikit-learn model to ONNX format.

    Args:
        fitted_model:  A fitted sklearn estimator (e.g. RandomForestClassifier).
        n_features:    Number of input features the model was trained on.
        model_name:    Name label for the ONNX model metadata.
        output_path:   If provided, save the .onnx file to this path.
        use_float32:   Use float32 input tensor (lighter); else float64.

    Returns:
        Dict with keys:
          - 'status':     'success' | 'unavailable' | 'error'
          - 'onnx_bytes': Raw ONNX protobuf bytes (if successful)
          - 'output_path': Path where the file was saved (if output_path given)
          - 'message':    Informational message
          - 'n_features': Number of input features
    """
    if not _HAS_SKL2ONNX:
        return {
            "status": "unavailable",
            "message": (
                "ONNX export requires skl2onnx. "
                "Install with: pip install skl2onnx onnxruntime"
            ),
            "n_features": n_features,
        }

    try:
        tensor_type = FloatTensorType if use_float32 else DoubleTensorType
        initial_type = [("float_input", tensor_type([None, n_features]))]

        onnx_model = convert_sklearn(
            fitted_model,
            initial_types=initial_type,
            target_opset=17,
        )

        onnx_bytes = onnx_model.SerializeToString()

        result: Dict[str, Any] = {
            "status": "success",
            "onnx_bytes": onnx_bytes,
            "n_features": n_features,
            "model_name": model_name,
            "message": f"Exported {type(fitted_model).__name__} to ONNX ({len(onnx_bytes):,} bytes).",
        }

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(onnx_bytes)
            result["output_path"] = output_path
            logger.info(f"ONNX model saved to: {output_path}")

        return result

    except Exception as exc:
        logger.error(f"ONNX export failed: {exc}")
        return {
            "status": "error",
            "message": str(exc),
            "n_features": n_features,
        }


def onnx_inference_check(onnx_bytes: bytes, sample_input: np.ndarray) -> Dict[str, Any]:
    """
    Run a quick inference sanity check on exported ONNX bytes.

    Args:
        onnx_bytes:    Raw ONNX protobuf bytes from export_model_to_onnx().
        sample_input:  2-D float32 numpy array of shape (n_samples, n_features).

    Returns:
        Dict with 'predictions' list and 'status'.
    """
    try:
        import onnxruntime as rt  # type: ignore
    except ImportError:
        return {
            "status": "unavailable",
            "message": "onnxruntime not installed. Install with: pip install onnxruntime",
        }

    try:
        sess = rt.InferenceSession(onnx_bytes)
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: sample_input.astype(np.float32)})
        return {
            "status": "success",
            "predictions": output[0].tolist(),
            "output_names": [o.name for o in sess.get_outputs()],
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def get_onnx_availability() -> Dict[str, bool]:
    """Return dict indicating which ONNX libraries are available."""
    result = {"skl2onnx": _HAS_SKL2ONNX, "onnxruntime": False}
    try:
        import onnxruntime  # noqa: F401
        result["onnxruntime"] = True
    except ImportError:
        pass
    return result


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
            if name == "OneHotEncoder":
                steps.append({"step": "encode", "method": "onehot", "columns": cols, "categories": cats,
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
    if not _HAS_SKL2ONNX:
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

    spec = build_preprocessing_spec(artifact, raw_df)
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
