"""
Config-driven re-execution and metric tolerance verification engine.
"""

import pandas as pd
from typing import Dict, Any, Tuple
from .pipeline_executor import execute_pipeline


def verify_reproducibility(
    df: pd.DataFrame,
    original_config: Dict[str, Any],
    original_metrics: Dict[str, float],
    tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Re-runs full pipeline using original config (bypassing cache) and checks if metrics match.
    """
    new_result = execute_pipeline(df, original_config, use_cache=False)
    new_metrics = new_result.get("metrics", {})

    diff = {}
    is_reproducible = True

    for metric_name, orig_val in original_metrics.items():
        if orig_val is None:
            continue
        new_val = new_metrics.get(metric_name)
        if new_val is None:
            diff[metric_name] = {"original": orig_val, "new": None, "delta": None, "match": False}
            is_reproducible = False
            continue

        delta = abs(float(orig_val) - float(new_val))
        match = delta <= tolerance
        diff[metric_name] = {
            "original": orig_val,
            "new": new_val,
            "delta": round(delta, 6),
            "match": match
        }
        if not match:
            is_reproducible = False

    message = "Pipeline results are 100% reproducible within tolerance." if is_reproducible else "Pipeline results differ from original run."

    return {
        "is_reproducible": is_reproducible,
        "tolerance": tolerance,
        "diff": diff,
        "message": message,
        "re-executed_metrics": new_metrics
    }
