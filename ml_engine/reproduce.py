"""
Reproducibility Verification Module
Re-runs an ML experiment using stored pipeline configurations
and verifies whether outputs match historical records within numerical tolerance.
"""

from typing import Any, Dict, Optional
import pandas as pd

from ml_engine.pipeline_executor import execute_pipeline
Config-driven re-execution and metric tolerance verification engine.
"""

import pandas as pd
from typing import Dict, Any, Tuple
from .pipeline_executor import execute_pipeline


def verify_reproducibility(
    df: pd.DataFrame,
<<<<<<< HEAD
    config: Dict[str, Any],
    original_metrics: Dict[str, float],
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """
    Re-runs an experiment from raw configuration and verifies output reproducibility.

    Args:
        df: Input pandas DataFrame.
        config: Full pipeline config dictionary (target_column, task_type, recipe, model_config).
        original_metrics: Dictionary of expected metrics recorded from previous run.
        tolerance: Maximum acceptable absolute difference between metric values.

    Returns:
        Dictionary detailing reproducibility status, metric differences, and evaluation output.
    """
    # Force uncached execution to test true reproducibility
    reproduced_result = execute_pipeline(df=df, config=config, use_cache=False)
    new_metrics = reproduced_result.get("metrics", {})

    discrepancies = {}
    is_reproducible = True

    for metric_name, expected_val in original_metrics.items():
        if metric_name not in new_metrics:
            is_reproducible = False
            discrepancies[metric_name] = {
                "expected": expected_val,
                "reproduced": None,
                "difference": None,
                "status": "MISSING_IN_NEW_RUN",
            }
            continue

        reproduced_val = new_metrics[metric_name]
        diff = abs(float(reproduced_val) - float(expected_val))
        passed = diff <= tolerance

        if not passed:
            is_reproducible = False

        discrepancies[metric_name] = {
            "expected": round(float(expected_val), 5),
            "reproduced": round(float(reproduced_val), 5),
            "difference": round(float(diff), 5),
            "passed": passed,
        }

    message = (
        f"Experiment verified: all {len(original_metrics)} metrics reproduced within tolerance ({tolerance})."
        if is_reproducible
        else f"Reproducibility check failed: {sum(1 for d in discrepancies.values() if not d.get('passed'))} metric(s) exceeded tolerance."
    )

    return {
        "is_reproducible": is_reproducible,
        "message": message,
        "tolerance": tolerance,
        "metrics_comparison": discrepancies,
        "reproduced_result": reproduced_result,
=======
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
>>>>>>> origin/main
    }
