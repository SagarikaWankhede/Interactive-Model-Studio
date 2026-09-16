"""
Reproducibility Verification Module
Re-runs an ML experiment using stored pipeline configurations
and verifies whether outputs match historical records within numerical tolerance.
"""

from typing import Any, Dict, Optional
import pandas as pd

from ml_engine.pipeline_executor import execute_pipeline


def verify_reproducibility(
    df: pd.DataFrame,
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
    }
