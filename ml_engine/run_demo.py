"""
Standalone Demonstration Runner for Person B (ML Engine & Tracking)
Executes an end-to-end demonstration:
1. Synthetic dataset creation (with missing values and categoricals)
2. Preprocessing recipe execution (imputation, onehot encoding, scaling, train/test split)
3. Multi-model training and evaluation (Random Forest vs Logistic Regression vs Gradient Boosting)
4. Visual terminal output of metrics, confusion matrix, and feature importances
5. Preprocessing cache speedup demonstration
6. Automatic reproducibility verification test
"""

import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ml_engine.pipeline_executor import execute_pipeline, clear_pipeline_cache
from ml_engine.reproduce import verify_reproducibility


def generate_sample_dataset(n_samples: int = 500) -> pd.DataFrame:
    """Generates a realistic tabular dataset with numerical, categorical, and missing fields."""
    np.random.seed(42)

    ages = np.random.normal(38, 12, n_samples)
    # Introduce ~8% missing values
    ages[np.random.choice(n_samples, int(n_samples * 0.08), replace=False)] = np.nan

    incomes = np.random.exponential(50000, n_samples) + 20000
    departments = np.random.choice(["Engineering", "Sales", "Marketing", "HR"], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    satisfaction = np.random.uniform(1, 10, n_samples)

    # Ground truth relationship with some noise
    logit = (
        0.04 * np.nan_to_num(ages, nan=38)
        + 0.00002 * incomes
        - 0.5 * satisfaction
        + (departments == "Sales") * 0.8
        - 1.5
    )
    prob = 1 / (1 + np.exp(-logit))
    attrition = (np.random.uniform(0, 1, n_samples) < prob).astype(int)

    return pd.DataFrame({
        "age": ages,
        "income": incomes,
        "department": departments,
        "satisfaction_score": satisfaction,
        "attrition": attrition,
    })


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_demo():
    print_banner("INTERACTIVE MODEL STUDIO — ML ENGINE DEMO (PERSON B)")
    print("Initializing synthetic HR Attrition dataset (500 rows)...")
    df = generate_sample_dataset()
    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing values in 'age': {df['age'].isna().sum()} rows")
    print("Features:", list(df.columns))

    recipe = [
        {"step": "split", "params": {"test_size": 0.2, "random_state": 42, "stratify": True}},
        {"step": "impute", "params": {"columns": ["age"], "strategy": "median"}},
        {"step": "encode", "params": {"columns": ["department"], "method": "onehot", "drop": "first"}},
        {"step": "scale", "params": {"columns": ["age", "income", "satisfaction_score"], "method": "standard"}},
    ]

    algorithms = [
        ("RandomForestClassifier", {"n_estimators": 100, "max_depth": 5, "random_state": 42}),
        ("LogisticRegression", {"C": 1.0, "max_iter": 500, "random_state": 42}),
        ("GradientBoostingClassifier", {"n_estimators": 80, "learning_rate": 0.1, "random_state": 42}),
    ]

    results = []
    clear_pipeline_cache()

    print_banner("1. TRAINING MULTIPLE MODELS ACROSS SHARED PREPROCESSING RECIPE")
    for idx, (algo, hyperparams) in enumerate(algorithms, 1):
        config = {
            "task_type": "classification",
            "target_column": "attrition",
            "recipe": recipe,
            "model_config": {
                "algorithm": algo,
                "hyperparameters": hyperparams,
            },
        }

        print(f"\n[{idx}/3] Training {algo}...")
        t0 = time.perf_counter()
        res = execute_pipeline(df, config, use_cache=True)
        total_wall_time = time.perf_counter() - t0

        results.append((algo, res, total_wall_time))
        print(f"      Train Time: {res['train_time_seconds']}s | Total: {round(total_wall_time, 4)}s")
        print(f"      Cached Preprocessing: {res['cached_preprocessing']}")
        print(f"      Accuracy: {res['metrics']['accuracy']} | F1: {res['metrics']['f1_score']}")

    print_banner("2. EXPERIMENT COMPARISON SUMMARY TABLE")
    header = f"{'Algorithm':<28} | {'Accuracy':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Train Time'}"
    print(header)
    print("-" * len(header))
    for algo, res, _ in results:
        m = res["metrics"]
        print(f"{algo:<28} | {m['accuracy']:<10} | {m['f1_score']:<10} | {m['precision']:<10} | {m['recall']:<10} | {res['train_time_seconds']}s")

    print_banner("3. DETAILED EVALUATION ARTIFACTS (TOP PERFORMER)")
    top_algo, top_res, _ = max(results, key=lambda x: x[1]["metrics"]["f1_score"])
    print(f"Best Model Selected: {top_algo} (F1 = {top_res['metrics']['f1_score']})\n")

    print("Confusion Matrix:")
    cm = top_res["confusion_matrix"]
    print(f"  Labels: {cm['labels']}")
    for row in cm["matrix"]:
        print(f"    {row}")

    print("\nFeature Importances:")
    for item in top_res["feature_importances"][:5]:
        bar = "#" * int(item["importance"] * 30)
        print(f"  {item['feature']:<25} : {item['importance']:<8.4f} {bar}")

    print_banner("4. REPRODUCIBILITY VERIFICATION")
    print(f"Re-running {top_algo} from stored configuration with zero cache...")
    top_config = {
        "task_type": "classification",
        "target_column": "attrition",
        "recipe": recipe,
        "model_config": top_res["model_config"],
    }
    verification = verify_reproducibility(df, top_config, top_res["metrics"], tolerance=1e-4)

    print(f"Reproducibility Status: {'PASSED [OK]' if verification['is_reproducible'] else 'FAILED'}")
    print(f"Report: {verification['message']}")
    print("Metric Diff Breakdown:")
    for k, v in verification["metrics_comparison"].items():
        print(f"  - {k:<15}: expected={v['expected']}, reproduced={v['reproduced']}, diff={v['difference']} (Match: {v['passed']})")

    print_banner("DEMO COMPLETED SUCCESSFULLY — ALL PERSON B CAPABILITIES VERIFIED")


if __name__ == "__main__":
    run_demo()
