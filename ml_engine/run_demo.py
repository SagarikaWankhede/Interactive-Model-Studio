import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
from ml_engine.pipeline_executor import execute_pipeline
from ml_engine.reproduce import verify_reproducibility


def generate_classification_data(n_samples: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    age = np.random.randint(18, 70, size=n_samples).astype(float)
    income = np.random.normal(50000, 15000, size=n_samples)
    gender = np.random.choice(["Male", "Female"], size=n_samples)
    
    # Introduce missing values
    age[::10] = np.nan
    
    target = ((pd.Series(age).fillna(40) * 0.05 + income * 0.00005 + (gender == "Female") * 2) > 4.5).astype(int)

    return pd.DataFrame({
        "age": age,
        "income": income,
        "gender": gender,
        "target": target
    })


def main():
    print("=" * 60)
    print("  INTERACTIVE MODEL STUDIO - ML ENGINE DEMO RUN  ")
    print("=" * 60)

    df = generate_classification_data()
    print(f"\n[1] Generated Synthetic Data: {df.shape[0]} rows, {df.shape[1]} columns")

    recipe = [
        {"step": "impute", "columns": ["age"], "params": {"strategy": "mean"}},
        {"step": "encode", "columns": ["gender"], "params": {"method": "onehot"}},
        {"step": "scale", "columns": ["age", "income"], "params": {"method": "standard"}},
        {"step": "split", "params": {"test_size": 0.25, "random_state": 42}}
    ]

    algorithms = ["RandomForestClassifier", "LogisticRegression", "GradientBoostingClassifier"]

    for algo in algorithms:
        config = {
            "task_type": "classification",
            "target_column": "target",
            "recipe": recipe,
            "model_config": {
                "algorithm": algo,
                "hyperparameters": {"random_state": 42}
            }
        }

        result = execute_pipeline(df, config)
        
        print(f"\n--- Model: {algo} ---")
        print(f"Train Time: {result['train_time_sec']} sec")
        print(f"Metrics: {result['metrics']}")
        print(f"Top Features: {result['feature_importances'][:2]}")

        # Reproducibility test
        repro = verify_reproducibility(df, config, result['metrics'])
        print(f"Reproducible: {repro['is_reproducible']} ({repro['message']})")

    print("\n=" * 60)
    print("  DEMO RUN COMPLETED SUCCESSFULLY!  ")
    print("=" * 60)


if __name__ == "__main__":
    main()
