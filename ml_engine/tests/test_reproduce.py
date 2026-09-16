import unittest
import pandas as pd
import numpy as np
from ml_engine.pipeline_executor import execute_pipeline
from ml_engine.reproduce import verify_reproducibility


class TestReproduce(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 80
        self.df = pd.DataFrame({
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "target": np.random.choice([0, 1], n)
        })

    def test_reproducibility(self):
        config = {
            "task_type": "classification",
            "target_column": "target",
            "recipe": [
                {"step": "split", "params": {"test_size": 0.25, "random_state": 42}}
            ],
            "model_config": {
                "algorithm": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 20, "random_state": 42}
            }
        }
        res = execute_pipeline(self.df, config)
        repro = verify_reproducibility(self.df, config, res["metrics"])
        self.assertTrue(repro["is_reproducible"])


if __name__ == "__main__":
    unittest.main()
