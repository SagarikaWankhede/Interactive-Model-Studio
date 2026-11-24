import unittest
import pandas as pd
import numpy as np
from ml_engine.pipeline_executor import execute_pipeline


class TestTraining(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            "feature1": np.random.normal(0, 1, n),
            "feature2": np.random.normal(5, 2, n),
            "target": np.random.choice([0, 1], n)
        })

    def test_classification_pipeline(self):
        config = {
            "task_type": "classification",
            "target_column": "target",
            "recipe": [
                {"step": "scale", "columns": ["feature1", "feature2"], "params": {"method": "standard"}},
                {"step": "split", "params": {"test_size": 0.2, "random_state": 42}}
            ],
            "model_config": {
                "algorithm": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 10, "random_state": 42}
            }
        }
        res = execute_pipeline(self.df, config)
        self.assertEqual(res["status"], "success")
        self.assertIn("accuracy", res["metrics"])
        self.assertGreater(len(res["feature_importances"]), 0)

    def test_regression_pipeline(self):
        self.df["reg_target"] = self.df["feature1"] * 2.5 + np.random.normal(0, 0.1, 100)
        config = {
            "task_type": "regression",
            "target_column": "reg_target",
            "recipe": [
                {"step": "split", "params": {"test_size": 0.2, "random_state": 42}}
            ],
            "model_config": {
                "algorithm": "LinearRegression",
                "hyperparameters": {}
            }
        }
        res = execute_pipeline(self.df, config)
        self.assertEqual(res["status"], "success")
        self.assertIn("r2", res["metrics"])
        self.assertGreater(res["metrics"]["r2"], 0.8)


if __name__ == "__main__":
    unittest.main()
