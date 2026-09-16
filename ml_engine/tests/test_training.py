"""
Unit Tests for Model Training, Evaluation, and Pipeline Executor
Verifies training across multiple algorithms, metric computation, and transform caching.
"""

import unittest
import numpy as np
import pandas as pd

from ml_engine.pipeline_executor import execute_pipeline, clear_pipeline_cache


class TestTrainingAndEvaluation(unittest.TestCase):

    def setUp(self):
        """Builds synthetic classification and regression datasets."""
        np.random.seed(42)
        n = 120

        # Classification dataset
        f1 = np.random.normal(0, 1, n)
        f2 = np.random.normal(5, 2, n)
        prob = 1 / (1 + np.exp(-(0.8 * f1 - 0.5 * f2 + 2.0)))
        y_cls = (prob > 0.5).astype(int)

        self.df_cls = pd.DataFrame({
            "feature_1": f1,
            "feature_2": f2,
            "target": y_cls,
        })

        # Regression dataset
        y_reg = 2.5 * f1 + 1.2 * f2 + np.random.normal(0, 0.2, n)
        self.df_reg = pd.DataFrame({
            "feature_1": f1,
            "feature_2": f2,
            "target": y_reg,
        })

    def tearDown(self):
        clear_pipeline_cache()

    def test_random_forest_classification(self):
        """Tests training and evaluation of RandomForestClassifier end-to-end."""
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
<<<<<<< HEAD
                {"step": "split", "params": {"test_size": 0.25, "random_state": 42}},
                {"step": "scale", "params": {"columns": ["feature_1", "feature_2"], "method": "standard"}}
            ],
            "model_config": {
                "algorithm": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 50, "max_depth": 4, "random_state": 42}
            }
        }

        result = execute_pipeline(self.df_cls, config)

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["algorithm"], "RandomForestClassifier")
        self.assertGreater(result["train_time_seconds"], 0.0)

        # Check metrics
        metrics = result["metrics"]
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.6)

        # Check confusion matrix
        cm = result["confusion_matrix"]
        self.assertIsNotNone(cm)
        self.assertEqual(len(cm["labels"]), 2)
        self.assertEqual(len(cm["matrix"]), 2)

        # Check feature importances
        fi = result["feature_importances"]
        self.assertEqual(len(fi), 2)
        self.assertGreaterEqual(fi[0]["importance"], fi[1]["importance"])

    def test_logistic_regression_classification(self):
        """Tests training of LogisticRegression with linear coefficient extraction."""
        config = {
            "task_type": "classification",
            "target_column": "target",
            "recipe": [
                {"step": "split", "params": {"test_size": 0.25, "random_state": 42}},
                {"step": "scale", "params": {"columns": ["feature_1", "feature_2"], "method": "standard"}}
            ],
            "model_config": {
                "algorithm": "LogisticRegression",
                "hyperparameters": {"C": 1.0, "random_state": 42}
            }
        }

        result = execute_pipeline(self.df_cls, config)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("accuracy", result["metrics"])
        self.assertEqual(len(result["feature_importances"]), 2)

    def test_random_forest_regression(self):
        """Tests regression pipeline with RMSE, MAE, R2 calculation."""
        config = {
            "task_type": "regression",
            "target_column": "target",
            "recipe": [
                {"step": "split", "params": {"test_size": 0.25, "random_state": 42}},
            ],
            "model_config": {
                "algorithm": "RandomForestRegressor",
                "hyperparameters": {"n_estimators": 50, "random_state": 42}
            }
        }

        result = execute_pipeline(self.df_reg, config)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("rmse", result["metrics"])
        self.assertIn("r2_score", result["metrics"])
        self.assertGreater(result["metrics"]["r2_score"], 0.8)

    def test_preprocessing_cache(self):
        """Verifies that running two models on the same recipe utilizes preprocessing cache."""
        config_1 = {
            "task_type": "classification",
            "target_column": "target",
            "recipe": [{"step": "split", "params": {"test_size": 0.2, "random_state": 42}}],
            "model_config": {"algorithm": "RandomForestClassifier", "hyperparameters": {"n_estimators": 10}}
        }
        config_2 = {
            "task_type": "classification",
            "target_column": "target",
            "recipe": [{"step": "split", "params": {"test_size": 0.2, "random_state": 42}}],
            "model_config": {"algorithm": "LogisticRegression", "hyperparameters": {}}
        }

        res_1 = execute_pipeline(self.df_cls, config_1, use_cache=True)
        self.assertFalse(res_1["cached_preprocessing"])

        res_2 = execute_pipeline(self.df_cls, config_2, use_cache=True)
        self.assertTrue(res_2["cached_preprocessing"])
=======
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
>>>>>>> origin/main


if __name__ == "__main__":
    unittest.main()
