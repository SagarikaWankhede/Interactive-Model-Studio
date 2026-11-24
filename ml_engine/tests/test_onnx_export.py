"""
Unit Tests for ONNX Model Export and Serialization (Phase 4 Kickstart).
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml_engine.export_onnx import export_model_to_onnx, verify_onnx_parity
from ml_engine.pipeline_executor import execute_pipeline
import pandas as pd


class TestONNXExport(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

        self.X = np.random.normal(0, 1, (n, 4)).astype(np.float32)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)

        self.df = pd.DataFrame(self.X, columns=[f"f_{i}" for i in range(4)])
        self.df["target"] = self.y

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_logistic_regression_to_onnx(self):
        """Verifies LogisticRegression converts cleanly to valid ONNX binary."""
        clf = LogisticRegression()
        clf.fit(self.X, self.y)

        out_path = self.output_dir / "lr_model.onnx"
        onnx_bytes, meta = export_model_to_onnx(
            fitted_model=clf,
            feature_count=4,
            feature_names=[f"f_{i}" for i in range(4)],
            output_path=out_path,
        )

        self.assertGreater(len(onnx_bytes), 0)
        self.assertTrue(out_path.exists())
        self.assertEqual(meta["model_type"], "LogisticRegression")
        self.assertEqual(meta["feature_count"], 4)

    def test_export_random_forest_to_onnx(self):
        """Verifies RandomForestClassifier converts cleanly to valid ONNX binary."""
        clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(self.X, self.y)

        out_path = self.output_dir / "rf_model.onnx"
        onnx_bytes, meta = export_model_to_onnx(
            fitted_model=clf,
            feature_count=4,
            output_path=out_path,
        )

        self.assertGreater(len(onnx_bytes), 0)
        self.assertTrue(out_path.exists())
        self.assertEqual(meta["model_type"], "RandomForestClassifier")

    def test_pipeline_executor_onnx_export(self):
        """Verifies pipeline_executor generates ONNX export when requested in config."""
        out_path = str(self.output_dir / "pipeline_model.onnx")
        config = {
            "task_type": "classification",
            "target_column": "target",
            "recipe": [
                {"step": "split", "params": {"test_size": 0.2, "random_state": 42}},
                {"step": "scale", "params": {"columns": ["f_0", "f_1", "f_2", "f_3"], "method": "standard"}}
            ],
            "model_config": {
                "algorithm": "LogisticRegression",
                "hyperparameters": {"C": 1.0}
            },
            "export_onnx": True,
            "onnx_output_path": out_path,
        }

        result = execute_pipeline(self.df, config)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("onnx_export", result)
        self.assertEqual(result["onnx_export"]["saved_path"], out_path)
        self.assertTrue(Path(out_path).exists())


if __name__ == "__main__":
    unittest.main()
