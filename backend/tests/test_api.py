import unittest
import io
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        
        # Create synthetic CSV data
        np.random.seed(42)
        df = pd.DataFrame({
            "age": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0],
            "income": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
            "target": [0, 0, 0, 0, 1, 1, 1, 1]
        })
        csv_bytes = io.BytesIO()
        df.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)
        cls.csv_bytes = csv_bytes.getvalue()

    def test_01_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_02_dataset_upload(self):
        files = {"file": ("test_data.csv", self.csv_bytes, "text/csv")}
        response = self.client.post("/api/datasets/upload", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("id", data)
        self.assertEqual(data["filename"], "test_data.csv")
        self.assertEqual(data["row_count"], 8)
        TestBackendAPI.dataset_id = data["id"]

    def test_03_preview_recipe(self):
        payload = {
            "dataset_id": self.dataset_id,
            "target_column": "target",
            "recipe": [
                {"step": "scale", "columns": ["age", "income"], "params": {"method": "standard"}},
                {"step": "encode", "columns": ["gender"], "params": {"method": "onehot"}},
                {"step": "split", "params": {"test_size": 0.25, "random_state": 42}}
            ]
        }
        response = self.client.post("/api/pipelines/preview", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("transformed_columns", data)

    def test_04_execute_run(self):
        payload = {
            "dataset_id": self.dataset_id,
            "task_type": "classification",
            "target_column": "target",
            "recipe": [
                {"step": "scale", "columns": ["age", "income"], "params": {"method": "standard"}},
                {"step": "encode", "columns": ["gender"], "params": {"method": "onehot"}},
                {"step": "split", "params": {"test_size": 0.25, "random_state": 42}}
            ],
            "model_config": {
                "algorithm": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 10, "random_state": 42}
            }
        }
        response = self.client.post("/api/runs/execute", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("id", data)
        self.assertIn("accuracy", data["metrics"])
        TestBackendAPI.run_id = data["id"]

    def test_05_predict(self):
        payload = {
            "features": {
                "age": 0.5,
                "income": 0.2,
                "gender_F": 1.0,
                "gender_M": 0.0
            }
        }
        response = self.client.post(f"/api/runs/{self.run_id}/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())

    def test_06_verify_reproducibility(self):
        response = self.client.post(f"/api/runs/{self.run_id}/verify-reproducibility")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["is_reproducible"])


if __name__ == "__main__":
    unittest.main()
