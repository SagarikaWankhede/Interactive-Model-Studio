import unittest
import pandas as pd
import numpy as np
from ml_engine.preprocessing import apply_imputation, apply_encoding, apply_scaling, apply_train_test_split
from ml_engine.recipe_runner import run_recipe


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            "age": [25.0, np.nan, 35.0, 45.0, 50.0],
            "city": ["NY", "LA", "NY", "SF", "LA"],
            "target": [0, 1, 0, 1, 0]
        })

    def test_imputation(self):
        df_out, imputer = apply_imputation(self.df, ["age"], strategy="mean")
        self.assertFalse(df_out["age"].isnull().any())
        self.assertEqual(df_out["age"].iloc[1], 38.75)

    def test_encoding(self):
        df_out, encoder = apply_encoding(self.df, ["city"], method="onehot")
        self.assertNotIn("city", df_out.columns)
        self.assertIn("city_LA", df_out.columns)

    def test_scaling(self):
        df_out, imputer = apply_imputation(self.df, ["age"], strategy="mean")
        df_scaled, scaler = apply_scaling(df_out, ["age"], method="standard")
        self.assertAlmostEqual(df_scaled["age"].mean(), 0.0, places=4)

    def test_recipe_determinism(self):
        recipe = [
            {"step": "impute", "columns": ["age"], "params": {"strategy": "mean"}},
            {"step": "encode", "columns": ["city"], "params": {"method": "onehot"}},
            {"step": "scale", "columns": ["age"], "params": {"method": "standard"}},
            {"step": "split", "params": {"test_size": 0.4, "random_state": 42}}
        ]
        
        X_tr1, X_te1, y_tr1, y_te1, _ = run_recipe(self.df, recipe, "target")
        X_tr2, X_te2, y_tr2, y_te2, _ = run_recipe(self.df, recipe, "target")

        pd.testing.assert_frame_equal(X_tr1, X_tr2)
        pd.testing.assert_series_equal(y_tr1, y_tr2)


if __name__ == "__main__":
    unittest.main()
