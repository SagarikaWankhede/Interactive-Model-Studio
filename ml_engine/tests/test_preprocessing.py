"""
Unit Tests for Preprocessing and Recipe Runner
Verifies deterministic transformation, handling of missing values, encoding, and scaling.
"""

import unittest
import numpy as np
import pandas as pd

from ml_engine.recipe_runner import run_recipe


class TestPreprocessingAndRecipe(unittest.TestCase):

    def setUp(self):
        """Creates a sample messy dataset for testing transformations."""
        np.random.seed(42)
        n_samples = 100
        self.df = pd.DataFrame({
            "age": [25.0, np.nan, 35.0, 40.0, np.nan, 22.0, 58.0, 31.0] * 12 + [29.0, 45.0, 33.0, 28.0],
            "income": np.random.uniform(30000, 120000, n_samples),
            "city": ["New York", "Paris", "London", "Tokyo"] * 25,
            "target": np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
        })

        self.recipe = [
            {
                "step": "split",
                "params": {"test_size": 0.2, "random_state": 42, "stratify": False}
            },
            {
                "step": "impute",
                "params": {"columns": ["age"], "strategy": "median"}
            },
            {
                "step": "encode",
                "params": {"columns": ["city"], "method": "onehot", "drop": "first"}
            },
            {
                "step": "scale",
                "params": {"columns": ["age", "income"], "method": "standard"}
            },
        ]

    def test_recipe_determinism(self):
        """Confirming recipe applied twice gives identical output (determinism check)."""
        X_train_1, X_test_1, y_train_1, y_test_1, _ = run_recipe(
            self.df, target_column="target", recipe=self.recipe
        )
        X_train_2, X_test_2, y_train_2, y_test_2, _ = run_recipe(
            self.df, target_column="target", recipe=self.recipe
        )

        # Assert identical training and testing frames
        pd.testing.assert_frame_equal(X_train_1, X_train_2)
        pd.testing.assert_frame_equal(X_test_1, X_test_2)
        pd.testing.assert_series_equal(y_train_1, y_train_2)
        pd.testing.assert_series_equal(y_test_1, y_test_2)

    def test_imputation_removes_nans(self):
        """Verifies that missing values are completely resolved."""
        X_train, X_test, _, _, _ = run_recipe(
            self.df, target_column="target", recipe=self.recipe
        )
        self.assertEqual(X_train["age"].isna().sum(), 0)
        self.assertEqual(X_test["age"].isna().sum(), 0)

    def test_encoding_creates_numeric_dummies(self):
        """Verifies that categorical columns are converted into numeric dummy columns."""
        X_train, X_test, _, _, _ = run_recipe(
            self.df, target_column="target", recipe=self.recipe
        )
        self.assertNotIn("city", X_train.columns)
        # Check that onehot dummy columns exist
        encoded_cols = [c for c in X_train.columns if c.startswith("city_")]
        self.assertGreater(len(encoded_cols), 0)
        for col in encoded_cols:
            self.assertTrue(np.issubdtype(X_train[col].dtype, np.number))

    def test_scaling_standardizes_features(self):
        """Verifies that standard scaling centers training distribution around 0."""
        X_train, _, _, _, _ = run_recipe(
            self.df, target_column="target", recipe=self.recipe
        )
        # Standard scaled feature should have mean near 0 and std near 1
        self.assertAlmostEqual(float(X_train["age"].mean()), 0.0, places=1)
        self.assertAlmostEqual(float(X_train["income"].mean()), 0.0, places=1)
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
