"""
Unit Tests for Advanced Feature Transformations:
Power/log transforms, binning, polynomial features, PCA, outlier clipping, and datetime decomposition.
"""

import unittest
import numpy as np
import pandas as pd

from ml_engine.preprocessing import (
    apply_power_transform,
    apply_binning,
    apply_polynomial_features,
    apply_pca,
    apply_outlier_clipping,
    apply_datetime_decomposition,
)
from ml_engine.recipe_runner import run_recipe


class TestAdvancedTransforms(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 150
        # Highly skewed positive exponential feature
        skewed = np.random.exponential(scale=10.0, size=n)
        # Normal feature with intentional extreme outliers
        normal_with_outliers = np.random.normal(50.0, 5.0, n)
        normal_with_outliers[0] = 500.0  # High outlier
        normal_with_outliers[1] = -200.0  # Low outlier

        self.df = pd.DataFrame({
            "skewed": skewed,
            "feature_x": np.random.uniform(1.0, 10.0, n),
            "feature_y": np.random.uniform(1.0, 10.0, n),
            "outlier_col": normal_with_outliers,
            "date_col": pd.date_range("2023-01-01", periods=n, freq="D"),
            "target": np.random.choice([0, 1], n),
        })

        self.X_train = self.df.iloc[:120].drop(columns=["target"]).copy()
        self.X_test = self.df.iloc[120:].drop(columns=["target"]).copy()

    def test_power_transform_log1p(self):
        """Verifies log1p compresses skewed features."""
        X_tr, X_te, _ = apply_power_transform(self.X_train, self.X_test, ["skewed"], method="log1p")
        self.assertLess(X_tr["skewed"].max(), self.X_train["skewed"].max())
        self.assertTrue(np.all(np.isfinite(X_tr["skewed"])))

    def test_power_transform_yeo_johnson(self):
        """Verifies Yeo-Johnson PowerTransformer."""
        X_tr, X_te, _ = apply_power_transform(self.X_train, self.X_test, ["skewed"], method="yeo-johnson")
        self.assertAlmostEqual(float(X_tr["skewed"].mean()), 0.0, places=1)

    def test_binning(self):
        """Verifies continuous column discretization into ordinal bins."""
        X_tr, X_te, _ = apply_binning(self.X_train, self.X_test, ["feature_x"], n_bins=4, strategy="quantile")
        unique_bins = set(X_tr["feature_x"].unique())
        self.assertTrue(unique_bins.issubset({0.0, 1.0, 2.0, 3.0}))

    def test_polynomial_features(self):
        """Verifies polynomial and interaction feature expansion."""
        cols = ["feature_x", "feature_y"]
        X_tr, X_te, poly = apply_polynomial_features(self.X_train, self.X_test, cols, degree=2, interaction_only=False)
        # Degree 2 of 2 features gives x, y, x^2, x*y, y^2 -> 5 non-bias features
        new_cols = [c for c in X_tr.columns if c not in ["skewed", "outlier_col", "date_col"]]
        self.assertEqual(len(new_cols), 5)

    def test_pca(self):
        """Verifies linear dimensionality reduction with PCA."""
        cols = ["feature_x", "feature_y"]
        X_tr, X_te, pca = apply_pca(self.X_train, self.X_test, cols, n_components=1)
        self.assertIn("pca_1", X_tr.columns)
        self.assertNotIn("feature_x", X_tr.columns)
        self.assertNotIn("feature_y", X_tr.columns)

    def test_outlier_clipping(self):
        """Verifies extreme outliers are bounded by IQR limits."""
        X_tr, X_te, _ = apply_outlier_clipping(self.X_train, self.X_test, ["outlier_col"], method="iqr", factor=1.5)
        self.assertLess(X_tr["outlier_col"].max(), 200.0)
        self.assertGreater(X_tr["outlier_col"].min(), -50.0)

    def test_datetime_decomposition(self):
        """Verifies datetime components are extracted and raw string dropped."""
        X_tr, X_te, _ = apply_datetime_decomposition(self.X_train, self.X_test, ["date_col"])
        self.assertIn("date_col_year", X_tr.columns)
        self.assertIn("date_col_month", X_tr.columns)
        self.assertIn("date_col_day", X_tr.columns)
        self.assertIn("date_col_dayofweek", X_tr.columns)
        self.assertIn("date_col_is_weekend", X_tr.columns)
        self.assertNotIn("date_col", X_tr.columns)

    def test_recipe_runner_with_all_transforms(self):
        """Verifies recipe_runner executes multiple advanced steps in sequence."""
        recipe = [
            {"step": "split", "params": {"test_size": 0.2, "random_state": 42}},
            {"step": "datetime", "params": {"columns": ["date_col"]}},
            {"step": "outlier_clip", "params": {"columns": ["outlier_col"], "method": "iqr"}},
            {"step": "power", "params": {"columns": ["skewed"], "method": "log1p"}},
            {"step": "scale", "params": {"columns": ["feature_x", "feature_y"], "method": "standard"}},
        ]
        X_tr, X_te, y_tr, y_te, transformers = run_recipe(self.df, target_column="target", recipe=recipe)
        self.assertEqual(len(X_tr) + len(X_te), len(self.df))
        self.assertIn("date_col_year", X_tr.columns)
        self.assertEqual(len(transformers), 4)


if __name__ == "__main__":
    unittest.main()
