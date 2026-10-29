"""
Unit Tests for Data Loader, Type Coercion, and Large Dataset Memory Optimization.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from ml_engine.data_loader import (
    load_dataset,
    optimize_memory,
    detect_and_coerce_types,
    load_dataset_chunked,
    get_dataset_type_summary,
)


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Diverse dataset with mixed types: floats, ints, bool strings, date strings, low-cardinality strings
        n = 200
        self.df = pd.DataFrame({
            "numeric_float": np.random.uniform(10.0, 100.0, n).astype(np.float64),
            "numeric_int": np.random.randint(1, 50, n).astype(np.int64),
            "bool_strings": ["yes", "no", "true", "false"] * 50,
            "date_strings": ["2023-01-15", "2023-02-20", "2023-03-25", "2023-04-30"] * 50,
            "category_strings": ["North", "South", "East", "West"] * 50,
            "target": np.random.choice([0, 1], n),
        })

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_type_coercion(self):
        """Verifies automatic detection and coercion of booleans and datetimes."""
        coerced = detect_and_coerce_types(self.df)
        self.assertTrue(pd.api.types.is_bool_dtype(coerced["bool_strings"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(coerced["date_strings"]))

    def test_memory_optimization(self):
        """Verifies downcasting reduces memory consumption significantly."""
        opt_df, report = optimize_memory(self.df)
        self.assertEqual(opt_df["numeric_float"].dtype, np.float32)
        # int64 with max 50 should downcast to int8
        self.assertIn(str(opt_df["numeric_int"].dtype), ["int8", "int16", "int32"])
        self.assertEqual(str(opt_df["category_strings"].dtype), "category")
        self.assertGreater(report["percent_reduction"], 30.0)

    def test_load_csv_and_parquet(self):
        """Verifies unified loader works across CSV and Parquet."""
        csv_path = self.dir_path / "test_data.csv"
        parquet_path = self.dir_path / "test_data.parquet"

        self.df.to_csv(csv_path, index=False)
        self.df.to_parquet(parquet_path, index=False)

        # Load CSV
        df_csv, meta_csv = load_dataset(csv_path)
        self.assertEqual(meta_csv["format"], ".csv")
        self.assertEqual(df_csv.shape, self.df.shape)

        # Load Parquet
        df_pq, meta_pq = load_dataset(parquet_path)
        self.assertEqual(meta_pq["format"], ".parquet")
        self.assertEqual(df_pq.shape, self.df.shape)

    def test_chunked_streaming(self):
        """Verifies chunked loading streams expected row counts."""
        csv_path = self.dir_path / "test_chunk.csv"
        self.df.to_csv(csv_path, index=False)

        chunks = list(load_dataset_chunked(csv_path, chunksize=50))
        self.assertEqual(len(chunks), 4)  # 200 rows / 50 = 4 chunks
        total_rows = sum(len(c) for c in chunks)
        self.assertEqual(total_rows, 200)

    def test_type_summary(self):
        """Verifies semantic category segmentation."""
        coerced = detect_and_coerce_types(self.df)
        opt_df, _ = optimize_memory(coerced)
        summary = get_dataset_type_summary(opt_df)

        self.assertIn("numeric_float", summary["numeric"])
        self.assertIn("bool_strings", summary["boolean"])
        self.assertIn("date_strings", summary["datetime"])
        self.assertIn("category_strings", summary["categorical"])


if __name__ == "__main__":
    unittest.main()
