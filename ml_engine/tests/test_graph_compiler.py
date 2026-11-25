import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml_engine.graph_compiler import compile_graph
from ml_engine.pipeline_executor import execute_pipeline


def _node(nid, kind, **params):
    return {"id": nid, "data": {"kind": kind, "params": params}}


def _chain(*nodes):
    edges = [{"source": a["id"], "target": b["id"]} for a, b in zip(nodes, nodes[1:])]
    return {"nodes": list(nodes), "edges": edges}


def _make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, n)
    income = rng.normal(50000, 15000, n)
    city = rng.choice(["Pune", "Delhi", "Mumbai"], n)
    plan = rng.choice(["basic", "pro"], n)
    age[rng.choice(n, 15, replace=False)] = np.nan
    city = city.astype(object)
    city[rng.choice(n, 10, replace=False)] = None
    target = ((age > 40) ^ (plan == "pro")).astype(int)
    price = income * 0.01 + rng.normal(0, 5, n)
    return pd.DataFrame({
        "customer_id": np.arange(n), "age": age, "income": income,
        "city": city, "plan": plan, "price": price, "churn": target,
    })


def _run_script(code, df, filename="data.csv"):
    with tempfile.TemporaryDirectory() as tmp:
        df.to_csv(os.path.join(tmp, filename), index=False)
        script = os.path.join(tmp, "train_model.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        proc = subprocess.run([sys.executable, script], cwd=tmp, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise AssertionError(f"Generated script failed:\n{proc.stderr}\n---\n{code}")
        out = proc.stdout
        return json.loads(out[out.index("{"): out.rindex("}") + 1])


class TestGraphCompiler(unittest.TestCase):
    def setUp(self):
        self.df = _make_df()

    def _classification_graph(self, split_position="middle", encode="onehot"):
        ds = _node("d", "dataset", dataset_id=1, target_column="churn", task_type="auto")
        drop = _node("x", "drop_columns", columns=["customer_id", "price"])
        imp_num = _node("i1", "impute", strategy="median", columns=["age", "income"])
        imp_cat = _node("i2", "impute", strategy="most_frequent", columns=["city"])
        enc = _node("e", "encode", method=encode, columns=["city", "plan"])
        split = _node("s", "split", test_size=0.25, random_state=7, stratify=True)
        scale = _node("sc", "scale", method="standard", columns=["age", "income"])
        model = _node("m", "model", algorithm="RandomForestClassifier", hyperparameters={"n_estimators": 30, "random_state": 0, "max_depth": ""})
        ev = _node("ev", "evaluate")
        if split_position == "middle":
            return _chain(ds, drop, imp_num, imp_cat, enc, split, scale, model, ev)
        return _chain(ds, drop, imp_num, imp_cat, enc, scale, model, ev)  # implicit split

    def _assert_script_matches_engine(self, graph, df, task_keys):
        result = compile_graph(graph, df=df, dataset_filename="data.csv")
        self.assertTrue(result["ok"], result["errors"])
        engine = execute_pipeline(df, result["config"], use_cache=False)
        script_metrics = _run_script(result["code"], df)
        for key in task_keys:
            self.assertAlmostEqual(script_metrics[key], engine["metrics"][key], places=4, msg=key)
        return result, engine

    def test_generated_script_matches_engine_post_split_steps(self):
        result, _ = self._assert_script_matches_engine(
            self._classification_graph("middle"), self.df,
            ["accuracy", "precision_macro", "recall_macro", "f1_macro"])
        # Scaler after split must be fitted on training rows only
        self.assertIn("X_test[cols] = scaler_1.transform(X_test[cols])", result["code"])
        self.assertEqual(result["config"]["recipe"][-2]["step"], "split")

    def test_generated_script_matches_engine_implicit_split_ordinal(self):
        result, _ = self._assert_script_matches_engine(
            self._classification_graph("none", encode="ordinal"), self.df,
            ["accuracy", "f1_macro"])
        self.assertIn("default 80/20 split", result["code"])

    def test_generated_script_matches_engine_regression(self):
        graph = _chain(
            _node("d", "dataset", dataset_id=1, target_column="price"),
            _node("x", "drop_columns", columns=["customer_id", "city", "plan"]),
            _node("i", "impute", strategy="mean", columns=["age"]),
            _node("s", "split", test_size=0.2, random_state=1),
            _node("m", "model", algorithm="LinearRegression", hyperparameters={}),
        )
        result, _ = self._assert_script_matches_engine(graph, self.df, ["rmse", "mae", "r2"])
        self.assertEqual(result["config"]["task_type"], "regression")

    def test_blocks_map_nodes_to_code_lines(self):
        result = compile_graph(self._classification_graph(), df=self.df, dataset_filename="data.csv")
        lines = result["code"].splitlines()
        block = next(b for b in result["blocks"] if b["node_id"] == "m")
        snippet = "\n".join(lines[block["start"] - 1: block["end"]])
        self.assertIn("RandomForestClassifier(", snippet)
        self.assertIn("model.fit(X_train, y_train)", snippet)

    def test_validation_errors(self):
        ds = _node("d", "dataset", dataset_id=1, target_column="churn")
        # no model
        r = compile_graph(_chain(ds, _node("s", "split")), df=self.df)
        self.assertFalse(r["ok"])
        self.assertTrue(any("Model node" in e["message"] for e in r["errors"]))
        # task mismatch + unencoded text columns
        r = compile_graph(_chain(ds, _node("m", "model", algorithm="LinearRegression")), df=self.df)
        msgs = " ".join(e["message"] for e in r["errors"])
        self.assertIn("regression model", msgs)
        self.assertIn("Text columns must be encoded", msgs)
        # branching
        g = _chain(ds, _node("a", "impute", columns=["age"]))
        g["nodes"].append(_node("b", "scale", columns=["age"]))
        g["edges"].append({"source": "d", "target": "b"})
        r = compile_graph(g, df=self.df)
        self.assertTrue(any("Branching" in e["message"] for e in r["errors"]))
        # mean on text column
        r = compile_graph(_chain(ds, _node("i", "impute", strategy="mean", columns=["city"])), df=self.df)
        self.assertTrue(any("only works on numeric" in e["message"] for e in r["errors"]))

    def test_disconnected_node_warns(self):
        g = self._classification_graph()
        g["nodes"].append(_node("lonely", "scale", columns=["age"]))
        r = compile_graph(g, df=self.df)
        self.assertTrue(r["ok"])
        self.assertTrue(any(w["node_id"] == "lonely" for w in r["warnings"]))


if __name__ == "__main__":
    unittest.main()
