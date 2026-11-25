"""End-to-end API tests: visual graph -> train -> ONNX -> data doctor -> active learning."""
import io
import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app


def _node(nid, kind, **params):
    return {"id": nid, "type": "mlNode", "position": {"x": 0, "y": 0}, "data": {"kind": kind, "params": params}}


class TestStudioAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        rng = np.random.default_rng(1)
        n = 300
        age = rng.normal(40, 10, n)
        age[:12] = np.nan
        salary = rng.normal(50000, 8000, n)
        salary[5] = 90_000_000  # the "₹9,00,00,000" typo
        plan = rng.choice(["basic", "pro", "max"], n)
        churn = np.where((np.nan_to_num(age, nan=40) + rng.normal(0, 6, n)) > 42, "Yes", "No")
        df = pd.DataFrame({"age": age, "birth_year": 2026 - np.nan_to_num(age, nan=40), "salary": salary,
                           "plan": plan, "churn": churn})
        df = pd.concat([df, df.iloc[:3]], ignore_index=True)  # 3 duplicates
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        res = cls.client.post("/api/datasets/upload", files={"file": ("customers.csv", buf.getvalue(), "text/csv")})
        assert res.status_code == 200, res.text
        cls.dataset_id = res.json()["id"]

    def graph(self, **overrides):
        ds = _node("d", "dataset", dataset_id=self.dataset_id, target_column="churn", task_type="auto")
        nodes = [
            ds,
            _node("x", "drop_columns", columns=["birth_year"]),
            _node("i", "impute", strategy="median", columns=["age"]),
            _node("e", "encode", method="onehot", columns=["plan"]),
            _node("s", "split", test_size=0.25, random_state=3, stratify=True),
            _node("sc", "scale", method="robust", columns=["age", "salary"]),
            _node("m", "model", algorithm=overrides.get("algorithm", "LogisticRegression"), hyperparameters={"max_iter": 500}),
            _node("ev", "evaluate"),
        ]
        edges = [{"id": f"e{i}", "source": a["id"], "target": b["id"]} for i, (a, b) in enumerate(zip(nodes, nodes[1:]))]
        return {"nodes": nodes, "edges": edges}

    def test_01_compile_and_run_graph(self):
        r = self.client.post("/api/graph/compile", json=self.graph())
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["ok"], body["errors"])
        self.assertIn('pd.read_csv("customers.csv")', body["code"])

        bad = self.graph(algorithm="LinearRegression")
        r = self.client.post("/api/graph/run", json=bad)
        self.assertEqual(r.status_code, 422)

        r = self.client.post("/api/graph/run", json=self.graph())
        self.assertEqual(r.status_code, 200, r.text)
        run = r.json()["run"]
        self.assertIn("accuracy", run["metrics"])
        TestStudioAPI.run_id = run["run_id"]

        r = self.client.get(f"/api/graph/runs/{self.run_id}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["graph"]["nodes"]), 8)

    def test_02_onnx_export_and_parity(self):
        r = self.client.post(f"/api/runs/{self.run_id}/onnx?quantize=true")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["parity"]["passed"], body["parity"])
        self.assertFalse(body["quantization"]["applied"])  # linear sklearn model: nothing to quantize
        spec = body["spec"]
        self.assertEqual([s["step"] for s in spec["steps"]], ["drop_columns", "impute", "encode", "scale"])
        self.assertEqual(sorted(spec["classes"]), ["No", "Yes"])
        self.assertEqual({i["name"] for i in spec["inputs"]}, {"age", "salary", "plan"})

        r = self.client.get(f"/api/runs/{self.run_id}/onnx/model")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(len(r.content), 100)
        self.assertEqual(self.client.get(f"/api/runs/{self.run_id}/onnx").status_code, 200)

    def test_03_data_doctor(self):
        r = self.client.get(f"/api/datasets/{self.dataset_id}/report?target=churn")
        self.assertEqual(r.status_code, 200, r.text)
        rep = r.json()
        checks = {c["id"]: c for c in rep["checks"]}
        self.assertEqual(rep["rows"], 303)
        self.assertNotEqual(checks["missing"]["status"], "ok")
        self.assertEqual(checks["duplicates"]["status"], "warn")
        self.assertIn("salary", [o["column"] for o in checks["outliers"]["details"]])
        self.assertIn(90_000_000, checks["outliers"]["details"][0]["examples"])
        pairs = checks["correlation"]["details"]["pairs"]
        self.assertTrue(any({p["a"], p["b"]} == {"age", "birth_year"} for p in pairs))

    def test_04_active_learning_loop(self):
        r = self.client.post(f"/api/runs/{self.run_id}/scan?threshold=0.7&limit=5")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertGreater(r.json()["queued"], 0)

        # A low-confidence prediction coming from the browser
        r = self.client.post("/api/edge-cases", json={
            "run_id": self.run_id, "source": "browser",
            "features": {"age": "41", "salary": 52000, "plan": "pro"},
            "predicted_label": "Yes", "confidence": 0.52, "probabilities": {"No": 0.48, "Yes": 0.52}})
        self.assertEqual(r.status_code, 200, r.text)

        cases = self.client.get(f"/api/edge-cases?run_id={self.run_id}&status=pending").json()
        scan_cases = [c for c in cases if c["source"] == "scan"]
        browser_case = next(c for c in cases if c["source"] == "browser")
        self.assertGreaterEqual(len(scan_cases), 2)
        self.assertEqual(scan_cases[0]["row_index"] is not None, True)
        for c in (scan_cases[0], browser_case):
            self.assertEqual(self.client.post(f"/api/edge-cases/{c['id']}/label", json={"label": "No"}).status_code, 200)
        self.assertEqual(self.client.post(f"/api/edge-cases/{scan_cases[-1]['id']}/dismiss").status_code, 200)

        r = self.client.post(f"/api/runs/{self.run_id}/retrain")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["appended_rows"], 1)
        self.assertEqual(body["new_dataset"]["rows"], 304)
        self.assertEqual(body["new_run"]["parent_run_id"], self.run_id)
        self.assertIn("_al", body["new_dataset"]["filename"])

        # Nothing left to apply
        self.assertEqual(self.client.post(f"/api/runs/{self.run_id}/retrain").status_code, 400)
        summary = self.client.get(f"/api/runs/{self.run_id}/active-learning").json()
        self.assertEqual(sorted(summary["classes"]), ["No", "Yes"])
        self.assertGreaterEqual(summary["counts"].get("applied", 0), 2)


if __name__ == "__main__":
    unittest.main()
