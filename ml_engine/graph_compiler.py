"""
Visual pipeline graph -> executable config + standalone Python script.

The frontend node editor sends a graph::

    {
      "nodes": [{"id": "n1", "data": {"kind": "dataset", "params": {...}}}, ...],
      "edges": [{"source": "n1", "target": "n2"}, ...]
    }

``compile_graph`` validates it, walks the chain from the dataset node and
produces:

* ``config``  – the exact ``{task_type, target_column, recipe, model_config}``
  dict that ``execute_pipeline`` runs when the user clicks *Train*;
* ``code``    – a self-contained Python script that performs the *same*
  operations in the *same* order (so the numbers it prints match the UI);
* ``blocks``  – which line range of ``code`` each node produced, so the UI can
  highlight "this node == these lines".

MVP scope: the graph must be a single linear chain (no branching/merging).
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PREPROCESS_KINDS = ("impute", "scale", "encode", "drop_columns")
NODE_KINDS = ("dataset", "split", "model", "evaluate") + PREPROCESS_KINDS

NODE_LABELS = {
    "dataset": "CSV Dataset",
    "impute": "Missing Value Imputer",
    "scale": "Feature Scaling",
    "encode": "Categorical Encoder",
    "drop_columns": "Drop Columns",
    "split": "Train / Test Split",
    "model": "Model",
    "evaluate": "Evaluation",
}

# algorithm -> (task_type, import line)
ALGORITHMS: Dict[str, Tuple[str, str]] = {
    "LogisticRegression": ("classification", "from sklearn.linear_model import LogisticRegression"),
    "DecisionTreeClassifier": ("classification", "from sklearn.tree import DecisionTreeClassifier"),
    "RandomForestClassifier": ("classification", "from sklearn.ensemble import RandomForestClassifier"),
    "GradientBoostingClassifier": ("classification", "from sklearn.ensemble import GradientBoostingClassifier"),
    "XGBClassifier": ("classification", "from xgboost import XGBClassifier"),
    "LinearRegression": ("regression", "from sklearn.linear_model import LinearRegression"),
    "DecisionTreeRegressor": ("regression", "from sklearn.tree import DecisionTreeRegressor"),
    "RandomForestRegressor": ("regression", "from sklearn.ensemble import RandomForestRegressor"),
    "GradientBoostingRegressor": ("regression", "from sklearn.ensemble import GradientBoostingRegressor"),
    "XGBRegressor": ("regression", "from xgboost import XGBRegressor"),
}

SCALERS = {"standard": "StandardScaler", "minmax": "MinMaxScaler", "robust": "RobustScaler"}
IMPUTE_STRATEGIES = ("mean", "median", "most_frequent", "constant")


def detect_task_type(series: pd.Series) -> str:
    """Same heuristic the rest of the backend uses."""
    s = series.dropna()
    if not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 10:
        return "classification"
    return "regression"


def py_literal(value: Any) -> str:
    """Render a JSON-ish value as Python source."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(py_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(str(k))}: {py_literal(v)}" for k, v in value.items()) + "}"
    return json.dumps(str(value))


def _clean_hyperparams(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (raw or {}).items():
        if v == "" or (isinstance(v, float) and v != v):
            continue
        if v in ("None", "none", "null"):
            v = None
        out[str(k)] = v
    return out


# ---------------------------------------------------------------------------
# Graph walking + validation
# ---------------------------------------------------------------------------

def _walk_chain(nodes: Dict[str, Dict], edges: List[Dict], errors: List[Dict]) -> List[str]:
    outgoing: Dict[str, List[str]] = {}
    incoming: Dict[str, List[str]] = {}
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s in nodes and t in nodes:
            outgoing.setdefault(s, []).append(t)
            incoming.setdefault(t, []).append(s)

    for nid, srcs in incoming.items():
        if len(srcs) > 1:
            errors.append({"node_id": nid, "message": "This node has more than one input. Merging branches is not supported yet — connect it to a single upstream node."})

    datasets = [nid for nid, n in nodes.items() if n["kind"] == "dataset"]
    if not datasets:
        errors.append({"node_id": None, "message": "Add a Dataset node — every pipeline starts from data."})
        return []
    if len(datasets) > 1:
        errors.append({"node_id": datasets[1], "message": "Only one Dataset node is supported per pipeline."})

    order = [datasets[0]]
    current = datasets[0]
    while True:
        nxt = outgoing.get(current, [])
        if not nxt:
            break
        if len(nxt) > 1:
            errors.append({"node_id": current, "message": "This node feeds more than one node. Branching pipelines are not supported yet — keep a single chain."})
            break
        current = nxt[0]
        if current in order:
            errors.append({"node_id": current, "message": "The graph contains a cycle. Pipelines must flow in one direction."})
            break
        order.append(current)
    return order


def compile_graph(
    graph: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    dataset_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate the graph and produce the engine config plus generated code."""
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    nodes: Dict[str, Dict[str, Any]] = {}
    for raw in graph.get("nodes", []) or []:
        data = raw.get("data", {}) or {}
        kind = data.get("kind")
        if kind not in NODE_KINDS:
            errors.append({"node_id": raw.get("id"), "message": f"Unknown node type '{kind}'."})
            continue
        nodes[str(raw["id"])] = {"id": str(raw["id"]), "kind": kind, "params": data.get("params", {}) or {}}

    order = _walk_chain(nodes, graph.get("edges", []) or [], errors)
    for nid in nodes:
        if nid not in order:
            warnings.append({"node_id": nid, "message": "Not connected to the pipeline — it will be ignored."})

    chain = [nodes[nid] for nid in order]

    # ---- dataset / target / task --------------------------------------------
    ds_params = chain[0]["params"] if chain else {}
    target = ds_params.get("target_column") or None
    task_type = ds_params.get("task_type") or "auto"
    filename = dataset_filename or ds_params.get("filename") or "your_dataset.csv"

    if chain:
        if df is None and ds_params.get("dataset_id") in (None, ""):
            errors.append({"node_id": chain[0]["id"], "message": "Choose a dataset for this node."})
        if not target:
            errors.append({"node_id": chain[0]["id"], "message": "Choose the target column (what the model should predict)."})
        elif df is not None and target not in df.columns:
            errors.append({"node_id": chain[0]["id"], "message": f"Target column '{target}' is not in the dataset."})
            target = None
    if task_type == "auto":
        task_type = detect_task_type(df[target]) if (df is not None and target) else "classification"

    # ---- ordering rules ------------------------------------------------------
    kinds = [n["kind"] for n in chain]
    model_idx = kinds.index("model") if "model" in kinds else None
    split_idxs = [i for i, k in enumerate(kinds) if k == "split"]

    if chain and model_idx is None:
        errors.append({"node_id": None, "message": "Connect a Model node to the end of the chain so there is something to train."})
    if len(split_idxs) > 1:
        errors.append({"node_id": chain[split_idxs[1]]["id"], "message": "Only one Train/Test Split node is allowed."})
    if split_idxs and model_idx is not None and split_idxs[0] > model_idx:
        errors.append({"node_id": chain[split_idxs[0]]["id"], "message": "The split must happen before the model is trained."})
    if model_idx is not None:
        for n in chain[model_idx + 1:]:
            if n["kind"] != "evaluate":
                errors.append({"node_id": n["id"], "message": "Only an Evaluation node can come after the model."})
    for n in chain[:model_idx if model_idx is not None else len(chain)]:
        if n["kind"] == "evaluate":
            errors.append({"node_id": n["id"], "message": "Evaluation needs a trained model — move it after the Model node."})

    # ---- build recipe while tracking which columns still exist ----------------
    available = [str(c) for c in df.columns] if df is not None else None
    recipe: List[Dict[str, Any]] = []
    model_config: Dict[str, Any] = {}
    steps_for_code: List[Dict[str, Any]] = []  # chain annotated with resolved columns

    for n in chain:
        kind, p = n["kind"], n["params"]
        if kind in PREPROCESS_KINDS:
            requested = [c for c in (p.get("columns") or []) if c != target]
            if available is not None:
                missing = [c for c in requested if c not in available]
                if missing:
                    warnings.append({"node_id": n["id"], "message": f"Ignoring columns no longer available at this point (dropped or encoded earlier): {', '.join(missing)}"})
                cols = [c for c in requested if c in available]
            else:
                cols = requested
            if not cols:
                warnings.append({"node_id": n["id"], "message": "No columns selected — this step will be skipped."})

            params: Dict[str, Any] = {}
            if kind == "impute":
                strategy = p.get("strategy", "mean")
                if strategy not in IMPUTE_STRATEGIES:
                    errors.append({"node_id": n["id"], "message": f"Unknown imputation strategy '{strategy}'."})
                params["strategy"] = strategy
                if strategy == "constant":
                    params["fill_value"] = p.get("fill_value", 0)
                if df is not None and strategy in ("mean", "median"):
                    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
                    if non_numeric:
                        errors.append({"node_id": n["id"], "message": f"'{strategy}' only works on numeric columns; {', '.join(non_numeric)} are text. Use 'most_frequent' for those."})
            elif kind == "scale":
                method = p.get("method", "standard")
                if method not in SCALERS:
                    errors.append({"node_id": n["id"], "message": f"Unknown scaling method '{method}'."})
                params["method"] = method
            elif kind == "encode":
                method = p.get("method", "onehot")
                if method not in ("onehot", "ordinal"):
                    errors.append({"node_id": n["id"], "message": f"Unknown encoding method '{method}'."})
                params["method"] = method

            if cols:
                recipe.append({"step": kind, "columns": cols, "params": params})
            if available is not None and cols and (kind == "drop_columns" or (kind == "encode" and params.get("method") == "onehot")):
                available = [c for c in available if c not in cols]
            steps_for_code.append({**n, "cols": cols, "resolved": params})

        elif kind == "split":
            test_size = float(p.get("test_size", 0.2))
            if not 0.05 <= test_size <= 0.5:
                errors.append({"node_id": n["id"], "message": "Test size should be between 5% and 50%."})
            params = {"test_size": test_size, "random_state": int(p.get("random_state", 42)), "stratify": bool(p.get("stratify", False))}
            if params["stratify"] and task_type != "classification":
                warnings.append({"node_id": n["id"], "message": "Stratification only applies to classification targets; it will be ignored."})
                params["stratify"] = False
            recipe.append({"step": "split", "params": params})
            steps_for_code.append({**n, "resolved": params})

        elif kind == "model":
            algorithm = p.get("algorithm") or "RandomForestClassifier"
            if algorithm not in ALGORITHMS:
                errors.append({"node_id": n["id"], "message": f"Unsupported algorithm '{algorithm}'."})
            elif ALGORITHMS[algorithm][0] != task_type:
                errors.append({"node_id": n["id"], "message": f"{algorithm} is a {ALGORITHMS[algorithm][0]} model, but target '{target}' is a {task_type} problem."})
            model_config = {"algorithm": algorithm, "hyperparameters": _clean_hyperparams(p.get("hyperparameters", {}))}
            steps_for_code.append({**n, "resolved": model_config})
        else:
            steps_for_code.append({**n, "resolved": {}})

    # Text columns that are never encoded will crash training — catch it early.
    if df is not None and available is not None and target and model_idx is not None:
        encoded_ordinal = {c for s in recipe if s["step"] == "encode" for c in s["columns"]}
        leftover_text = [c for c in available if c != target and c not in encoded_ordinal and not pd.api.types.is_numeric_dtype(df[c])]
        if leftover_text:
            errors.append({"node_id": None, "message": f"Text columns must be encoded or dropped before training: {', '.join(leftover_text)}."})

    config = {
        "task_type": task_type,
        "target_column": target,
        "recipe": recipe,
        "model_config": model_config,
    }

    code, blocks = generate_code(steps_for_code, filename, target or "target", task_type)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "order": order,
        "config": config,
        "dataset_id": ds_params.get("dataset_id"),
        "code": code,
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _banner(title: str) -> str:
    text = f"# -- {title} "
    return text + "-" * max(4, 78 - len(text))


def generate_code(chain: List[Dict[str, Any]], filename: str, target: str, task_type: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Emit a standalone script mirroring ``run_recipe`` + training + evaluation."""
    imports = {"import json", "import pandas as pd", "from sklearn.model_selection import train_test_split"}
    sections: List[Tuple[Optional[str], List[str]]] = []
    counters: Dict[str, int] = {}
    split_done = False
    has_model = False
    algorithm = None

    def var(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}_{counters[prefix]}"

    def split_lines(params: Dict[str, Any], implicit: bool) -> List[str]:
        lines = []
        if implicit:
            lines.append("# No Split node in the graph, so the default 80/20 split is used.")
        lines += [
            "X = df.drop(columns=[target])",
            "y = df[target]",
            "X_train, X_test, y_train, y_test = train_test_split(",
            f"    X, y, test_size={py_literal(params['test_size'])}, random_state={params['random_state']}"
            + (", stratify=y" if params.get("stratify") else ""),
            ")",
            'print(f"Train rows: {len(X_train):,}   Test rows: {len(X_test):,}")',
        ]
        return lines

    for node in chain:
        kind, nid = node["kind"], node["id"]
        label = NODE_LABELS[kind]
        lines: List[str] = []

        if kind == "dataset":
            reader = "read_parquet" if filename.lower().endswith(".parquet") else "read_csv"
            lines += [
                _banner(f"[{label}] Load the data"),
                f"df = pd.{reader}({py_literal(filename)})",
                f"target = {py_literal(target)}  # the column we want to predict",
                'print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")',
            ]

        elif kind in PREPROCESS_KINDS:
            cols = node.get("cols", [])
            p = node.get("resolved", {})
            stage = "learned from training rows only (after the split)" if split_done else "fitted on the full dataset (before the split)"
            title = {
                "impute": f"Fill missing values ({p.get('strategy')})",
                "scale": f"Scale numeric features ({SCALERS.get(p.get('method'), '?')})",
                "encode": f"Encode categories ({p.get('method')})",
                "drop_columns": "Drop columns",
            }[kind]
            lines.append(_banner(f"[{label}] {title}"))
            if not cols:
                lines.append("# Skipped: no columns selected for this node.")
            elif kind == "drop_columns":
                lines.append(f"cols = {py_literal(cols)}")
                if split_done:
                    lines += ["X_train = X_train.drop(columns=cols)", "X_test = X_test.drop(columns=cols)"]
                else:
                    lines.append("df = df.drop(columns=cols)")
            else:
                if kind == "impute":
                    imports.add("from sklearn.impute import SimpleImputer")
                    name = var("imputer")
                    args = f"strategy={py_literal(p['strategy'])}"
                    if p["strategy"] == "constant":
                        args += f", fill_value={py_literal(p.get('fill_value'))}"
                    ctor = f"SimpleImputer({args})"
                elif kind == "scale":
                    cls = SCALERS[p["method"]]
                    imports.add(f"from sklearn.preprocessing import {cls}")
                    name = var("scaler")
                    ctor = f"{cls}()"
                else:
                    name = var("encoder")
                    if p["method"] == "onehot":
                        imports.add("from sklearn.preprocessing import OneHotEncoder")
                        ctor = 'OneHotEncoder(sparse_output=False, handle_unknown="ignore")'
                    else:
                        imports.add("from sklearn.preprocessing import OrdinalEncoder")
                        ctor = 'OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)'

                lines.append(f"# {stage}")
                lines.append(f"cols = {py_literal(cols)}")
                lines.append(f"{name} = {ctor}")
                onehot = kind == "encode" and p["method"] == "onehot"
                if not split_done:
                    if onehot:
                        lines += [
                            f"encoded = pd.DataFrame({name}.fit_transform(df[cols]), columns={name}.get_feature_names_out(cols), index=df.index)",
                            "df = df.drop(columns=cols).join(encoded)",
                        ]
                    else:
                        lines.append(f"df[cols] = {name}.fit_transform(df[cols])")
                else:
                    if onehot:
                        lines += [
                            f"train_enc = {name}.fit_transform(X_train[cols])",
                            f"test_enc = {name}.transform(X_test[cols])",
                            f"names = {name}.get_feature_names_out(cols)",
                            "X_train = X_train.drop(columns=cols).join(pd.DataFrame(train_enc, columns=names, index=X_train.index))",
                            "X_test = X_test.drop(columns=cols).join(pd.DataFrame(test_enc, columns=names, index=X_test.index))",
                        ]
                    else:
                        lines += [
                            f"X_train[cols] = {name}.fit_transform(X_train[cols])  # learn from training data",
                            f"X_test[cols] = {name}.transform(X_test[cols])        # reuse what was learned",
                        ]

        elif kind == "split":
            lines.append(_banner(f"[{label}] Hold out data for an honest test"))
            lines += split_lines(node["resolved"], implicit=False)
            split_done = True

        elif kind == "model":
            if not split_done:
                sections.append((None, [_banner("[Train / Test Split] (default)")] + split_lines({"test_size": 0.2, "random_state": 42}, implicit=True)))
                split_done = True
            algorithm = node["resolved"].get("algorithm")
            if algorithm in ALGORITHMS:
                imports.add(ALGORITHMS[algorithm][1])
            hp = node["resolved"].get("hyperparameters", {})
            args = ", ".join(f"{k}={py_literal(v)}" for k, v in hp.items())
            lines += [
                _banner(f"[{label}] Train {algorithm}"),
                f"model = {algorithm}({args})",
                "model.fit(X_train, y_train)",
                'print("Training complete.")',
            ]
            has_model = True

        elif kind == "evaluate":
            lines += _evaluation_lines(task_type, imports)

        sections.append((nid, lines))

    if has_model and not any(n["kind"] == "evaluate" for n in chain):
        sections.append((None, _evaluation_lines(task_type, imports)))
    if has_model:
        imports.add("import joblib")
        sections.append((None, [
            _banner("Save the trained model"),
            'joblib.dump(model, "model.joblib")',
            'print("Saved model.joblib")',
        ]))
    elif chain:
        sections.append((None, ["# Connect a Model node to generate the training code."]))

    needs_xgb = algorithm is not None and algorithm.startswith("XGB")
    header = [
        '"""',
        "Auto-generated by Interactive Model Studio from your visual pipeline.",
        "Each block is tagged with the node that produced it, and runs the same",
        "steps in the same order as the Studio's Train button.",
        "",
        "Run it with:",
        "    pip install pandas scikit-learn joblib" + (" xgboost" if needs_xgb else ""),
        "    python train_model.py",
        '"""',
        "",
    ]
    import_lines = sorted(i for i in imports if i.startswith("import ")) + sorted(i for i in imports if i.startswith("from "))

    out: List[str] = header + import_lines + [""]
    blocks: List[Dict[str, Any]] = []
    for nid, lines in sections:
        start = len(out) + 1
        out.extend(lines)
        if nid is not None:
            blocks.append({"node_id": nid, "start": start, "end": len(out)})
        out.append("")
    return "\n".join(out).rstrip() + "\n", blocks


def _evaluation_lines(task_type: str, imports: set) -> List[str]:
    if task_type == "regression":
        imports.add("import numpy as np")
        imports.add("from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score")
        return [
            _banner("[Evaluation] Score the model on the held-out test set"),
            "y_pred = model.predict(X_test)",
            "metrics = {",
            '    "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),',
            '    "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),',
            '    "r2": round(float(r2_score(y_test, y_pred)), 4),',
            "}",
            "print(json.dumps(metrics, indent=2))",
        ]
    imports.add("from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score")
    return [
        _banner("[Evaluation] Score the model on the held-out test set"),
        "y_pred = model.predict(X_test)",
        "metrics = {",
        '    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),',
        '    "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),',
        '    "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),',
        '    "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),',
        "}",
        "print(json.dumps(metrics, indent=2))",
    ]
