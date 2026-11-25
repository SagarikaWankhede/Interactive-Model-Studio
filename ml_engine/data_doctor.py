"""
Data Doctor — automatic dataset sanity report.

Flags problems that silently ruin models (missing data, duplicates, class
imbalance, outliers, redundant or leaky features, numbers stored as text).
It never modifies the data: every finding is a prompt to investigate.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Thresholds (kept in one place so the report can explain them)
MISSING_WARN_PCT = 5.0
MISSING_CRIT_PCT = 30.0
IMBALANCE_WARN_RATIO = 3.0      # majority / minority
IMBALANCE_CRIT_RATIO = 10.0
CORR_WARN = 0.9
LEAKAGE_CORR = 0.95
IQR_K = 3.0                     # "far out" fences — flags extreme values only
MAX_EXAMPLES = 5


def _status(level: str) -> str:
    return {"ok": "ok", "warn": "warn", "crit": "critical"}[level]


def _to_py(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def detect_target(df: pd.DataFrame, target: Optional[str]) -> str:
    if target and target in df.columns:
        return target
    for c in df.columns:
        if str(c).lower() in ("target", "label", "churn", "class", "y"):
            return c
    return df.columns[-1]


def analyze_dataset(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Return a structured data-quality report for ``df``."""
    rows, cols = df.shape
    target = detect_target(df, target_column) if cols else None
    checks: List[Dict[str, Any]] = []

    # ---- Missing values ------------------------------------------------------
    missing_per_col = []
    total_cells = rows * cols
    total_missing = int(df.isna().sum().sum())
    missing_pct = round(total_missing / total_cells * 100, 2) if total_cells else 0.0
    for c in df.columns:
        n = int(df[c].isna().sum())
        if n:
            missing_per_col.append({"column": str(c), "count": n, "pct": round(n / rows * 100, 2)})
    missing_per_col.sort(key=lambda d: d["pct"], reverse=True)
    worst = missing_per_col[0]["pct"] if missing_per_col else 0.0
    level = "crit" if worst >= MISSING_CRIT_PCT else ("warn" if worst >= MISSING_WARN_PCT or missing_pct > 0 else "ok")
    checks.append({
        "id": "missing",
        "label": "Missing values",
        "status": _status(level),
        "value": f"{missing_pct}% of cells",
        "summary": (f"{len(missing_per_col)} column(s) have gaps; worst is '{missing_per_col[0]['column']}' at {worst}%."
                    if missing_per_col else "No missing values."),
        "why": "Most models cannot handle empty cells. How you fill them (mean, median, most frequent) changes what the model learns.",
        "suggestion": "Add a Missing Value Imputer node; consider dropping columns that are mostly empty." if missing_per_col else None,
        "details": missing_per_col,
    })

    # ---- Duplicates ----------------------------------------------------------
    dup_count = int(df.duplicated().sum()) if rows else 0
    dup_pct = round(dup_count / rows * 100, 2) if rows else 0.0
    checks.append({
        "id": "duplicates",
        "label": "Duplicate rows",
        "status": _status("warn" if dup_count else "ok"),
        "value": f"{dup_count:,} ({dup_pct}%)",
        "summary": f"{dup_count:,} exact duplicate rows found." if dup_count else "No duplicate rows.",
        "why": "Duplicates can land in both the training and test sets, making the test score look better than reality.",
        "suggestion": "Check whether duplicates are real repeated events or a data-collection bug." if dup_count else None,
        "details": [],
    })

    # ---- Data types ----------------------------------------------------------
    type_issues = []
    for c in df.columns:
        s = df[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            non_null = s.dropna().astype(str).str.strip()
            if len(non_null) == 0:
                continue
            as_num = pd.to_numeric(non_null.str.replace(",", "", regex=False), errors="coerce")
            parse_rate = float(as_num.notna().mean())
            if parse_rate >= 0.9:
                bad = non_null[as_num.isna()].unique()[:MAX_EXAMPLES]
                type_issues.append({
                    "column": str(c),
                    "issue": "numbers stored as text",
                    "detail": f"{parse_rate*100:.0f}% of values are numeric" + (f"; non-numeric examples: {', '.join(map(str, bad))}" if len(bad) else ""),
                })
        if s.nunique(dropna=True) <= 1 and rows > 1:
            type_issues.append({"column": str(c), "issue": "constant column", "detail": "Only one distinct value — it carries no information."})
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    checks.append({
        "id": "types",
        "label": "Data types",
        "status": _status("warn" if type_issues else "ok"),
        "value": f"{len(numeric_cols)} numeric / {len(categorical_cols)} categorical",
        "summary": f"{len(type_issues)} column(s) look mistyped or useless." if type_issues else "Column types look consistent.",
        "why": "A numeric column read as text gets one-hot encoded into hundreds of useless features instead of being used as a number.",
        "suggestion": "Clean the source file (e.g. stray '?' or 'N/A' strings) or drop constant columns." if type_issues else None,
        "details": type_issues,
    })

    # ---- Class balance -------------------------------------------------------
    task_type = None
    class_balance = None
    if target is not None:
        t = df[target].dropna()
        task_type = "classification" if (not pd.api.types.is_numeric_dtype(t) or t.nunique() <= 10) else "regression"
        if task_type == "classification" and len(t):
            counts = t.value_counts()
            ratio = float(counts.iloc[0] / counts.iloc[-1]) if counts.iloc[-1] else float("inf")
            class_balance = [{"label": str(k), "count": int(v), "pct": round(v / len(t) * 100, 2)} for k, v in counts.items()]
            level = "crit" if ratio >= IMBALANCE_CRIT_RATIO else ("warn" if ratio >= IMBALANCE_WARN_RATIO else "ok")
            majority = class_balance[0]
            checks.append({
                "id": "balance",
                "label": "Class balance",
                "status": _status(level),
                "value": f"{ratio:.1f} : 1",
                "summary": (f"'{majority['label']}' is {majority['pct']}% of rows. A model that always predicts it scores {majority['pct']}% accuracy while being useless."
                            if level != "ok" else f"{len(counts)} classes, reasonably balanced."),
                "why": "With imbalanced classes, accuracy is misleading. Look at recall / F1 for the minority class instead.",
                "suggestion": "Turn on 'stratify' in the Split node and judge the model by F1 / recall, not accuracy." if level != "ok" else None,
                "details": class_balance,
            })

    # ---- Outliers (IQR "far out" fences) -------------------------------------
    outliers = []
    for c in numeric_cols:
        if c == target:
            continue
        s = df[c].dropna()
        if len(s) < 10 or s.nunique() < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
        mask = (s < lo) | (s > hi)
        n = int(mask.sum())
        if n:
            flagged = s[mask]
            # most extreme first (furthest from the median)
            examples = flagged.loc[(flagged - s.median()).abs().sort_values(ascending=False).index][:MAX_EXAMPLES]
            outliers.append({
                "column": str(c),
                "count": n,
                "pct": round(n / len(s) * 100, 2),
                "median": _to_py(s.median()),
                "lower_fence": _to_py(lo),
                "upper_fence": _to_py(hi),
                "examples": [_to_py(v) for v in examples.tolist()],
            })
    outliers.sort(key=lambda d: d["count"], reverse=True)
    checks.append({
        "id": "outliers",
        "label": "Outliers",
        "status": _status("warn" if outliers else "ok"),
        "value": f"{len(outliers)} feature(s)",
        "summary": (f"Extreme values in {', '.join(o['column'] for o in outliers[:4])}{'…' if len(outliers) > 4 else ''}."
                    if outliers else "No extreme values detected."),
        "why": f"Values beyond Q1 − {IQR_K:g}·IQR or Q3 + {IQR_K:g}·IQR are flagged. They may be typos (₹9,00,00,000 instead of ₹90,000) or genuine rare cases — investigate, don't blindly delete.",
        "suggestion": "Inspect the flagged rows; a Robust scaler is less sensitive to outliers than Standard." if outliers else None,
        "details": outliers,
    })

    # ---- Correlation / leakage -----------------------------------------------
    corr_pairs = []
    leakage = []
    feature_num = [c for c in numeric_cols if c != target]
    if len(feature_num) >= 2:
        corr = df[feature_num].corr().abs()
        for i, a in enumerate(feature_num):
            for b in feature_num[i + 1:]:
                v = corr.loc[a, b]
                if pd.notna(v) and v >= CORR_WARN:
                    corr_pairs.append({"a": str(a), "b": str(b), "corr": round(float(v), 3)})
        corr_pairs.sort(key=lambda d: d["corr"], reverse=True)
    if target is not None and pd.api.types.is_numeric_dtype(df[target]):
        for c in feature_num:
            try:
                v = abs(df[c].corr(df[target]))
            except Exception:
                continue
            if pd.notna(v) and v >= LEAKAGE_CORR:
                leakage.append({"column": str(c), "corr": round(float(v), 3)})
    level = "crit" if leakage else ("warn" if corr_pairs else "ok")
    summary_bits = []
    if leakage:
        summary_bits.append(f"Possible target leakage: {', '.join(l['column'] for l in leakage)} track the target almost perfectly.")
    if corr_pairs:
        top = corr_pairs[0]
        summary_bits.append(f"'{top['a']}' and '{top['b']}' are {top['corr']:.2f} correlated" + (f" (+{len(corr_pairs)-1} more pairs)." if len(corr_pairs) > 1 else "."))
    checks.append({
        "id": "correlation",
        "label": "Correlation",
        "status": _status(level),
        "value": f"{len(corr_pairs)} pair(s)" + (f", {len(leakage)} leak(s)" if leakage else ""),
        "summary": " ".join(summary_bits) or "No highly correlated features.",
        "why": "Highly correlated features (e.g. Age and BirthYear) carry the same information. A feature that matches the target almost exactly usually means the answer leaked into the inputs.",
        "suggestion": "Drop one of each redundant pair; drop leaky columns before training." if (corr_pairs or leakage) else None,
        "details": {"pairs": corr_pairs, "leakage": leakage},
    })

    order = {"critical": 0, "warn": 1, "ok": 2}
    score = 100 - sum({"critical": 25, "warn": 8, "ok": 0}[c["status"]] for c in checks)

    return {
        "rows": rows,
        "columns": cols,
        "target_column": str(target) if target is not None else None,
        "task_type": task_type,
        "health_score": max(0, score),
        "checks": sorted(checks, key=lambda c: order[c["status"]]),
        "column_summary": [
            {
                "name": str(c),
                "type": "numeric" if c in numeric_cols else "categorical",
                "missing_pct": round(float(df[c].isna().mean() * 100), 2) if rows else 0.0,
                "unique": int(df[c].nunique(dropna=True)),
            }
            for c in df.columns
        ],
        "class_balance": class_balance,
    }
