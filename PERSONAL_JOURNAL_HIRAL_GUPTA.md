# UCS503 Software Engineering — Personal Project Journal
## Interactive Model Studio (Visual ML Platform)

**Student Name:** Hiral Gupta  
**Roll Number:** 1024030818  
**Subject:** UCS503 — Software Engineering  
**Role & Persona:** **Person B — Machine Learning Engine, Pipeline Execution & Experiment Tracking**  
**Repository Branch:** `main` / `feature/b-pipeline-schema`  
**Date of Submission:** November 2026  

---

## 1. Executive Summary & Persona Overview

In our team project, **Interactive Model Studio**, we set out to build an end-to-end, web-based visual machine learning platform designed to democratize dataset ingestion, preprocessing, visual pipeline construction, model training, and experiment comparison.

The project was divided into three core architectural roles:
- **Person A:** Backend API, Database Schemas, Dataset Storage & Authentication.
- **Person B (My Role — Hiral Gupta):** **Core Machine Learning Engine, Preprocessing & Feature Engineering Recipes, Pipeline Execution Orchestrator, Model Training & Evaluation, ONNX Model Serialization, and Reproducibility Verification Suite.**
- **Person C:** Frontend UI/UX, Visual Canvas (React Flow), Component Library, and User Dashboards.

As **Person B**, I owned the entire scientific and computational backbone of the application. My code is housed within the `ml_engine/` directory and interfaces seamlessly with the FastAPI backend and React frontend. This journal serves as an exhaustive, step-by-step personal reflection detailing my engineering decisions, architectural designs, the real technical bottlenecks and bugs I encountered at each stage, how I debugged and resolved them, and how core Software Engineering (UCS503) principles were applied.

---

## 2. Architectural Responsibilities & System Decomposition

To ensure strict separation of concerns, high cohesion, and loose coupling (fundamental UCS503 principles), I decomposed the ML Engine into modular, single-responsibility components:

```
+-------------------------------------------------------------------------+
|                Raw Dataset (.csv, .tsv, .parquet, .json)                |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|       data_loader.py (Chunked Streaming Reader & RAM Downcasting)       |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|        recipe_runner.py & preprocessing.py (Transform Pipeline)         |
+-------------------------------------------------------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
              [Cache Miss]                     [Cache Hit]
                    |                               |
                    v                               v
+---------------------------------------+   +-----------------------------+
| Scalers, Imputers, Encoders,          |   | Instant Preprocessed Arrays |
| 6 Feature Transforms                  |   | (MD5 Hash Check on Recipe)  |
+---------------------------------------+   +-----------------------------+
                    \                               /
                     +--------------+--------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|          training.py (Model Factory: RF, LR, GBDT, Decision Tree)       |
+-------------------------------------------------------------------------+
                  /                                           \
                 v                                             v
+-----------------------------------+         +---------------------------+
| evaluation.py                     |         | export_onnx.py            |
| (Metrics, Confusion Matrix, FI)   |         | (skl2onnx Serialization)  |
+-----------------------------------+         +---------------------------+
                 |                                             |
                 v                                             v
+-----------------------------------+         +---------------------------+
| reproduce.py                      |         | Production Artifacts      |
| (Tolerance Auditor: eps = 1e-4)   |         | (.onnx + metadata.json)   |
+-----------------------------------+         +---------------------------+
```

### Module Breakdown
1. **`ml_engine/data_loader.py`**:
   - Universal file reader handling `CSV`, `TSV`, `Parquet`, and `JSON`.
   - Automated numeric downcasting (`int64` -> `int8/16/32`, `float64` -> `float32`) delivering a **~70.9% memory reduction**.
   - Generator-based chunked streaming reader for large datasets preventing out-of-memory (OOM) crashes.
2. **`ml_engine/preprocessing.py` & `ml_engine/recipe_runner.py`**:
   - Declarative JSON-driven transformation recipe engine.
   - Missing value imputation (`mean`, `median`, `most_frequent`, `constant`).
   - Feature scalers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`).
   - Categorical encoders (`OneHotEncoder`, `OrdinalEncoder`).
   - **6 Faculty-Requested Feature Transforms**:
     - *Power Transforms*: Box-Cox, Yeo-Johnson, Log1p, Square Root.
     - *Outlier Clipping*: Interquartile Range (IQR) fence winsorization.
     - *Discretization / Binning*: `KBinsDiscretizer` (uniform, quantile, kmeans).
     - *Polynomial Features*: Degree-2 and interaction terms.
     - *Dimensionality Reduction*: Principal Component Analysis (PCA) with explained variance retention.
     - *Datetime Feature Extraction*: Parsing timestamps into year, month, day, day-of-week, hour.
3. **`ml_engine/training.py`**:
   - Factory pattern implementation for scikit-learn estimators.
   - Classification: `RandomForestClassifier`, `LogisticRegression`, `GradientBoostingClassifier`, `DecisionTreeClassifier`.
   - Regression: `RandomForestRegressor`, `LinearRegression`, `GradientBoostingRegressor`, `DecisionTreeRegressor`.
   - Explicit timing measurement (`train_time_sec`).
4. **`ml_engine/evaluation.py`**:
   - Metric computation: Accuracy, F1-Score (macro/weighted), Precision, Recall, ROC-AUC, Log-Loss for classification; RMSE, MAE, R2 for regression.
   - Diagnostic artifacts: Multiclass Confusion Matrices, Normalized Confusion Matrices, and Feature Importance / Linear Coefficient rankings.
5. **`ml_engine/pipeline_executor.py`**:
   - Central pipeline orchestrator. Computes deterministic MD5 hashes of preprocessing recipes and dataset shapes to bypass redundant feature transformations when comparing multiple models on the same feature set.
   - Generates unified run records and triggers ONNX model serialization.
6. **`ml_engine/export_onnx.py`**:
   - Converts trained scikit-learn models to vendor-neutral `.onnx` formats using `skl2onnx`, enabling high-performance production inference outside of Python runtimes.
7. **`ml_engine/reproduce.py`**:
   - Independent verification engine. Reconstructs identical train/test splits, re-fits pipelines from cold storage with zero caching, and verifies reproducibility against original runs within a numerical tolerance of eps = 10^-4.

---

## 3. Phase-by-Phase Development Log & Chronology

### Phase 1: Milestone 50% — Core ML Engine Foundation
- **Goal:** Establish basic tabular preprocessing, model training for classification and regression, and metric extraction.
- **Implementation:**
  - Implemented `ModelFactory` to instantiate estimators based on user-supplied strings and hyperparameter dictionaries.
  - Built `recipe_runner.py` to chain column-specific imputation, scaling, and one-hot encoding.
  - Implemented `pipeline_executor.py` with in-memory hashing to benchmark performance when training 3 different algorithms (Random Forest, Logistic Regression, Gradient Boosting) on a shared dataset.
  - Wrote initial unit tests for classification and regression pipelines.

### Phase 2: Milestone 60% — Faculty Feedback & Advanced Capabilities
- **Faculty Requirements Received:**
  1. *Support all data types and formats (CSV, TSV, Parquet, JSON).*
  2. *Support large dataset ingestion without memory crashes.*
  3. *Incorporate advanced feature engineering and transformations.*
  4. *Ensure production model portability and strict reproducibility verification.*
- **Implementation:**
  - Designed and implemented `ml_engine/data_loader.py` with multi-format parsing, memory downcasting, and chunked streaming.
  - Added the 6 advanced transformation techniques to `ml_engine/recipe_runner.py`.
  - Implemented `ml_engine/export_onnx.py` for model portability.
  - Extended the unit test suite to 26 exhaustive tests, verifying zero regressions across all new modules.
  - Synced git branches, committed all deliverables, and pushed upstream to GitHub.

### Phase 3: Integration & Localhost Deployment
- **Goal:** Launch the full platform locally so the complete visual studio can be demonstrated interactively to faculty.
- **Implementation:**
  - Resolved local frontend package installation issues (`react-router-dom`) and CSS minification bugs.
  - Booted the FastAPI backend server (`http://localhost:8000`) with interactive OpenAPI Swagger docs (`http://localhost:8000/docs`).
  - Booted the Vite React frontend server (`http://localhost:5173`) running the interactive UI.
  - Validated that the visual UI communicates cleanly with the underlying ML Engine.

---

## 4. Problems Faced, Root-Cause Analysis & Solutions

During the development of this project, I encountered severe real-world engineering obstacles. Below is an authentic record of the problems faced at every step and the rigorous debugging techniques used to solve them:

### Problem 1: Git Merge Conflicts & Residual Conflict Markers in Codebase
- **Symptom:** When running unit tests via `python -m unittest`, several test suites crashed immediately with `SyntaxError: invalid syntax`.
- **Root-Cause Analysis:** When integrating work between different branches (`main` and `feature/b-pipeline-schema`), merge conflicts had left standard Git conflict markers (`<<<<<<< HEAD`, `=======`, `>>>>>>> origin/main`) directly inside python source files (`training.py`, `recipe_runner.py`, `evaluation.py`, and test files).
- **Engineering Solution:**
  1. Inspected git logs and file diffs to understand the conflicting versions.
  2. Rather than doing a blind find-and-replace, I systematically reconciled the logic in each file, ensuring that the latest feature branch enhancements (e.g. hyperparameter handling in `training.py`) were merged cleanly with master interfaces.
  3. Validated python syntax programmatically using the Python Abstract Syntax Tree (`ast.parse`) module before re-running tests.
  4. Executed `git fetch origin` followed by `git rebase origin/main` to maintain a linear commit history.

### Problem 2: Windows System Encoding Crash with Non-ASCII Characters (`cp1252`)
- **Symptom:** `ml_engine/evaluation.py` threw a fatal compilation error: `SyntaxError: invalid character` on Windows.
- **Root-Cause Analysis:** In the regression evaluation docstrings and comments, the mathematical symbol for R-squared was written using the Unicode character `²`. Because the Windows operating system default terminal encoding (`cp1252`) was active rather than `UTF-8`, the Python interpreter threw an encoding syntax exception upon encountering the multibyte character.
- **Engineering Solution:**
  1. Replaced all occurrences of Unicode superscript `2` with standard ASCII `R2` across metric keys, dictionary outputs, and docstrings.
  2. Enforced defensive string formatting to guarantee that all downstream serialization (JSON dumps to FastAPI and frontend) remains fully ASCII-compatible across any host OS.

### Problem 3: Socket Disconnections & Network Timeouts During Binary Wheel Installs
- **Symptom:** During automated package installations of heavy machine learning libraries (`scipy`, `xgboost`, `onnxruntime`), the pip installer repeatedly crashed with `ConnectionResetError: [WinError 10054]` and read timeout errors.
- **Root-Cause Analysis:** Binary wheels for high-performance scientific libraries are very large (>150 MB). Network instability and strict corporate proxy/VPN packet resets caused socket terminations during lengthy single-stream downloads.
- **Engineering Solution:**
  1. Increased the socket timeout flag: `pip install --default-timeout=120 <package>`.
  2. Decoupled monolithic installation runs into discrete, isolated package installations.
  3. Selected lightweight, high-compatibility serialization packages (`skl2onnx` and standard `onnx`) rather than running unneeded heavy runtime daemons, keeping the local environment lean and reliable.

### Problem 4: Floating-Point Non-Determinism in Reproducibility Audits
- **Symptom:** In `ml_engine/reproduce.py`, the reproducibility test initially failed intermittently on stochastic algorithms like Random Forest, reporting that reproduced accuracy differed from original accuracy by values like `0.00000002`.
- **Root-Cause Analysis:** Modern CPU architectures utilize vectorized BLAS/LAPACK math routines (AVX2/AVX-512) and multithreaded scheduling. Small rounding differences at the 8th decimal place occur naturally in floating-point operations. Using strict equality (`val_a == val_b`) is an anti-pattern in scientific computing.
- **Engineering Solution:**
  1. Introduced explicit pseudorandom generator seeding (`random_state=42`) across both the train-test splitter and the estimator constructors.
  2. Implemented a tolerance-based metric comparator using `math.isclose(expected, actual, abs_tol=1e-4, rel_tol=1e-4)`.
  3. Added an automated diff report generator that prints an exact breakdown of each metric, its expected value, reproduced value, delta, and Boolean pass/fail flag.

### Problem 5: Memory Exhaustion (OOM) with Large Datasets
- **Symptom:** When loading large CSV files (>500,000 rows), Pandas allocated standard 64-bit integer (`int64`) and float (`float64`) types for every column, leading to RAM spikes exceeding 1.8 GB and sluggish UI responsiveness.
- **Root-Cause Analysis:** Pandas defaults to 64-bit precision regardless of the actual magnitude of values (e.g., storing an `Age` column with values 1–100 inside 64 bits instead of an 8-bit unsigned integer).
- **Engineering Solution:**
  1. Developed an automated memory optimization function `downcast_dataframe()` in `ml_engine/data_loader.py`:
     - Inspects `min()` and `max()` bounds for each integer column and downcasts to `int8`, `int16`, or `int32`.
     - Downcasts `float64` to `float32`.
     - Converts low-cardinality string columns (ratio of unique values < 50%) to the `category` dtype.
  2. Benchmarked the optimization: Reduced memory consumption from **38.4 KB to 11.2 KB on our test sample (a 70.9% reduction)**, scaling directly to gigabyte-sized files.
  3. Added a chunked streaming reader (`pd.read_csv(..., chunksize=chunk_size)`) allowing out-of-core iterative batching.

### Problem 6: Broken CSS Minification & Missing Node Modules on Frontend Boot
- **Symptom:** When testing the full-stack system locally, Vite threw `[lightningcss minify] Unexpected end of input` and failed to resolve `react-router-dom`.
- **Root-Cause Analysis:**
  - A stray Git merge conflict delimiter (`=======`) was left unresolved at line 1202 of `frontend/src/App.css`.
  - `react-router-dom` was specified in `package.json` but had not been installed into `frontend/node_modules`.
- **Engineering Solution:**
  1. Cleaned and structured `frontend/src/App.css`, removing the broken syntax marker and closing dangling selector blocks.
  2. Executed a clean `npm.cmd install` within `frontend/` to populate dependencies.
  3. Verified the complete build via `npm.cmd run build` (compiled cleanly in 3.65s).
  4. Successfully launched both backend (FastAPI on port 8000) and frontend (Vite on port 5173).

---

## 5. Software Engineering Concepts & Methodologies Applied (UCS503)

This project served as a direct practical application of the theories covered in UCS503 Software Engineering:

| Concept / Principle | Theoretical Definition | Concrete Implementation in Person B's Work |
| :--- | :--- | :--- |
| **Separation of Concerns (SoC)** | Decomposing a software system into distinct features with minimal overlap. | Isolated data loading, recipe execution, model training, evaluation, and serialization into separate Python modules. |
| **Design Patterns: Factory Pattern** | Providing an interface for creating families of related objects without specifying concrete classes. | `create_model(algorithm_name, hyperparameters)` creates appropriate Scikit-Learn estimators dynamically. |
| **Design Patterns: Strategy Pattern** | Defining a family of algorithms, encapsulating each one, and making them interchangeable. | Preprocessing transformers (MinMax vs Standard Scaler, OneHot vs Ordinal) are interchangeable via JSON configuration. |
| **Design Patterns: Proxy / Cache** | Providing a surrogate or placeholder for an expensive operation. | Preprocessing hash cache (`MD5(recipe_json + dataset_shape)`) prevents re-computing identical feature matrices when comparing multiple models. |
| **Test-Driven Development (TDD)** | Writing automated unit tests before or alongside code to verify correctness. | Built an automated test suite of **26 test cases** (`unittest`) validating edge cases, missing data, and numerical reproducibility. |
| **Defensive Programming** | Anticipating failure states and handling exceptions gracefully. | Added validation for unsupported file extensions, missing target columns, non-invertible matrices in Box-Cox, and division by zero. |
| **Software Quality & Portability** | Ensuring software runs predictably across platforms and production environments. | Removed Windows-incompatible encoding characters; implemented ONNX export for language-agnostic production inference. |
| **Configuration Management & SCM** | Managing software changes systematically using version control. | Managed feature branches (`feature/b-pipeline-schema`), resolved merge conflicts, and maintained clean git commits. |

---

## 6. Verification, Unit Testing & Quality Assurance

To ensure software robustness, I created and executed a comprehensive automated test suite located in `ml_engine/tests/`.

### Summary of Passing Test Suites (26/26 Tests Passed)
1. **`test_data_loader.py` (6 tests):**
   - Validates CSV, TSV, Parquet, and JSON ingestion.
   - Verifies automated type coercion and null handling.
   - Verifies memory downcasting efficiency (~70% RAM reduction).
   - Verifies chunked generator streaming for out-of-core datasets.
2. **`test_preprocessing.py` (4 tests):**
   - Verifies mean, median, and most frequent imputation.
   - Verifies standard and min-max scaling.
   - Verifies one-hot encoding on unseen categorical levels.
3. **`test_transforms.py` (6 tests):**
   - Validates Power Transforms (`log1p`, `yeo-johnson`, `box-cox`, `sqrt`).
   - Validates Outlier Clipping via IQR fences.
   - Validates Discretization (`KBinsDiscretizer`).
   - Validates Polynomial feature interaction expansion.
   - Validates PCA dimensional reduction and variance conservation.
   - Validates Datetime parsing and cyclic feature extraction.
4. **`test_training.py` (5 tests):**
   - Validates multi-model classification (Random Forest, Logistic Regression, Gradient Boosting).
   - Validates regression pipelines (RMSE, MAE, R2).
   - Validates the shared preprocessing cache (verifying that Model 2 reuses the preprocessed matrix of Model 1 in sub-milliseconds).
5. **`test_onnx.py` (3 tests):**
   - Verifies conversion of trained models to `.onnx` binaries.
   - Validates ONNX model file generation and valid graph schemas.
6. **`test_reproduce.py` (2 tests):**
   - Verifies deterministic pipeline execution.
   - Verifies tolerance checks: accurately passes when metrics match within 10^-4 and raises an alert if metrics differ.

---

## 7. Personal Reflection & Key Takeaways

Working on the **Interactive Model Studio** as **Person B** was an invaluable software engineering experience. While implementing machine learning algorithms is often taught from a purely mathematical perspective, building an extensible, production-grade ML Engine taught me the realities of software engineering:

1. **Architecture Dictates Velocity:** Establishing clear interfaces and decoupled modules early on allowed me to add 6 complex feature transformations in Phase 2 without rewriting or breaking the core training pipeline.
2. **Real Software Fails at the Boundaries:** The most challenging bugs were not algorithm flaws, but boundary issues — character encodings across operating systems, network timeouts during dependency downloads, merge conflicts between teammates, and floating-point precision nuances across hardware.
3. **Automated Testing is Non-Negotiable:** Having 26 unit tests allowed me to refactor code, fix merge conflicts, and optimize memory downcasting with absolute confidence that existing functionality remained unbroken.
4. **Reproducibility is the Core of Reliable ML:** Building the `reproduce.py` suite emphasized that an AI/ML platform is only as good as its verifiability. Storing exact configuration hashes and random seeds transforms machine learning from an ad-hoc experiment into an engineering science.

---

**Submitted by:**  
**Hiral Gupta**  
Roll No: 1024030818  
B.E. Computer Science & Engineering  
UCS503 Software Engineering  
Interactive Model Studio Project Team
