"""
ML Engine Registry
Single source of truth mapping step names and algorithm names
to actual scikit-learn and XGBoost classes and functions.
Used by both the execution engine and the code generator.
"""

from typing import Any, Dict, List, Optional, Tuple, Type


def get_preprocessor_class(step_name: str, method_or_strategy: str = "default"):
    """
    Dynamically resolves a preprocessing transformer class.
    Deferred imports ensure fast startup and clean error handling.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        RobustScaler,
        OneHotEncoder,
        OrdinalEncoder,
    )

    mapping = {
        ("impute", "mean"): SimpleImputer,
        ("impute", "median"): SimpleImputer,
        ("impute", "most_frequent"): SimpleImputer,
        ("impute", "constant"): SimpleImputer,
        ("scale", "standard"): StandardScaler,
        ("scale", "minmax"): MinMaxScaler,
        ("scale", "robust"): RobustScaler,
        ("encode", "onehot"): OneHotEncoder,
        ("encode", "ordinal"): OrdinalEncoder,
    }

    key = (step_name.lower(), method_or_strategy.lower())
    if key not in mapping:
        raise ValueError(
            f"Unknown preprocessor: step='{step_name}', method/strategy='{method_or_strategy}'. "
            f"Supported combinations: {list(mapping.keys())}"
        )
    return mapping[key]


def get_model_class(model_name: str) -> Type[Any]:
    """
    Returns the estimator class for a given model name.
    Supports classical models from scikit-learn and XGBoost.
    """
    model_map: Dict[str, str] = {
        # Classification
        "RandomForestClassifier": "sklearn.ensemble.RandomForestClassifier",
        "LogisticRegression": "sklearn.linear_model.LogisticRegression",
        "DecisionTreeClassifier": "sklearn.tree.DecisionTreeClassifier",
        "GradientBoostingClassifier": "sklearn.ensemble.GradientBoostingClassifier",
        "XGBClassifier": "xgboost.XGBClassifier",
        # Regression
        "RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
        "LinearRegression": "sklearn.linear_model.LinearRegression",
        "DecisionTreeRegressor": "sklearn.tree.DecisionTreeRegressor",
        "GradientBoostingRegressor": "sklearn.ensemble.GradientBoostingRegressor",
        "XGBRegressor": "xgboost.XGBRegressor",
    }

    if model_name not in model_map:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {list(model_map.keys())}"
        )

    module_path, class_name = model_map[model_name].rsplit(".", 1)
    
    if module_path == "xgboost":
        try:
            import xgboost as xgb
            return getattr(xgb, class_name)
        except ImportError:
            raise ImportError(
                "xgboost is not installed. Please install it using 'pip install xgboost' "
                "or choose a scikit-learn algorithm like 'GradientBoostingClassifier'."
            )
            
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# Metadata describing models and their default hyperparameters
MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "RandomForestClassifier": {
        "task_type": "classification",
        "library": "sklearn",
        "default_params": {"n_estimators": 100, "max_depth": None, "random_state": 42},
        "description": "Ensemble of decision trees trained with bagging and random feature subspaces."
    },
    "LogisticRegression": {
        "task_type": "classification",
        "library": "sklearn",
        "default_params": {"C": 1.0, "max_iter": 1000, "random_state": 42},
        "description": "Linear classification model using a logistic sigmoid loss function."
    },
    "DecisionTreeClassifier": {
        "task_type": "classification",
        "library": "sklearn",
        "default_params": {"max_depth": 5, "random_state": 42},
        "description": "Non-parametric tree-structured decision model."
    },
    "GradientBoostingClassifier": {
        "task_type": "classification",
        "library": "sklearn",
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        "description": "Additive boosting ensemble of shallow decision trees."
    },
    "XGBClassifier": {
        "task_type": "classification",
        "library": "xgboost",
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        "description": "Scalable, distributed gradient boosting library."
    },
    "RandomForestRegressor": {
        "task_type": "regression",
        "library": "sklearn",
        "default_params": {"n_estimators": 100, "max_depth": None, "random_state": 42},
        "description": "Random forest regression ensemble."
    },
    "LinearRegression": {
        "task_type": "regression",
        "library": "sklearn",
        "default_params": {"fit_intercept": True},
        "description": "Ordinary least squares linear regression."
    },
    "DecisionTreeRegressor": {
        "task_type": "regression",
        "library": "sklearn",
        "default_params": {"max_depth": 5, "random_state": 42},
        "description": "Single decision tree regression."
    },
    "GradientBoostingRegressor": {
        "task_type": "regression",
        "library": "sklearn",
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        "description": "Gradient boosting regression ensemble."
    },
    "XGBRegressor": {
        "task_type": "regression",
        "library": "xgboost",
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        "description": "XGBoost gradient boosting regression."
    },
}


def list_supported_models(task_type: Optional[str] = None) -> List[str]:
    """Returns a list of model names, optionally filtered by task type."""
    if task_type is None:
        return list(MODEL_METADATA.keys())
    return [name for name, meta in MODEL_METADATA.items() if meta["task_type"] == task_type]


def get_model_defaults(model_name: str) -> Dict[str, Any]:
    """Returns the default hyperparameters for a given model."""
    if model_name not in MODEL_METADATA:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_METADATA[model_name]["default_params"].copy()
Registry dictionary mapping step types and model names to constructor factories.
Single source of truth for all supported transformers and estimators.
"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

TRANSFORMER_REGISTRY = {
    "impute": {
        "simple": SimpleImputer,
    },
    "scale": {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    },
    "encode": {
        "onehot": OneHotEncoder,
        "ordinal": OrdinalEncoder,
    }
}

MODEL_REGISTRY = {
    "classification": {
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "GradientBoostingClassifier": GradientBoostingClassifier,
    },
    "regression": {
        "RandomForestRegressor": RandomForestRegressor,
        "LinearRegression": LinearRegression,
        "GradientBoostingRegressor": GradientBoostingRegressor,
    }
}

if HAS_XGBOOST:
    MODEL_REGISTRY["classification"]["XGBoostClassifier"] = XGBClassifier
    MODEL_REGISTRY["classification"]["XGBClassifier"] = XGBClassifier
    MODEL_REGISTRY["regression"]["XGBoostRegressor"] = XGBRegressor
    MODEL_REGISTRY["regression"]["XGBRegressor"] = XGBRegressor


def get_transformer(step_type: str, method_name: str):
    """Retrieve transformer class from registry."""
    if step_type not in TRANSFORMER_REGISTRY:
        raise ValueError(f"Unknown preprocessing step type: {step_type}")
    methods = TRANSFORMER_REGISTRY[step_type]
    if method_name not in methods:
        raise ValueError(f"Unsupported method '{method_name}' for step '{step_type}'. Available: {list(methods.keys())}")
    return methods[method_name]


def get_model_class(task_type: str, model_name: str):
    """Retrieve model estimator class from registry."""
    if task_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown task type: '{task_type}'. Must be 'classification' or 'regression'.")
    models = MODEL_REGISTRY[task_type]
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not supported for task '{task_type}'. Available: {list(models.keys())}")
    return models[model_name]
