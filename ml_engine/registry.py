"""
ML Engine Registry
Single source of truth mapping step names and algorithm names
to actual scikit-learn and XGBoost classes and functions.
Used by both the execution engine and the code generator.
"""

from typing import Any, Dict, List, Optional, Tuple, Type
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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

TRANSFORMER_REGISTRY = {
    "impute": {
        "simple": SimpleImputer,
        "mean": SimpleImputer,
        "median": SimpleImputer,
        "most_frequent": SimpleImputer,
        "constant": SimpleImputer,
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
        "DecisionTreeClassifier": DecisionTreeClassifier,
    },
    "regression": {
        "RandomForestRegressor": RandomForestRegressor,
        "LinearRegression": LinearRegression,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor,
    }
}

if HAS_XGBOOST:
    MODEL_REGISTRY["classification"]["XGBoostClassifier"] = XGBClassifier
    MODEL_REGISTRY["classification"]["XGBClassifier"] = XGBClassifier
    MODEL_REGISTRY["regression"]["XGBoostRegressor"] = XGBRegressor
    MODEL_REGISTRY["regression"]["XGBRegressor"] = XGBRegressor


def get_preprocessor_class(step_name: str, method_or_strategy: str = "default"):
    """Dynamically resolves a preprocessing transformer class."""
    key = step_name.lower()
    if key in TRANSFORMER_REGISTRY:
        methods = TRANSFORMER_REGISTRY[key]
        if method_or_strategy in methods:
            return methods[method_or_strategy]
    raise ValueError(f"Unknown preprocessor: step='{step_name}', method='{method_or_strategy}'")


def get_transformer(step_type: str, method_name: str):
    """Retrieve transformer class from registry."""
    if step_type not in TRANSFORMER_REGISTRY:
        raise ValueError(f"Unknown preprocessing step type: {step_type}")
    methods = TRANSFORMER_REGISTRY[step_type]
    if method_name not in methods:
        raise ValueError(f"Unsupported method '{method_name}' for step '{step_type}'. Available: {list(methods.keys())}")
    return methods[method_name]


def get_model_class(task_type_or_name: str, model_name: Optional[str] = None) -> Type[Any]:
    """
    Returns the estimator class for a given model name.
    Supports get_model_class('RandomForestClassifier') and get_model_class('classification', 'RandomForestClassifier').
    """
    if model_name is None:
        model_name = task_type_or_name
        task_type = "classification"
    else:
        task_type = task_type_or_name

    model_map: Dict[str, Type[Any]] = {
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "LinearRegression": LinearRegression,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "DecisionTreeRegressor": DecisionTreeRegressor,
    }

    if HAS_XGBOOST:
        model_map["XGBoostClassifier"] = XGBClassifier
        model_map["XGBClassifier"] = XGBClassifier
        model_map["XGBoostRegressor"] = XGBRegressor
        model_map["XGBRegressor"] = XGBRegressor

    if model_name in model_map:
        return model_map[model_name]

    if task_type in MODEL_REGISTRY and model_name in MODEL_REGISTRY[task_type]:
        return MODEL_REGISTRY[task_type][model_name]

    raise ValueError(f"Unsupported model '{model_name}' for task '{task_type}'. Available: {list(model_map.keys())}")


MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "RandomForestClassifier": {
        "task_type": "classification",
        "library": "sklearn",
        "default_params": {"n_estimators": 100, "max_depth": None, "random_state": 42},
        "description": "Ensemble of decision trees trained with bagging."
    },
    "LogisticRegression": {
        "task_type": "classification",
        "library": "sklearn",
        "default_params": {"C": 1.0, "max_iter": 1000, "random_state": 42},
        "description": "Linear classification model."
    },
    "GradientBoostingClassifier": {
        "task_type": "classification",
        "library": "sklearn",
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        "description": "Boosting ensemble of decision trees."
    },
    "DecisionTreeClassifier": {
        "task_type": "classification",
        "library": "sklearn",
        "default_params": {"max_depth": 5, "random_state": 42},
        "description": "Single decision tree; fully interpretable."
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
}


def list_supported_models(task_type: Optional[str] = None) -> List[str]:
    """Returns a list of model names, optionally filtered by task type."""
    if task_type is None:
        return list(MODEL_METADATA.keys())
    return [name for name, meta in MODEL_METADATA.items() if meta["task_type"] == task_type]


def get_model_defaults(model_name: str) -> Dict[str, Any]:
    """Returns the default hyperparameters for a given model."""
    if model_name not in MODEL_METADATA:
        return {"random_state": 42}
    return MODEL_METADATA[model_name]["default_params"].copy()
