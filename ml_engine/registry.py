"""
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
