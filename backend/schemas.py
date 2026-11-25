from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import datetime


class RecipeStep(BaseModel):
    step: str = Field(..., description="Step type: impute, encode, scale, split")
    columns: Optional[List[str]] = Field(default=[], description="Target columns")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Step parameters")


class ModelConfig(BaseModel):
    algorithm: str = Field(..., description="Name of algorithm from registry")
    hyperparameters: Optional[Dict[str, Any]] = Field(default={}, description="Algorithm hyperparameters")


class ExecutionRequest(BaseModel):
    dataset_id: int
    task_type: str = Field(..., description="classification or regression")
    target_column: str
    recipe: List[RecipeStep]
    model_settings: ModelConfig = Field(..., alias="model_config", description="Model configuration (algorithm + hyperparams)")


class PreviewRecipeRequest(BaseModel):
    dataset_id: int
    target_column: str
    recipe: List[RecipeStep]


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Key-value feature inputs for single sample inference")


class DatasetResponse(BaseModel):
    id: int
    user_id: Optional[int] = None
    filename: str
    row_count: int
    col_count: int
    column_types: Dict[str, str]
    missing_counts: Dict[str, int]
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class RunResponse(BaseModel):
    id: int
    user_id: Optional[int] = None
    dataset_id: int
    task_type: str
    algorithm: str
    train_time_sec: float
    metrics: Dict[str, Any]
    confusion_matrix: Optional[Dict[str, Any]] = None
    feature_importances: Optional[List[Dict[str, Any]]] = None
    onnx_model_path: Optional[str] = None
    is_reproducible: bool
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    username: str  # username or email
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    token: str
    user: UserResponse
    message: str = "Authenticated successfully"

