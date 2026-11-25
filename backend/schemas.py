from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
import datetime


class RecipeStep(BaseModel):
    step: str = Field(..., description="Step type: impute, encode, scale, split")
    columns: Optional[List[str]] = Field(default=[], description="Target columns")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Step parameters")


class ModelSettings(BaseModel):
    algorithm: str = Field(..., description="Name of algorithm from registry")
    hyperparameters: Optional[Dict[str, Any]] = Field(default={}, description="Algorithm hyperparameters")


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset_id: int
    task_type: str = Field(default="classification", description="classification or regression")
    target_column: str
    recipe: List[RecipeStep]
    model_settings: ModelSettings = Field(..., alias="model_config", description="Model configuration (algorithm + hyperparams)")
    use_cache: Optional[bool] = Field(default=True, description="Whether to use cached preprocessed data")


class PreviewRecipeRequest(BaseModel):
    dataset_id: int
    target_column: str
    recipe: List[RecipeStep]


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Key-value feature inputs for single sample inference")


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    user_id: Optional[int] = None
    filename: str
    name: Optional[str] = None
    row_count: int
    col_count: int
    column_count: Optional[int] = None
    column_types: Dict[str, str]
    missing_counts: Dict[str, int]
    target_column: Optional[str] = None
    schema_info: Optional[List[Dict[str, Any]]] = Field(default=None, alias="schema")
    diagnostics: Optional[List[Dict[str, Any]]] = None
    created_at: datetime.datetime


class RunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    user_id: Optional[int] = None
    dataset_id: int
    task_type: str
    algorithm: str
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    status: Optional[str] = "completed"
    train_time_sec: float
    train_time: Optional[float] = None
    metrics: Dict[str, Any]
    confusion_matrix: Optional[Any] = None
    feature_importances: Optional[List[Dict[str, Any]]] = None
    onnx_model_path: Optional[str] = None
    recipe: Optional[List[Dict[str, Any]]] = None
    model_settings: Optional[Dict[str, Any]] = Field(default=None, alias="model_config")
    is_reproducible: bool
    created_at: datetime.datetime


class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    username: str  # username or email
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime.datetime


class AuthResponse(BaseModel):
    token: str
    user: UserResponse
    message: str = "Authenticated successfully"


class ModelMetadataResponse(BaseModel):
    task_type: Optional[str] = None
    models: List[Dict[str, Any]]
