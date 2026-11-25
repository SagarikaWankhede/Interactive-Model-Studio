import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .database import Base


class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    datasets = relationship("DatasetDB", back_populates="user")
    runs = relationship("RunDB", back_populates="user")


class DatasetDB(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    row_count = Column(Integer, nullable=False)
    col_count = Column(Integer, nullable=False)
    column_types = Column(JSON, nullable=False)
    missing_counts = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("UserDB", back_populates="datasets")


class ExperimentDB(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    task_type = Column(String(50), nullable=False)
    target_column = Column(String(255), nullable=False)
    recipe_json = Column(JSON, nullable=False)
    model_config_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class RunDB(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    task_type = Column(String(50), nullable=False)
    algorithm = Column(String(100), nullable=False)
    train_time_sec = Column(Float, nullable=False)
    metrics_json = Column(JSON, nullable=False)
    confusion_matrix_json = Column(JSON, nullable=True)
    feature_importances_json = Column(JSON, nullable=True)
    recipe_json = Column(JSON, nullable=True)
    model_config_json = Column(JSON, nullable=True)
    model_artifact_path = Column(String(512), nullable=True)
    onnx_model_path = Column(String(512), nullable=True)
    is_reproducible = Column(Boolean, default=True)
    graph_json = Column(JSON, nullable=True)        # visual node graph that produced this run
    generated_code = Column(Text, nullable=True)    # Python script generated from that graph
    parent_run_id = Column(Integer, nullable=True)  # set when produced by active-learning retraining
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("UserDB", back_populates="runs")

    @property
    def metrics(self):
        return self.metrics_json

    @property
    def confusion_matrix(self):
        return self.confusion_matrix_json

    @property
    def feature_importances(self):
        return self.feature_importances_json

    @property
    def train_time(self):
        return self.train_time_sec

    @property
    def model_name(self):
        return self.algorithm

    @property
    def model_type(self):
        return self.algorithm

    @property
    def status(self):
        return "completed"

    @property
    def recipe(self):
        return self.recipe_json or []

    @property
    def model_config(self):
        return self.model_config_json or {}

    @property
    def hyperparams(self):
        if self.model_config_json and "hyperparameters" in self.model_config_json:
            return self.model_config_json["hyperparameters"]
        return {}


class EdgeCaseDB(Base):
    """A low-confidence prediction waiting for a human label (active learning queue)."""
    __tablename__ = "edge_cases"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False, index=True)
    source = Column(String(32), nullable=False, default="browser")  # browser | scan | api
    row_index = Column(Integer, nullable=True)       # dataset row, when found by a dataset scan
    features_json = Column(JSON, nullable=False)     # raw input features
    predicted_label = Column(String(255), nullable=True)
    confidence = Column(Float, nullable=True)
    probabilities_json = Column(JSON, nullable=True)  # {class_label: probability}
    dataset_label = Column(String(255), nullable=True)  # label already in the dataset (scan only)
    status = Column(String(32), nullable=False, default="pending")  # pending | labeled | dismissed | applied
    corrected_label = Column(String(255), nullable=True)
    retrain_run_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    labeled_at = Column(DateTime, nullable=True)
