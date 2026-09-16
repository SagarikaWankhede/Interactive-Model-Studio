import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from .database import Base


class DatasetDB(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    row_count = Column(Integer, nullable=False)
    col_count = Column(Integer, nullable=False)
    column_types = Column(JSON, nullable=False)
    missing_counts = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


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
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    task_type = Column(String(50), nullable=False)
    algorithm = Column(String(100), nullable=False)
    train_time_sec = Column(Float, nullable=False)
    metrics_json = Column(JSON, nullable=False)
    confusion_matrix_json = Column(JSON, nullable=True)
    feature_importances_json = Column(JSON, nullable=True)
    model_artifact_path = Column(String(512), nullable=True)
    is_reproducible = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    @property
    def metrics(self):
        return self.metrics_json

    @property
    def confusion_matrix(self):
        return self.confusion_matrix_json

    @property
    def feature_importances(self):
        return self.feature_importances_json
