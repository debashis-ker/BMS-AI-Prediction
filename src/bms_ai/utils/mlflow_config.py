import os
import mlflow
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class MLflowConfig:
    # Project Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    MLFLOW_DIR = PROJECT_ROOT / "mlruns"
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    MLFLOW_HOST = os.getenv("MLFLOW_HOST", "127.0.0.1")
    MLFLOW_PORT = int(os.getenv("MLFLOW_PORT", "5000"))
    DEFAULT_EXPERIMENT_NAME = "BMS_AI_Default"
    ARTIFACT_LOCATION = os.getenv("MLFLOW_ARTIFACT_LOCATION", "./mlartifacts")
    REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)
    TRACKING_URI = MLFLOW_TRACKING_URI  # Alias for compatibility
    AUTOLOG_SKLEARN = os.getenv("MLFLOW_AUTOLOG_SKLEARN", "true").lower() == "true"
    AUTOLOG_XGBOOST = os.getenv("MLFLOW_AUTOLOG_XGBOOST", "true").lower() == "true"
    
    @classmethod
    def setup_mlflow(cls):
        mlflow.set_tracking_uri(cls.MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(cls.REGISTRY_URI)
        Path(cls.ARTIFACT_LOCATION).mkdir(parents=True, exist_ok=True)
        return mlflow.get_tracking_uri()
    
    @classmethod
    def get_or_create_experiment(cls, experiment_name=None):
        if experiment_name is None:
            experiment_name = cls.DEFAULT_EXPERIMENT_NAME
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location=os.path.join(cls.ARTIFACT_LOCATION, experiment_name))
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        return experiment_id
    
    @classmethod
    def get_mlflow_ui_url(cls):
        return f"http://{cls.MLFLOW_HOST}:{cls.MLFLOW_PORT}"
