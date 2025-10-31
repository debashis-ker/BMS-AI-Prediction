"""
MLflow Tracking Utilities for API Endpoints
"""

import mlflow
from functools import wraps
from typing import Callable, Any
import time
from src.bms_ai.utils.mlflow_config import MLflowConfig
from src.bms_ai.logger_config import setup_logger

log = setup_logger(__name__)


def track_prediction(func: Callable) -> Callable:
    """
    Decorator to track prediction requests in MLflow.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        MLflowConfig.setup_mlflow()
        experiment_id = MLflowConfig.get_or_create_experiment(
            MLflowConfig.PREDICTION_EXPERIMENT_NAME
        )
        
        start_time = time.time()
        
        with mlflow.start_run(experiment_id=experiment_id, run_name="prediction"):
            try:
                result = func(*args, **kwargs)
                
                elapsed_time = time.time() - start_time
                mlflow.log_metric("prediction_time_seconds", elapsed_time)
                mlflow.log_metric("success", 1)
                
                mlflow.set_tags({
                    "endpoint": func.__name__,
                    "status": "success"
                })
                
                return result
                
            except Exception as e:
                mlflow.log_metric("success", 0)
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e))
                raise
    
    return wrapper


def track_optimization(func: Callable) -> Callable:
    """
    Decorator to track optimization requests in MLflow.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        MLflowConfig.setup_mlflow()
        experiment_id = MLflowConfig.get_or_create_experiment(
            MLflowConfig.OPTIMIZATION_EXPERIMENT_NAME
        )
        
        start_time = time.time()
        
        with mlflow.start_run(experiment_id=experiment_id, run_name="optimization"):
            try:
                result = func(*args, **kwargs)
                
                elapsed_time = time.time() - start_time
                
                if hasattr(result, 'dict'):
                    result_dict = result.dict()
                else:
                    result_dict = result
                
                mlflow.log_metrics({
                    "optimization_time_seconds": result_dict.get('optimization_time_seconds', elapsed_time),
                    "total_combinations_tested": result_dict.get('total_combinations_tested', 0),
                    "min_fan_power_kw": result_dict.get('min_fan_power_kw', 0),
                    "success": 1
                })
                
                mlflow.log_param("optimization_method", result_dict.get('optimization_method', 'unknown'))
                
                if 'best_setpoints' in result_dict:
                    for key, value in result_dict['best_setpoints'].items():
                        mlflow.log_param(f"best_{key}", value)
                
                mlflow.set_tags({
                    "endpoint": func.__name__,
                    "status": "success"
                })
                
                return result
                
            except Exception as e:
                mlflow.log_metric("success", 0)
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e))
                raise
    
    return wrapper


def track_health_check(func: Callable) -> Callable:
    """
    Decorator to track health check predictions in MLflow.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        MLflowConfig.setup_mlflow()
        experiment_id = MLflowConfig.get_or_create_experiment(
            MLflowConfig.HEALTH_CHECK_EXPERIMENT_NAME
        )
        
        start_time = time.time()
        
        with mlflow.start_run(experiment_id=experiment_id, run_name="health_prediction"):
            try:
                result = func(*args, **kwargs)
                
                elapsed_time = time.time() - start_time
                
                if hasattr(result, 'dict'):
                    result_dict = result.dict()
                else:
                    result_dict = result
                
                mlflow.log_metrics({
                    "prediction_time_seconds": elapsed_time,
                    "failure_threshold": result_dict.get('failure_threshold', 0),
                    "n_predictions": len(result_dict.get('resampled_predicted_data', [])),
                    "success": 1
                })
                
                if 'earliest_end_of_life' in result_dict and result_dict['earliest_end_of_life']:
                    mlflow.log_param("earliest_eol", result_dict['earliest_end_of_life'])
                    mlflow.log_metric("has_failure_prediction", 1)
                else:
                    mlflow.log_metric("has_failure_prediction", 0)
                
                mlflow.set_tags({
                    "endpoint": func.__name__,
                    "status": "success",
                    "message": result_dict.get('message', '')
                })
                
                return result
                
            except Exception as e:
                mlflow.log_metric("success", 0)
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e))
                raise
    
    return wrapper


class MLflowContextManager:
    """
    Context manager for MLflow runs.
    """
    
    def __init__(self, experiment_name: str, run_name: str = None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = None
        
    def __enter__(self):
        MLflowConfig.setup_mlflow()
        experiment_id = MLflowConfig.get_or_create_experiment(self.experiment_name)
        self.run = mlflow.start_run(experiment_id=experiment_id, run_name=self.run_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(exc_val))
        mlflow.end_run()
        return False
    
    def log_params(self, params: dict):
        """Log parameters to MLflow."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: dict):
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics)
    
    def log_tags(self, tags: dict):
        """Log tags to MLflow."""
        mlflow.set_tags(tags)
