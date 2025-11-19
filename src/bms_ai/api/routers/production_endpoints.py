from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.bms_ai.logger_config import setup_logger
from pathlib import Path
from src.bms_ai.api.routers.heatlh_check import Damper_health_analysis
from src.bms_ai.pipelines.damper_optimization_pipeline import (
    train as damper_train,
    optimize as damper_optimize
)
from src.bms_ai.pipelines.generic_optimization_pipeline import (
    train_generic,
    optimize_generic
)

import pandas as pd
import joblib
import warnings
import math
import time

log = setup_logger(__name__)

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/prod", tags=["Prescriptive Optimization"])

damper_forecast_model = None

try:
    damper_forecast_model = joblib.load("artifacts/production_models/AHU1_Damper_health.joblib")
    log.info("Model loaded successfully.")
except Exception as e:
    log.error(f"Error loading forecast model: {e}")
    print(f"Error loading forecast model: {e}")

class PredictionRequest(BaseModel):
    periods: int = Field(3, description="Number of future time periods to predict.")
    failure_threshold: float = Field(6.0, description="Threshold at which End of Life will be predicted.")

class ResampledData(BaseModel):
    date: str
    prediction_mean: float

class PredictionResponse(BaseModel):
    message: str = Field(..., description="Status.")
    earliest_end_of_life: Optional[str] = Field(None, description="The date of the earliest potential End of Life.")
    failure_threshold: float = Field(..., description="Threshold used for prediction.")
    resampled_predicted_data: List[ResampledData] = Field(..., description="Resampled 15-day average prediction data.")

@router.post('/damper_health_prediction', response_model=PredictionResponse)
def Damper_health_prediction(
    request_data: PredictionRequest
):
    start = time.time()
    log.info(f"input Data: {request_data.dict()}") 
    result = Damper_health_analysis(request_data)
    end = time.time()
    log.info(f"VOX AHU1 Damper Prediction completed in {end - start:.2f} seconds") 
    return result


@router.get('/status')
def model_status():
    """
    Check if the forecast model is loaded and ready to use.
    Returns model status information.
    """
    if damper_forecast_model is None:
        return {
            "status": "unavailable",
            "model_loaded": False,
            "message": "Damper Forecast model is not loaded. Please check server logs."
        }

    return {
        "status": "ready",
        "model_loaded": True,
        "Damper_Health_model_type": type(damper_forecast_model).__name__,
        "message": "Forecast model is loaded and ready for predictions."
    }



class DamperTrainRequest(BaseModel):
    data_path: str = Field("C:\\Users\\debas\\OneDrive\\Desktop\\actual_data.csv", description="Path to training data CSV file")
    equipment_id: str = Field("Ahu1", description="Equipment ID to filter")
    test_size: float = Field(0.2, description="Fraction of data for testing")
    search_method: str = Field("random", description="Hyperparameter search method: 'random' or 'grid'")
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    n_iter: int = Field(20, description="Number of iterations for RandomizedSearchCV")


class DamperTrainResponse(BaseModel):
    status: str
    best_model_name: str
    metrics: Dict[str, Any]
    model_path: str
    scaler_path: str


@router.post('/damper_train', response_model=DamperTrainResponse)
def damper_train_endpoint(request_data: DamperTrainRequest):
    """
    Train the damper optimization surrogate model with multiple algorithms and hyperparameter tuning.
    
    Args:
        data_path: Path to CSV file with training data
        equipment_id: Equipment ID to filter
        test_size: Fraction for test split
        search_method: 'random' for RandomizedSearchCV or 'grid' for GridSearchCV
        cv_folds: Number of cross-validation folds
        n_iter: Number of iterations for RandomizedSearchCV
        
    Returns:
        Training results including metrics and artifact paths
    """
    start = time.time()
    log.info(f"Damper training request: {request_data.dict()}")
    print(f"Damper training request: {request_data.dict()}")
    try:
        result = damper_train(
            data_path=request_data.data_path,
            equipment_id=request_data.equipment_id,
            test_size=request_data.test_size,
            search_method=request_data.search_method,
            cv_folds=request_data.cv_folds,
            n_iter=request_data.n_iter
        )
        
        end = time.time()
        log.info(f"Damper model training completed in {end - start:.2f} seconds")
        
        return DamperTrainResponse(**result)
        
    except Exception as e:
        log.error(f"Damper training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DamperOptimizeRequest(BaseModel):
    current_conditions: Dict[str, Any] = Field(..., description="Current system state")
    search_space: Optional[Dict[str, List[float]]] = Field(None, description="Setpoint ranges to search")
    optimization_method: Optional[str] = Field("random", description="Optimization method: 'grid', 'random', or 'hybrid'")
    n_iterations: Optional[int] = Field(1000, description="Number of iterations for random/hybrid search")


class DamperOptimizeResponse(BaseModel):
    best_setpoints: Dict[str, float]
    min_fbfad: float
    total_combinations_tested: int
    optimization_method: str
    optimization_time_seconds: float


@router.post('/damper_optimize', response_model=DamperOptimizeResponse)
def damper_optimize_endpoint(request_data: DamperOptimizeRequest):
    """
    Optimize damper setpoints to minimize FbFAD (Fresh Air Damper Feedback).
    
    Args:
        current_conditions: Current system state with all required features
        search_space: Optional setpoint ranges (defaults provided if not specified)
        optimization_method: 'grid', 'random', or 'hybrid'
        n_iterations: Number of iterations for random/hybrid methods
        
    Returns:
        Best setpoints and minimum FbFAD value
    """
    start = time.time()
    log.info(f"Damper optimization request: method={request_data.optimization_method}")
    
    try:
        result = damper_optimize(
            current_conditions=request_data.current_conditions,
            search_space=request_data.search_space,
            optimization_method=request_data.optimization_method or "random",
            n_iterations=request_data.n_iterations or 1000
        )
        
        end = time.time()
        log.info(f"Damper optimization completed in {end - start:.2f} seconds")
        log.info(f"Best setpoints: {result['best_setpoints']}, Min FbFAD: {result['min_fbfad']:.4f}")
        
        return DamperOptimizeResponse(**result)
        
    except Exception as e:
        log.error(f"Damper optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Generic Optimization Endpoints

class GenericTrainRequest(BaseModel):
    data_path: str = Field("C:\\Users\\debas\\OneDrive\\Desktop\\actual_data.csv", description="Path to training data CSV file")
    equipment_id: str = Field("Ahu1", description="Equipment ID to filter")
    target_variable: str = Field(..., description="Target variable to optimize (must be numeric)")
    test_size: float = Field(0.2, description="Fraction of data for testing")
    search_method: str = Field("random", description="Hyperparameter search method: 'random' or 'grid'")
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    n_iter: int = Field(20, description="Number of iterations for RandomizedSearchCV")


class GenericTrainResponse(BaseModel):
    status: str
    best_model_name: str
    selected_features: List[str]
    metrics: Dict[str, Any]
    model_path: str
    scaler_path: str
    correlation_plot: str
    mi_plot: str


@router.post('/generic_train', response_model=GenericTrainResponse)
def generic_train_endpoint(request_data: GenericTrainRequest):
    """
    Train a generic optimization surrogate model with automatic feature selection.
    
    Uses correlation and mutual information analysis to automatically select the top 20 most
    relevant features from the dataset. Always includes mandatory setpoints: SpMinVFD, SpTREff, SpTROcc.
    
    Args:
        data_path: Path to CSV file with training data
        equipment_id: Equipment ID to filter
        target_variable: Target variable to optimize (must be numeric)
        test_size: Fraction for test split
        search_method: 'random' for RandomizedSearchCV or 'grid' for GridSearchCV
        cv_folds: Number of cross-validation folds
        n_iter: Number of iterations for RandomizedSearchCV
        
    Returns:
        Training results including selected features, metrics, and artifact paths
    """
    start = time.time()
    log.info(f"Generic training request: {request_data.dict()}")
    print(f"Generic training request: {request_data.dict()}")
    
    try:
        result = train_generic(
            data_path=request_data.data_path,
            equipment_id=request_data.equipment_id,
            target_column=request_data.target_variable,  # API uses target_variable, pipeline uses target_column
            test_size=request_data.test_size,
            search_method=request_data.search_method,
            cv_folds=request_data.cv_folds,
            n_iter=request_data.n_iter
        )
        
        end = time.time()
        log.info(f"Generic model training completed in {end - start:.2f} seconds")
        log.info(f"Selected {len(result.get('selected_features', []))} features for {request_data.target_variable}")
        
        return GenericTrainResponse(**result)
        
    except Exception as e:
        log.error(f"Generic training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class GenericOptimizeRequest(BaseModel):
    current_conditions: Dict[str, Any] = Field(..., description="Current system state")
    equipment_id: str = Field(..., description="Equipment ID (must match training)")
    target_variable: str = Field(..., description="Target variable to minimize (must match training)")
    search_space: Optional[Dict[str, List[float]]] = Field(None, description="Setpoint ranges to search")
    optimization_method: Optional[str] = Field("random", description="Optimization method: 'grid', 'random', or 'hybrid'")
    n_iterations: Optional[int] = Field(1000, description="Number of iterations for random/hybrid search")


class GenericOptimizeResponse(BaseModel):
    best_setpoints: Dict[str, float]
    min_target_value: float
    target_variable: str
    total_combinations_tested: int
    optimization_method: str
    optimization_time_seconds: float


@router.post('/generic_optimize', response_model=GenericOptimizeResponse)
def generic_optimize_endpoint(request_data: GenericOptimizeRequest):
    """
    Optimize AHU setpoints to minimize any specified target variable.
    
    Args:
        current_conditions: Current system state with all required features
        target_variable: Target variable to minimize
        search_space: Optional setpoint ranges (defaults provided if not specified)
        optimization_method: 'grid', 'random', or 'hybrid'
        n_iterations: Number of iterations for random/hybrid methods
        
    Returns:
        Best setpoints and minimum target value
    """
    start = time.time()
    log.info(f"Generic optimization request: equipment={request_data.equipment_id}, target={request_data.target_variable}, method={request_data.optimization_method}")
    
    try:
        result = optimize_generic(
            current_conditions=request_data.current_conditions,
            equipment_id=request_data.equipment_id,
            target_column=request_data.target_variable,  # API uses target_variable, pipeline uses target_column
            search_space=request_data.search_space,
            optimization_method=request_data.optimization_method or "random",
            n_iterations=request_data.n_iterations or 1000
        )
        
        end = time.time()
        log.info(f"Generic optimization completed in {end - start:.2f} seconds")
        log.info(f"Best setpoints: {result['best_setpoints']}, Min {request_data.target_variable}: {result['min_target_value']:.4f}")
        
        return GenericOptimizeResponse(**result)
        
    except Exception as e:
        log.error(f"Generic optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
