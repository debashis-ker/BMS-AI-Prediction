from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.bms_ai.logger_config import setup_logger
from pathlib import Path
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
fan_forecast_model = None
anamoly_model = None

try:
    damper_forecast_model = joblib.load("artifacts/production_models/AHU1_Damper_health.joblib")
    log.info("AHU1 Damper Model loaded successfully.")
except Exception as e:
    log.error(f"Error loading Damper forecast model: {e}")
    print(f"Error loading Damper forecast model: {e}")

try:
    fan_forecast_model = joblib.load("artifacts/production_models/AHU1_Fan_Speed_Health.joblib")
    log.info("Ahu1 Fan Speed Model loaded successfully.")
except Exception as e:
    log.error(f"Error loading Fan Speed forecast model: {e}")
    print(f"Error loading Fan Speed forecast model: {e}")

try:
    anamoly_model = joblib.load("artifacts/production_models/Anamoly_model.joblib")
    log.info("Anamoly Detection Model loaded successfully.")
except Exception as e:
    log.error(f"Error loading Anamoly Detection  model: {e}")
    print(f"Error loading Anamoly Detection model: {e}")

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


class MonitoringDataRecord(BaseModel):
    asset_code: str
    datapoint: str
    monitoring_data: str
    data_received_on: str

class AnamolyPredictionRequest(BaseModel):
    queryResponse: List[MonitoringDataRecord] = Field(
        ...,
        description="List of raw monitoring data records to check for anomalies."
    )
    asset: str = Field('OS04-AHU-06', description="Asset column of data.")
    feature: str = Field('Co2RA', description="Feature on which Anomaly should be detected.")

class AnamolyPredictionResponse(BaseModel):
    asset_code: str = Field(..., description="The asset code analyzed.")
    feature: str = Field(..., description="The specific feature analyzed.")
    predictions: List[Dict[str, Any]] = Field(..., description="List of time series records including the anomaly flag (1=Normal, -1=Anomaly).")
    total_anomalies: int = Field(..., description="Total count of unique anomalies detected in the input data.")



def Anamoly_data_pipeline(data: Dict[str, Any], date_column: str, target_asset_code: str) -> pd.DataFrame:
    try:
        records = data.get("data", {}).get("queryResponse", [])
        if not records:
            raise ValueError("Input JSON is missing the 'data' or 'queryResponse' key, or the record list is empty.")
            
        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError("DataFrame is empty after extracting records.")

        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found.")
        
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        if df[date_column].dt.tz is not None:
            df[date_column] = df[date_column].dt.tz_localize(None)
            
        df.rename(columns={date_column: "data_received_on"}, inplace=True)
        
        if 'monitoring_data' in df.columns:
            mapping = {'inactive': 0.0, 'active': 1.0}
            df['monitoring_data'] = df['monitoring_data'].replace(mapping, regex=False)
            df['monitoring_data'] = pd.to_numeric(df['monitoring_data'], errors='coerce')

        if 'system_type' in df.columns:
            df = df[df['system_type'] == "AHU"].copy()
        
        if 'asset_code' in df.columns:
            df = df[df['asset_code'] == target_asset_code].copy()

        if df.empty:
            print(f"Warning: No data found for system_type = AHU and asset_code='{target_asset_code}'.")
            return pd.DataFrame()
        
        required = ["data_received_on", 'asset_code', 'datapoint', 'monitoring_data']
        missing = [col for col in required if col not in df.columns]
        if missing:
             raise ValueError(f"Required columns for aggregation are missing: {', '.join(missing)}")

        aggregated_scores = df.groupby(["data_received_on", 'asset_code', 'datapoint'])['monitoring_data'].agg('first')
        result_df = aggregated_scores.unstack(level='datapoint').reset_index()
        
        return result_df

    except Exception as e:
        log.error(f"Data pipeline failed: {e}")
        raise Exception(f"Data pipeline failed: {e}")
    
def Anomaly_detection_analysis(request_data: AnamolyPredictionRequest) -> AnamolyPredictionResponse:
    if anamoly_model is None:
        log.error("Anomaly Detection model is unavailable.")
        raise HTTPException(status_code=503, detail="Anomaly Detection model is currently unavailable.")

    asset_code = request_data.asset
    feature = request_data.feature
    
    try:
        model_package = anamoly_model[feature][asset_code]
    except KeyError:
        log.warning(f"Model package not found for Feature: {feature} and Asset: {asset_code}")
        raise HTTPException(
            status_code=404, 
            detail=f"Trained model not found for asset '{asset_code}' and feature '{feature}'. Available models: {list(anamoly_model.keys())}"
        )

    try:
        wrapped_data = {"data": {"queryResponse": [rec.dict() for rec in request_data.queryResponse]}}
        
        date_col_name = "data_received_on" 
        
        df_wide = Anamoly_data_pipeline(wrapped_data, date_col_name, asset_code)
        
        if df_wide.empty:
            return AnamolyPredictionResponse(
                asset_code=asset_code, feature=feature, predictions=[], total_anomalies=0
            )
            
    except Exception as e:
        log.error(f"Data pipeline error during anomaly detection: {e}")
        raise HTTPException(status_code=400, detail=f"Data processing failed: {e}")

    X_df = df_wide[[feature, "data_received_on"]].copy().dropna(subset=[feature])
    
    if X_df.empty:
        log.warning(f"No valid, non-null data found for feature {feature} after preprocessing.")
        return AnamolyPredictionResponse(
            asset_code=asset_code, feature=feature, predictions=[], total_anomalies=0
        )
        
    X_scaled = model_package['scaler'].transform(X_df[[feature]])
    
    predictions = model_package['model'].predict(X_scaled)

    X_df['Anomaly_Flag'] = predictions
    X_df['Data_received_on'] = X_df["data_received_on"].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    X_df.drop(columns=["data_received_on"], inplace=True)
    
    total_anomalies = (X_df['Anomaly_Flag'] == -1).sum()
    log.info(f"Asset {asset_code}, Feature {feature}: Detected {total_anomalies} anomalies.")

    report_list = X_df.to_dict('records')
    
    for record in report_list:
        record['Anomaly_Flag'] = int(record['Anomaly_Flag'])
        record[feature] = float(record[feature])

    return AnamolyPredictionResponse(
        asset_code=asset_code,
        feature=feature,
        predictions=report_list, # type: ignore
        total_anomalies=total_anomalies
    )

def get_resample_rule(months_input: int) -> str:
    """
    This function takes
        months_input : int (months)
    It returns
        resample_rule_days : string    
    """
    DAYS_IN_MONTH = 30.4375
    TARGET_POINTS = 11 
    
    total_days = months_input * DAYS_IN_MONTH
    resample_rule_days = max(1, round(total_days / TARGET_POINTS))
    
    return f'{resample_rule_days}D'

def Fan_health_analysis(request_data: PredictionRequest):
    """
    This function takes 
        periods : int (months)
        failure_threshold : float (Assumed Damper Difference at which End of Lifecycle happens)
    It returns 
        failure_message : A user understandable message like : Earliest Potential End of Life: 2027-02-26 ,
        earliest_end_of_life_cycle : Date at which, the failure threshold reached
        failure_threshold : The value of failure threshold 
        resampled_predicted_data : The resampled data of the forecasted dataframe defined by months * 30.4375 (Days in a month) / 11
    """
    if fan_forecast_model is None:
        raise HTTPException(status_code=503, detail="Fan Health Prediction model is currently unavailable.")
    
    months_input = request_data.periods 
    eol_threshold = request_data.failure_threshold

    total_days = int(math.ceil(months_input * 30.4375)) 
    resample_rule = get_resample_rule(months_input)

    log.info(f"Using dynamic resample rule: {resample_rule}")

    future = fan_forecast_model.make_future_dataframe(periods=total_days, freq='D') 
    forecast = fan_forecast_model.predict(future)
    
    forecast.rename(columns={'yhat_upper': 'prediction'}, inplace=True) 
    
    upper_bounds = forecast[['ds', 'prediction']]
    failure_pred_series = upper_bounds[upper_bounds["prediction"] >= eol_threshold].copy()
    
    if failure_pred_series.empty:
        failure_pred = pd.NaT
        failure_message = "No Failure chances in the Fan in next predicted time interval"
        earliest_end_of_life = None
    else:
        failure_pred = failure_pred_series['ds'].min()
        earliest_end_of_life = failure_pred.strftime('%Y-%m-%d') 
        failure_message = f"Earliest Potential End of Life: {earliest_end_of_life}"
    
    resample_data = forecast[['ds', 'prediction']].copy()
    resample_data.set_index('ds', inplace=True)
    
    resampled_forecast = resample_data.resample(resample_rule).mean().dropna(how='all').reset_index()

    resampled_forecast.rename(columns={
        'ds': 'date', 
        'prediction': 'prediction_mean' 
    }, inplace=True)

    resampled_forecast['date'] = resampled_forecast['date'].dt.strftime('%Y-%m-%d')
    
    resampled_predicted_data = resampled_forecast.to_dict('records')

    return PredictionResponse(
        message=failure_message,
        earliest_end_of_life=earliest_end_of_life,
        failure_threshold=eol_threshold,
        resampled_predicted_data=resampled_predicted_data
    )

def Damper_health_analysis(request_data: PredictionRequest):
    """
    This function takes 
        periods : int (months)
        failure_threshold : float (Assumed Damper Difference at which End of Lifecycle happens)
    It returns 
        failure_message : A user understandable message like : Earliest Potential End of Life: 2027-02-26 ,
        earliest_end_of_life_cycle : Date at which, the failure threshold reached
        failure_threshold : The value of failure threshold 
        resampled_predicted_data : The resampled data of the forecasted dataframe defined by months * 30.4375 (Days in a month) / 11
    """
    if damper_forecast_model is None:
        raise HTTPException(status_code=503, detail="Damper Health Prediction model is currently unavailable.")
    
    months_input = request_data.periods 
    eol_threshold = request_data.failure_threshold

    total_days = int(math.ceil(months_input * 30.4375)) 
    resample_rule = get_resample_rule(months_input)

    log.info(f"Using dynamic resample rule: {resample_rule}")

    future = damper_forecast_model.make_future_dataframe(periods=total_days, freq='D') 
    forecast = damper_forecast_model.predict(future)
    
    forecast.rename(columns={'yhat_upper': 'prediction'}, inplace=True) 
    
    upper_bounds = forecast[['ds', 'prediction']]
    failure_pred_series = upper_bounds[upper_bounds["prediction"] >= eol_threshold].copy()
    
    if failure_pred_series.empty:
        failure_pred = pd.NaT
        failure_message = "No Failure chances in the Damper in next predicted time interval"
        earliest_end_of_life = None
    else:
        failure_pred = failure_pred_series['ds'].min()
        earliest_end_of_life = failure_pred.strftime('%Y-%m-%d') 
        failure_message = f"Earliest Potential End of Life: {earliest_end_of_life}"
    
    resample_data = forecast[['ds', 'prediction']].copy()
    resample_data.set_index('ds', inplace=True)
    
    resampled_forecast = resample_data.resample(resample_rule).mean().dropna(how='all').reset_index()

    resampled_forecast.rename(columns={
        'ds': 'date', 
        'prediction': 'prediction_mean' 
    }, inplace=True)

    resampled_forecast['date'] = resampled_forecast['date'].dt.strftime('%Y-%m-%d')
    
    resampled_predicted_data = resampled_forecast.to_dict('records')

    return PredictionResponse(
        message=failure_message,
        earliest_end_of_life=earliest_end_of_life,
        failure_threshold=eol_threshold,
        resampled_predicted_data=resampled_predicted_data
    )



@router.post('/damper_health_prediction', response_model=PredictionResponse)
def Damper_health_prediction(
    request_data: PredictionRequest
):
    start = time.time()
    log.info(f"input Data: {request_data.dict()}") 
    result = Damper_health_analysis(request_data)
    end = time.time()
    log.info(f"VOX AHU1 Damper End of Life Prediction completed in {end - start:.2f} seconds") 
    return result

@router.post('/Fan_Speed_health_prediction', response_model=PredictionResponse)
def Fan_Speed_health_prediction(
    request_data: PredictionRequest
):
    start = time.time()
    log.info(f"input Data: {request_data.dict()}") 
    result = Fan_health_analysis(request_data)
    end = time.time()
    log.info(f"VOX AHU1 Fan Speed End of Life Prediction completed in {end - start:.2f} seconds") 
    return result

@router.post('/anomaly_detection_prediction', response_model=AnamolyPredictionResponse)
def Anomaly_detection_endpoint(
    request_data: AnamolyPredictionRequest
):
    start = time.time()
    log.info(f"Anomaly detection initiated for Asset: {request_data.asset}, Feature: {request_data.feature}") 
    result = Anomaly_detection_analysis(request_data)
    end = time.time()
    log.info(f"Anomaly Detection completed in {end - start:.2f} seconds. Anomalies found: {result.total_anomalies}") 
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
    
    if fan_forecast_model is None:
        return {
            "status": "unavailable",
            "model_loaded": False,
            "message": "Fan Speed Forecast model is not loaded. Please check server logs."
        }

    return {
        "status": "ready",
        "model_loaded": True,
        "Damper_Health_model_type": type(damper_forecast_model).__name__,
        "Fan_Speed_Health_model_type": type(fan_forecast_model).__name__,
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
            target_column=request_data.target_variable,  
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
            target_column=request_data.target_variable,  
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
