from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from src.bms_ai.logger_config import setup_logger
from pathlib import Path

import pandas as pd
import joblib
import warnings
import math
import time

log = setup_logger(__name__)

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/health_check", tags=["Prescriptive Optimization"])

damper_forecast_model = None
fan_forecast_model = None

try:
    damper_forecast_model = joblib.load("artifacts/Damper_health_check_forecast_model.joblib")
    fan_forecast_model = joblib.load("artifacts/Fan_Speed_Health_Check_Model.joblib")
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

@router.post('/Damper_health_prediction', response_model=PredictionResponse)
def Damper_health_prediction(
    request_data: PredictionRequest
):
    start = time.time()
    log.info(f"input Data: {request_data.dict()}") 
    result = Damper_health_analysis(request_data)
    end = time.time()
    log.info(f"Damper Prediction completed in {end - start:.2f} seconds") 
    return result

@router.post('/Fan_health_prediction', response_model=PredictionResponse)
def Fan_health_prediction(
    request_data: PredictionRequest
):
    start = time.time()
    log.info(f"input Data: {request_data.dict()}") 
    result = Fan_health_analysis(request_data)
    end = time.time()
    log.info(f"Fan Prediction completed in {end - start:.2f} seconds") 
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
            "message": "Fan Forecast model is not loaded. Please check server logs."
        }

    return {
        "status": "ready",
        "model_loaded": True,
        "Damper_Health_model_type": type(damper_forecast_model).__name__,
        "Fan_Health_model_type": type(fan_forecast_model).__name__,
        "message": "Forecast model is loaded and ready for predictions."
    }
