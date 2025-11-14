from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from src.bms_ai.logger_config import setup_logger
from pathlib import Path
from src.bms_ai.api.routers.heatlh_check import get_resample_rule, Damper_health_analysis

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

try:
    damper_forecast_model = joblib.load("artifacts/AHU1_Damper_health.joblib")
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

@router.post('/Damper_health_prediction', response_model=PredictionResponse)
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
