from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from datetime import datetime
import requests
from dotenv import load_dotenv
from src.bms_ai.logger_config import setup_logger

load_dotenv()
log = setup_logger(__name__)

router = APIRouter(prefix="/lstm", tags=["LSTM Predictions"])

model = None
feature_scaler = None
target_scaler = None
TIMESTEPS = 24

class PredictionResult(BaseModel):
    timestamp: str
    TrAvg: float
    HuAvg1: float

class PredictionListResponse(BaseModel):
    predictions: List[PredictionResult]
    total_predictions: int

class HourPredictionRequest(BaseModel):
    hour_index: int

class WeatherPredictionRequest(BaseModel):
    city: str = "Sharjah"
    days: int = 3

def load_lstm_artifacts():
    """Load LSTM model and scalers from artifacts folder"""
    global model, feature_scaler, target_scaler
    
    try:
        model_path = "artifacts/lstm_indoor_prediction_model.keras"
        if not os.path.exists(model_path):
            model_path = "artifacts/lstm_indoor_prediction_model.h5"
        
        model = load_model(model_path)
        log.info(f"LSTM model loaded successfully from {model_path}")
        
        feature_scaler = joblib.load('artifacts/feature_scaler.pkl')
        target_scaler = joblib.load('artifacts/target_scaler.pkl')
        log.info("Scalers loaded successfully")
        
        return True
    except Exception as e:
        log.error(f"Failed to load LSTM artifacts: {e}")
        return False

@router.on_event("startup")
async def startup_event():
    """Initialize model on router startup"""
    load_lstm_artifacts()

@router.get("/health")
def health_check():
    """Check if LSTM model is loaded and ready"""
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "scalers_loaded": feature_scaler is not None and target_scaler is not None
    }

@router.post("/predict_from_weather", response_model=PredictionListResponse)
def predict_from_weather(request: WeatherPredictionRequest):
    """
    Get indoor temperature and humidity predictions based on weather forecast
    
    Args:
        city: City name for weather forecast (default: Dubai)
        days: Number of days to forecast (default: 3, max: 3)
    
    Returns:
        List of predictions with timestamps
    """
    try:
        if model is None or feature_scaler is None or target_scaler is None:
            if not load_lstm_artifacts():
                raise HTTPException(status_code=500, detail="Model not loaded. Please check artifacts folder.")
        
        weather_api = os.getenv("WEATHER_API_KEY")
        if not weather_api:
            raise HTTPException(status_code=500, detail="WEATHER_API_KEY not found in environment variables")
        
        log.info(f"Fetching weather forecast for {request.city}, {request.days} days")
        forecast_url = f"https://api.weatherapi.com/v1/forecast.json?key={weather_api}&q={request.city}&days={request.days}&aqi=yes&alerts=yes"
        forecast_response = requests.get(url=forecast_url)
        
        if forecast_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Weather API error: {forecast_response.status_code}")
        
        weather_data = forecast_response.json()
        
        outdoor_temps = []
        outdoor_humidity = []
        hours = []
        timestamps = []
        
        for day in weather_data['forecast']['forecastday']:
            for hour_data in day['hour']:
                outdoor_temps.append(hour_data['temp_c'])
                outdoor_humidity.append(hour_data['humidity'])
                
                time_obj = datetime.strptime(hour_data['time'], '%Y-%m-%d %H:%M')
                hours.append(time_obj.hour)
                timestamps.append(hour_data['time'])
        
        log.info(f"Extracted {len(outdoor_temps)} hourly forecasts")
        
        predictions = []
        
        for i in range(len(outdoor_temps)):
            if i < TIMESTEPS - 1:
                temp_seq = [outdoor_temps[0]] * (TIMESTEPS - i - 1) + outdoor_temps[:i+1]
                hum_seq = [outdoor_humidity[0]] * (TIMESTEPS - i - 1) + outdoor_humidity[:i+1]
                hour_seq = [hours[0]] * (TIMESTEPS - i - 1) + hours[:i+1]
            else:
                temp_seq = outdoor_temps[i-TIMESTEPS+1:i+1]
                hum_seq = outdoor_humidity[i-TIMESTEPS+1:i+1]
                hour_seq = hours[i-TIMESTEPS+1:i+1]
            
            input_features = np.column_stack([temp_seq, hum_seq, hour_seq])
            
            scaled_input = feature_scaler.transform(input_features)
            
            scaled_input = scaled_input.reshape(1, TIMESTEPS, 3)
            
            prediction_scaled = model.predict(scaled_input, verbose=0)
            
            prediction_actual = target_scaler.inverse_transform(prediction_scaled)
            
            predictions.append(PredictionResult(
                timestamp=timestamps[i],
                TrAvg=float(prediction_actual[0, 0]),
                HuAvg1=float(prediction_actual[0, 1])
            ))
        
        log.info(f"Generated {len(predictions)} predictions successfully")
        
        return PredictionListResponse(
            predictions=predictions,
            total_predictions=len(predictions)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error in predict_from_weather: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get_hour_prediction", response_model=PredictionResult)
def get_hour_prediction(request: HourPredictionRequest):
    """
    Get prediction for a specific hour index from the latest weather forecast
    
    Args:
        hour_index: The hour index (0-71 for 3-day forecast)
    
    Returns:
        Single prediction for the specified hour
    """
    try:
        weather_request = WeatherPredictionRequest(city="Dubai", days=3)
        full_predictions = predict_from_weather(weather_request)
        
        if request.hour_index < 0 or request.hour_index >= full_predictions.total_predictions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid hour_index. Must be between 0 and {full_predictions.total_predictions - 1}"
            )
        
        return full_predictions.predictions[request.hour_index]
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error in get_hour_prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict_custom")
def predict_custom(
    outdoor_temp_sequence: List[float],
    outdoor_humidity_sequence: List[float],
    hour_sequence: List[int]
):
    """
    Make prediction with custom input sequences
    
    Args:
        outdoor_temp_sequence: List of outdoor temperatures (length = 24)
        outdoor_humidity_sequence: List of outdoor humidity values (length = 24)
        hour_sequence: List of hour values 0-23 (length = 24)
    
    Returns:
        Prediction for indoor temperature and humidity
    """
    try:
        if model is None or feature_scaler is None or target_scaler is None:
            if not load_lstm_artifacts():
                raise HTTPException(status_code=500, detail="Model not loaded. Please check artifacts folder.")
        
        if len(outdoor_temp_sequence) != TIMESTEPS:
            raise HTTPException(status_code=400, detail=f"outdoor_temp_sequence must have {TIMESTEPS} values")
        if len(outdoor_humidity_sequence) != TIMESTEPS:
            raise HTTPException(status_code=400, detail=f"outdoor_humidity_sequence must have {TIMESTEPS} values")
        if len(hour_sequence) != TIMESTEPS:
            raise HTTPException(status_code=400, detail=f"hour_sequence must have {TIMESTEPS} values")
        
        input_features = np.column_stack([outdoor_temp_sequence, outdoor_humidity_sequence, hour_sequence])
        
        scaled_input = feature_scaler.transform(input_features)
        
        scaled_input = scaled_input.reshape(1, TIMESTEPS, 3)
        
        prediction_scaled = model.predict(scaled_input, verbose=0)
        
        prediction_actual = target_scaler.inverse_transform(prediction_scaled)
        
        return {
            "TrAvg": float(prediction_actual[0, 0]),
            "HuAvg1": float(prediction_actual[0, 1])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error in predict_custom: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model_info")
def get_model_info():
    """Get information about the loaded LSTM model"""
    try:
        if model is None:
            return {"error": "Model not loaded"}
        
        import json
        with open('artifacts/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return {
            "model_loaded": True,
            "features": metadata.get('features', []),
            "targets": metadata.get('targets', []),
            "timesteps": metadata.get('timesteps', TIMESTEPS),
            "metrics": metadata.get('metrics', {}),
            "train_samples": metadata.get('train_samples', 0),
            "test_samples": metadata.get('test_samples', 0)
        }
    except Exception as e:
        log.error(f"Error loading model info: {e}")
        return {"error": str(e)}
