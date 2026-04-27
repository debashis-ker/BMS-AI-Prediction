# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from typing import List
# import traceback

# from src.bms_ai.components.inference_pipeline import AHUInferencePipeline, InferenceConfig
# from src.bms_ai.logger_config import setup_logger

# log = setup_logger(__name__)

# router = APIRouter(prefix="/lstm", tags=["LSTM Predictions"])

# class WeatherPredictionRequest(BaseModel):
#     equipment_id: str = "Ahu1"  
#     screen_id: str = "Screen 1"
#     building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2"
#     ticket: str = "53b15409-2f32-4d8b-a753-3422a76f3802"
#     ticket_type: str = "jobUser"
#     lookback_minutes: int = 1440
#     city: str = "Sharjah"
#     days: int = 3


# class PredictionResult(BaseModel):
#     timestamp: str
#     TrAvg: float

# class PredictionListResponse(BaseModel):
#     total_predictions: int
#     predictions: List[PredictionResult]


# @router.post("/predict_from_weather_occupancy", response_model=PredictionListResponse)
# def predict_from_weather_occupancy(request: WeatherPredictionRequest):
#     log.info(f"Received forecast request for {request.equipment_id} ({request.screen_id})")
    
#     try:
        
#         config = InferenceConfig(
#             equipment_id=request.equipment_id,
#             screen_id=request.screen_id,
#             building_id=request.building_id,
#             ticket=request.ticket,
#             ticket_type=request.ticket_type,
#             lookback_minutes=request.lookback_minutes,
#             city=request.city,
#             forecast_days=request.days
#         )
        
#         # Initialize the pipeline and execute
#         predictor = AHUInferencePipeline(config=config)
#         forecast_df = predictor.execute_forecast()
        
        
#         if forecast_df is None or forecast_df.empty:
#             raise HTTPException(
#                 status_code=500, 
#                 detail="Pipeline failed to generate forecast. Check server logs."
#             )

#         # 
#         #Format the Pandas DataFrame to match your new schema
#         forecast_df = forecast_df.reset_index()
        
        
#         forecast_df.rename(columns={
#             'index': 'timestamp',
#             'Predicted_TrAvg': 'TrAvg'
#         }, inplace=True)
        
#         # Convert timestamps to strings
#         forecast_df['timestamp'] = forecast_df['timestamp'].astype(str)
        
        
#         final_df = forecast_df[['timestamp', 'TrAvg']]
        
#         # Convert to dictionary format
#         forecast_records = final_df.to_dict(orient="records")
        
#         return PredictionListResponse(
#             total_predictions=len(forecast_records),
#             predictions=forecast_records
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         log.error(f"Critical API Error: {str(e)}")
#         log.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import traceback

from components.lstm_inference_pipeline import (
    AHUInferencePipeline,
    InferenceConfig
)
from src.bms_ai.logger_config import setup_logger

log = setup_logger(__name__)

router = APIRouter(prefix="/lstm_occupancy", tags=["LSTM Predictions"])


class RequestModel(BaseModel):
    equipment_id: str = "Ahu1"
    screen_id: str = "Screen 1"
    building_id: str
    ticket: str
    ticket_type: str
    lookback_minutes: int = 1440
    city: str = "Sharjah"
    days: int = 3


class Prediction(BaseModel):
    timestamp: str
    TrAvg: float


class ResponseModel(BaseModel):
    total_predictions: int
    predictions: List[Prediction]


@router.post("/predict_from_weather_occupancy", response_model=ResponseModel)
def predict(request: RequestModel):
    try:
        config = InferenceConfig(
            equipment_id=request.equipment_id,
            screen_id=request.screen_id,
            building_id=request.building_id,
            ticket=request.ticket,
            ticket_type=request.ticket_type,
            lookback_minutes=request.lookback_minutes,
            city=request.city,
            forecast_days=request.days
        )

        pipeline = AHUInferencePipeline(config)
        df = pipeline.execute_forecast()

        if df is None or df.empty:
            raise HTTPException(status_code=500, detail="Forecast failed")

        df = df.reset_index()
        df.rename(columns={
            'index': 'timestamp',
            'Predicted_TrAvg': 'TrAvg'
        }, inplace=True)

        df['timestamp'] = df['timestamp'].astype(str)

        records = df[['timestamp', 'TrAvg']].to_dict(orient="records")

        return ResponseModel(
            total_predictions=len(records),
            predictions=records
        )

    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
