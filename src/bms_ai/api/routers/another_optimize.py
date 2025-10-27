from fastapi import FastAPI, Body, APIRouter
from typing import Dict, Any
from src.bms_ai.logger_config import setup_logger

import joblib
import numpy as np
import warnings
import time


log = setup_logger(__name__)


warnings.filterwarnings('ignore')

router = APIRouter(prefix="/another_optimize", tags=["Prescriptive Optimization"])

# app = FastAPI()

log.error("Loading model and scaler...")

try:
    loaded_model = joblib.load("artifacts/ElasticNet.joblib")
    loaded_scaler = joblib.load("artifacts/minmaxScaler")

    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    log.error("Model or scaler file not found. Using dummy implementations.")
    
except Exception as e:
    log.error(f"Error loading model or scaler: {e}")
    print("Error loading model or scaler. Using dummy implementations.")
 
def create_input_array(conditions: Dict[str, Any], setpoints: Dict[str, float]):
    input_list = [
        conditions['OA Flow'], conditions['OA Humid'], conditions['OA Temp'],
        setpoints['RA temperature setpoint'], 
        setpoints['RA CO2 setpoint'], 
        conditions['RA CO2'], conditions['RA Damper feedback'], conditions['RA Temp'],
        conditions['SA Fan Speed feedback'], 
        setpoints['SA Pressure setpoint'], 
        conditions['SA pressure'], conditions['SA temp'], conditions['Trip status'],
        conditions['date'], conditions['month'], conditions['year'], conditions['hour'], conditions['minute']
    ]
    return np.array([input_list])

def optimize_setpoints(model, conditions: Dict[str, Any], x_scaler, n_iterations=1000):
    search_space = {
        'RA temperature setpoint': (20.0, 27.0),
        'RA CO2 setpoint': (500.0, 800.0),
        'SA Pressure setpoint': (500.0, 1200.0)
    }

    current_setpoints = {
        'RA temperature setpoint': conditions['RA temperature setpoint'],
        'RA CO2 setpoint': conditions['RA CO2 setpoint'],
        'SA Pressure setpoint': conditions['SA Pressure setpoint']
    }
    
    current_input_data = create_input_array(conditions, current_setpoints)
    current_input_data_scaled = x_scaler.transform(current_input_data)
    current_predictions = model.predict(current_input_data_scaled)
    current_power = current_predictions[0][3]

    best_power = current_power
    best_setpoints = current_setpoints.copy()

    for _ in range(n_iterations):
        random_setpoints = {
            'RA temperature setpoint': np.random.uniform(*search_space['RA temperature setpoint']),
            'RA CO2 setpoint': np.random.uniform(*search_space['RA CO2 setpoint']),
            'SA Pressure setpoint': np.random.uniform(*search_space['SA Pressure setpoint'])
        }

        input_data = create_input_array(conditions, random_setpoints)
        input_data_scaled = x_scaler.transform(input_data)
        predictions = model.predict(input_data_scaled)
        predicted_power = predictions[0][3]
        if predicted_power < best_power:
                best_power = predicted_power
                best_setpoints = random_setpoints

    return best_setpoints, best_power

def prediction_with_optimization(current_conditions: Dict[str, Any], model, X_scaler):
    start_time = time.time()
    print(f"X_Scaler: {X_scaler is not None}, Model: {model is not None}")
    optimal_setpoints, minimal_power = optimize_setpoints(model, current_conditions, X_scaler)
    log.info(f"Optimal Setpoints: {optimal_setpoints}, Minimal Fan Power: {minimal_power}")
    
    optimal_input = create_input_array(current_conditions, optimal_setpoints)
    optimal_input_scaled = X_scaler.transform(optimal_input)
    optimal_commands = model.predict(optimal_input_scaled)

    optimal_commands[0][0] = np.clip(optimal_commands[0][0], 0, 10)
    optimal_commands[0][1] = np.clip(optimal_commands[0][1], 0, 100)
    optimal_commands[0][2] = np.clip(optimal_commands[0][2], 0, 100)
    end_time = time.time()  
    time_taken = end_time - start_time
    log.info(f"Optimization completed in {time_taken:.2f} seconds")

    return {
        "best_setpoints": {
            "RA  temperature setpoint": optimal_setpoints['RA temperature setpoint'],
            "RA CO2 setpoint": optimal_setpoints['RA CO2 setpoint'],
            "SA Pressure setpoint": optimal_setpoints['SA Pressure setpoint']
        },
        "min_fan_power_kw": minimal_power,
        "total_combinations_tested": 1000,
        "optimization_time_seconds": time_taken
    }

@router.post('/')
def predict_with_new_data(
    current_conditions: Dict[str, Any] = Body(),
    model = loaded_model,
    X_scaler=loaded_scaler
):
    start = time.time()
    print(f"Current conditions received: {current_conditions}")
    result = prediction_with_optimization(current_conditions, model, X_scaler)
    end = time.time()
    print(f"Prediction completed in {end - start:.2f} seconds")
    return result