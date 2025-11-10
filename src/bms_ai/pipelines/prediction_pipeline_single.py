import sys
import pandas as pd
import numpy as np
from src.bms_ai.exception import CustomException
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.main_utils import load_object
import os

log = setup_logger(__name__)

class PredictionPipelineSingle:
    """
    Prediction pipeline for single output (Fan Power) model.
    """
    
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model_single.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor_single.pkl")
        
    def predict(self, features):
        """
        Make predictions using the trained single output model.
        
        Args:
            features: pd.DataFrame with input features
            
        Returns:
            Predicted Fan Power value(s)
        """
        try:
            log.info("Loading model and preprocessor for single output prediction...")
            
            # Load model and preprocessor
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            
            log.info(f"Input features shape: {features.shape}")
            
            # Transform features
            data_scaled = preprocessor.transform(features)
            log.info(f"Scaled features shape: {data_scaled.shape}")
            
            # Make prediction
            prediction = model.predict(data_scaled)
            log.info(f"Prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'scalar'}")
            
            return prediction
            
        except Exception as e:
            log.error(f"Exception occurred in predict: {e}")
            raise CustomException(e, sys)


class CustomDataSingle:
    """
    Custom data class for single output prediction input.
    """
    
    def __init__(self,
                 oa_flow: float,
                 oa_humid: float,
                 oa_temp: float,
                 ra_temp_setpoint: float,
                 ra_co2: float,
                 ra_co2_setpoint: float,
                 ra_damper_feedback: float,
                 ra_temp: float,
                 sa_fan_speed_feedback: float,
                 sa_pressure_setpoint: float,
                 sa_pressure: float,
                 sa_temp: float,
                 sup_fan_cmd: int = 1,
                 plant_enable: int = 1,
                 trip_status: float = 1.0,
                 airflow_status: int = 1,
                 auto_status: int = 1,
                 bag_filter_dirty: int = 0,
                 pre_filter_dirty: int = 0,
                 hour: int = 12,
                 dayofweek: int = 0,
                 month: int = 1,
                 dayofyear: int = 1):
        """
        Initialize custom data for Fan Power prediction.
        
        All parameters match the features expected by the model.
        """
        self.oa_flow = oa_flow
        self.oa_humid = oa_humid
        self.oa_temp = oa_temp
        self.ra_temp_setpoint = ra_temp_setpoint
        self.ra_co2 = ra_co2
        self.ra_co2_setpoint = ra_co2_setpoint
        self.ra_damper_feedback = ra_damper_feedback
        self.ra_temp = ra_temp
        self.sa_fan_speed_feedback = sa_fan_speed_feedback
        self.sa_pressure_setpoint = sa_pressure_setpoint
        self.sa_pressure = sa_pressure
        self.sa_temp = sa_temp
        self.sup_fan_cmd = sup_fan_cmd
        self.plant_enable = plant_enable
        self.trip_status = trip_status
        self.airflow_status = airflow_status
        self.auto_status = auto_status
        self.bag_filter_dirty = bag_filter_dirty
        self.pre_filter_dirty = pre_filter_dirty
        self.hour = hour
        self.dayofweek = dayofweek
        self.month = month
        self.dayofyear = dayofyear
        
    def get_data_as_dataframe(self):
        """
        Convert input data to DataFrame with engineered features.
        """
        try:
            # Create cyclical features
            hour_sin = np.sin(2 * np.pi * self.hour / 24)
            hour_cos = np.cos(2 * np.pi * self.hour / 24)
            month_sin = np.sin(2 * np.pi * self.month / 12)
            month_cos = np.cos(2 * np.pi * self.month / 12)
            dayofweek_sin = np.sin(2 * np.pi * self.dayofweek / 7)
            dayofweek_cos = np.cos(2 * np.pi * self.dayofweek / 7)
            
            custom_data_input_dict = {
                "OA Flow": [self.oa_flow],
                "OA Humid": [self.oa_humid],
                "OA Temp": [self.oa_temp],
                "RA  temperature setpoint": [self.ra_temp_setpoint],
                "RA CO2": [self.ra_co2],
                "RA CO2 setpoint": [self.ra_co2_setpoint],
                "RA Damper feedback": [self.ra_damper_feedback],
                "RA Temp": [self.ra_temp],
                "SA Fan Speed feedback": [self.sa_fan_speed_feedback],
                "SA Pressure setpoint": [self.sa_pressure_setpoint],
                "SA pressure": [self.sa_pressure],
                "SA temp": [self.sa_temp],
                "Sup fan cmd": [self.sup_fan_cmd],
                "Plant enable": [self.plant_enable],
                "Trip status": [self.trip_status],
                "airflow Status": [self.airflow_status],
                "auto Status": [self.auto_status],
                "Bag filter dirty status": [self.bag_filter_dirty],
                "pre Filter dirty staus": [self.pre_filter_dirty],
                "hour": [self.hour],
                "dayofweek": [self.dayofweek],
                "month": [self.month],
                "dayofyear": [self.dayofyear],
                "hour_sin": [hour_sin],
                "hour_cos": [hour_cos],
                "month_sin": [month_sin],
                "month_cos": [month_cos],
                "dayofweek_sin": [dayofweek_sin],
                "dayofweek_cos": [dayofweek_cos]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            log.error(f"Exception occurred in get_data_as_dataframe: {e}")
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    # Test prediction with sample data
    custom_data = CustomDataSingle(
        oa_flow=1000.89,
        oa_humid=80.93,
        oa_temp=40.93,
        ra_temp_setpoint=24.5,
        ra_co2=500,
        ra_co2_setpoint=750.0,
        ra_damper_feedback=15.0,
        ra_temp=30.38,
        sa_fan_speed_feedback=100.0,
        sa_pressure_setpoint=750.0,
        sa_pressure=297.92,
        sa_temp=18.36,
        sup_fan_cmd=1,
        plant_enable=1,
        trip_status=1.0,
        airflow_status=1,
        auto_status=1,
        bag_filter_dirty=0,
        pre_filter_dirty=0,
        hour=6,
        dayofweek=2,
        month=5,
        dayofyear=142
    )
    
    # Get dataframe
    pred_df = custom_data.get_data_as_dataframe()
    print("Input DataFrame:")
    print(pred_df)
    
    # Make prediction
    pipeline = PredictionPipelineSingle()
    prediction = pipeline.predict(pred_df)
    
    print(f"\nPredicted Fan Power: {prediction[0]:.4f} KW")
