# import pandas as pd
# import numpy as np
# import joblib
# import requests
# from datetime import datetime, timedelta, timezone
# from dataclasses import dataclass
# from typing import List
# from tensorflow.keras.models import load_model
# import os
# from dotenv import load_dotenv  
# from src.bms_ai.logger_config import setup_logger
# from src.bms_ai.utils.setpoint_optimization_utils import fetch_movie_schedule, get_occupancy_status_for_timestamp
# load_dotenv()  
# log = setup_logger(__name__)

# @dataclass
# class InferenceConfig:
#     equipment_id: str = "Ahu1"
#     screen_id: str = "Screen 1"
#     building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2"
#     bms_api_url: str = "https://ikoncloud.keross.com/bms-express-server/data"
#     weather_api_key: str = os.getenv("WEATHER_API_KEY")
#     ticket: str = "53b15409-2f32-4d8b-a753-3422a76f3802"
#     ticket_type: str = "jobUser"
#     lookback_minutes: int = 1440 
#     required_datapoints = ['TrAvg']
#     city: str = "sharjah"
#     forecast_days: int = 3 
        

    

# # =============================================================================
# # 2. BMS DATA FETCHER (FOR ANCHORING PAST TIMESTAMPS)
# # =============================================================================
# class BMSDataFetcher:
#     def __init__(self, config: InferenceConfig):
#         self.config = config
        
#     def fetch_last_24_hours(self) -> pd.DataFrame:
#         now_utc = datetime.now(timezone.utc)
#         from_time = now_utc - timedelta(minutes=self.config.lookback_minutes)
        
#         from_str = from_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + " UTC"
#         to_str = now_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + " UTC"
        
#         cleaned_building_id = self.config.building_id.replace("-", "")
#         table_name = f"datapoint_live_monitoring_{cleaned_building_id}"
        
#         datapoints_str = ",".join([f"'{dp}'" for dp in self.config.required_datapoints])
        
#         query = f"""
#             SELECT * FROM {table_name} 
#             WHERE system_type = 'AHU' 
#             AND data_received_on >= '{from_str}' 
#             AND data_received_on <= '{to_str}' 
#             AND equipment_id = '{self.config.equipment_id}' 
#             AND datapoint IN ({datapoints_str}) 
#             ALLOW FILTERING;
#         """
#         print(f"query={query}")
#         log.info(f"Fetching BMS anchor data from {from_str} to {to_str}")
        
#         try:
#             response = requests.post(
#                 self.config.bms_api_url,
#                 json={"query": query.strip()},
#                 headers={"Content-Type": "application/json"},
#                 timeout=30
#             )
#             response.raise_for_status()
#             data = response.json()
#             if not data: return pd.DataFrame()
#             return self._parse_and_resample(data, from_time, now_utc)
#         except Exception as e:
#             log.error(f"BMS API request failed: {e}")
#             return pd.DataFrame()

#     def _parse_and_resample(self, data: list, start_time, end_time) -> pd.DataFrame:
#         df = pd.DataFrame(data)
#         if 'monitoring_data' not in df.columns: return df
            
#         df['data_received_on'] = pd.to_datetime(df['data_received_on'])
#         if df['data_received_on'].dt.tz is not None:
#             df['data_received_on'] = df['data_received_on'].dt.tz_convert(None)
            
#         start_time = start_time.replace(tzinfo=None)
#         end_time = end_time.replace(tzinfo=None)
        
#         df['monitoring_data'] = pd.to_numeric(df['monitoring_data'], errors='coerce')
#         pivot_df = df.pivot_table(index='data_received_on', columns='datapoint', values='monitoring_data')
#         resampled_df = pivot_df.resample('15min').mean()
        
#         pd_start, pd_end = pd.Timestamp(start_time), pd.Timestamp(end_time)
#         expected_index = pd.date_range(start=pd_start.ceil('15min'), end=pd_end.floor('15min'), freq='15min')
        
#         final_df = resampled_df.reindex(expected_index).ffill().bfill()
#         final_df.index.name = 'data_received_on'
#         return final_df

# # =============================================================================
# # 3. AI INFERENCE PIPELINE
# # =============================================================================
# class AHUInferencePipeline:
#     def __init__(self, config: InferenceConfig, model_path='artifacts/champion_lstm_model.keras', scaler_path='artifacts/champion_scaler.pkl'):
#         self.config = config
#         self.model = load_model(model_path)
#         self.scaler = joblib.load(scaler_path)
        
#         # EXACTLY YOUR 5 FEATURES
#         self.features = ['outdoor_temp', 'outdoor_humidity','occupancy_status', 'hour_of_day', 'minute_of_hour']
#         self.look_back = 96
#         self.forecast_steps = 288  

#     def _fetch_weather(self, target_dates, is_historical=False):
#         """Fetches either historical or future weather and interpolates to 15-mins."""
#         endpoint = "history.json" if is_historical else "forecast.json"
#         log.info(f"Fetching {'historical' if is_historical else 'future'} weather from WeatherAPI...")
        
#         hourly_records = []
#         unique_dates = pd.Series(target_dates).dt.date.unique()
        
#         for date_obj in unique_dates:
#             url = f"https://api.weatherapi.com/v1/{endpoint}"
            
#             params = {"key": self.config.weather_api_key, "q": self.config.city} 
            
#             if is_historical:
#                 params["dt"] = date_obj.strftime('%Y-%m-%d')
#             else:
                
#                 params["days"] = self.config.forecast_days
                
#             response = requests.get(url, params=params)
#             if response.status_code == 200:
#                 data = response.json()
                
                
#                 if is_historical:
#                     # Historical only returns 1 day at a time, so [0] is correct here
#                     for hour in data['forecast']['forecastday'][0]['hour']:
#                         hourly_records.append({
#                             'data_received_on': pd.to_datetime(hour['time']),
#                             'outdoor_temp': hour['temp_c'],
#                             'outdoor_humidity': hour['humidity']
#                         })
#                 else:
#                     # Forecast returns 3 days, so we must loop through all of them!
#                     for day in data['forecast']['forecastday']:
#                         for hour in day['hour']:
#                             hourly_records.append({
#                                 'data_received_on': pd.to_datetime(hour['time']),
#                                 'outdoor_temp': hour['temp_c'],
#                                 'outdoor_humidity': hour['humidity']
#                             })
#             if not is_historical: break # Forecast returns 3 days in one API call
                
#         if not hourly_records:
#             return np.full(len(target_dates), 35.0), np.full(len(target_dates), 60.0)

#         weather_df = pd.DataFrame(hourly_records).set_index('data_received_on')
#         weather_df.index = weather_df.index.tz_localize(None)
        
#         combined_index = weather_df.index.union(target_dates).sort_values().unique()
#         weather_df = weather_df.reindex(combined_index).interpolate(method='linear')
        
#         matched_weather = weather_df.reindex(target_dates).ffill().bfill()
#         return matched_weather['outdoor_temp'].values, matched_weather['outdoor_humidity'].values

#     def _apply_occupancy(self, df):
#         """Applies occupancy logic using the Ticketing API."""
#         schedule = fetch_movie_schedule(ticket=self.config.ticket, ticket_type=self.config.ticket_type)
#         if schedule:
#             df['occupancy_status'] = [
#                 get_occupancy_status_for_timestamp(schedule, self.config.screen_id, ts).get('status', 0) 
#                 for ts in df.index
#             ]
            
#             statuses = df['occupancy_status'].tolist()
#             for i in range(len(statuses)):
#                 if statuses[i] == 1:
#                     if i + 2 < len(statuses) and statuses[i+1] == 0 and statuses[i+2] == 1:
#                         statuses[i+1] = 1  
#                     elif i + 3 < len(statuses) and statuses[i+1] == 0 and statuses[i+2] == 0 and statuses[i+3] == 1:
#                         statuses[i+1], statuses[i+2] = 1, 1  
#             df['occupancy_status'] = statuses
#         else:
#             df['occupancy_status'] = 0
#         return df
    
#     def execute_forecast(self):
#         log.info("INITIATING FORECAST: Collecting your 5 features...")
        
#         # 1. Anchor timestamps with BMS API
#         live_history_df = BMSDataFetcher(self.config).fetch_last_24_hours()
#         if live_history_df.empty or len(live_history_df) < self.look_back:
#             log.error("Failed to get time anchors from BMS database.")
#             return None

#         live_history_df['hour_of_day'] = live_history_df.index.hour
#         live_history_df['minute_of_hour'] = live_history_df.index.minute
#         live_history_df = self._apply_occupancy(live_history_df)
#         past_temps, past_hums = self._fetch_weather(live_history_df.index, is_historical=True)
#         live_history_df['outdoor_temp'] = past_temps
#         live_history_df['outdoor_humidity'] = past_hums
        
#         recent_history = live_history_df[self.features] 

#         start_future = recent_history.index[-1] + pd.Timedelta(minutes=15)
#         future_dates = pd.date_range(start=start_future, periods=self.forecast_steps, freq='15min')
#         future_df = pd.DataFrame(index=future_dates)
        
#         future_df['hour_of_day'] = future_df.index.hour
#         future_df['minute_of_hour'] = future_df.index.minute
#         future_df = self._apply_occupancy(future_df)
#         future_temps, future_hums = self._fetch_weather(future_dates, is_historical=False)
#         future_df['outdoor_temp'] = future_temps
#         future_df['outdoor_humidity'] = future_hums
        
#         future_features_locked = future_df[self.features]

    
#         combined_features = pd.concat([recent_history, future_features_locked])
#         scaled_features = self.scaler.transform(combined_features)
        
#         X_future = np.array([scaled_features[i-self.look_back:i] for i in range(self.look_back, len(scaled_features))])
        
#         log.info("Predicting Future TrAvg...")
#         future_df['Predicted_TrAvg'] = self.model.predict(X_future, verbose=0).flatten()
        
#         log.info("FORECAST COMPLETE!")
#         return future_df[['outdoor_temp', 'occupancy_status', 'Predicted_TrAvg']]

# if __name__ == "__main__":    
#     predictor = AHUInferencePipeline(config=InferenceConfig())
#     forecast_df = predictor.execute_forecast()
#     if forecast_df is not None:
#         log.info(f"\n{forecast_df.head(10) }")
#         print(forecast_df)
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.setpoint_optimization_utils import (
    fetch_movie_schedule,
    get_occupancy_status_for_timestamp
)

load_dotenv()
log = setup_logger(__name__)



current_dir = os.path.dirname(os.path.abspath(__file__))

artifacts_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'artifacts'))
MODEL_PATH = os.path.join(artifacts_dir, 'champion_lstm_model.keras')
SCALER_PATH = os.path.join(artifacts_dir, 'champion_scaler.pkl')

# 4. Load the models
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@dataclass
class InferenceConfig:
    equipment_id: str = "Ahu1"
    screen_id: str = "Screen 1"
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2"
    bms_api_url: str = "https://ikoncloud.keross.com/bms-express-server/data"
    weather_api_key: str = os.getenv("WEATHER_API_KEY")
    ticket: str = "62768dd6-85dc-42d9-87c6-a1d3a870e509"
    ticket_type: str = "jobUser"
    lookback_minutes: int = 1440
    required_datapoints = ['TrAvg']
    city: str = "sharjah"
    forecast_days: int = 3


# =========================
# BMS FETCHER
# =========================
class BMSDataFetcher:
    def __init__(self, config: InferenceConfig):
        self.config = config

    def fetch_last_24_hours(self):
        now_utc = datetime.now(timezone.utc)
        from_time = now_utc - timedelta(minutes=self.config.lookback_minutes)

        from_str = from_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + " UTC"
        to_str = now_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + " UTC"

        cleaned_building_id = self.config.building_id.replace("-", "")
        table_name = f"datapoint_live_monitoring_{cleaned_building_id}"

        datapoints_str = ",".join([f"'{dp}'" for dp in self.config.required_datapoints])

        query = f"""
        SELECT * FROM {table_name}
        WHERE system_type = 'AHU'
        AND data_received_on >= '{from_str}'
        AND data_received_on <= '{to_str}'
        AND equipment_id = '{self.config.equipment_id}'
        AND datapoint IN ({datapoints_str})
        ALLOW FILTERING;
        """

        try:
            response = requests.post(
                self.config.bms_api_url,
                json={"query": query.strip()},
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                return pd.DataFrame()

            return self._parse_and_resample(data, from_time, now_utc)

        except Exception as e:
            log.error(f"BMS API failed: {e}")
            return pd.DataFrame()

    def _parse_and_resample(self, data, start_time, end_time):
        df = pd.DataFrame(data)

        if 'monitoring_data' not in df.columns:
            return df

        df['data_received_on'] = pd.to_datetime(df['data_received_on'])
        df = df.sort_values('data_received_on')

        df = df.drop_duplicates(subset=['data_received_on', 'datapoint'])

        if df['data_received_on'].dt.tz is not None:
            df['data_received_on'] = df['data_received_on'].dt.tz_convert(None)

        df['monitoring_data'] = pd.to_numeric(df['monitoring_data'], errors='coerce')

        pivot_df = df.pivot_table(
            index='data_received_on',
            columns='datapoint',
            values='monitoring_data'
        )

        resampled_df = pivot_df.resample('15min').mean()

        start_time = start_time.replace(tzinfo=None)
        end_time = end_time.replace(tzinfo=None)

        expected_index = pd.date_range(
            start=pd.Timestamp(start_time).ceil('15min'),
            end=pd.Timestamp(end_time).floor('15min'),
            freq='15min'
        )

        final_df = resampled_df.reindex(expected_index).ffill().bfill()
        return final_df



class AHUInferencePipeline:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = model
        self.scaler = scaler

        self.features = [
            'outdoor_temp',
            'outdoor_humidity',
            'occupancy_status',
            'hour_of_day',
            'minute_of_hour'
        ]

        self.look_back = 96
        self.forecast_steps = 288

    def _fetch_weather(self, dates, is_historical=False):
        try:
            url = f"https://api.weatherapi.com/v1/{'history.json' if is_historical else 'forecast.json'}"

            params = {
                "key": self.config.weather_api_key,
                "q": self.config.city
            }

            if not is_historical:
                params["days"] = self.config.forecast_days
            else:
                params["dt"] = dates[0].strftime('%Y-%m-%d')

            response = requests.get(url, params=params, timeout=10)
            
            data = response.json()

            hourly = []
            for day in data['forecast']['forecastday']:
                for hour in day['hour']:
                    hourly.append({
                        "data_received_on": pd.to_datetime(hour['time']),
                        "outdoor_temp": hour['temp_c'],
                        "outdoor_humidity": hour['humidity']
                    })

            df = pd.DataFrame(hourly).set_index('data_received_on')
            df.index = df.index.tz_localize(None)

            df = df.reindex(df.index.union(dates)).interpolate()
            df = df.reindex(dates).ffill().bfill()

            return df['outdoor_temp'].values, df['outdoor_humidity'].values

        except Exception:
            log.warning("Weather API failed → using defaults")
            return np.full(len(dates), 35.0), np.full(len(dates), 60.0)

    def _apply_occupancy(self, df, schedule):
        
        if schedule:
            df['occupancy_status'] = [
                get_occupancy_status_for_timestamp(schedule, self.config.screen_id, ts).get('status', 0) 
                for ts in df.index
            ]
            
            statuses = df['occupancy_status'].tolist()
            for i in range(len(statuses)):
                if statuses[i] == 1:
                    if i + 2 < len(statuses) and statuses[i+1] == 0 and statuses[i+2] == 1:
                        statuses[i+1] = 1  
                    elif i + 3 < len(statuses) and statuses[i+1] == 0 and statuses[i+2] == 0 and statuses[i+3] == 1:
                        statuses[i+1], statuses[i+2] = 1, 1  
            df['occupancy_status'] = statuses
        else:
            df['occupancy_status'] = 0
        return df

    def execute_forecast(self):
        log.info("INITIATING FORECAST: Collecting your 5 features...")
        
        # 1. FETCH SCHEDULE EXACTLY ONCE HERE!
        log.info("Fetching movie schedule from Ticketing API...")
        master_schedule = fetch_movie_schedule(ticket=self.config.ticket, ticket_type=self.config.ticket_type)
        
        
        live_history_df = BMSDataFetcher(self.config).fetch_last_24_hours()
        if live_history_df.empty or len(live_history_df) < self.look_back:
            log.error("Failed to get time anchors from BMS database.")
            return None

        live_history_df['hour_of_day'] = live_history_df.index.hour
        live_history_df['minute_of_hour'] = live_history_df.index.minute
        
        # 3. Pass the master_schedule to the PAST dataframe
        live_history_df = self._apply_occupancy(live_history_df, master_schedule)
        
        past_temps, past_hums = self._fetch_weather(live_history_df.index, is_historical=True)
        live_history_df['outdoor_temp'] = past_temps
        live_history_df['outdoor_humidity'] = past_hums
        
        recent_history = live_history_df[self.features] 

        start_future = recent_history.index[-1] + pd.Timedelta(minutes=15)
        future_dates = pd.date_range(start=start_future, periods=self.forecast_steps, freq='15min')
        future_df = pd.DataFrame(index=future_dates)
        
        future_df['hour_of_day'] = future_df.index.hour
        future_df['minute_of_hour'] = future_df.index.minute
        
        # 4. Pass the exact same master_schedule to the FUTURE dataframe
        future_df = self._apply_occupancy(future_df, master_schedule)
        
        future_temps, future_hums = self._fetch_weather(future_dates, is_historical=False)
        future_df['outdoor_temp'] = future_temps
        future_df['outdoor_humidity'] = future_hums
        
        future_features_locked = future_df[self.features]

        combined_features = pd.concat([recent_history, future_features_locked])
        scaled_features = self.scaler.transform(combined_features)
        
        X_future = np.array([scaled_features[i-self.look_back:i] for i in range(self.look_back, len(scaled_features))])
        
        log.info("Predicting Future TrAvg...")
        future_df['Predicted_TrAvg'] = self.model.predict(X_future, verbose=0).flatten()
        
        log.info("FORECAST COMPLETE!")
        return future_df[['outdoor_temp', 'occupancy_status', 'Predicted_TrAvg']]