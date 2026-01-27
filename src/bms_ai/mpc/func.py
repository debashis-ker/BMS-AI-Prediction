import pandas as pd
from typing import List, Dict, Any
import requests
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def add_open_meteo_weather(df, lat, lon):
    df['data_received_on'] = pd.to_datetime(df['data_received_on'])
    start_date = df['data_received_on'].min().strftime('%Y-%m-%d')
    end_date = df['data_received_on'].max().strftime('%Y-%m-%d')
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "auto"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()
        weather_df = pd.DataFrame({
            'ds_weather': pd.to_datetime(data['hourly']['time']),
            'outside_temp': data['hourly']['temperature_2m'],
            'outside_humidity': data['hourly']['relative_humidity_2m']
        })
        
        df = df.sort_values('data_received_on')
        combined_df = pd.merge_asof(
            df, weather_df.sort_values('ds_weather'), 
            left_on='data_received_on', right_on='ds_weather', 
            direction='nearest'
        )
        return combined_df.drop(columns=['ds_weather'])
    except Exception as e:
        print(f"Weather fetch failed: {e}")
        df['outside_temp'] = 0
        df['outside_humidity'] = 0
        return df


def new_data_pipeline(records: List[Dict[str, Any]], STANDARD_DATE_COLUMN: str = "data_received_on", target : str = "FbVFD", setpoint : str = "SpTREff", occupancy_factor : str = "TsOn") -> pd.DataFrame:
    """Preprocesses raw data: Fixes leakage and adds thermal lags."""
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    
    if 'monitoring_data' in df.columns:
        mapping = {'inactive': 0.0, 'active': 1.0}
        df['monitoring_data'] = df['monitoring_data'].replace(mapping, regex=False)
        df['monitoring_data'] = pd.to_numeric(df['monitoring_data'], errors='coerce')
    
    df[STANDARD_DATE_COLUMN] = pd.to_datetime(df[STANDARD_DATE_COLUMN], errors='coerce')
    if df[STANDARD_DATE_COLUMN].dt.tz is not None:
        df[STANDARD_DATE_COLUMN] = df[STANDARD_DATE_COLUMN].dt.tz_localize(None)

    df = add_open_meteo_weather(df, 25.33, 55.39) 

    aggregated = df.groupby([STANDARD_DATE_COLUMN, 'datapoint'])['monitoring_data'].agg('first')
    result_df = aggregated.unstack(level='datapoint').reset_index()
    
    weather = df[[STANDARD_DATE_COLUMN, 'outside_temp', 'outside_humidity']].drop_duplicates()
    result_df = pd.merge(result_df, weather, on=STANDARD_DATE_COLUMN, how='left')

    result_df = result_df.sort_values(STANDARD_DATE_COLUMN).set_index(STANDARD_DATE_COLUMN)
    result_df = result_df.resample('10min').mean().ffill()

    result_df['hour'] = result_df.index.hour
    result_df['day_of_week'] = result_df.index.dayofweek
    result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
    result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
    result_df['day_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
    result_df['day_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)

    if occupancy_factor in result_df.columns:
        result_df['Is_Starting_Up'] = ((result_df[occupancy_factor] == 1) & (result_df[occupancy_factor].shift(1) == 0)).astype(int)
    
    result_df['Target_Temp'] = result_df['TempSp1'].shift(-1)
    
    result_df['TempSp1_lag_10m'] = result_df['TempSp1'].shift(1)
    result_df[f'{setpoint}_lag_10_min'] = result_df[setpoint].shift(1)
            
    return result_df.dropna(subset=['Target_Temp', 'TempSp1_lag_10m']).reset_index()


def fetch_data(
        building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2",
        url: str = f"https://ikoncloud.keross.com/bms-express-server/data",
        zone: str = "",
        equipment_id: str = "",
        datapoints: List[str] = [],
        system_type: str = "AHU",
        from_date: str = "",
        to_date: str = ""
):
    cleaned_id = building_id.replace("-", "").lower()
    location_table_name = f"datapoint_live_monitoring_{cleaned_id}"

    query=(
        f"select * from {location_table_name} where "
    )

    if equipment_id:
        query += f"equipment_id = '{equipment_id}' and "

    if datapoints:
        datapoint_list = ', '.join([f"'{f}'" for f in datapoints])
        query += f"datapoint IN ({datapoint_list}) and "

    if system_type:
        query += f"system_type = '{system_type}' and "

    if zone:
        query += f"zone = '{zone}' and "

    if from_date and to_date:
        query += f"data_received_on >= '{from_date}' and data_received_on <= '{to_date}' "
        
    if not from_date and not to_date:
        previous_week_day_in_UTC = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S.%f%z')
        query += f"data_received_on >= '{previous_week_day_in_UTC}' and " 
        query += f"data_received_on <= '{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S.%f%z')}'"

    query += f"allow filtering;"

    API_PAYLOAD = {"query": query}
    try:
        response = requests.post(url, json=API_PAYLOAD, timeout=60)
        response.raise_for_status()
        raw_api_response = response.json()
        
        data_list = []
        if isinstance(raw_api_response, list):
            data_list = raw_api_response
        elif isinstance(raw_api_response, dict) and 'queryResponse' in raw_api_response:
            data_list = raw_api_response.get('queryResponse', [])
        
        if not isinstance(data_list, list):
            raise ValueError("API response data list format is invalid.")
            
        print(f"[A1] Total raw records fetched: {len(data_list)}")
        return data_list
    
    except Exception as e:
        print(f"Failed to fetch batch data from API: {str(e)}")




df = fetch_data(datapoints=['SpTREff','FbVFD','TempSu','TempSp1','SpTROcc','FbFAD'],equipment_id="Ahu13",from_date="2025-11-29 07:00:00",to_date="2026-01-27 07:00:00")

new_df = pd.DataFrame(df)

new_df["datapoint"].value_counts()