from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
from src.bms_ai.logger_config import setup_logger
from pathlib import Path
import json
from src.bms_ai.utils.cassandra_utils import fetch_data_from_metadata, get_metadata, fetch_data
from src.bms_ai.pipelines.damper_optimization_pipeline import (
    train as damper_train,
    optimize as damper_optimize
)
from src.bms_ai.pipelines.generic_optimization_pipeline import (
    train_generic,
    optimize_generic
)

from src.bms_ai.utils.ikon_apis import fetch_and_find_data_points
from src.bms_ai.utils.save_cassandra_data import save_data_to_cassandraV2

import requests
import pandas as pd
import joblib
import warnings
import math
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder


log = setup_logger(__name__)

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/prod", tags=["Prescriptive Optimization"])

damper_forecast_model = None
fan_forecast_model = None
anamoly_model = None

try:
    damper_forecast_model = joblib.load("artifacts/production_models/ahu1_damper_model.joblib")
    log.info("AHU1 Damper Model loaded successfully.")
except Exception as e:
    log.error(f"Error loading Damper forecast model: {e}")
    print(f"Error loading Damper forecast model: {e}")

try:
    fan_forecast_model = joblib.load("artifacts/production_models/ahu1_fan_speed_model.joblib")
    log.info("Ahu1 Fan Speed Model loaded successfully.")
except Exception as e:
    log.error(f"Error loading Fan Speed forecast model: {e}")
    print(f"Error loading Fan Speed forecast model: {e}")

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

FIXED_SYSTEM_TYPE = "AHU"
DEFAULT_BUILDING_ID = "36c27828-d0b4-4f1e-8a94-d962d342e7c2"

class AnamolyPredictionRequest(BaseModel):
    feature: str = Field(..., description="Feature on which Anomaly should be detected (e.g., TSu).")
    site: str = Field(..., description="Site/Zone of the equipment.")
    equipment_id: str = Field(..., description="Equipment ID for which anomaly detection is requested.")
    system_type: str = Field(FIXED_SYSTEM_TYPE, description="System type of the equipment (defaults to AHU).")
    building_id: Optional[str] = Field(DEFAULT_BUILDING_ID, description="Optional building ID for filtering.")

API_URL = "https://ikoncloud.keross.com/bms-express-server/data"
ALL_AVAILABLE_FEATURES = ['TSu', 'Co2RA', 'FbFAD', 'FbVFD', 'HuAvg1']
STANDARD_DATE_COLUMN = "data_received_on"
CO2RA_CEILING_THRESHOLD = 850.0
FBVFD_NORMAL_MAX = 1.0 
BASE_MODEL_PATH = "artifacts/generic_anamoly_models/"
EMISSION_FACTOR = 0.4041

FEATURE_FALLBACKS = {
    'TSu': ['TempSu'],
    'Co2RA': ['Co2Avg'],
    'HuAvg1': ['HuR1', 'HuRt']
}

QUERY_FEATURES = set(ALL_AVAILABLE_FEATURES)
for fallbacks in FEATURE_FALLBACKS.values():
    QUERY_FEATURES.update(fallbacks)

CONSOLIDATION_MAP = {}
for master_feat, fallbacks in FEATURE_FALLBACKS.items():
    CONSOLIDATION_MAP[master_feat] = master_feat
    for fb in fallbacks:
        CONSOLIDATION_MAP[fb] = master_feat

for feat in ALL_AVAILABLE_FEATURES:
    if feat not in CONSOLIDATION_MAP:
        CONSOLIDATION_MAP[feat] = feat
        
MASTER_ANAMOLY_MODELS: Dict[str, Dict[str, Any]] = {} 

log.info("Starting model loading and consolidation...")

all_model_keys_to_load = set(ALL_AVAILABLE_FEATURES)
for fallbacks in FEATURE_FALLBACKS.values():
    all_model_keys_to_load.update(fallbacks)

for model_key_in_file in all_model_keys_to_load:
    model_file = f"{BASE_MODEL_PATH}{model_key_in_file}_model.joblib"
    master_key = CONSOLIDATION_MAP.get(model_key_in_file) 
    
    if master_key is None:
        continue

    try:
        feature_models = joblib.load(model_file)
        if not feature_models:
             log.warning(f"[SKIP] Model file '{model_key_in_file}' loaded but contains NO asset models (empty dictionary).")
             continue 

        if master_key not in MASTER_ANAMOLY_MODELS:
            MASTER_ANAMOLY_MODELS[master_key] = {}
        
        MASTER_ANAMOLY_MODELS[master_key].update(feature_models)
        
        log.info(f"Loaded '{model_key_in_file}' models (Assets: {len(feature_models)}) and consolidated under MASTER KEY: '{master_key}'.")
        
    except FileNotFoundError:
        log.warning(f"Model file not found for {model_key_in_file}. Skipping.")
    except Exception as e:
        log.error(f"FATAL: Error loading model {model_key_in_file}. Skipping: {e}")
        
anamoly_model = MASTER_ANAMOLY_MODELS

class AnomalyVizRequest(BaseModel):
    chart_type: str = Field('pie', description="Type of visualization data requested: 'pie' or 'line'.")

class AnomalyVizResponse(BaseModel):
    chart_type: str
    data: Dict[str, Any] = Field(..., description="Data structure for the requested visualization.")

class EmissionRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Raw input data, expected to contain 'queryResponse'.")
    equipment_id: Optional[str] = Field(None, description="Optional filter for a specific equipment ID.")
    zone: Optional[str] = Field(None, description="Optional filter for a specific site/zone.")

class StaticEmissionRequest(BaseModel):
    equipment_id: Optional[str] = Field(None, description="Optional filter for a specific equipment ID.")
    zone: Optional[str] = Field(None, description="Optional filter for a specific site/zone.")

class StaticEmissionResponse(BaseModel):
    carbon_emission_kg: float
    energy_consumed_kwh: float
    breakdown_by_equipment_and_zone: List[Dict[str, Any]]

class EmissionResponse(BaseModel):
    data: Dict[str, Any] = Field(..., description="The final structured emission report.")

def fetch_all_ahu_data(
    building_id: str = DEFAULT_BUILDING_ID,
    url: str = API_URL
) -> List[Dict]:
    """Fetches ALL historical data for AHUs in a single API call."""
    cleaned_id = building_id.replace("-", "").lower()
    location_table_name = f"datapoint_live_monitoring_values{cleaned_id}"
    
    datapoint_list = ', '.join([f"'{f}'" for f in QUERY_FEATURES])
    
    query = (
        f"select * from {location_table_name} "
        f"where system_type = '{FIXED_SYSTEM_TYPE}' "
        f"and datapoint IN ({datapoint_list}) "
        f"allow filtering;"
    )

    API_PAYLOAD = {"query": query}
    try:
        response = requests.post(url, json=API_PAYLOAD, timeout=60)
        response.raise_for_status()
        raw_api_response = response.json()
        
        if isinstance(raw_api_response, list):
            data_list = raw_api_response
        elif isinstance(raw_api_response, dict) and 'queryResponse' in raw_api_response:
            data_list = raw_api_response.get('queryResponse')
        else:
            raise ValueError(f"API response format unexpected: {raw_api_response}")

        if not isinstance(data_list, list):
            raise ValueError(f"'queryResponse' key found but its value is not a list, or the top-level object was empty.")
        
        return data_list
    
    except Exception as e:
        log.error(f"Failed to fetch batch data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch batch data: {str(e)}")
    
def fetch_all_ahu_dataV2(
        building_id: str = DEFAULT_BUILDING_ID,
        url: str = API_URL,
        site: str = "",
        system_type: str = FIXED_SYSTEM_TYPE,
        equipment_id: Optional[str] = None
    ) -> List[Dict]:
    """Fetches ALL historical data for AHUs in a single API call."""
    cleaned_id = building_id.replace("-", "").lower()
    location_table_name = f"datapoint_live_monitoring_{cleaned_id}"
    datapoint_list = ', '.join([f"'{f}'" for f in ALL_AVAILABLE_FEATURES])
    previous_week_day_in_UTC = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S.%f%z')
    log.debug(f"Fetching data since: {previous_week_day_in_UTC}")
    query = (
        f"select * from {location_table_name} "
        f"where site = '{site}' " + 
        f"and system_type = '{system_type}' " +
        (f"and equipment_id = '{equipment_id}' " if equipment_id else ' ') +
        f"and data_received_on >= '{previous_week_day_in_UTC}' "
        #f"and data_received_on >= '2025-11-08T00:00:00.000 UTC' " +
        f"allow filtering;"
    )
    log.debug(f"Constructed Query: {query}")

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
            
        log.info(f"[A1] Total raw records fetched: {len(data_list)}")
        return data_list
    
    except Exception as e:
        log.error(f"Failed to fetch batch data from API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch batch data: {str(e)}")
    
def fetch_adjustment_history(
    building_id: str,
    site: str,
    equipment_id: str,
    url: str = API_URL
) -> pd.DataFrame:
    """Fetches adjustment history for a specific equipment."""
    cleaned_id = building_id.replace("-", "").lower()
    location_table_name = f"historical_data_opt_{cleaned_id}"
    
    query = (
        f"select * from {location_table_name} "
        f"where site = '{site}' "
        f"and equipment_id = '{equipment_id}' "
        f"and system_type = '{FIXED_SYSTEM_TYPE}' "
        f"allow filtering;"
    )

    API_PAYLOAD = {"query": query}
    try:
        response = requests.post(url, json=API_PAYLOAD, timeout=60)
        response.raise_for_status()
        raw_api_response = response.json()
        
        if isinstance(raw_api_response, list):
            data_list = raw_api_response
        elif isinstance(raw_api_response, dict) and 'queryResponse' in raw_api_response:
            data_list = raw_api_response.get('queryResponse', [])
        else:
            raise ValueError(f"API response format unexpected: {raw_api_response}")

        df = pd.DataFrame(data_list)
        return df
    
    except Exception as e:
        log.error(f"Failed to fetch adjustment history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch adjustment history: {str(e)}")

def anomaly_detection(raw_data: List[Dict[str, Any]], asset_code: str, feature: str) -> List[Dict[str, Any]]:
    if feature not in anamoly_model: #type:ignore
        raise HTTPException(status_code=503, detail=f"Anomaly Detection model for {feature} is unavailable.")

    try:
        model_package = anamoly_model[feature].get(asset_code) #type:ignore
        
        if model_package is None:
             raise KeyError("Model not found for specific asset.")

        model_features = model_package.get('feature_cols', [feature]) 
        log.info(f"[{asset_code}/{feature}] [B1] Model FOUND. Features needed: {model_features}")
        
    except KeyError as e:
        log.warning(f"[{asset_code}/{feature}] Model package not found for key: {e}. Returning empty list.")
        return [] 
    
    try:
        df_wide = anamoly_data_pipeline(raw_data, asset_code, model_package, feature)
            
        if df_wide.empty:
            return []
            
    except Exception as e:
        log.error(f"[{asset_code}/{feature}] Data pipeline error: {e}")
        return [] 
            
    data_column = model_features[0] 
    cols_to_select = [col for col in model_features if col in df_wide.columns] + [STANDARD_DATE_COLUMN]
    X_df = df_wide[cols_to_select].copy().dropna(subset=[data_column])
    
    if X_df.empty:
        log.info(f"[{asset_code}/{feature}] No valid data points left after dropping NaN from primary column.")
        return []
            
    X_for_model = X_df[[col for col in model_features if col in X_df.columns]].copy()
    log.info(f"[{asset_code}/{feature}] [B3] Rows sent to IF model: {len(X_for_model)}. Input cols: {X_for_model.columns.tolist()}")

    try:
        X_scaled = model_package['scaler'].transform(X_for_model)
        predictions = model_package['model'].predict(X_scaled)
    except Exception as e:
        log.error(f"[{asset_code}/{feature}] Prediction scaling/run failed: {e}")
        return []

    X_df['Anomaly_Flag'] = predictions
    log.info(f"[{asset_code}/{feature}] [C1] Raw IF anomalies: {(predictions == -1).sum()}")
    
    # 5. Post-processing Overrides (Ceiling Layer)
    
    # Co2RA Ceiling Check (UPDATED LOGIC from training pipeline)
    if feature == 'Co2RA' and data_column in X_df.columns:
        co2_values_numeric = pd.to_numeric(X_df[data_column], errors='coerce')
        
        condition_high = co2_values_numeric > CO2RA_CEILING_THRESHOLD
        condition_low = co2_values_numeric < CO2RA_CEILING_THRESHOLD
        
        if condition_high.any():
            override_count_to_anomaly = (condition_high & (X_df['Anomaly_Flag'] == 1)).sum()
            log.warning(f"[{asset_code}/{feature}] Co2RA override: Flagged {override_count_to_anomaly} normal points as -1 (Anomaly) because value > {CO2RA_CEILING_THRESHOLD}.")
            X_df.loc[condition_high, 'Anomaly_Flag'] = -1
        
        if condition_low.any():
            override_count_to_normal = (condition_low & (X_df['Anomaly_Flag'] == -1)).sum()
            log.warning(f"[{asset_code}/{feature}] Co2RA override: Flagged {override_count_to_normal} anomaly points as 1 (Normal) because value < {CO2RA_CEILING_THRESHOLD}.")
            X_df.loc[condition_low, 'Anomaly_Flag'] = 1 


    # FbVFD Normal Range Override (Unchanged)
    if feature == 'FbVFD' and data_column in X_df.columns:
        fbvfd_values_numeric = pd.to_numeric(X_df[data_column], errors='coerce')
        normal_fbvfd_condition = (fbvfd_values_numeric >= 0) & (fbvfd_values_numeric <= FBVFD_NORMAL_MAX)
        if normal_fbvfd_condition.any():
            override_count = (normal_fbvfd_condition & (X_df['Anomaly_Flag'] == -1)).sum()
            if override_count > 0:
                log.warning(f"[{asset_code}/{feature}] FbVFD override: Reverted {override_count} anomalies to normal (1) because value is 0-{FBVFD_NORMAL_MAX}.")
            X_df.loc[normal_fbvfd_condition, 'Anomaly_Flag'] = 1
    
    # 6. Final Formatting
    
    # FIX: Keep ALL rows (Normal and Anomaly) as requested.
    final_df = X_df.copy() 
    
    if final_df.empty:
        log.info(f"[{asset_code}/{feature}] Final report size: 0. No data points processed.")
        return []
    
    final_df['data_received_on_str'] = final_df[STANDARD_DATE_COLUMN].dt.strftime('%Y-%m-%d %H:%M:%S.%f') #type:ignore

    report_list = []
    for _, row in final_df.iterrows():
        record = {
            "data_received_on": row['data_received_on_str'],
            "Anomaly_Flag": str(int(row['Anomaly_Flag']))
        }
        # FIX: Key the value by the actual data column name (data_column), removing the redundant primary name.
        data_column = model_features[0] 
        record[data_column] = str(float(row[data_column])) 
        
        report_list.append(record)

    log.info(f"[{asset_code}/{feature}] [C2] Final report list size: {len(report_list)}")
    return report_list

def anamoly_data_pipeline(
    records: List[Dict[str, Any]], 
    asset_code: str, 
    model_package: Dict[str, Any],
    feature_name: str
) -> pd.DataFrame:
    """ Processes long-format records into a wide-format DataFrame, adds temporal, and handles encoding."""
    
    if not records:
        log.warning(f"[{asset_code}/{feature_name}] Pipeline received zero records.")
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    
    df[STANDARD_DATE_COLUMN] = pd.to_datetime(df[STANDARD_DATE_COLUMN], errors='coerce')
    df = df.dropna(subset=[STANDARD_DATE_COLUMN])
    
    if df[STANDARD_DATE_COLUMN].dt.tz is not None: #type:ignore
        df[STANDARD_DATE_COLUMN] = df[STANDARD_DATE_COLUMN].dt.tz_localize(None) #type:ignore
        
    if 'monitoring_data' in df.columns:
        mapping = {'inactive': 0.0, 'active': 1.0}
        df['monitoring_data'] = df['monitoring_data'].replace(mapping, regex=False)
        df['monitoring_data'] = pd.to_numeric(df['monitoring_data'], errors='coerce')
    
    aggregated_scores = df.groupby([STANDARD_DATE_COLUMN, 'asset_code', 'datapoint'])['monitoring_data'].agg('first')
    result_df = aggregated_scores.unstack(level='datapoint').reset_index()
    
    result_df['hour'] = result_df[STANDARD_DATE_COLUMN].dt.hour #type:ignore
    result_df['weekday_name'] = result_df[STANDARD_DATE_COLUMN].dt.day_name() #type:ignore
    result_df['is_weekend'] = result_df[STANDARD_DATE_COLUMN].dt.dayofweek.isin([5, 6]).astype(int) #type:ignore
    
    if 'asset_code' in result_df.columns:
        result_df[['site', 'equipment_id']] = result_df['asset_code'].str.split('_', n=1, expand=True)

    label_encoders = model_package.get('label_encoders', {})
    
    for cat_col in ['site', 'equipment_id', 'weekday_name']:
        if cat_col in result_df.columns and cat_col in label_encoders:
            le: LabelEncoder = label_encoders[cat_col]
            def transform_with_fallback(value):
                try:
                    return le.transform([value])[0] #type:ignore
                except ValueError:
                    return -1
                except TypeError:
                    return -1
            
            result_df[cat_col + '_encoded'] = result_df[cat_col].apply(transform_with_fallback) #type:ignore
        else:
             result_df[cat_col + '_encoded'] = -1 

    log.info(f"[{asset_code}/{feature_name}] [B2] Wide DF rows after pivot: {len(result_df)}")
    return result_df

def anamoly_detection_chart(request_data: AnomalyVizRequest) -> AnomalyVizResponse:
    chart_type = request_data.chart_type.lower()
    
    features = ALL_AVAILABLE_FEATURES 
    
    BASE_REPORT_DIR = Path(r"src/bms_ai/utils/ahu1_stored_anamoly")
    
    all_feature_data = {} 
    
    for feature in features:
        file_path = BASE_REPORT_DIR / f"{feature}.json" 
        
        if not file_path.exists():
            log.warning(f"Static anomaly file NOT FOUND for feature '{feature}'. Checked path: {file_path.resolve()}. Skipping.")
            continue
            
        try:
            with open(file_path, 'r') as f:
                raw_list = json.load(f)
                
            df = pd.DataFrame(raw_list)
            
            if df.empty or 'Anamoly_Flag' not in df.columns or 'data_received_on' not in df.columns:
                 log.warning(f"File {feature}.json is empty or missing required columns. Skipping.")
                 continue
                 
            df['Anamoly_Flag'] = pd.to_numeric(df['Anamoly_Flag'], errors='coerce').fillna(1).astype(int)
            df['date'] = pd.to_datetime(df['data_received_on'], errors='coerce')
            
            total_anomalies = df[df['Anamoly_Flag'] == -1].shape[0]
            
            if feature not in df.columns:
                 log.warning(f"Raw data column '{feature}' not found in {feature}.json. Skipping.")
                 continue
                 
            all_feature_data[feature] = {
                'total': total_anomalies,
                'df': df.copy() 
            }
            log.info(f"Successfully loaded and processed static data for feature: {feature} with {total_anomalies} anomalies.")
            
        except Exception as e:
            log.error(f"Failed to process anomaly file {file_path}: {e}")
            continue

    if not all_feature_data:
        raise HTTPException(status_code=404, detail="No anomaly data could be loaded from static files for visualization.")

    if chart_type == 'pie':
        feature_totals = {f: 0 for f in features} 
        
        for feature, data in all_feature_data.items():
            feature_totals[feature] = data['total']
            
        response_data = {
            'feature_anomalies': feature_totals,
            'total_anomalies_across_all_features': sum(feature_totals.values())
        }
        
        return AnomalyVizResponse(chart_type='pie', data=response_data)

    elif chart_type == 'line':
        
        integrated_data_by_feature = {}
        
        for feature, data in all_feature_data.items():
            df = data['df'].copy()
            
            if feature not in df.columns:
                 log.warning(f"Raw data column '{feature}' missing in DataFrame for report. Skipping.")
                 continue

            df_report = df[['date', 'Anamoly_Flag', feature]].copy()
            
            if not df_report['date'].isna().all():
                max_timestamp = df_report['date'].max()
                cutoff_timestamp = max_timestamp - pd.Timedelta(days=1)
                df_report = df_report[df_report['date'] >= cutoff_timestamp].copy()
                log.info(f"Feature '{feature}': Filtered to last 24h. Max: {max_timestamp}, Cutoff: {cutoff_timestamp}, Records: {len(df_report)}")
            
            df_report.rename(columns={
                'date': 'timestamp', 
                'Anamoly_Flag': 'Anamoly_Flag'
            }, inplace=True)
            
            df_report['timestamp'] = df_report['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f') + '+00:00'
            
            df_report['Anamoly_Flag'] = df_report['Anamoly_Flag'].astype(str)
            df_report[feature] = df_report[feature].astype(str)
            
            integrated_data_by_feature[feature] = df_report.to_dict('records')
            
        
        return AnomalyVizResponse(
            chart_type='line', 
            data={
                'historical_data': integrated_data_by_feature
            }
        )

    else:
        raise HTTPException(status_code=400, detail=f"Invalid chart_type: {chart_type}. Must be 'pie' or 'line'.")
    
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

def fan_health_analysis(request_data: PredictionRequest):
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

def damper_health_analysis(request_data: PredictionRequest):
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

def transform_to_dataframe(json_input_data: Dict[str, Any]) -> pd.DataFrame:
    GROUPING_COLS: List[str] = ['data_received_on', 'equipment_id', 'site']
    empty_df = pd.DataFrame(columns=GROUPING_COLS + ['Eg'])
    
    try:
        records: List[Dict[str, Any]] = (
            json_input_data.get("data", {}).get("queryResponse", []) or 
            json_input_data.get("queryResponse", []) or 
            json_input_data.get("data", [])
        )
        if not records:
            log.warning("No records found in the input JSON structure.")
            return empty_df

        df = pd.DataFrame(records)
        
        if 'system_type' not in df.columns:
            log.error("'system_type' column missing in raw data.")
            return empty_df
            
        mtr_df = df[df["system_type"] == "MtrEMU"].copy() 
        if mtr_df.empty:
            log.warning("No records found for 'MtrEMU' system type.")
            return empty_df

        mtr_df["data_received_on"] = pd.to_datetime(mtr_df["data_received_on"], errors='coerce')
        if mtr_df["data_received_on"].dt.tz is not None: #type: ignore
             mtr_df["data_received_on"] = mtr_df["data_received_on"].dt.tz_localize(None) #type: ignore
        
        mtr_df.dropna(subset=["data_received_on"], inplace=True)
        
        if 'monitoring_data' in mtr_df.columns:
            mtr_df['monitoring_data'] = mtr_df['monitoring_data'].astype(str).str.strip()
        else:
            log.error("'monitoring_data' column missing.")
            return empty_df

        pivoted_df = mtr_df.pivot_table(
            index=['data_received_on', 'equipment_id', 'site'],
            columns='datapoint',
            values='monitoring_data',
            aggfunc='first' #type:ignore
        ).reset_index()

        if isinstance(pivoted_df.columns, pd.MultiIndex):
             pivoted_df.columns = [
                col[-1] if col[-1] else str(col[0]) for col in pivoted_df.columns
             ]
        
        if 'Eg' not in pivoted_df.columns:
            log.error("The required 'Eg' datapoint is missing after pivoting.")
            return empty_df

        pivoted_df['Eg'] = pd.to_numeric(pivoted_df['Eg'], errors='coerce')
        pivoted_df.dropna(subset=['Eg'], inplace=True)
        
        if pivoted_df.empty:
            log.warning("All records were dropped after final 'Eg' cleanup.")
            return empty_df
        
        required_cols = ['data_received_on', 'equipment_id', 'site', 'Eg']
        missing_cols = [col for col in required_cols if col not in pivoted_df.columns]
        
        if missing_cols:
            log.error(f"Final DataFrame is missing required columns: {missing_cols}")
            return empty_df

        log.info(f"Data processing successful. {len(pivoted_df)} rows ready for aggregation.")
        return pivoted_df[required_cols].copy()
        
    except Exception as e:
        log.error(f"Critical error in transform_to_dataframe: {e}", exc_info=True)
        raise RuntimeError(f"Data transformation failed: {e}")

def calculate_aggregate_emissions(df_input: pd.DataFrame) -> pd.DataFrame:
    if df_input.empty:
        return pd.DataFrame(columns=['equipment_id', 'site', 'min_Eg', 'max_Eg', 'count', 
                                     'Energy_Range_kWh', 'carbon_emission_kg'])
    
    try:
        df_input['Eg'] = df_input['Eg'].astype(np.float64) 
        
        emission_summary = df_input.groupby(['equipment_id', 'site']).agg(
            min_Eg=('Eg', 'min'),
            max_Eg=('Eg', 'max'),
            count=('Eg', 'size')
        ).reset_index()

        emission_summary['Energy_Range_kWh'] = emission_summary['max_Eg'] - emission_summary['min_Eg']
        
        emission_summary = emission_summary[emission_summary['Energy_Range_kWh'] > 0].copy()
        
        emission_summary['carbon_emission_kg'] = emission_summary['Energy_Range_kWh'] * EMISSION_FACTOR
        
        return emission_summary

    except Exception as e:
        log.error(f"Error during emission calculation: {e}", exc_info=True)
        raise RuntimeError(f"Aggregation failed: {e}")

def get_emission_report(json_input_data: Dict[str, Any], equipment_id: Optional[str], zone: Optional[str]) -> Dict[str, Any]:
    try:
        df_processed = transform_to_dataframe(json_input_data)
        
        if df_processed.empty:
            log.info("Returning empty report due to no processed data.")
            return {
                "Request_Parameters": {"equipment_id": equipment_id, "zone": zone},
                "Target_Emission_Report": {"Target_ID": "All", "Target_Type": "Global", "carbon_emission_kg": 0, "energy_consumed_kwh": 0, "breakdown_by_equipment_and_zone": []},
                "Total_Site_Emission_Report": {"Total_CO2_Emission_kg": 0, "Total_Energy_Consumed_kWh": 0, "Emission_Factor_kgCO2_per_kWh": EMISSION_FACTOR},
                "Processing_Status": "No Data Processed"
            }

        df_emissions = calculate_aggregate_emissions(df_processed)

        total_emission_kg = df_emissions['carbon_emission_kg'].sum()
        total_energy_kwh = df_emissions['Energy_Range_kWh'].sum()
        
        total_report = {
            "Total_CO2_Emission_kg": round(total_emission_kg, 2),
            "Total_Energy_Consumed_kWh": round(total_energy_kwh, 2),
            "Emission_Factor_kgCO2_per_kWh": EMISSION_FACTOR
        }
        
        df_filtered = df_emissions.copy()

        if equipment_id:
            df_filtered = df_filtered[df_filtered['equipment_id'] == equipment_id]
            
        if zone:
            df_filtered = df_filtered[df_filtered['site'] == zone] 

        specific_emission_kg = df_filtered['carbon_emission_kg'].sum()
        specific_energy_kwh = df_filtered['Energy_Range_kWh'].sum()

        target_type = "Global"
        target_id = "All"
        if equipment_id and zone:
            target_type = "Equipment_ID_and_Zone"
            target_id = f"{equipment_id} @ {zone}"
        elif equipment_id:
            target_type = "Equipment_ID"
            target_id = equipment_id
        elif zone:
            target_type = "Zone"
            target_id = zone

        df_breakdown = df_filtered.copy()
        df_breakdown['Energy_Range_kWh'] = df_breakdown['Energy_Range_kWh'].round(2)
        df_breakdown['carbon_emission_kg'] = df_breakdown['carbon_emission_kg'].round(2)
        
        specific_report = {
            "Target_ID": target_id,
            "Target_Type": target_type,
            "carbon_emission_kg": round(specific_emission_kg, 2),
            "energy_consumed_kwh": round(specific_energy_kwh, 2),
            "breakdown_by_equipment_and_zone": df_breakdown[['equipment_id', 'site', 'Energy_Range_kWh', 'carbon_emission_kg']].to_dict('records')
        }

        final_report = {
            "Request_Parameters": {"equipment_id": equipment_id, "zone": zone},
            "Target_Emission_Report": specific_report,
            "Total_Site_Emission_Report": total_report,
            "Processing_Status": "Success"
        }
        
        return final_report
    
    except RuntimeError as e:
        log.error(f"API processing failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        log.critical(f"Unexpected critical error in get_emission_report: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during report generation."
        )

def get_emission_report_from_json(equipment_id: Optional[str] = None, zone: Optional[str] = None) -> Dict[str, Any]:
    try:
        df_emissions = pd.read_json('src/bms_ai/utils/carbon_emission/carbon_emission.json', orient='index')
        
        cleaned_index = (
            df_emissions.index.to_series()
            .str.strip("()")  
            .str.replace("'", "", regex=False) 
            .str.replace(" ", "", regex=False)
        )
        
        df_emissions[['equipment_id', 'site']] = cleaned_index.str.split(',', expand=True)
        df_emissions = df_emissions.reset_index(drop=True)
        df_emissions.rename(columns={'diff': 'Energy_Range_kWh'}, inplace=True)
        
        df_emissions['carbon_emission_kg'] = df_emissions['Energy_Range_kWh'] * EMISSION_FACTOR
        
        df_filtered = df_emissions.copy()

        if equipment_id:
            df_filtered = df_filtered[df_filtered['equipment_id'] == equipment_id]
            
        if zone:
            df_filtered = df_filtered[df_filtered['site'] == zone]
            
        if df_filtered.empty:
            return {
                "carbon_emission_kg": 0,
                "energy_consumed_kwh": 0,
                "breakdown_by_equipment_and_zone": []
            }

        specific_emission_kg = df_filtered['carbon_emission_kg'].sum()
        specific_energy_kwh = df_filtered['Energy_Range_kWh'].sum()
        
        df_breakdown = df_filtered.copy()
        
        df_breakdown.rename(columns={'Energy_Range_kWh': 'energy_range_kwh'}, inplace=True)
        
        final_minimal_report = {
            "carbon_emission_kg": round(specific_emission_kg, 2),
            "energy_consumed_kwh": round(specific_energy_kwh, 2),
            "breakdown_by_equipment_and_zone": df_breakdown[['equipment_id', 'site', 'energy_range_kwh', 'carbon_emission_kg']].round(2).to_dict('records')
        }
        
        return final_minimal_report
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"Configuration Error: The static data file was not"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Critical processing error: {e}"
        )

try:
    STATIC_PEAK_DATA = pd.read_json('src/bms_ai/utils/peak_demand/peak_demand_results.json', orient='index')
except Exception as e:
    print(f"Error loading data: {e}. STATIC_PEAK_DATA initialized as empty DataFrame.")
    STATIC_PEAK_DATA = pd.DataFrame()

@router.get("/get_peak_demand", response_model=Dict[str, Dict[str, Any]])
def get_all_peak_demand() -> Dict[str, Dict[str, Any]]:
    if STATIC_PEAK_DATA.empty:
        raise HTTPException(
            status_code=500, 
            detail="Peak demand data is not loaded or is empty."
        )
    return STATIC_PEAK_DATA.to_dict(orient='index') #type: ignore

@router.post('/carbon_emission_evaluation', response_model=EmissionResponse)
def carbon_emission_evaluation(
    request_data: EmissionRequest
) -> EmissionResponse:
    
    start = time.time()
    
    input_data = request_data.data
    equipment_id = request_data.equipment_id
    zone = request_data.zone
    
    log.info(f"Request initiated for Equipment: {equipment_id}, Zone: {zone}")
    
    try:
        emission_report_dict = get_emission_report(
            input_data, 
            equipment_id, 
            zone
        )
    
    except HTTPException:
        raise HTTPException(
            status_code=500,
            detail="Error in generating Emission Report."
        )
    
    except Exception as e:
        log.error(f"Unhandled error in API endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="A critical error occurred."
        )

    end = time.time()
    log.info(f"Evaluation completed in {end - start:.2f} seconds.")
    
    return EmissionResponse(data=emission_report_dict)

@router.post('/carbon_emission_evaluation_static', response_model=StaticEmissionResponse)
def carbon_emission_evaluation_static(
    request_data: StaticEmissionRequest
) -> Dict[str, Any]:
    
    equipment_id = request_data.equipment_id
    zone = request_data.zone
    
    log.info(f"Static Report Request initiated for Equipment: {equipment_id}, Zone: {zone}")
    
    emission_report_dict = get_emission_report_from_json(
        equipment_id, 
        zone
    )
    
    log.info("Static Evaluation completed.")
    
    return emission_report_dict

@router.post('/damper_health_prediction', response_model=PredictionResponse)
def damper_health_prediction(
    request_data: PredictionRequest
):
    start = time.time()
    log.info(f"input Data: {request_data.dict()}") 
    result = damper_health_analysis(request_data)
    end = time.time()
    log.info(f"VOX AHU1 Damper End of Life Prediction completed in {end - start:.2f} seconds") 
    return result

@router.post('/fan_speed_health_prediction', response_model=PredictionResponse)
def fan_speed_health_prediction(
    request_data: PredictionRequest
):
    start = time.time()
    log.info(f"input Data: {request_data.dict()}") 
    result = fan_health_analysis(request_data)
    end = time.time()
    log.info(f"VOX AHU1 Fan Speed End of Life Prediction completed in {end - start:.2f} seconds") 
    return result


@router.post('/anomaly_detection_all_ahu')
def anomaly_detection_all_ahu() -> Dict[str, Any]:
    start_time = time.time()
    all_asset_results: Dict[str, Any] = {}
    
    all_data_records = fetch_all_ahu_data(DEFAULT_BUILDING_ID)
    
    if not all_data_records:
        return {"data": {"historical_data": {}}, "message": "No data found for AHU systems."}
        
    grouped_data_by_asset: Dict[str, List[Dict]] = defaultdict(list)
    for record in all_data_records:
        site = record.get('site')
        equipment_name = record.get('equipment_name')
        
        if site and equipment_name:
            asset_code_key = f"{site}_{equipment_name}" 
            record['asset_code'] = asset_code_key 
            grouped_data_by_asset[asset_code_key].append(record)

    log.info(f"[A2] Total grouped assets: {len(grouped_data_by_asset)}")
    
    for asset_code_key, asset_records in grouped_data_by_asset.items():
        
        try:
            site, equipment_id = asset_code_key.split('_', 1) 
        except ValueError:
            site = "Unknown"; equipment_id = asset_code_key

        asset_historical_data: Dict[str, List[Dict[str, Any]]] = {}
        
        for feature in ALL_AVAILABLE_FEATURES:
            
            datapoint_name = None
            if feature in FEATURE_FALLBACKS:
                for fallback_name in FEATURE_FALLBACKS[feature]:
                    if any(r.get('datapoint') == fallback_name for r in asset_records):
                        datapoint_name = fallback_name
                        break
            if datapoint_name is None:
                datapoint_name = feature
            
            feature_raw_data = [r for r in asset_records if r.get('datapoint') == datapoint_name]
            log.debug(f"Records for {asset_code_key}/{feature} (using data col: {datapoint_name}): {len(feature_raw_data)}")
            
            if not feature_raw_data:
                continue

            try:
                feature_results = anomaly_detection(feature_raw_data, asset_code_key, feature)

                if feature_results:
                    asset_historical_data[datapoint_name] = feature_results
                            
            except HTTPException:
                raise
            except Exception as e:
                log.error(f"Prediction logic failed for {asset_code_key}/{feature}: {e}")
                continue
        
        if asset_historical_data:
            all_asset_results[asset_code_key] = {
                "data": {
                    "historical_data": asset_historical_data
                },
                "site": site,
                "equipment_id": equipment_id,
                "system_type": FIXED_SYSTEM_TYPE
            }
    
    final_output = {
        "data": {
            "all_anomalies_by_asset": all_asset_results
        },
        "total_assets_processed": len(grouped_data_by_asset),
        "anomalous_assets_count": len(all_asset_results)
    }
    
    log.info(f"Anomaly detection completed in {time.time() - start_time:.2f} seconds.")
    return final_output

@router.post('/anomaly_detection_single_asset')
def anomaly_detection_single_asset(request: AnamolyPredictionRequest) -> Dict[str, Any]:
    start_time = time.time()
    asset_code = f"{request.site}_{request.equipment_id}"
    feature = request.feature
    
    datapoint_name = feature
    if feature in FEATURE_FALLBACKS:
        all_data_records = fetch_all_ahu_data(request.building_id or DEFAULT_BUILDING_ID)
        
        found_datapoint = next((
            r.get('datapoint')
            for r in all_data_records
            if r.get('site') == request.site
            and r.get('equipment_name') == request.equipment_id
            and r.get('datapoint') in FEATURE_FALLBACKS[feature]
        ), feature)

        datapoint_name = found_datapoint

    all_data_records = fetch_all_ahu_data(request.building_id or DEFAULT_BUILDING_ID)
    
    feature_raw_data = [
        r for r in all_data_records 
        if r.get('site') == request.site 
        and r.get('equipment_name') == request.equipment_id
        and r.get('datapoint') == datapoint_name 
    ]
    
    if not feature_raw_data:
        return {"data": {"historical_data": {datapoint_name: []}}, "message": f"No data found for {asset_code}/{datapoint_name}."}

    try:
        feature_results = anomaly_detection(feature_raw_data, asset_code, feature)

        log.info(f"Single asset prediction for {asset_code}/{feature} completed in {time.time() - start_time:.2f} seconds.")

        return {
            "data": {
                "historical_data": {datapoint_name: feature_results}
            },
            "site": request.site,
            "equipment_id": request.equipment_id,
            "system_type": request.system_type
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Single asset prediction failed for {asset_code}/{feature}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed due to internal error: {e}")

@router.post('/anomaly_chart_data', response_model=AnomalyVizResponse)
def anomaly_chart_data(
    request_data: AnomalyVizRequest
):
    """
    Runs anomaly detection on multiple assets/features and aggregates the results 
    for visualization (Pie chart for totals, Line chart for time series).
    """
    start = time.time()
    log.info(f"Visualization data request initiated for chart_type: {request_data.chart_type}") 
    
    result = anamoly_detection_chart(request_data)
    end = time.time()
    log.info(f"Visualization data generation completed in {end - start:.2f} seconds.") 
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
    # total_combinations_tested: int
    # optimization_method: str
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
    #data_path: str = Field("C:\\Users\\debas\\OneDrive\\Desktop\\actual_data.csv", description="Path to training data CSV file")
    data_path: str = Field("D:\\My Donwloads\\bacnet_latest_data\\bacnet_latest_data.csv", description="Path to training data CSV file")
    equipment_id: str = Field("Ahu1", description="Equipment ID to filter")
    target_variable: str = Field(..., description="Target variable to optimize (must be numeric)")
    test_size: float = Field(0.2, description="Fraction of data for testing")
    search_method: str = Field("random", description="Hyperparameter search method: 'random' or 'grid'")
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    n_iter: int = Field(20, description="Number of iterations for RandomizedSearchCV")
    setpoints: Optional[List[str]] = Field(None, description="Optional list of setpoints to include in selection (defaults to primary setpoints)")

class GenericTrainRequestV2(BaseModel):
    #data_path: str = Field("D:\\My Donwloads\\bacnet_latest_data\\bacnet_latest_data.csv", description="Path to training data CSV file")
    #data : List[dict] = Field(..., description="Input data in JSON format")
    ticket: str = Field(..., description="Ticket ID for tracking the training job")
    account_id: str = Field(..., description="Account ID associated with the training job")
    software_id: str = Field(..., description="Software ID for versioning")
    building_id: str = Field(..., description="Building ID where the equipment is located")
    site: str = Field(..., description="Site name where the equipment is located")
    system_type: str = Field(..., description="System type (e.g., 'AHU', 'RTU')")
    equipment_id: str = Field("Ahu1", description="Equipment ID to filter")
    target_variable_tag: List[List[str]] = Field(..., description="Target variable to optimize (must be numeric)")
    test_size: float = Field(0.2, description="Fraction of data for testing")
    search_method: str = Field("random", description="Hyperparameter search method: 'random' or 'grid'")
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    n_iter: int = Field(20, description="Number of iterations for RandomizedSearchCV")
    setpoints: Optional[List[List[str]]] = Field(None, description="Optional list of setpoints to include in selection (defaults to primary setpoints)")    


class GenericTrainResponse(BaseModel):
    status: str
    best_model_name: str
    selected_features: List[str]
    setpoints: List[str]
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
            n_iter=request_data.n_iter,
            setpoints=request_data.setpoints
        )
        
        end = time.time()
        log.info(f"Generic model training completed in {end - start:.2f} seconds")
        log.info(f"Selected {len(result.get('selected_features', []))} features for {request_data.target_variable}")
        if result.get('setpoints'):
            log.info(f"Tracked {len(result['setpoints'])} setpoints: {result['setpoints']}")
        
        return GenericTrainResponse(**result)
        
    except Exception as e:
        log.error(f"Generic training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/generic_trainV2', response_model=GenericTrainResponse)
def generic_train_endpointV2(request_data: GenericTrainRequestV2):
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
    matches_tags = []
    matches_tags = matches_tags + request_data.target_variable_tag
    print(f"Target variable tags: {matches_tags}")
    if request_data.setpoints != None:
        matches_tags = matches_tags + request_data.setpoints
        print(f"Setpoint tags: {matches_tags}")

    results = fetch_and_find_data_points(
            building_id=request_data.building_id,
            floor_id=None,
            equipment_id=request_data.equipment_id,
            #search_tag_groups= request_data.target_variable_tag,
            search_tag_groups= matches_tags,
            ticket=request_data.ticket,
            software_id=request_data.software_id,
            account_id=request_data.account_id,
            system_type=request_data.system_type,
            env="prod"
        )
    
    target_variable = ""
    setpoints = []

    print(f"Fetched data points: {results}")


    if len(results) > 0:
        #target_variable = results[0].get("dataPointName")
        try:
            target_variable = results[0]["dataPointName"]
            print(f"Using target variable: {target_variable}")
            if len(results) > 1:
                print("Additional fetched data points (likely setpoints):")
                for res in results[1:]:
                    print(f"- {res.get('dataPointName')}")
                    setpoints.append(res.get('dataPointName'))
        except KeyError:
            print("Data point name not found in results.")
            print(f"Results content: {results[0]}")
            raise HTTPException(status_code=500, detail="Data point name not found in results.")
    else:
        raise HTTPException(status_code=404, detail="No data points found for the specified target variable tags.")
      
    data = fetch_all_ahu_dataV2(
            building_id=request_data.building_id,
            site=request_data.site,
            system_type=request_data.system_type,
            equipment_id=request_data.equipment_id
        )
    
    log.debug(f"Data fetched for training: {data}")
    

    
    start = time.time()
    log.info(f"Generic training request: {request_data.dict()}")
    print(f"Generic training request: {request_data.dict()}")
    log.info(f"Using target variable: {target_variable}")
    log.info(f"Using setpoints: {setpoints}")
    
    try:
        result = train_generic(
            #data_path=request_data.data_path,
            #data=request_data.data,
            data=data,
            equipment_id=request_data.equipment_id,
            target_column=target_variable,  
            test_size=request_data.test_size,
            search_method=request_data.search_method,
            cv_folds=request_data.cv_folds,
            n_iter=request_data.n_iter,
            #setpoints=request_data.setpoints
            setpoints=setpoints
        )
        
        end = time.time()
        log.info(f"Generic model training completed in {end - start:.2f} seconds")
        log.info(f"Selected {len(result.get('selected_features', []))} features for {target_variable}")
        if result.get('setpoints'):
            log.info(f"Tracked {len(result['setpoints'])} setpoints: {result['setpoints']}")
        
        return GenericTrainResponse(**result)
        
    except Exception as e:
        log.error(f"Generic training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class GenericOptimizeRequest(BaseModel):
    current_conditions: Dict[str, Any] = Field(..., description="Current system state")
    equipment_id: str = Field("Ahu1", description="Equipment ID (must match training)")
    target_variable: str = Field(..., description="Target variable to optimize (must match training)")
    search_space: Optional[Dict[str, List[float]]] = Field(None, description="Setpoint ranges to search")
    optimization_method: Optional[str] = Field("random", description="Optimization method: 'grid' or 'random'")
    n_iterations: Optional[int] = Field(1000, description="Number of iterations for random search (ignored for grid)")
    direction: Optional[str] = Field("minimize", description="Optimization direction: 'minimize' or 'maximize'")

class GenericOptimizeRequestV2(BaseModel):
    current_conditions: Dict[str, Any] = Field(..., description="Current system state")
    ticket: str = Field(..., description="Ticket ID for tracking the training job")
    account_id: str = Field(..., description="Account ID associated with the training job")
    software_id: str = Field(..., description="Software ID for versioning")
    building_id: Optional[str] = Field(None, description="Building ID (optional)")
    site: str = Field(..., description="Site name where the equipment is located")
    system_type: str = Field(..., description="System type (e.g., 'AHU', 'RTU')")
    equipment_id: str = Field("Ahu1", description="Equipment ID (must match training)")
    target_variable_tag: List[List[str]] = Field(..., description="Target haystacks to optimize (must match training)")
    search_space: Optional[Dict[str, List[float]]] = Field(None, description="Setpoint ranges to search")
    optimization_method: Optional[str] = Field("random", description="Optimization method: 'grid' or 'random'")
    n_iterations: Optional[int] = Field(1000, description="Number of iterations for random search (ignored for grid)")
    direction: Optional[str] = Field("minimize", description="Optimization direction: 'minimize' or 'maximize'")


class GenericOptimizeResponse(BaseModel):
    best_setpoints: Dict[str, float]
    best_target_value: float
    target_variable: str
    # optimization_direction: str
    # total_combinations_tested: int
    # optimization_method: str
    optimization_time_seconds: float


@router.post('/generic_optimize', response_model=GenericOptimizeResponse)
def generic_optimize_endpoint(request_data: GenericOptimizeRequest):
    """
    Optimize AHU setpoints to minimize or maximize any specified target variable.
    
    Args:
        current_conditions: Current system state with all required features
        target_variable: Target variable to optimize
        search_space: Optional setpoint ranges (defaults provided if not specified)
        optimization_method: 'grid' or 'random'
        n_iterations: Number of iterations for random search
        direction: 'minimize' or 'maximize' the target variable
        
    Returns:
        Best setpoints and optimized target value
    """
    start = time.time()
    log.info(f"Generic optimization request: equipment={request_data.equipment_id}, target={request_data.target_variable}, method={request_data.optimization_method}, direction={request_data.direction}")
    
    try:
        result = optimize_generic(
            current_conditions=request_data.current_conditions,
            equipment_id=request_data.equipment_id,
            target_column=request_data.target_variable,  
            search_space=request_data.search_space,
            optimization_method=request_data.optimization_method or "random",
            n_iterations=request_data.n_iterations or 1000,
            direction=request_data.direction or "minimize"
        )
        
        end = time.time()
        log.info(f"Generic optimization completed in {end - start:.2f} seconds")
        log.info(f"Best setpoints: {result['best_setpoints']}, Best {request_data.target_variable}: {result['best_target_value']:.4f}")
        
        return GenericOptimizeResponse(**result)
        
    except Exception as e:
        log.error(f"Generic optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/generic_optimizeV2', response_model=GenericOptimizeResponse)
def generic_optimize_endpoint(request_data: GenericOptimizeRequestV2):
    """
    Optimize AHU setpoints to minimize or maximize any specified target variable.
    
    Args:
        current_conditions: Current system state with all required features
        target_variable: Target variable to optimize
        search_space: Optional setpoint ranges (defaults provided if not specified)
        optimization_method: 'grid' or 'random'
        n_iterations: Number of iterations for random search
        direction: 'minimize' or 'maximize' the target variable
    Returns:
        Best setpoints and optimized target value
    """

    results = fetch_and_find_data_points( building_id=request_data.building_id,
            equipment_id=request_data.equipment_id,
            floor_id=None,
            search_tag_groups=request_data.target_variable_tag,
            ticket=request_data.ticket,
            software_id=request_data.software_id,
            account_id=request_data.account_id,
            system_type=request_data.system_type,
            env="prod")
    
    target_variable = ""
    print(f"Fetched data points: {results}")
    if len(results) > 0:
        try:
            target_variable = results[0]["dataPointName"]
            print(f"Using target variable: {target_variable}")
        except KeyError:
            print("Data point name not found in results.")
            print(f"Results content: {results[0]}")
            raise HTTPException(status_code=500, detail="Data point name not found in results.")
    start = time.time()
    log.info(f"Generic optimization request: equipment={request_data.equipment_id}, target={target_variable}, method={request_data.optimization_method}, direction={request_data.direction}")
    
    try:
        result = optimize_generic(
            current_conditions=request_data.current_conditions,
            equipment_id=request_data.equipment_id,
            target_column=target_variable,  
            search_space=request_data.search_space,
            optimization_method=request_data.optimization_method or "random",
            n_iterations=request_data.n_iterations or 1000,
            direction=request_data.direction or "minimize"
        )
    
        end = time.time()
        log.info(f"Generic optimization completed in {end - start:.2f} seconds")
        log.info(f"Best setpoints: {result['best_setpoints']}, Best {target_variable}: {result['best_target_value']:.4f}")

        save_data_to_cassandraV2(
            data_chunk=[result],
            building_id=request_data.building_id,
            metadata={
                "site": request_data.site,
                "equipment_id": request_data.equipment_id,
                "system_type": request_data.system_type
            }
        )
        
        return GenericOptimizeResponse(**result)
        
    except Exception as e:
        log.error(f"Generic optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class OptimizationResultsResponse(BaseModel):
    optimized_percentage: float = Field(..., description="Average difference between actual and predicted values as percentage")
    results: List[Dict[str, Any]] = Field(..., description="Optimization results with timestamps, actual values, and predicted values")


@router.get("/optimization_results", response_model=OptimizationResultsResponse)
async def get_optimization_results():
    """
    Returns the optimization results with optimized percentage.
    
    Returns:
        JSON containing optimization results with timestamps, actual values,
        predicted values, setpoint comparisons, and optimized percentage.
    """
    file_path = Path.cwd() / "test_optimization_results.json"
    
    if not file_path.exists():
        log.error(f"Optimization results file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Optimization results file not found")
    
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        differences = [r['difference_actual_and_pred'] for r in results if r.get('difference_actual_and_pred') is not None]
        avg_difference = sum(differences) / len(differences) if differences else 0
        
        log.info(f"Serving optimization results from {file_path} with avg difference: {avg_difference:.2f}")
        
        return OptimizationResultsResponse(
            optimized_percentage=round(avg_difference, 2)*10,
            results=results
        )
    except Exception as e:
        log.error(f"Error reading optimization results: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading optimization results: {str(e)}")
    

class AdjustmentHistoryResponse(BaseModel):
    building_id: str = Field(..., description="Building ID")
    site: str = Field(..., description="Site name")
    system_type: str = Field(..., description="System type (e.g., 'AHU', 'RTU')")
    equipment_id: str = Field(..., description="Equipment ID for which adjustments are fetched")
    adjustments: List[Dict[str, Any]] = Field(..., description="List of adjustment history records")

@router.get("/adjustment_history", response_model=AdjustmentHistoryResponse)
async def get_adjustment_history(
    building_id: str,
    site: str,
    system_type :str,
    equipment_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: Optional[int] = 100  
):
    """
    Fetches adjustment history for a given equipment within an optional date range.
    
    Args:
        equipment_id: Equipment ID to filter adjustments
        start_date: Optional start date for filtering (inclusive)
        end_date: Optional end date for filtering (inclusive)
        limit: Maximum number of records to return (default 100)
    Returns:

        JSON containing adjustment history records
    """
    try:
        adjustments = fetch_adjustment_hisoryData(
            equipment_id=equipment_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        log.info(f"Fetched {len(adjustments)} adjustment records for equipment_id={equipment_id}")
        
        return AdjustmentHistoryResponse(
            equipment_id=equipment_id,
            adjustments=adjustments
        )
        
    except Exception as e:
        log.error(f"Error fetching adjustment history: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching adjustment history: {str(e)}")
