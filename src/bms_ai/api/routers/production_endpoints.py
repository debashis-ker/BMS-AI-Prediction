from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.bms_ai.logger_config import setup_logger
from pathlib import Path
import json
from src.bms_ai.api.routers.routers import anamoly
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
import os
from dotenv import load_dotenv

load_dotenv()

log = setup_logger(__name__)

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/prod", tags=["Prescriptive Optimization"])
router.include_router(anamoly.router)

EMISSION_FACTOR = 0.4041

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

def fetch_all_ahu_dataV2(
        building_id: str,
        url: str = "",
        site: str = "",
        system_type: str = "",
        equipment_id: Optional[str] = None
    ) -> List[Dict]:
    """Fetches ALL historical data for AHUs in a single API call."""
    cleaned_id = building_id.replace("-", "").lower()
    location_table_name = f"datapoint_live_monitoring_{cleaned_id}"
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
    
def anamoly_detection_chart(request_data: AnomalyVizRequest) -> AnomalyVizResponse:
    chart_type = request_data.chart_type.lower()
    
    features = ['TSu', 'Co2RA', 'FbFAD', 'FbVFD', 'HuAvg1'] 
    
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
            
            if df.empty or 'Anomaly_Flag' not in df.columns or 'data_received_on' not in df.columns:
                 log.warning(f"File {feature}.json is empty or missing required columns. Skipping.")
                 continue
                 
            df['Anamoly_Flag'] = pd.to_numeric(df['Anomaly_Flag'], errors='coerce').fillna(1).astype(int)
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
        
        emission_summary['carbon_emission_kg'] = emission_summary['Energy_Range_kWh'] * EMISSION_FACTOR #type:ignore
        
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
            
        if df_filtered.empty:
            return {
                "carbon_emission_kg": 0,
                "energy_consumed_kwh": 0,
                "breakdown_by_equipment_and_zone": []
            }

        mdb_filtered = df_filtered[(df_filtered['site'] == "OS01") & (df_filtered['equipment_id'] == "EMU03")]
        smdb_filtered = df_filtered[(df_filtered['site'] == "OS01") & (df_filtered['equipment_id'] == "EMU06")]
        mdb_carbon_emission_kg = mdb_filtered["carbon_emission_kg"].sum()
        mdb_energy_kwh = mdb_filtered["Energy_Range_kWh"].sum()
        smdb_carbon_emission_kg = smdb_filtered["carbon_emission_kg"].sum()
        smdb_energy_kwh = smdb_filtered["Energy_Range_kWh"].sum()

        df_breakdown = df_filtered.copy()
        
        df_breakdown.rename(columns={'Energy_Range_kWh': 'energy_range_kwh'}, inplace=True)
        
        final_minimal_report = {
            "carbon_emission_kg": round((mdb_carbon_emission_kg + smdb_carbon_emission_kg),2),
            "energy_consumed_kwh": round((mdb_energy_kwh + smdb_energy_kwh),2),
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
            data_path=request_data.data_path, #type:ignore
            equipment_id=request_data.equipment_id,
            target_column=request_data.target_variable,  
            test_size=request_data.test_size,
            search_method=request_data.search_method,
            cv_folds=request_data.cv_folds,
            n_iter=request_data.n_iter,
            setpoints=request_data.setpoints
        ) #type:ignore
        
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
def generic_optimize_endpoint(request_data: GenericOptimizeRequest): #type:ignore
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

    results = fetch_and_find_data_points( building_id=request_data.building_id, #type:ignore
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
            building_id=request_data.building_id, #type:ignore
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
