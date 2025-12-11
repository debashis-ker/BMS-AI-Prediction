from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple, Iterator
from collections import defaultdict
from src.bms_ai.logger_config import setup_logger # Assuming logger setup is correct
import warnings
import pandas as pd
import time
import requests
import logging
from dateutil import parser
from dateutil import tz
from sklearn.preprocessing import LabelEncoder
from src.bms_ai.utils.cassandra_utils import fetch_all_ahu_data # Assuming this fetches raw data
import joblib
from cassandra.cluster import Cluster

log = setup_logger(__name__)

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/anomalies",tags=["Anomalies"])

# --- API CONSTANTS ---
FIXED_SYSTEM_TYPE = "AHU"
DEFAULT_BUILDING_ID = "36c27828-d0b4-4f1e-8a94-d962d342e7c2"
API_INTERNAL_URL = "http://127.0.0.1:8000/prod/anomalies/anomaly_detection_all_ahu"

# --- CASSANDRA CONSTANTS ---
CASSANDRA_HOST = ['127.0.0.1']
CASSANDRA_PORT = 9041
KEYSPACE_NAME = 'anamoly'
CHUNK_SIZE = 100
HISTORY_TABLE_SUFFIX = "historical_data"
ANAMOLY_TABLE_SUFFIX = "anamoly_data"
ANAMOLY_TTL_SECONDS = 30 * 24 * 60 * 60
CASSANDRA_DEFAULT_TZ = tz.gettz('UTC')
BUILDING_IDS = ["36c27828-d0b4-4f1e-8a94-d962d342e7c2"] # Used for ingestion
DEFAULT_METADATA = {"equipment_id": "", "system_type": "AHU", "site": ""}

# --- MODEL CONSTANTS ---
ALL_AVAILABLE_FEATURES = ['TSu', 'Co2RA', 'FbFAD', 'FbVFD', 'HuAvg1'] # Base features only
STANDARD_DATE_COLUMN = "data_received_on"
CO2RA_CEILING_THRESHOLD = 850.0
FBVFD_NORMAL_MAX = 1.0 

FEATURE_FALLBACKS = {
    'TSu': ['TempSu'],
    'Co2RA': ['Co2Avg'],
    'HuAvg1': ['HuR1', 'HuRt']
}

# Used to retrieve the LOGICAL feature name from a PHYSICAL/FALLBACK name (e.g., 'HuR1' -> 'HuAvg1')
REVERSE_FEATURE_MAP = {}
for master_feat, fallbacks in FEATURE_FALLBACKS.items():
    for fb in fallbacks:
        REVERSE_FEATURE_MAP[fb] = master_feat
REVERSE_FEATURE_MAP['Co2Avg'] = 'Co2RA' # Handle specific Co2 fallback naming


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
all_model_keys_to_load.add('Co2Avg') # Ensure Co2Avg is included if used as a model file

for model_key_in_file in all_model_keys_to_load:
    model_file = f"artifacts/generic_anamoly_models/{model_key_in_file}_model.joblib"
    master_key = CONSOLIDATION_MAP.get(model_key_in_file) 
    
    if master_key is None:
        continue

    try:
        # Assuming model loading is fixed
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

def anomaly_detection(raw_data: List[Dict[str, Any]], asset_code: str, feature: str) -> List[Dict[str, Any]]:
    if feature not in anamoly_model: #type:ignore
        raise HTTPException(status_code=503, detail=f"Anomaly Detection model for {feature} is unavailable.")

    try:
        model_package = anamoly_model[feature].get(asset_code) #type:ignore
        if model_package is None: raise KeyError("Model not found for specific asset.")
        model_features = model_package.get('feature_cols', [feature]) 
    except KeyError as e:
        log.warning(f"[{asset_code}/{feature}] Model package not found for key: {e}. Returning empty list.")
        return [] 
    
    try:
        df_wide = anamoly_data_pipeline(raw_data, asset_code, model_package, feature)
        if df_wide.empty: return []
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

    try:
        X_scaled = model_package['scaler'].transform(X_for_model)
        predictions = model_package['model'].predict(X_scaled)
    except Exception as e:
        log.error(f"[{asset_code}/{feature}] Prediction scaling/run failed: {e}")
        return []

    X_df['Anomaly_Flag'] = predictions
    
    # Post-processing Overrides (Co2RA and FbVFD safety rules)
    if feature == 'Co2RA' and data_column in X_df.columns:
        co2_values_numeric = pd.to_numeric(X_df[data_column], errors='coerce')
        condition_high = co2_values_numeric > CO2RA_CEILING_THRESHOLD
        condition_low = co2_values_numeric < CO2RA_CEILING_THRESHOLD
        X_df.loc[condition_high, 'Anomaly_Flag'] = -1
        X_df.loc[condition_low, 'Anomaly_Flag'] = 1 

    if feature == 'FbVFD' and data_column in X_df.columns:
        fbvfd_values_numeric = pd.to_numeric(X_df[data_column], errors='coerce')
        normal_fbvfd_condition = (fbvfd_values_numeric >= 0) & (fbvfd_values_numeric <= FBVFD_NORMAL_MAX)
        X_df.loc[normal_fbvfd_condition, 'Anomaly_Flag'] = 1
    
    final_df = X_df.copy() 
    
    if final_df.empty: return []
    
    final_df['data_received_on_str'] = final_df[STANDARD_DATE_COLUMN].dt.strftime('%Y-%m-%d %H:%M:%S.%f') #type:ignore

    report_list = []
    for _, row in final_df.iterrows():
        record = {
            "data_received_on": row['data_received_on_str'],
            "Anomaly_Flag": str(int(row['Anomaly_Flag']))
        }
        data_column = model_features[0] 
        record[data_column] = str(float(row[data_column])) 
        report_list.append(record)
    return report_list

def anamoly_data_pipeline(records: List[Dict[str, Any]], asset_code: str, model_package: Dict[str, Any], feature_name: str) -> pd.DataFrame:
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    
    if 'monitoring_data' in df.columns:
        mapping = {'inactive': 0.0, 'active': 1.0}
        df['monitoring_data'] = df['monitoring_data'].replace(mapping, regex=False)
        df['monitoring_data'] = pd.to_numeric(df['monitoring_data'], errors='coerce')
    
    if 'site' in df.columns and 'equipment_name' in df.columns:
        df['asset_code'] = df['site'].astype(str) + '_' + df['equipment_name'].astype(str)
    elif 'asset_code' not in df.columns:
        df['asset_code'] = asset_code 

    df[STANDARD_DATE_COLUMN] = pd.to_datetime(df[STANDARD_DATE_COLUMN], errors='coerce')
    df = df[df['asset_code'] == asset_code].copy()
    df = df.dropna(subset=[STANDARD_DATE_COLUMN, 'monitoring_data'])
    
    if df[STANDARD_DATE_COLUMN].dt.tz is not None: #type:ignore
        df[STANDARD_DATE_COLUMN] = df[STANDARD_DATE_COLUMN].dt.tz_localize(None) #type:ignore
        
    if df.empty: return pd.DataFrame()

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
                try: return le.transform([value])[0] #type:ignore
                except ValueError: return -1
                except TypeError: return -1
            result_df[cat_col + '_encoded'] = result_df[cat_col].apply(transform_with_fallback) #type:ignore
        else:
             result_df[cat_col + '_encoded'] = -1 
    return result_df

def create_safety_checker_json(grouped_data_by_asset: Dict[str, List[Dict]]) -> Dict[str, Any]:
    safety_checker_results: Dict[str, Any] = {}
    
    for asset_code_key, asset_records in grouped_data_by_asset.items():
        site, equipment_id = asset_code_key.split('_', 1) 
        asset_historical_data: Dict[str, List[Dict[str, Any]]] = {}
        
        for feature in ALL_AVAILABLE_FEATURES:
            
            datapoint_name = feature
            if feature in FEATURE_FALLBACKS:
                datapoint_name = next((fb for fb in FEATURE_FALLBACKS[feature] if any(r.get('datapoint') == fb for r in asset_records)), feature)
            
            feature_raw_data = [r for r in asset_records if r.get('datapoint') == datapoint_name]
            
            if feature_raw_data:
                latest_record = max(feature_raw_data, key=lambda x: x.get(STANDARD_DATE_COLUMN, ''))
                data_value = latest_record.get('monitoring_data', 'NaN')
                
                try:
                    float_value = float(data_value)
                    
                    record = {
                        "data_received_on": latest_record.get(STANDARD_DATE_COLUMN).replace(' UTC', '.000000') if latest_record.get(STANDARD_DATE_COLUMN) else "", #type:ignore
                        "Anomaly_Flag": "1",
                        datapoint_name: str(float_value) # Use physical name here
                    }
                    logical_feature_key = CONSOLIDATION_MAP.get(datapoint_name, datapoint_name) 
                    asset_historical_data[logical_feature_key] = [record]
                except (ValueError, TypeError):
                    pass 
                
        if asset_historical_data:
            safety_checker_results[asset_code_key] = {
                "data": asset_historical_data,
                "site": site,
                "equipment_id": equipment_id,
                "system_type": FIXED_SYSTEM_TYPE
            }
            
    log.info(f"[SafetyChecker] Created baseline for {len(safety_checker_results)} assets.")
    return safety_checker_results

@router.post('/anomaly_detection_all_ahu')
def anomaly_detection_all_ahu() -> Dict[str, Any]:
    start_time = time.time()
    
    all_data_records = fetch_all_ahu_data(DEFAULT_BUILDING_ID)
    
    if not all_data_records:
        return {"data": {"all_anomalies_by_asset": {}}, "message": "No data found for AHU systems."}
        
    grouped_data_by_asset: Dict[str, List[Dict]] = defaultdict(list)
    for record in all_data_records:
        site = record.get('site')
        equipment_name = record.get('equipment_name')
        if site and equipment_name:
            asset_code_key = f"{site}_{equipment_name}" 
            record['asset_code'] = asset_code_key 
            grouped_data_by_asset[asset_code_key].append(record)

    log.info(f"[A2] Total grouped assets: {len(grouped_data_by_asset)}")

    reconciled_results = create_safety_checker_json(grouped_data_by_asset)
    
    total_anomalous_assets = 0
    
    for asset_code_key, asset_records in grouped_data_by_asset.items():
        
        found_anomaly_for_asset = False
        current_asset_anomalies: Dict[str, List[Dict[str, Any]]] = {}

        for feature in ALL_AVAILABLE_FEATURES:
            
            datapoint_name = feature
            if feature in FEATURE_FALLBACKS:
                datapoint_name = next((fb for fb in FEATURE_FALLBACKS[feature] if any(r.get('datapoint') == fb for r in asset_records)), feature)
            
            feature_raw_data = [r for r in asset_records if r.get('datapoint') == datapoint_name]
            
            if not feature_raw_data:
                continue

            try:
                feature_results = anomaly_detection(feature_raw_data, asset_code_key, feature)
                
                if feature_results:
                    # The prediction output keys are the actual data column names (TempSu, HuR1, etc.)
                    current_asset_anomalies[datapoint_name] = feature_results 
                    
                    if any(r.get('Anomaly_Flag') == "-1" for r in feature_results):
                        found_anomaly_for_asset = True
                            
            except HTTPException:
                raise
            except Exception as e:
                log.error(f"Prediction logic failed for {asset_code_key}/{feature}: {e}")
                continue
        
        if asset_code_key in reconciled_results and 'data' in reconciled_results[asset_code_key]:
             reconciled_results[asset_code_key]['data'].update(current_asset_anomalies)
        elif current_asset_anomalies:
             # Handle case where safety checker failed to create metadata but prediction succeeded
             site, equipment_id = asset_code_key.split('_', 1) 
             reconciled_results[asset_code_key] = {
                 "data": current_asset_anomalies,
                 "site": site,
                 "equipment_id": equipment_id,
                 "system_type": FIXED_SYSTEM_TYPE
             }
             
        if found_anomaly_for_asset:
             total_anomalous_assets += 1
    
    # 3. Format final output structure (removing the redundant 'historical_data' wrapper)
    output_assets = {
        key: {
            "data": value['data'],
            "site": value['site'],
            "equipment_id": value['equipment_id'],
            "system_type": value['system_type']
        }
        for key, value in reconciled_results.items()
        if 'data' in value and value['data'] # Ensure data map is not empty
    }
    
    final_output = {
        "data": {
            "all_anomalies_by_asset": output_assets
        },
        "total_assets_processed": len(grouped_data_by_asset),
        "anomalous_assets_count": total_anomalous_assets
    }
    
    log.info(f"Anomaly detection completed in {time.time() - start_time:.2f} seconds.")
    return final_output

class IngestionResponse(BaseModel):
    status: str = Field(...)
    total_assets_processed: int
    total_records_inserted: int
    duration_seconds: float
    message: str

def fetch_data_in_chunks(url: str, chunk_size: int) -> Iterator[Tuple[str, Dict[str, str], Iterator[List[Dict]]]]:
    print(f"Fetching data from batch endpoint {url}...")
    
    response = requests.post(url, json={})
    response.raise_for_status()
    api_data: Dict[str, Any] = response.json()
    
    all_assets_results = api_data.get('data', {}).get('all_anomalies_by_asset', {})

    if not all_assets_results:
        raise ValueError("Error: 'all_anomalies_by_asset' key not found or empty in API response.")

    for asset_code, asset_details in all_assets_results.items():
        
        asset_metadata = {
            "site": asset_details.get('site', DEFAULT_METADATA['site']),
            "equipment_id": asset_details.get('equipment_id', DEFAULT_METADATA['equipment_id']),
            "system_type": asset_details.get('system_type', DEFAULT_METADATA['system_type']),
        }
        
        building_id = BUILDING_IDS[0] 
        
        # Access the feature map directly from the inner 'data' key
        historical_data_map = asset_details.get('data', {})
        
        if not historical_data_map:
            print(f"Warning: No feature map found for asset {asset_code}. Skipping.")
            continue
            
        all_readings = []
        for feature_name, readings in historical_data_map.items():
            for reading in readings:
                
                # 1. Try the primary feature key (which is the physical datapoint name in the final API JSON)
                value = reading.get(feature_name)
                
                # CRITICAL FIX: If value is None, check ALL possible fallback/dynamic keys
                if value is None and feature_name in FEATURE_FALLBACKS:
                    # Check for HuR1, HuRt, TempSu, Co2RA1
                    for fallback_key in FEATURE_FALLBACKS[feature_name]:
                        value = reading.get(fallback_key)
                        if value is not None:
                            break
                            
                # Fallback check for known edge cases like Co2Avg
                if value is None and feature_name == 'Co2RA' and 'Co2Avg' in reading:
                    value = reading.get('Co2Avg')

                flat_record = {
                    'feature_name': feature_name, 
                    'data_value': value, # Now reliably holds the numeric string value
                    'data_received_on_str': reading.get('data_received_on'),
                    'Anomaly_Flag': reading.get('Anomaly_Flag')
                }
                all_readings.append(flat_record)
        
        chunk_iterator = (all_readings[i:i + chunk_size] 
                          for i in range(0, len(all_readings), chunk_size))
        
        yield building_id, asset_metadata, chunk_iterator

def save_data_to_cassandra(data_chunk: List[Dict], building_id: str, metadata: dict) -> int:
    """Saves data chunk to Cassandra. Returns the number of rows inserted (into history)."""
    cluster = None
    rows_inserted = 0
    
    site_value = metadata["site"]
    equipment_id_value = metadata["equipment_id"]
    system_type_value = metadata["system_type"]

    history_table = f"{building_id.replace('-', '').lower()}_{HISTORY_TABLE_SUFFIX}"
    anamoly_table = f"{building_id.replace('-', '').lower()}_{ANAMOLY_TABLE_SUFFIX}"
    
    CREATE_BASE_CQL = """
    CREATE TABLE IF NOT EXISTS {keyspace}."{table_name}" ( 
        datapoint text, 
        timestamp timestamp, 
        value text, 
        anomaly_flag int, 
        site text, 
        equipment_id text, 
        system_type text,
        PRIMARY KEY ((site, system_type, equipment_id), datapoint, timestamp)) 
        WITH CLUSTERING ORDER BY (datapoint ASC, timestamp ASC) {ttl_option};
    """
    
    INSERT_BASE_CQL = f"""
    INSERT INTO {KEYSPACE_NAME}."{{table_name}}" 
    (datapoint, timestamp, value, anomaly_flag, site, equipment_id, system_type) 
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        cluster = Cluster(CASSANDRA_HOST, port=CASSANDRA_PORT)
        session = cluster.connect(KEYSPACE_NAME)
        
        session.execute(CREATE_BASE_CQL.format(keyspace=KEYSPACE_NAME, table_name=history_table, ttl_option=""))
        session.execute(CREATE_BASE_CQL.format(keyspace=KEYSPACE_NAME, table_name=anamoly_table, ttl_option=f"AND default_time_to_live = {ANAMOLY_TTL_SECONDS}"))
        
        history_stmt = session.prepare(INSERT_BASE_CQL.format(table_name=history_table))
        anamoly_stmt = session.prepare(INSERT_BASE_CQL.format(table_name=anamoly_table))
        
        for reading in data_chunk:
            timestamp_str = reading.get('data_received_on_str')
            
            # --- Timestamp Cleaning and Parsing ---
            if timestamp_str and timestamp_str.count('.') > 1 and 'T' in timestamp_str:
                try:
                    first_dot_pos = timestamp_str.find('.')
                    second_dot_pos = timestamp_str.find('.', first_dot_pos + 1)
                    if second_dot_pos != -1:
                        timestamp_str = timestamp_str[:second_dot_pos]
                except Exception:
                    pass

            try:
                timestamp_dt = parser.parse(timestamp_str) if timestamp_str else None
                if timestamp_dt and (timestamp_dt.tzinfo is None or timestamp_dt.tzinfo.utcoffset(timestamp_dt) is None):
                    timestamp_dt = timestamp_dt.replace(tzinfo=CASSANDRA_DEFAULT_TZ)
            except Exception as e:
                print(f"SKIPPED (Timestamp Error): Asset {metadata['equipment_id']} Feature {reading.get('feature_name')} Time {timestamp_str}. Error: {e}")
                timestamp_dt = None
            
            if timestamp_dt is None: continue 

            # --- Data Point Selection (The Final Fix: Store Physical Name) ---
            physical_datapoint = str(reading.get('feature_name'))
            datapoint = physical_datapoint # Store the actual name (HuR1, TempSu, Co2Avg, etc.)
            
            # --- Value and Flag Extraction ---
            value_to_insert = str(reading.get('data_value')) if reading.get('data_value') is not None else None
            if value_to_insert is None: 
                print(f"SKIPPED (Null Value): Asset {metadata['equipment_id']} Feature {physical_datapoint} Time {timestamp_str}.")
                continue 
            
            anomaly_flag_value = reading.get('Anomaly_Flag')
            try:
                anomaly_flag = int(anomaly_flag_value) if anomaly_flag_value is not None else 0
            except ValueError:
                anomaly_flag = 0 
            
            data_tuple = (datapoint, timestamp_dt, value_to_insert, anomaly_flag, 
                          site_value, equipment_id_value, system_type_value)
            
            # Insert into HISTORY table (All data)
            session.execute(history_stmt, data_tuple)
            rows_inserted += 1

            # Insert into ANAMOLY table (All data, as requested)
            session.execute(anamoly_stmt, data_tuple)
            
    except Exception as e:
        # NOTE: Using print for local execution debugging
        print(f"Cassandra insert error: {type(e).__name__}: {e}")
        # print(traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=f"Cassandra insert error: {e}")
    finally:
        if cluster:
            cluster.shutdown()
            
    return rows_inserted

# --- NEW ENDPOINT FOR INGESTION ---
@router.post("/store_anamolies")
def store_anamolies_endpoint() -> IngestionResponse:
    """
    Triggers the full batch process: fetches anomaly detection results from the 
    external API and inserts them into Cassandra's historical and anomaly tables.
    """
    start_time = time.time()
    total_assets_processed = 0
    total_records_inserted = 0
    
    try:
        # Use API_INTERNAL_URL to call the prediction endpoint within the same app
        api_iterator = fetch_data_in_chunks(API_INTERNAL_URL, CHUNK_SIZE)
        
        for building_id, api_metadata, api_chunks in api_iterator:
            total_assets_processed += 1
            
            log.info(f"Processing Asset {total_assets_processed}: {api_metadata['site']}/{api_metadata['equipment_id']}")
            
            for i, chunk in enumerate(api_chunks):
                inserted_count = save_data_to_cassandra(chunk, building_id, api_metadata)
                total_records_inserted += inserted_count
                
        duration = time.time() - start_time
        
        return { #type:ignore
            "status": "SUCCESS",
            "total_assets_processed": total_assets_processed,
            "total_records_inserted": total_records_inserted,
            "duration_seconds": round(duration, 3),
            "message": "Batch ingestion completed successfully."
        }
        
    except requests.exceptions.RequestException as req_err:
        log.error(f"External API fetch failed: {req_err}")
        raise HTTPException(status_code=500, detail=f"External API fetch failed: {req_err}")
    except ValueError as val_err:
        log.error(f"Data processing error: {val_err}")
        raise HTTPException(status_code=500, detail=f"Data processing error: {val_err}")
    except Exception as e:
        log.error(f"An unexpected error occurred during batch execution: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during batch execution: {e}")