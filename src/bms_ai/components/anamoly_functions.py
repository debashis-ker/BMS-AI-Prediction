import os
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
import pandas as pd
import joblib
from fastapi import HTTPException
from dateutil import parser, tz
from sklearn.preprocessing import LabelEncoder
from cassandra.cluster import Session, ConsistencyLevel
from cassandra.query import SimpleStatement
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.cassandra_utils import fetch_all_ahu_data
from src.bms_ai.utils.ikon_apis import fetch_and_find_data_points
import time
import requests

log = setup_logger(__name__)

FIXED_SYSTEM_TYPE = "AHU"
DEFAULT_BUILDING_ID = os.getenv('DEFAULT_BUILDING_ID', '36c27828-d0b4-4f1e-8a94-d962d342e7c2')
API_INTERNAL_URL = os.getenv('ANOMALY_DETECTION_URL')
BUILDING_IDS = [DEFAULT_BUILDING_ID]
CASSANDRA_HOST = os.getenv('CASSANDRA_HOST')
CASSANDRA_PORT = os.getenv('CASSANDRA_PORT')
KEYSPACE_NAME = os.getenv('CASSANDRA_KEYSPACE')
CASSANDRA_DEFAULT_TZ = tz.gettz('UTC')
HISTORY_TABLE_SUFFIX = "historical_data"
ANAMOLY_TABLE_SUFFIX = "anamoly_data"
ANAMOLY_TTL_SECONDS = 2592000
STANDARD_DATE_COLUMN = "data_received_on"
CO2RA_CEILING_THRESHOLD = 850.0
FBVFD_NORMAL_MAX = 1.0
CHART_AGGREGATION_MAP = {
    'TSu': ['TSu', 'TempSu'],
    'Co2RA': ['Co2RA', 'Co2Avg', 'RtCo2'],
    'HuAvg1': ['HuAvg1', 'HuRt', 'HuR1'],
    'FbFAD': ['FbFAD'],
    'FbVFD': ['FbVFD']
}
# Unit_Mapping
# UNIT_MAPPING = {
#     'TSu' : "celsius",
#     'Co2RA' : "ppm",
#     'FbFAD' : "%",
#     'FbVFD' : "hertz",
#     'HuAvg1' : "%",
#     'TempSu' : "celsius",
#     'Co2Avg' : "ppm",
#     'HuR1' : "%",
#     'HuRt' : "%"
# }

MASTER_LOGICAL_FEATURES = ['TSu', 'Co2RA', 'FbFAD', 'FbVFD', 'HuAvg1', 'TempSu', 'Co2Avg', 'HuR1', 'HuRt']
KNOWN_MODEL_FILE_NAMES = ['TempSu', 'Co2Avg', 'HuR1', 'HuRt']
ALL_AVAILABLE_FEATURES = MASTER_LOGICAL_FEATURES
DEFAULT_METADATA = {"equipment_id": "", "system_type": "AHU", "site": ""}

PRIORITY_GROUPS = {
    'humidity': ['HuAvg1', 'HuRt', 'HuR1'],
    'temperature': ['TSu', 'TempSu'],
    'co2': ['Co2RA', 'Co2Avg']
}

def wrap_data_list(data_list: List[Dict]) -> Dict[str, Any]:
    """Wraps the list of datapoint records into the dictionary structure expected by build_dynamic_features."""
    return {"success": True, "data": data_list, "count": len(data_list)}

def build_dynamic_features(json_record: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
    dynamic_fallbacks = defaultdict(list)
    consolidation_map = {}
    
    for record in json_record.get('data', []):
        physical_name = record.get('dataPointName')
        tags = set(record.get('queryTags', []))
        if not physical_name: continue
            
        master_feature = None

        if 'temp' in tags and 'supply' in tags:
            master_feature = 'TSu'
        elif physical_name == "TempSu": 
            master_feature = 'TempSu'
        elif physical_name == "TSu":
            master_feature = 'TSu'

        elif 'co2' in tags:
            master_feature = 'Co2RA'

        elif 'humidity' in tags and ('space' in tags or 'average' in tags or 'return' in tags):
            master_feature = 'HuAvg1'

        elif ('speed' in tags or 'vfd' in tags) and 'temp' not in tags:
            master_feature = 'FbVFD'
        
        elif 'damper' in tags and 'outside' in tags:
            master_feature = 'FbFAD'

        if master_feature is None and physical_name in MASTER_LOGICAL_FEATURES:
             master_feature = physical_name

        if master_feature:
            consolidation_map[physical_name] = master_feature
            if physical_name != master_feature:
                if physical_name not in dynamic_fallbacks[master_feature]:
                    dynamic_fallbacks[master_feature].append(physical_name)
    
    return list(consolidation_map.keys()), dict(dynamic_fallbacks), consolidation_map

def initialize_anomaly_detection_models(building_id: str,
    floor_id: Optional[str],
    equipment_id: str,
    search_tag_groups: List[List[str]],
    ticket: str,
    software_id: str,
    account_id: str,
    system_type: Optional[str] = None,
    env: Optional[str] = "prod",
    ticket_type: Optional[str] = None):
    """
    Fetches datapoint mapping from Ikon, builds feature consolidation maps,
    and loads all required anomaly detection models into global state.
    """
    global ALL_QUERY_FEATURES, FEATURE_FALLBACKS, CONSOLIDATION_MAP
    global REVERSE_FEATURE_MAP, MASTER_ANAMOLY_MODELS
    global ALL_AVAILABLE_FEATURES
    ALL_AVAILABLE_FEATURES = MASTER_LOGICAL_FEATURES

    log.info("Starting anomaly model initialization...")

    try:
        raw_data_list = fetch_and_find_data_points(building_id=building_id, floor_id=floor_id, equipment_id=equipment_id,
            search_tag_groups=search_tag_groups,
            ticket=ticket, software_id = software_id, account_id = account_id, system_type=system_type, env=env, ticket_type=ticket_type) #type:ignore
    except requests.exceptions.RequestException as req_err:
        log.error(f"IKON API fetch failed during initialization: {req_err}")
        raise HTTPException(status_code=503, detail=f"Failed to fetch datapoints from Ikon API: {req_err}")
    except Exception as e:
        log.error(f"Unexpected error during IKON API fetch: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during external API call: {e}")

    json_record = wrap_data_list(raw_data_list)
    
    if not json_record.get('data'):
        log.warning(f"No datapoints found for AHU: {equipment_id} in building: {building_id}.")
        raise HTTPException(status_code=503, detail=f"No Datapoint found due to invalid IKON fetch datapoint API request.")
    
    (
        ALL_QUERY_FEATURES,
        FEATURE_FALLBACKS,
        CONSOLIDATION_MAP
    ) = build_dynamic_features(json_record)

    ALL_AVAILABLE_FEATURES = [f for f in MASTER_LOGICAL_FEATURES if f not in CONSOLIDATION_MAP or CONSOLIDATION_MAP[f] == f]

    REVERSE_FEATURE_MAP = {
        physical_name: master_name
        for physical_name, master_name in CONSOLIDATION_MAP.items()
    }

    for master_feat in MASTER_LOGICAL_FEATURES:
        if master_feat not in REVERSE_FEATURE_MAP:
            REVERSE_FEATURE_MAP[master_feat] = master_feat

    MASTER_ANAMOLY_MODELS = {}
    log.info("Starting model loading and consolidation...")

    all_model_keys_to_load = set(MASTER_LOGICAL_FEATURES)
    for fallbacks in FEATURE_FALLBACKS.values():
        all_model_keys_to_load.update(fallbacks)

    all_model_keys_to_load.update(KNOWN_MODEL_FILE_NAMES)

    models_loaded_count = 0
    for model_key_in_file in all_model_keys_to_load:
        model_file = f"artifacts/generic_anamoly_models/{model_key_in_file}_model.joblib"
        master_key = model_key_in_file

        if master_key is None: continue

        try:
            feature_models = joblib.load(model_file)
            if master_key not in MASTER_ANAMOLY_MODELS:
                MASTER_ANAMOLY_MODELS[master_key] = {}
            
            MASTER_ANAMOLY_MODELS[master_key].update(feature_models)
            log.info(f"Loaded '{model_key_in_file}' models under KEY: '{master_key}'")
            
            logical_master = CONSOLIDATION_MAP.get(model_key_in_file)
            if logical_master and logical_master != master_key:
                if logical_master not in MASTER_ANAMOLY_MODELS:
                    MASTER_ANAMOLY_MODELS[logical_master] = {}
                MASTER_ANAMOLY_MODELS[logical_master].update(feature_models)
            models_loaded_count += 1

        except FileNotFoundError:
            log.warning(f"Model file not found for {model_key_in_file}. Skipping.")
        except Exception as e:
            log.error(f"FATAL: Error loading model {model_key_in_file}. Skipping: {e}")

    log.info("Anomaly model initialization complete.")
    
    if not MASTER_ANAMOLY_MODELS:
         raise HTTPException(status_code=503, detail="No anomaly detection models could be loaded. Service is unavailable.")
         
    global anamoly_model 
    anamoly_model = MASTER_ANAMOLY_MODELS

def anomaly_detection(raw_data: List[Dict[str, Any]], asset_code: str, feature: str) -> List[Dict[str, Any]]:
    """Runs the loaded model prediction pipeline on raw data for a specific asset and feature."""
    if feature not in anamoly_model: #type:ignore
        raise HTTPException(status_code=503, detail=f"Anomaly Detection model for master feature '{feature}' is unavailable. Please check service initialization.")

    try:
        model_package = anamoly_model[feature].get(asset_code) #type:ignore
        if model_package is None: 
            log.warning(f"[{asset_code}/{feature}] No specific model found for this asset. Returning empty list.")
            return []

        model_features = model_package.get('feature_cols', [feature]) 
    except KeyError as e:
        log.warning(f"[{asset_code}/{feature}] Model package key error: {e}. Returning empty list.")
        return [] 
    
    try:
        df_wide = anamoly_data_pipeline(raw_data, asset_code, model_package, feature)
        if df_wide.empty: 
            log.info(f"[{asset_code}/{feature}] Data pipeline returned empty DataFrame.")
            return []
    except Exception as e:
        log.error(f"[{asset_code}/{feature}] Data pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal data pipeline failure for asset {asset_code}/{feature}: {e}")
        
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
        raise HTTPException(status_code=500, detail=f"Model prediction failed for asset {asset_code}/{feature}: {e}")

    X_df['Anomaly_Flag'] = predictions
    
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
            "Anomaly_Flag": str(int(row['Anomaly_Flag'])),
            feature: str(float(row[data_column]))
        }
        report_list.append(record)
    return report_list

def anamoly_data_pipeline(records: List[Dict[str, Any]], asset_code: str, model_package: Dict[str, Any], feature_name: str) -> pd.DataFrame:
    """Preprocesses raw data into a wide format DataFrame suitable for model prediction."""
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

def anamoly_evaluation(building_id: str, floor_id: Optional[str], equipment_id: str, search_tags: List[List[str]], ticket_id: str, software_id: str, account_id: str, system_type: Optional[str] = None, env: Optional[str] = "prod", ticket_type: Optional[str] = None):
    try:
        initialize_anomaly_detection_models(building_id=building_id, floor_id=floor_id, equipment_id=equipment_id, search_tag_groups=search_tags, ticket=ticket_id, software_id=software_id, account_id=account_id, system_type=system_type, env=env, ticket_type=ticket_type)
    except Exception as e:
        log.error(f"Initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    all_data_records = fetch_all_ahu_data(building_id=building_id or DEFAULT_BUILDING_ID)
    if not all_data_records: return {"data": {"all_anomalies_by_asset": {}}}

    grouped_data_by_asset = defaultdict(list)
    for record in all_data_records:
        if record.get('site') and record.get('equipment_name'):
            asset_key = f"{record['site']}_{record['equipment_name']}"
            record['asset_code'] = asset_key
            grouped_data_by_asset[asset_key].append(record)

    reconciled_results = create_safety_checker_json(grouped_data_by_asset)
    total_anomalous_assets = 0
    
    for asset_code_key, asset_records in grouped_data_by_asset.items():
        found_anomaly_for_asset = False
        current_asset_anomalies = {}
        processed_physical_points = set()

        active_selections = {}
        for group, p_list in PRIORITY_GROUPS.items():
            for feat in p_list:
                target_physical = feat
                if feat in FEATURE_FALLBACKS:
                    target_physical = next((fb for fb in FEATURE_FALLBACKS[feat] 
                                          if any(r.get('datapoint') == fb for r in asset_records)), feat)
                
                if any(r.get('datapoint') == target_physical for r in asset_records):
                    active_selections[group] = feat
                    break

        for feature in ALL_AVAILABLE_FEATURES:
            in_group = False
            for group, p_list in PRIORITY_GROUPS.items():
                if feature in p_list:
                    in_group = True
                    if feature != active_selections.get(group):
                        continue

            datapoint_to_pull = feature
            if feature in FEATURE_FALLBACKS:
                datapoint_to_pull = next((fb for fb in FEATURE_FALLBACKS[feature] 
                                         if any(r.get('datapoint') == fb for r in asset_records)), feature)
            
            if datapoint_to_pull in processed_physical_points:
                continue

            feature_raw_data = [r for r in asset_records if r.get('datapoint') == datapoint_to_pull]
            if not feature_raw_data:
                continue

            model_key = feature
            if model_key not in anamoly_model:
                model_key = CONSOLIDATION_MAP.get(model_key, model_key)
                if model_key not in anamoly_model: continue

            try:
                feature_results = anomaly_detection(feature_raw_data, asset_code_key, model_key)
                if feature_results:
                    current_asset_anomalies[datapoint_to_pull] = feature_results 
                    processed_physical_points.add(datapoint_to_pull)
                    if any(r.get('Anomaly_Flag') == "-1" for r in feature_results):
                        found_anomaly_for_asset = True
            except Exception as e:
                log.error(f"Error for {asset_code_key}/{feature}: {e}")

        if asset_code_key in reconciled_results:
            reconciled_results[asset_code_key]['data'].update(current_asset_anomalies)
        elif current_asset_anomalies:
            site, eq_id = asset_code_key.split('_', 1) 
            reconciled_results[asset_code_key] = {"data": current_asset_anomalies, "site": site, "equipment_id": eq_id, "system_type": FIXED_SYSTEM_TYPE}
             
        if found_anomaly_for_asset: total_anomalous_assets += 1
    
    return {
        "data": {"all_anomalies_by_asset": reconciled_results},
        "total_assets_processed": len(grouped_data_by_asset),
        "anomalous_assets_count": total_anomalous_assets
    }

def create_safety_checker_json(grouped_data_by_asset: Dict[str, List[Dict]]) -> Dict[str, Any]:
    safety_checker_results: Dict[str, Any] = {}
    priority_groups = {
        'humidity': ['HuAvg1', 'HuRt', 'HuR1'],
        'temperature': ['TSu', 'TempSu'],
        'co2': ['Co2RA', 'Co2Avg']
    }
    
    for asset_code_key, asset_records in grouped_data_by_asset.items():
        site, equipment_id = asset_code_key.split('_', 1) 
        asset_historical_data = {}
        processed_physical_names = set()

        active_selections = {}
        for group, p_list in priority_groups.items():
            for feat in p_list:
                target = feat
                if feat in FEATURE_FALLBACKS:
                    target = next((fb for fb in FEATURE_FALLBACKS[feat] if any(r.get('datapoint') == fb for r in asset_records)), feat)
                if any(r.get('datapoint') == target for r in asset_records):
                    active_selections[group] = feat
                    break

        for feature in ALL_AVAILABLE_FEATURES:
            skip_feat = False
            for group, p_list in priority_groups.items():
                if feature in p_list and feature != active_selections.get(group):
                    skip_feat = True
            if skip_feat: continue

            datapoint_name = feature
            if feature in FEATURE_FALLBACKS:
                datapoint_name = next((fb for fb in FEATURE_FALLBACKS[feature] if any(r.get('datapoint') == fb for r in asset_records)), feature)
            
            if datapoint_name in processed_physical_names: continue

            feature_raw_data = [r for r in asset_records if r.get('datapoint') == datapoint_name]
            if feature_raw_data:
                latest_record = max(feature_raw_data, key=lambda x: x.get(STANDARD_DATE_COLUMN, ''))
                val = latest_record.get('monitoring_data')
                if val is not None and str(val).lower() != 'null':
                    logical_key = CONSOLIDATION_MAP.get(datapoint_name, feature)
                    asset_historical_data[logical_key] = [{
                        "data_received_on": latest_record.get(STANDARD_DATE_COLUMN).replace(' UTC', '.000000'), #type:ignore
                        "Anomaly_Flag": "1",
                        logical_key: str(val)
                    }]
                    processed_physical_names.add(datapoint_name)
                 
        if asset_historical_data:
            safety_checker_results[asset_code_key] = {"data": asset_historical_data, "site": site, "equipment_id": equipment_id, "system_type": FIXED_SYSTEM_TYPE}
    return safety_checker_results

def process_asset_anomalies_for_storage(asset_details: Dict[str, Any], asset_code: str) -> List[Dict]:
    all_readings = []
    
    asset_metadata = {
        "site": asset_details.get('site', DEFAULT_METADATA['site']),
        "equipment_id": asset_details.get('equipment_id', DEFAULT_METADATA['equipment_id']),
        "system_type": asset_details.get('system_type', DEFAULT_METADATA['system_type']),
    }
    
    historical_data_map = asset_details.get('data', {})
    
    if not historical_data_map:
        log.warning(f"Warning: No feature map found in evaluation result for asset {asset_code}. Skipping.")
        return []
    
    for feature_name, readings in historical_data_map.items():
        for reading in readings:
            
            data_value = reading.get(feature_name)
            
            if data_value is None:
                log.warning(f"Value missing for {asset_code}/{feature_name} in reading. Skipping record.")
                continue

            flat_record = {
                'feature_name': feature_name, 
                'data_value': data_value,
                'data_received_on_str': reading.get('data_received_on'),
                'Anomaly_Flag': reading.get('Anomaly_Flag'),
                'metadata': asset_metadata
            }
            all_readings.append(flat_record)
            
    return all_readings

def store_data(data_chunk: List[Dict], building_id: str, metadata: dict, session: Session) -> int:
    """Saves data chunk to Cassandra using prepared statements."""
    rows_inserted = 0
    
    site_value = metadata["site"]
    equipment_id_value = metadata["equipment_id"]
    system_type_value = metadata["system_type"]

    if not KEYSPACE_NAME:
         raise HTTPException(status_code=500, detail="Cassandra keyspace is not configured (KEYSPACE_NAME environment variable missing).")

    history_table = f"{HISTORY_TABLE_SUFFIX}_{building_id.replace('-', '').lower()}"
    anamoly_table = f"{ANAMOLY_TABLE_SUFFIX}_{building_id.replace('-', '').lower()}"
    
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
        session.execute(CREATE_BASE_CQL.format(keyspace=KEYSPACE_NAME, table_name=history_table, ttl_option=""))
        session.execute(CREATE_BASE_CQL.format(keyspace=KEYSPACE_NAME, table_name=anamoly_table, ttl_option=f"AND default_time_to_live = {ANAMOLY_TTL_SECONDS}"))
        
        history_stmt = session.prepare(INSERT_BASE_CQL.format(table_name=history_table))
        anamoly_stmt = session.prepare(INSERT_BASE_CQL.format(table_name=anamoly_table))
        
        for reading in data_chunk:
            timestamp_str = reading.get('data_received_on_str')
            
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
                log.warning(f"Failed to parse timestamp '{timestamp_str}': {e}. Skipping record.")
                timestamp_dt = None
            
            if timestamp_dt is None: continue 

            physical_datapoint = str(reading.get('feature_name'))
            datapoint = physical_datapoint
            
            value_to_insert = str(reading.get('data_value')) if reading.get('data_value') is not None else None
            if value_to_insert is None: 
                log.warning(f"Value to insert is None for {datapoint} at {timestamp_str}. Skipping record.")
                continue 
            
            anomaly_flag_value = reading.get('Anomaly_Flag')
            try:
                anomaly_flag = int(anomaly_flag_value) if anomaly_flag_value is not None else 0
            except ValueError:
                log.warning(f"Could not parse anomaly flag '{anomaly_flag_value}'. Defaulting to 0.")
                anomaly_flag = 0 
            
            data_tuple = (datapoint, timestamp_dt, value_to_insert, anomaly_flag, 
                          site_value, equipment_id_value, system_type_value)
            
            session.execute(history_stmt, data_tuple)
            rows_inserted += 1

            session.execute(anamoly_stmt, data_tuple)
            
    except Exception as e:
        log.error(f"Cassandra operation failed: {e}")
        raise HTTPException(status_code=503, detail=f"Cassandra insert error: {e}")
            
    return rows_inserted

def save_data_to_cassandra(
    session: Session,
    building_id: str,
    floor_id: Optional[str],
    equipment_id: str,
    search_tags: List[List[str]],
    ticket_id: str,
    software_id: str,
    account_id: str,
    system_type: Optional[str] = None,
    env: Optional[str] = "prod",
    ticket_type: Optional[str] = None
    ):

    start_time = time.time()
    total_assets_processed = 0
    total_records_inserted = 0
    CASSANDRA_CHUNK_SIZE = 100

    building_id = building_id or DEFAULT_BUILDING_ID
    
    try:
        log.info(f"Starting anomaly evaluation for building: {building_id}")
        result_json = anamoly_evaluation(
            building_id=building_id,
            floor_id=floor_id,
            equipment_id=equipment_id,
            search_tags=search_tags,
            ticket_id=ticket_id,
            software_id=software_id,
            account_id=account_id,
            system_type=system_type,
            ticket_type=ticket_type
        )
        
        all_assets_results = result_json.get('data', {}).get('all_anomalies_by_asset', {})

        if not all_assets_results:
            return {
                "status":"SUCCESS",
                "total_assets_processed":0,
                "total_records_inserted":0,
                "duration_seconds":round(time.time() - start_time, 3),
                "message":"Anomaly evaluation returned no assets with data."
            }

        for asset_code, asset_details in all_assets_results.items():
            total_assets_processed += 1
            log.info(f"Processing Asset {total_assets_processed} for storage: {asset_code}")
            
            all_readings = process_asset_anomalies_for_storage(asset_details, asset_code)
            
            asset_metadata = {
                "site": asset_details.get('site', DEFAULT_METADATA['site']),
                "equipment_id": asset_details.get('equipment_id', DEFAULT_METADATA['equipment_id']),
                "system_type": asset_details.get('system_type', FIXED_SYSTEM_TYPE),
            }

            for i in range(0, len(all_readings), CASSANDRA_CHUNK_SIZE):
                chunk = all_readings[i:i + CASSANDRA_CHUNK_SIZE]
                
                if session is None:
                     raise HTTPException(status_code=503, detail="Cassandra session is not initialized or available.")

                inserted_count = store_data(chunk, building_id, asset_metadata, session)
                total_records_inserted += inserted_count
        
        duration = time.time() - start_time
        
        return{
            "status":"SUCCESS",
            "total_assets_processed":total_assets_processed,
            "total_records_inserted":total_records_inserted,
            "duration_seconds":round(duration, 3),
            "message":f"Batch ingestion completed successfully. Processed {total_assets_processed} assets."
        }
    
    except HTTPException:
        raise
    except requests.exceptions.RequestException as req_err:
        log.error(f"External API fetch failed: {req_err}")
        raise HTTPException(status_code=503, detail=f"External API fetch failed: {req_err}")
    except ValueError as val_err:
        log.error(f"Data processing error: {val_err}")
        raise HTTPException(status_code=400, detail=f"Data processing error: {val_err}")
    except Exception as e:
        log.error(f"An unexpected error occurred during batch execution: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during batch execution: {e}")

def format_cassandra_output(data_records: List[Dict], limit: Optional[int]) -> Dict[str, Any]:
    """Formats raw Cassandra query results into the desired API output structure."""
    historical_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for record in data_records:
         datapoint = record.pop('datapoint') 
         
         record_output = {
             "timestamp": record['timestamp'],
             "Anamoly_Flag": str(record['anomaly_flag']),
             datapoint: record['value'] 
         }
         
         historical_data[datapoint].append(record_output)

    final_historical_data = {}
    if limit is not None:
         for feature, records in historical_data.items():
             final_historical_data[feature] = records[:limit]
    else:
         final_historical_data = dict(historical_data)

    # Unit_Mapping
    # active_units = {
    #     feature: UNIT_MAPPING.get(feature, "unknown") 
    #     for feature in final_historical_data.keys()
    # }

    return {
        "data": {
            "historical_data": final_historical_data,
        }
        # "units": active_units # Unit_Mapping
    }

def fetch_data_from_cassandra(params: Dict[str, Any], table_suffix: str, session: Session) -> List[Dict]:
    """
    Constructs and executes a CQL query against the specified Cassandra table.
    Dynamically fetches feature list if none is provided, formats timestamp, 
    and returns results ordered by timestamp.
    """
    features_to_query = None

    building_id = params.get('building_id')
    if not building_id:
        raise HTTPException(status_code=400, detail="Missing required parameter: 'building_id'.")
        
    limit = params.get('limit', 1000)
    
    if not KEYSPACE_NAME:
         raise HTTPException(status_code=500, detail="Cassandra keyspace is not configured (KEYSPACE_NAME environment variable missing).")

    cleaned_building_id = building_id.replace("-", "").lower()
    table_name = f'"{table_suffix}_{cleaned_building_id}"'

    where_clauses = []
    features_to_query = params.get('feature')
    
    if not features_to_query:
        log.info("Feature list missing in request. Attempting dynamic feature list generation...")
        
        ticket = params.get('ticket')
        account_id = params.get('account_id')
        software_id = params.get('software_id')
        search_tag_groups = params.get('search_tag_groups')
        ticket_type = params.get('ticket_type')
        
        if ticket and account_id and software_id and search_tag_groups:
            try:
                raw_data_list = fetch_and_find_data_points(
                    building_id=params.get('building_id', DEFAULT_BUILDING_ID),
                    equipment_id=params.get('equipment_id', ""),
                    system_type=params.get('system_type', FIXED_SYSTEM_TYPE),
                    floor_id=params.get('floor_id',""),
                    ticket=ticket,
                    account_id=account_id,
                    software_id=software_id,
                    search_tag_groups=search_tag_groups,
                    ticket_type=ticket_type
                )
                json_record = wrap_data_list(raw_data_list)
                (features_to_query, _, _) = build_dynamic_features(json_record)
                log.debug(f"Dynamic fetch yielded features: {features_to_query}")

            except Exception as e:
                log.error(f"Dynamic feature fetch failed: {e}. Falling back to querying with available filters.")
                features_to_query = [] 
        else:
            log.warning("IKON API parameters missing from DataQueryRequest for dynamic lookup. Proceeding without feature filter.")
            features_to_query = []

    if features_to_query:
         if isinstance(features_to_query, list):
              if features_to_query:
                   feature_list_str = ", ".join([f"'{f}'" for f in features_to_query])
                   where_clauses.append(f"datapoint IN ({feature_list_str})")
         else:
              where_clauses.append(f"datapoint = '{features_to_query}'")

    if params.get('site'): where_clauses.append(f"site = '{params['site']}'")
    if params.get('equipment_id'): where_clauses.append(f"equipment_id = '{params['equipment_id']}'")
    if params.get('system_type'): where_clauses.append(f"system_type = '{params['system_type']}'")
    if params.get('start_date'): where_clauses.append(f"timestamp >= '{params['start_date']}'")
    if params.get('end_date'): where_clauses.append(f"timestamp <= '{params['end_date']}'")
    
    where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    
    cql_query = f"""
        SELECT datapoint, timestamp, value, anomaly_flag
        FROM {KEYSPACE_NAME}.{table_name}
        {where_clause}
        ALLOW FILTERING;
    """

    results_list = []
    
    try:
        if session is None:
             raise HTTPException(status_code=503, detail="Cassandra session is not initialized or available.")
             
        statement = SimpleStatement(cql_query, consistency_level=ConsistencyLevel.LOCAL_QUORUM)
        rows = session.execute(statement, timeout=30.0) 
        
        for row in rows:
            timestamp_dt = row.timestamp
            formatted_timestamp = None
            
            if timestamp_dt is not None:
                milliseconds = timestamp_dt.strftime('%f')[:4]
                
                base_time_str = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
                
                formatted_timestamp = f"{base_time_str}.{milliseconds}0+00:00"
            
            results_list.append({
                "datapoint": row.datapoint,
                "timestamp": formatted_timestamp, 
                "value": row.value,
                "anomaly_flag": row.anomaly_flag
            })
            
    except Exception as e:
        log.error(f"Cassandra query failed for table {table_name} with query: {cql_query}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Cassandra query failed for table {table_name}: {e}")
    
    results_list.sort(key=lambda x: x['timestamp'])
    
    return results_list

from dateutil import parser
from cassandra.query import SimpleStatement, ConsistencyLevel
import pandas as pd

def dynamic_anomaly_detection_chart(
    building_id: str,
    floor_id: Optional[str],
    site: str,
    equipment_id: str,
    search_tag_groups: List[List[str]],
    ticket: str,
    software_id: str,
    account_id: str,
    system_type: Optional[str] = None,
    ticket_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    chart_type: Optional[str] = "pie",
    session: Session = None) -> Dict[str, Any]: #type:ignore

    chart_type = chart_type.lower() if chart_type else "pie"
    
    try:
        raw_api_points = fetch_and_find_data_points(
            building_id=building_id,
            floor_id=floor_id,
            equipment_id=equipment_id,
            search_tag_groups=search_tag_groups,
            ticket=ticket,
            software_id=software_id,
            account_id=account_id,
            system_type=system_type,
            ticket_type=ticket_type
        )
        json_record = wrap_data_list(raw_api_points)
        all_phys_features, _, consolidation_map = build_dynamic_features(json_record)
        
        if not site and raw_api_points:
            site = raw_api_points[0].get('site', '')
            
    except Exception as e:
        log.error(f"Metadata fetch failed: {e}")
        all_phys_features = MASTER_LOGICAL_FEATURES 
        consolidation_map = {}

    if not all_phys_features:
        all_phys_features = MASTER_LOGICAL_FEATURES

    cleaned_id = building_id.replace("-", "").lower()
    table_name = f'"{ANAMOLY_TABLE_SUFFIX}_{cleaned_id}"' 
    datapoint_list_str = ", ".join([f"'{f}'" for f in all_phys_features])
    
    active_system_type = system_type if system_type else FIXED_SYSTEM_TYPE
    where_clauses = [f"system_type = '{active_system_type}'"]

    if equipment_id and equipment_id.strip() != "":
        where_clauses.append(f"equipment_id = '{equipment_id}'")
    
    if site and site.strip() != "":
        where_clauses.append(f"site = '{site}'")
    
    date_filters = [('start_date', start_date, '>='), ('end_date', end_date, '<=')]
    for label, val, operator in date_filters:
        if val and str(val).strip() != "":
            try:
                clean_date = parser.parse(str(val)).strftime('%Y-%m-%d %H:%M:%S')
                where_clauses.append(f"timestamp {operator} '{clean_date}'")
            except Exception as d_err:
                log.warning(f"Invalid date format for {label}: {val}. Skipping.")

    where_clauses.append(f"datapoint IN ({datapoint_list_str})")
    
    query = f"SELECT site, equipment_id, datapoint, timestamp, value, anomaly_flag FROM {KEYSPACE_NAME}.{table_name} "
    query += "WHERE " + " AND ".join(where_clauses) + " ALLOW FILTERING;"
    
    log.debug(f"Executing Chart Query: {query}")

    try:
        statement = SimpleStatement(query, consistency_level=ConsistencyLevel.LOCAL_QUORUM)
        rows = session.execute(statement)
        data_records = [{"site": r.site, "equipment_id": r.equipment_id, "datapoint": r.datapoint, 
                         "timestamp": r.timestamp, "value": r.value, "anomaly_flag": r.anomaly_flag} for r in rows]
        
        if not data_records:
            return {"chart_type": chart_type, "data": {}}

        df = pd.DataFrame(data_records)
        df.columns = [c.lower() for c in df.columns] 
        df['anomaly_flag'] = pd.to_numeric(df['anomaly_flag'], errors='coerce').fillna(1).astype(int)
        
    except Exception as e:
        log.error(f"Database fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Database fetch failed.")

    if chart_type == 'pie':
        final_counts = {group_root: 0 for group_root in CHART_AGGREGATION_MAP.keys()}
        
        for _, group_df in df.groupby(['site', 'equipment_id', 'timestamp']):
            for group_root, priority_list in CHART_AGGREGATION_MAP.items():
                active_sensor = None
                
                for p_feat in priority_list:
                    if any(group_df['datapoint'] == p_feat):
                        active_sensor = p_feat
                        break 

                if active_sensor:
                    sensor_row = group_df[group_df['datapoint'] == active_sensor].iloc[0]
                    if sensor_row['anomaly_flag'] == -1:
                        final_counts[group_root] += 1

        return {
            "chart_type": "pie",
            "data": {
                "feature_anomalies": final_counts,
                "total_anomalies_across_all_features": sum(final_counts.values())
            }
        }

    elif chart_type == 'line':
        historical_output = {}
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        for group_root, priority_list in CHART_AGGREGATION_MAP.items():
            active_physical = None
            for p_feat in priority_list:
                if any(df['datapoint'] == p_feat):
                    active_physical = p_feat
                    break
            
            if active_physical:
                df_feat = df[df['datapoint'] == active_physical].copy()
                df_feat.sort_values('timestamp', inplace=True)
                
                records = []
                for _, row in df_feat.iterrows():
                    records.append({
                        "timestamp": row['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f') + '+00:00',
                        "Anamoly_Flag": str(row['anomaly_flag']),
                        group_root: str(row['value'])
                    })
                historical_output[group_root] = records

        return {"chart_type": "line", "data": {"historical_data": historical_output}}

    return {"chart_type": chart_type, "data": {}}