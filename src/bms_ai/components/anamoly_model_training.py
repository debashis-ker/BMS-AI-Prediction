import os
import pandas as pd
import warnings
import joblib
from typing import List, Dict, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from fastapi import BackgroundTasks, HTTPException
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.ikon_apis import fetch_and_find_data_points
from src.bms_ai.utils.cassandra_utils import fetch_all_ahu_historical_data

warnings.filterwarnings('ignore')
log = setup_logger(__name__)

TAG_FALLBACK_PRIORITIES = {
    "Co2RA": [
        ["co2", "sensor", "space"],           
        ["co2", "average"],                  
        ["co2"]                              
    ],
    "TSu": [
        ["temp", "supply", "ahu"],           
        ["temp", "air"]                      
    ],
    "FbVFD": [
        ["sensor", "speed", "vfd"],          
        ["sp", "speed", "vfd"],
        ["sensor", "fan", "vfd", "ahu"]      # New Fallback for FbVFDSf/FbVFDSf1
    ],
    "FbFAD": [
        ["sensor", "damper", "outside"]      
    ],
    "HuAvg1": [
        ["humidity", "average"],             
        ["humidity", "return"],              
        ["humidity", "space"],               
        ["humidity", "sensor"],
        ["humidity", "return", "ahu"],       # New Fallback for HumRt
        ["humidity", "supply", "ahu"]        # New Fallback for HuSu
    ]
}

def extract_features_based_on_tags(raw_ikon_list: List[Dict]) -> List[str]:
    final_features = []
    for sensor in raw_ikon_list:
        phys_name = sensor.get('dataPointName')
        tags = set(sensor.get('queryTags', []))
        if not phys_name: continue
        is_valid_sensor = False
        for groups in TAG_FALLBACK_PRIORITIES.values():
            for tag_group in groups:
                if set(tag_group).issubset(tags):
                    is_valid_sensor = True
                    break
            if is_valid_sensor: break
        
        if is_valid_sensor:
            final_features.append(phys_name)
            
    return list(set(final_features))

def data_processing(data_subset, user_target_features):
    df = data_subset.copy()
    
    df["data_received_on"] = pd.to_datetime(df["data_received_on"], errors='coerce')
    df = df.dropna(subset=['data_received_on'])
    df["data_received_on"] = df["data_received_on"].dt.tz_localize(None)
    df = df.set_index("data_received_on").sort_index(ascending=True).reset_index()

    aggregated_scores = df.groupby(['data_received_on', 'asset_code', 'datapoint'])['monitoring_data'].agg('first')
    result_df = aggregated_scores.unstack(level='datapoint').reset_index()
    
    result_df['hour'] = result_df['data_received_on'].dt.hour
    result_df['is_weekend'] = result_df['data_received_on'].dt.dayofweek.isin([5, 6]).astype(int)
    result_df['weekday_name'] = result_df['data_received_on'].dt.day_name()
    
    result_df[['site', 'equipment_id']] = result_df['asset_code'].str.split('_', n=1, expand=True)
    
    try:
        df_clean, available_feature_cols, label_encoders = data_cleaning(result_df, user_target_features)
    except Exception as e:
        log.error(f"Data cleaning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data cleaning error: {e}")

    return df_clean, available_feature_cols, label_encoders

def data_cleaning(df, user_target_features):
    features_to_keep = [feat for feat in user_target_features if feat in df.columns]
    
    categorical_cols = ['site', 'equipment_id', 'weekday_name']
    cols_to_include = ['data_received_on', 'asset_code', 'site', 'equipment_id', 'weekday_name'] + ['hour', 'is_weekend'] + features_to_keep
    cols_to_include = [col for col in cols_to_include if col in df.columns]
    
    df_filtered = df[cols_to_include].copy()
    for col in features_to_keep:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    
    label_encoders = {}
    for cat_col in categorical_cols:
        if cat_col in df_filtered.columns:
            le = LabelEncoder()
            le.fit(df_filtered[cat_col].dropna().unique())
            df_filtered[cat_col + '_encoded'] = df_filtered[cat_col].map(lambda x: le.transform([x])[0] if pd.notna(x) else -1) #type:ignore
            label_encoders[cat_col] = le
    
    constant_cols = df_filtered[features_to_keep].loc[:, df_filtered[features_to_keep].std() == 0].columns.tolist()
    if constant_cols:
        df_filtered.drop(columns=constant_cols, inplace=True)
        features_to_keep = [col for col in features_to_keep if col not in constant_cols]

    return df_filtered, features_to_keep, label_encoders

def model_training(df, available_features, label_encoders, raw_metadata_list):
    """
    Core math logic: Uses tags to determine training parameters dynamically.
    """
    tag_lookup = {item['dataPointName']: item.get('queryTags', []) for item in raw_metadata_list}
    
    TAG_CONTAM_RATES = {
        'temp': 0.04, 'co2': 0.05, 'humidity': 0.04, 'speed': 0.02, 'vfd': 0.02, 'damper': 0.02, 'DEFAULT': 0.01
    }
    
    FEATURE_EQUIP_CONTAM_MAP = {
        'co2': {
            'OS01_Ahu1': 0.05,
            'OS01_Ahu2': 0.05,
            'OS01_Ahu3': 0.03,
            'OS02_Ahu15': 0.05,
            'OS02_Ahu16': 0.02,
            'OS02_Ahu5': 0.05,
            'OS04_Ahu13': 0.02,
            'OS04_Ahu6': 0.05,
            'OS04_Ahu7': 0.05,
            'OS04_Ahu8': 0.05,
            'OS05_Ahu10': 0.05,
            'OS05_Ahu11': 0.02,
            'OS05_Ahu12': 0.05,
            'OS05_Ahu9': 0.05,
            'OS02_Ahu4': 0.05
        },
        'temp': {
            'OS01_Ahu17': 0.04,
            'OS01_Ahu3': 0.04,
            'OS02_Ahu14': 0.04,
            'OS02_Ahu15': 0.04,
            'OS02_Ahu16': 0.04,
            'OS02_Ahu4': 0.04,
            'OS02_Ahu5': 0.04,
            'OS04_Ahu13': 0.04,
            'OS04_Ahu6': 0.04,
            'OS04_Ahu7': 0.04,
            'OS04_Ahu8': 0.04,
            'OS05_Ahu10': 0.04,
            'OS05_Ahu11': 0.04,
            'OS05_Ahu12': 0.04,
            'OS05_Ahu9': 0.04,
            'OS01_Ahu1': 0.04,
            'OS01_Ahu2': 0.04
        },
        'damper': {
            'OS01_Ahu1': 0.02,
            'OS01_Ahu17': 0.02,
            'OS01_Ahu2': 0.02,
            'OS01_Ahu3': 0.02,
            'OS02_Ahu14': 0.02,
            'OS02_Ahu15': 0.02,
            'OS02_Ahu16': 0.02,
            'OS02_Ahu4': 0.02,
            'OS02_Ahu5': 0.02,
            'OS04_Ahu13': 0.02,
            'OS04_Ahu6': 0.02,
            'OS04_Ahu7': 0.02,
            'OS04_Ahu8': 0.02,
            'OS05_Ahu10': 0.02,
            'OS05_Ahu11': 0.02,
            'OS05_Ahu12': 0.02,
            'OS05_Ahu9': 0.02
        },
        'speed': {
            'OS01_Ahu1': 0.02,
            'OS01_Ahu17': 0.02,
            'OS01_Ahu2': 0.02,
            'OS01_Ahu3': 0.02,
            'OS02_Ahu14': 0.02,
            'OS02_Ahu15': 0.02,
            'OS02_Ahu16': 0.02,
            'OS02_Ahu4': 0.02,
            'OS02_Ahu5': 0.02,
            'OS04_Ahu13': 0.02,
            'OS04_Ahu6': 0.02,
            'OS04_Ahu7': 0.02,
            'OS04_Ahu8': 0.02,
            'OS05_Ahu10': 0.02,
            'OS05_Ahu11': 0.02,
            'OS05_Ahu12': 0.02,
            'OS05_Ahu9': 0.02
        },
        'humidity' : {
            'OS01_Ahu1': 0.04,
            'OS01_Ahu2': 0.04,
            'OS01_Ahu3': 0.04,
            'OS02_Ahu4': 0.04,
            'OS02_Ahu5': 0.04,
            'OS02_Ahu15': 0.04,
            'OS02_Ahu16': 0.04,
            'OS04_Ahu13': 0.04,
            'OS04_Ahu6': 0.04,
            'OS04_Ahu7': 0.04,
            'OS04_Ahu8': 0.04,
            'OS05_Ahu10': 0.04,
            'OS05_Ahu11': 0.04,
            'OS05_Ahu12': 0.04,
            'OS05_Ahu9': 0.04,
            'OS01_Ahu17': 0.04,
            'OS02_AHU14': 0.04
        }
    }

    asset_code = df['asset_code'].iloc[0] if not df.empty else "Unknown"
    asset_model_package = {} 

    for feature in available_features:
        sensor_tags = tag_lookup.get(feature, [])
        contam_rate = TAG_CONTAM_RATES['DEFAULT']
        
        for tag in sensor_tags:
            if tag in TAG_CONTAM_RATES:
                contam_rate = TAG_CONTAM_RATES[tag]
            if tag in FEATURE_EQUIP_CONTAM_MAP:
                override = FEATURE_EQUIP_CONTAM_MAP[tag].get(asset_code)
                if override is not None:
                    contam_rate = override
                    break

        feature_cols = [feature, 'hour', 'is_weekend']
        encoded_cols = [f'{c}_encoded' for c in ['site', 'equipment_id', 'weekday_name'] if f'{c}_encoded' in df.columns]
        training_cols = feature_cols + encoded_cols
        
        X_df = df[training_cols].dropna(subset=[feature])
        if len(X_df) < 50: continue 

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_df)
        
        model = IsolationForest(n_estimators=100, contamination=contam_rate, random_state=42)
        model.fit(X_scaled)

        asset_model_package[feature] = {
            'model': model,
            'scaler': scaler,
            'feature_cols': training_cols,
            'label_encoders': label_encoders,
            'meta': {"tags": sensor_tags, "rate": contam_rate}
        }
            
    return asset_model_package

def model_saving(raw_data, discovered_features, raw_metadata_list):
    log.info("--- Starting Dynamic Model Training Workflow ---")
    df_ahu_raw = raw_data[raw_data['system_type'] == "AHU"].copy()
    
    master_model_package = {}
    target_iterator = df_ahu_raw[['site', 'equipment_name']].drop_duplicates().itertuples(index=False)
    
    for site, equipment_name in target_iterator:
        compound_key = f"{site}_{equipment_name}" 
        data_subset = df_ahu_raw[(df_ahu_raw['site'] == site) & (df_ahu_raw['equipment_name'] == equipment_name)].copy()
        if data_subset.empty: continue
            
        data_subset['asset_code'] = compound_key 
        
        try:
            df_clean, features, encoders = data_processing(data_subset, discovered_features)
        except Exception as e:
            log.error(f"Data processing failed for {compound_key}: {e}")
            continue

        try:
            asset_models = model_training(df_clean, features, encoders, raw_metadata_list)
        except Exception as e:
            log.error(f"Model training failed for {compound_key}: {e}")
            continue

        for feature, package in asset_models.items():
            if feature not in master_model_package:
                master_model_package[feature] = {}
            master_model_package[feature][compound_key] = package 

    for feature, assets in master_model_package.items():
        output_model_path = f"artifacts/test_anamoly_models/{feature}_model.joblib"
        joblib.dump(assets, output_model_path)

def training_pipeline(
    background_tasks: BackgroundTasks,
    building_id: str,
    ticket: str,
    software_id: str,
    account_id: str,
    search_tag_groups: List[List[str]],
    ticket_type : Optional[str]
):
    try:
        log.info(f"Discovering active sensors for building {building_id}...")
        
        raw_ikon_list = fetch_and_find_data_points(
            floor_id=None,
            equipment_id="",
            building_id=building_id, 
            ticket=ticket, 
            software_id=software_id,
            account_id=account_id, 
            search_tag_groups=search_tag_groups,
            env="prod",
            ticket_type=ticket_type
        )

        if not raw_ikon_list:
            raise HTTPException(status_code=503, detail=f"Failed to fetch datapoints from Ikon API")
    
        discovered_features = extract_features_based_on_tags(raw_ikon_list)

        if not discovered_features:
            raise HTTPException(status_code=404, detail="No suitable sensors found even with tag fallbacks.")

        log.info(f"Fetching training data for discovered sensors: {discovered_features}")
        raw_records = fetch_all_ahu_historical_data(building_id=building_id)
        if not raw_records:
            raise HTTPException(status_code=404, detail="No historical records found for training.")

        df_raw = pd.DataFrame(raw_records)

        if df_raw.empty:
            log.error("Empty DataFrame")
            raise HTTPException(status_code=404, detail="Failed to construct DataFrame due to empty data")

        background_tasks.add_task(
            model_saving, 
            raw_data=df_raw, 
            discovered_features=discovered_features,
            raw_metadata_list=raw_ikon_list
        )

        previous_time_duration = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S.%f%z')
        current_time_duration = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S.%f%z')

        return {
            "status": "Success",
            "building_id": building_id,
            "details": f"Model Saving triggered on data between {previous_time_duration} to {current_time_duration} in background tasks.",
        }
    except Exception as e:
        log.error(f"Training endpoint execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))