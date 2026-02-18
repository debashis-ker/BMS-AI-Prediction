import os
import pandas as pd
import warnings
import joblib
import requests
import numpy as np
import itertools
from typing import List, Dict, Optional
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.preprocessing import MinMaxScaler
from fastapi import BackgroundTasks, HTTPException
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.ikon_apis import fetch_and_find_data_points
from src.bms_ai.utils.cassandra_utils import DEFAULT_BUILDING_ID, fetch_all_ahu_historical_data, fetch_mahu_data

warnings.filterwarnings('ignore')
log = setup_logger(__name__)

TAG_FALLBACK_PRIORITIES = {
    "Co2RA": [["co2", "sensor", "space"], ["co2", "average"], ["co2"]],
    "TSu": [["temp", "supply", "ahu"], ["temp", "air"]],
    "FbVFD": [["sensor", "speed", "vfd"], ["sp", "speed", "vfd"], ["sensor", "fan", "vfd", "ahu"],["point", "ahu", "airHandlingEquip", "sensor", "fan", "vfd", "supply"]],
    "FbFAD": [["sensor", "damper", "outside"]],
    "HuAvg1": [["humidity", "average"], ["humidity", "return"], ["humidity", "space"], ["humidity", "sensor"], ["humidity", "return", "ahu"], ["humidity", "supply", "ahu"]]
}

def add_open_meteo_weather(df, lat=25.33, lon=55.39):
    if df.empty: return df
    df['data_received_on'] = pd.to_datetime(df['data_received_on'])
    start_date = df['data_received_on'].min().strftime('%Y-%m-%d')
    end_date = df['data_received_on'].max().strftime('%Y-%m-%d')
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date, "hourly": "temperature_2m,relative_humidity_2m", "timezone": "auto"}
    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()
        weather_df = pd.DataFrame({'ds_weather': pd.to_datetime(data['hourly']['time']), 'outside_temp': data['hourly']['temperature_2m'], 'outside_humidity': data['hourly']['relative_humidity_2m']})
        df = df.sort_values('data_received_on')
        combined_df = pd.merge_asof(df, weather_df.sort_values('ds_weather'), left_on='data_received_on', right_on='ds_weather', direction='nearest')
        return combined_df.drop(columns=['ds_weather'])
    except Exception as e:
        log.error(f"Weather fetch failed: {e}")
        df['outside_temp'] = 0.0
        df['outside_humidity'] = 0.0
        return df

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
        if is_valid_sensor: final_features.append(phys_name)
    return list(set(final_features))

def data_processing(data_subset, user_target_features):
    df = data_subset.copy()
    df["data_received_on"] = pd.to_datetime(df["data_received_on"], errors='coerce')
    df = df.dropna(subset=['data_received_on'])
    df["data_received_on"] = df["data_received_on"].dt.tz_localize(None)
    df = add_open_meteo_weather(df)
    aggregated_scores = df.groupby(['data_received_on', 'asset_code', 'datapoint'])['monitoring_data'].agg('first')
    result_df = aggregated_scores.unstack(level='datapoint').reset_index()
    weather_cols = df[['data_received_on', 'outside_temp', 'outside_humidity']].drop_duplicates()
    result_df = pd.merge(result_df, weather_cols, on='data_received_on', how='left')
    result_df[['site', 'equipment_id']] = result_df['asset_code'].str.split('_', n=1, expand=True)
    df_clean, available_feature_cols = data_cleaning(result_df, user_target_features)
    return df_clean, available_feature_cols

def data_cleaning(df, user_target_features):
    features_to_keep = [feat for feat in user_target_features if feat in df.columns]
    cols_to_include = ['data_received_on', 'asset_code', 'outside_temp', 'outside_humidity'] + features_to_keep
    df_filtered = df[cols_to_include].copy()
    for col in features_to_keep + ['outside_temp', 'outside_humidity']:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['outside_temp', 'outside_humidity'])
    return df_filtered, features_to_keep

def tune_best_parameters(train_data):
    if train_data['y'].nunique() <= 1:
        return {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'}
    param_grid = {'changepoint_prior_scale': [0.01, 0.05, 0.1], 'seasonality_prior_scale': [0.1, 1.0, 10.0], 'seasonality_mode': ['additive', 'multiplicative'], 'interval_width': [0.95]}
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []
    for params in all_params:
        m = Prophet(**params)
        m.add_regressor('outside_temp')
        m.add_regressor('outside_humidity')
        m.fit(train_data)
        try:
            df_cv = cross_validation(m, initial='7 days', period='1 days', horizon='1 day', parallel="processes")
            df_p = performance_metrics(df_cv, metrics=['rmse'])
            rmses.append(df_p['rmse'].mean())
        except:
            rmses.append(float('inf'))
    return all_params[np.argmin(rmses)]

def model_training(df, available_features):
    asset_model_package = {}
    regressors = ['outside_temp', 'outside_humidity']
    for feature in available_features:
        prep_df = df[['data_received_on', feature] + regressors].rename(columns={'data_received_on': 'ds', feature: 'y'}).dropna()
        if len(prep_df) < 50: continue
        scaler = MinMaxScaler()
        prep_df[regressors] = scaler.fit_transform(prep_df[regressors])
        # best_params = tune_best_parameters(prep_df)
        best_params = {'changepoint_prior_scale': 0.01, 'interval_width': 0.95, 'seasonality_mode': 'additive', 'seasonality_prior_scale': 10.0}
        model = Prophet(**best_params)
        for reg in regressors: model.add_regressor(reg)
        model.fit(prep_df)
        asset_model_package[feature] = {'model': model, 'scaler': scaler, 'best_params': best_params, 'feature_cols': regressors}
    return asset_model_package

def model_saving(raw_data, discovered_features):
    accepted_types = ["AHU", "MAHU1"]
    df_ahu_raw = raw_data[raw_data['system_type'].isin(accepted_types)].copy()
    
    master_model_package = {}
    target_iterator = df_ahu_raw[['site', 'equipment_name']].drop_duplicates().itertuples(index=False)
    for site, equipment_name in target_iterator:
        compound_key = f"{site}_{equipment_name}"
        data_subset = df_ahu_raw[(df_ahu_raw['site'] == site) & (df_ahu_raw['equipment_name'] == equipment_name)].copy()
        if data_subset.empty: continue
        data_subset['asset_code'] = compound_key
        try:
            df_clean, features = data_processing(data_subset, discovered_features)
            asset_models = model_training(df_clean, features)
            for feature, package in asset_models.items():
                if feature not in master_model_package: master_model_package[feature] = {}
                master_model_package[feature][compound_key] = package
        except Exception as e:
            log.error(f"Workflow failed for {compound_key}: {e}")
            continue
    for feature, assets in master_model_package.items():
        output_model_path = f"artifacts/Anamoly_Prophet_Models/Model_Training_Pipeline/{feature}_package.joblib"
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        joblib.dump(assets, output_model_path)

def training_pipeline(background_tasks: BackgroundTasks, building_id: str, ticket: str, software_id: str, account_id: str, search_tag_groups: List[List[str]], ticket_type: Optional[str]):
    try:
        raw_ikon_list = fetch_and_find_data_points(floor_id=None, equipment_id="", building_id=building_id, ticket=ticket, software_id=software_id, account_id=account_id, search_tag_groups=search_tag_groups, env="prod", ticket_type=ticket_type)
        if not raw_ikon_list: raise HTTPException(status_code=503, detail="Ikon fetch failed")
        discovered_features = extract_features_based_on_tags(raw_ikon_list)
        if not discovered_features: raise HTTPException(status_code=404, detail="No sensors found")
        raw_records = fetch_all_ahu_historical_data(building_id=building_id)
        mahu_records = fetch_mahu_data(building_id=building_id or DEFAULT_BUILDING_ID, equipment_id="AhuMkp1")
        if mahu_records:
            for record in mahu_records:
                record['site'] = "OS02"
                record['equipment_name'] = "AhuMkp1"
                record['system_type'] = "MAHU1" # Set to actual MAHU type
            
            if not raw_records:
                raw_records = mahu_records
            else:
                raw_records.extend(mahu_records)

        if not raw_records: raise HTTPException(status_code=404, detail="No historical records")
        df_raw = pd.DataFrame(raw_records)
        background_tasks.add_task(model_saving, raw_data=df_raw, discovered_features=discovered_features)
        return {"status": "Success", "building_id": building_id}
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))