import requests
from typing import List, Dict, Any
import json
from src.bms_ai.logger_config import setup_logger
from fastapi import HTTPException
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

FIXED_SYSTEM_TYPE = "AHU"
DEFAULT_BUILDING_ID = "36c27828-d0b4-4f1e-8a94-d962d342e7c2"
ALL_AVAILABLE_FEATURES = ['TSu', 'Co2RA', 'FbFAD', 'FbVFD', 'HuAvg1', 'TempSu', 'Co2Avg', 'HuR1', 'HuRt']
FEATURE_FALLBACKS = {
    'TSu': ['TSu', 'TempSu'],
    'Co2RA': ['Co2RA', 'Co2Avg', 'RtCo2'],
    'HuAvg1': ['HuAvg1', 'HuRt', 'HuR1', 'HumRt', 'HuSu'], # Added HumRt, HuSu
    'FbFAD': ['FbFAD'],
    'FbVFD': ['FbVFD', 'FbVFDSf', 'FbVFDSf1'] # Added FbVFDSf, FbVFDSf1 
}
QUERY_FEATURES = set(ALL_AVAILABLE_FEATURES)
for fallbacks in FEATURE_FALLBACKS.values():
    QUERY_FEATURES.update(fallbacks)

log = setup_logger(__name__)

def get_metadata(raw_data: List[Dict[str, Any]]) -> Dict[str, str]:
    if not raw_data:
        return {"site": "", "equipment_id": "", "system_type": "", "asset_code": ""}
        
    first_record = raw_data[0]
    return {
        "site": first_record.get("site", ""),
        "equipment_id": first_record.get("equipment_id", ""),
        "system_type": first_record.get("system_type", ""),
        "asset_code": first_record.get("asset_code", ""),
    }

def fetch_data(url: str = f"{os.getenv('IKON_BASE_URL_PROD')}/bms-express-server/data") -> Dict[str, Any] | List[Dict]:
    API_PAYLOAD = {
        "query": "select * from datapoint_live_monitoring_values36c27828d0b44f1e8a94d962d342e7c2 where site = 'OS01' and system_type = 'AHU' and equipment_id = 'Ahu17' and datapoint IN ('TSu', 'Co2RA', 'FbFAD', 'FbVFD', 'HuAvg1') allow filtering;",
    }
    
    try:
        response = requests.post(url, json=API_PAYLOAD) 
        
        response.raise_for_status()
        
        return response.json()

    except requests.exceptions.RequestException as req_err:
        log.error(f"An HTTP error occurred during API fetch: {req_err}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from external API: {req_err}")
    except ValueError as val_err:
        log.error(f"Data validation error: {val_err}")
        raise HTTPException(status_code=500, detail=str(val_err))

def fetch_data_with_query(query: str, url: str = f"{os.getenv('IKON_BASE_URL_PROD')}/bms-express-server/data", wrap_result: bool = True) -> Dict[str, Any] | List[Dict]:
    API_PAYLOAD = {"query": query}
    
    try:
        response = requests.post(url, json=API_PAYLOAD) 
        response.raise_for_status()
        
        raw_api_response = response.json() 
        
        if not wrap_result:
            data_list = raw_api_response.get('queryResponse')
            
            if not isinstance(data_list, list):
                error_detail = raw_api_response.get("message") or raw_api_response.get("error") or str(raw_api_response)
                log.error(f"API returned data wrapper error: {error_detail}")
                raise HTTPException(status_code=500, detail=f"Data Extraction Error: 'queryResponse' not found or is not a list. Details: {error_detail}")
            
            return data_list

        return raw_api_response
        
    except requests.exceptions.HTTPError as http_err:
        log.error(f"HTTP Error during API fetch: {http_err}. Response: {response.text}")
        try:
            error_response = response.json()
            error_msg = error_response.get("error") or error_response.get("detail") or "Unknown HTTP error."
        except json.JSONDecodeError:
            error_msg = response.text
        
        raise HTTPException(status_code=response.status_code, detail=f"External API HTTP Error: {error_msg}")
        
    except Exception as e:
        log.error(f"An unexpected error occurred during fetching: {e}")
        raise HTTPException(status_code=500, detail=f"API Request failed: {str(e)}")
   
def fetch_all_ahu_data(
    building_id: str = DEFAULT_BUILDING_ID,
    url: str = f"{os.getenv('IKON_BASE_URL_PROD')}/bms-express-server/data",
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
    
def fetch_mahu_data(
    building_id: str = DEFAULT_BUILDING_ID,
    url: str = f"{os.getenv('IKON_BASE_URL_PROD')}/bms-express-server/data",
    equipment_id: str = "AhuMkp1"
) -> List[Dict]:
    """Fetches ALL historical data for AHUs in a single API call."""
    cleaned_id = building_id.replace("-", "").lower()
    location_table_name = f"datapoint_live_monitoring_values{cleaned_id}"
    
    datapoint_list = ', '.join([f"'{f}'" for f in QUERY_FEATURES])
    
    query = (
        f"select * from {location_table_name} "
        f"where equipment_id = '{equipment_id}' "
        f"and datapoint IN ({datapoint_list}) "
        f"allow filtering;"
    )

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

def fetch_cassandra_data(
        building_id: str = DEFAULT_BUILDING_ID,
        url: str = f"{os.getenv('IKON_BASE_URL_PROD')}/bms-express-server/data",
        zone: str = "",
        equipment_id: str = "",
        datapoints: List[str] = [],
        system_type: str = "AHU",
        from_date: str = "",
        to_date: str = ""
) -> List[Dict]:
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

    log.debug(f"Cassandra Query: {query}")

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

def fetch_all_ahu_historical_data(
    building_id: str = DEFAULT_BUILDING_ID,
    url: str = f"{os.getenv('IKON_BASE_URL_PROD')}/bms-express-server/data"
) -> List[Dict]:
    """Fetches ALL historical data for AHUs in a single API call."""
    cleaned_id = building_id.replace("-", "").lower()
    location_table_name = f"datapoint_live_monitoring_{cleaned_id}"
    
    datapoint_list = ', '.join([f"'{f}'" for f in QUERY_FEATURES])
    # previous_week_day_in_UTC = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S.%f%z')

    query = (
        f"select * from {location_table_name} "
        f"where system_type = '{FIXED_SYSTEM_TYPE}' "
        f"and datapoint IN ({datapoint_list}) "
        # f"and data_received_on >= '{previous_week_day_in_UTC}' " 
        f"allow filtering;"
    )

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