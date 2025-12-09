import requests
from typing import Optional, List, Dict, Any
import json
from src.bms_ai.logger_config import setup_logger
from fastapi import HTTPException

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

def fetch_data(url: str = "https://ikoncloud.keross.com/bms-express-server/data") -> Dict[str, Any] | List[Dict]:
    API_PAYLOAD = {
        "query": "select * from datapoint_live_monitoring_values36c27828d0b44f1e8a94d962d342e7c2 where site = 'OS01' and system_type = 'AHU' and equipment_id = 'Ahu17' and datapoint IN ('TSu', 'Co2RA', 'FbFAD', 'FbVFD', 'HuAvg1') allow filtering;",
    }
    
    try:
        print(f"Sending POST request to {url} with query body: {API_PAYLOAD}")
        
        response = requests.post(url, json=API_PAYLOAD) 
        
        response.raise_for_status()
        
        return response.json()

    except requests.exceptions.RequestException as req_err:
        log.error(f"An HTTP error occurred during API fetch: {req_err}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from external API: {req_err}")
    except ValueError as val_err:
        log.error(f"Data validation error: {val_err}")
        raise HTTPException(status_code=500, detail=str(val_err))

def fetch_data_with_query(query: str, url: str = "https://ikoncloud.keross.com/bms-express-server/data", wrap_result: bool = True) -> Dict[str, Any] | List[Dict]:
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
        
def fetch_data_from_metadata(
        
    metadata: Dict[str, str], 
    table_name: str = "datapoint_live_monitoring_values",
    building_id: str = "36c27828d0b44f1e8a94d962d342e7c2", 
    url: str = "https://ikoncloud.keross.com/bms-express-server/data",
    wrap_result: Optional[bool] = True
) -> Dict[str, Any] | List[Dict]:
    
    site = metadata.get("site")
    equipment_id = metadata.get("equipment_id")
    system_type = metadata.get("system_type")
    location_table_name = table_name + building_id.replace("-", "").lower()

    if not all([site, equipment_id, system_type]):
        missing_keys = [k for k, v in metadata.items() if not v]
        error_msg = f"Missing required metadata keys: {', '.join(missing_keys)}"
        log.error(error_msg)
        raise ValueError(error_msg)

    query = (
        f"select * from {location_table_name} "
        f"where site = '{site}' "
        f"and system_type = '{system_type}' "
        f"and equipment_id = '{equipment_id}' "
        f"and datapoint IN ('TSu', 'Co2RA', 'FbFAD', 'FbVFD', 'HuAvg1')"
        f"allow filtering;"
    )
    
    log.info(f"Generated query: {query}")

    return fetch_data_with_query(query=query, url=url, wrap_result=wrap_result if wrap_result is not None else True)