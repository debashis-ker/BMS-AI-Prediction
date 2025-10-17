from typing import Dict, Any, List, Optional
import os
import json
import requests
from src.bms_ai.logger_config import setup_logger

log = setup_logger(__name__)

def get_my_instances_v2(ticket: str, process_name: str, software_id: str, account_id: str,
                         predefined_filters: Optional[Dict] = None, 
                         process_variable_filters: Optional[Dict] = None,
                         task_variable_filters: Optional[Dict] = None, 
                         mongo_where_clause: Optional[str] = None,
                         projections: Optional[List[str]] = None, 
                         all_instances: bool = False) -> Optional[Dict]:
    """
    
    Args:
        ticket: Authentication ticket
        process_name: Name of the process
        account_id: Account identifier
        predefined_filters: Optional predefined filters
        process_variable_filters: Optional process variable filters
        task_variable_filters: Optional task variable filters
        mongo_where_clause: Optional MongoDB where clause
        projections: Optional list of fields to project
        all_instances: Whether to fetch all instances
        
    Returns:
        Response JSON from the IKON service or None if request fails
        
    Raises:
        EnvironmentError: If IKON_BASE_URL or SOFTWARE_ID is not set
    """
    ikon_base_url = os.getenv("IKON_BASE_URL")
    if not software_id:
        software_id = os.getenv("SOFTWARE_ID")
    
    if not ikon_base_url:
        log.error("IKON_BASE_URL is not set in environment variables.")
        raise EnvironmentError("IKON_BASE_URL is not set in environment variables.")
    
    if not software_id:
        log.error("SOFTWARE_ID is not set in environment variables.")
        raise EnvironmentError("SOFTWARE_ID is not set in environment variables.")

    url = f"{ikon_base_url}/rest?inZip=false&outZip=true&inFormat=freejson&outFormat=freejson" \
          f"&service=processRuntimeService&operation=getMyInstancesV2" \
          f"&locale=en_US&activeAccountId={account_id}&softwareId={software_id}&ticket={ticket}"

    log.debug(f"Request URL: {url}")

    arguments_list = [
        process_name,
        account_id,
        predefined_filters,
        process_variable_filters,
        task_variable_filters,
        mongo_where_clause,
        projections or [],
        all_instances
    ]

    data = {
        "arguments": json.dumps(arguments_list)
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    log.debug(f"Request arguments: {data}")
    
    try:
        log.info(f"Sending POST request to IKON service")
        response = requests.post(url, data=data, headers=headers, timeout=30)
        response.raise_for_status()
        log.info(f"Successfully fetched instances for process: {process_name}")
        
        response_data = response.json()
        log.debug(f"Response data type: {type(response_data)}")
        return response_data
        
    except requests.exceptions.Timeout as e:
        log.error(f"Request timeout: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        log.error(f"HTTP error: {e}, Status code: {response.status_code}, Response: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        log.error(f"Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse JSON response: {e}, Response text: {response.text}")
        return None
    except Exception as e:
        log.error(f"Unexpected error in get_my_instances_v2: {e}", exc_info=True)
        return None