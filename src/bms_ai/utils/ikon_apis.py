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
                         projections: Optional[List[str]] = ["Data"], 
                         all_instances: bool = False,
                         env : Optional[str] = "dev") -> Optional[Dict]:
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
        env: Environment to use ("dev" or "prod")
    Returns:
        Response JSON from the IKON service or None if request fails
        
    Raises:
        EnvironmentError: If IKON_BASE_URL or SOFTWARE_ID is not set
    """
    if env == "prod":
        ikon_base_url = os.getenv("IKON_BASE_URL_PROD")
    elif env == "dev":
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


def get_building_hierarchy_data(
    building_id: str,
    ticket: str,
    software_id: str,
    account_id: str,
    env: Optional[str] = "dev"
) -> Dict[str, Any]:
    """
    Fetches and merges building hierarchy data from the BMS Building Data Hierarchy process.
    
    Args:
        building_id: The ID of the building to fetch hierarchy data for
        ticket: Authentication ticket
        software_id: Software ID
        account_id: Account ID
        env: Environment (dev or prod)
        
    Returns:
        A dictionary containing buildingId and merged hierarchyData
    """
    log.info(f"Fetching building hierarchy data for buildingId: {building_id}")
    
    try:
        building_hierarchy_data_raw = get_my_instances_v2(
            ticket=ticket,
            process_name="BMS Building Data Hierarchy",
            software_id=software_id,
            account_id=account_id,
            predefined_filters={"taskName": "Building Hierarchy Task"},
            process_variable_filters={"buildingId": building_id},
            env=env
        )
        
        log.debug(f"Building hierarchy data raw: {building_hierarchy_data_raw}")
        
        if not building_hierarchy_data_raw or not isinstance(building_hierarchy_data_raw, list) or len(building_hierarchy_data_raw) == 0:
            log.warning("No building hierarchy data found.")
            return {
                "buildingId": building_id,
                "hierarchyData": {}
            }
        
        combined_hierarchy_data = {}
        
        for current_instance in building_hierarchy_data_raw:
            instance_hierarchy = {}
            if isinstance(current_instance, dict):
                data = current_instance.get('data', {})
                if isinstance(data, dict):
                    instance_hierarchy = data.get('hierarchyData', {})
            
            if isinstance(instance_hierarchy, dict):
                combined_hierarchy_data.update(instance_hierarchy)
        
        final_result = {
            "buildingId": building_id,
            "hierarchyData": combined_hierarchy_data
        }
        
        log.info(f"Successfully merged hierarchy data for building: {building_id}")
        return final_result
        
    except Exception as error:
        log.error(f"Error fetching or processing building hierarchy data: {error}", exc_info=True)
        return {
            "buildingId": building_id,
            "hierarchyData": {}
        }


def find_data_points_by_floor_and_tags(
    hierarchy_data: Dict[str, Any],
    floor_name: str,
    target_equipment_id: str,
    search_tag_groups: List[List[str]],
    system_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Finds data points within a specific floor and equipment that match Haystack tags.
    
    Args:
        hierarchy_data: The object containing the site and system hierarchy
        floor_name: The name of the floor/site to search within
        target_equipment_id: The ID/key of the specific equipment
        search_tag_groups: An array of arrays of Haystack tags to search for
        system_type: Optional system type (e.g., "FCU", "AHU")
        
    Returns:
        List of matched data point objects
    """
    results = []
    
    # Initial Validations
    if not search_tag_groups or len(search_tag_groups) == 0 or any(not group or len(group) == 0 for group in search_tag_groups):
        log.warning("Search tag groups are empty or invalid. Returning no results.")
        return []
    
    if not hierarchy_data or not isinstance(hierarchy_data, dict):
        log.error("Hierarchy data is invalid.")
        return []
    
    if not floor_name or not isinstance(floor_name, str):
        log.warning("Floor name is invalid. Returning no results.")
        return []
    
    if target_equipment_id and not isinstance(target_equipment_id, str):
        log.warning("Target Equipment ID is invalid when provided.")
        return []
    
    if system_type and not isinstance(system_type, str):
        log.warning("System Type is invalid when provided.")
        return []
    
    # Floor Selection
    log.debug(f"Looking for floor: '{floor_name}' in hierarchy data")
    log.debug(f"Available keys in hierarchy data: {list(hierarchy_data.keys())}")
    
    site_data_to_search = None
    if floor_name in hierarchy_data:
        site_data_to_search = hierarchy_data[floor_name]
        log.debug(f"Floor found as direct key: {floor_name}")
    else:
        for key, value in hierarchy_data.items():
            if isinstance(value, dict) and value.get('site') == floor_name:
                site_data_to_search = value
                log.debug(f"Floor found via 'site' property in key: {key}")
                break
    
    if not site_data_to_search:
        log.warning(f'Floor "{floor_name}" not found in hierarchyData.')
        log.warning(f'Available floors: {[v.get("site") if isinstance(v, dict) else None for v in hierarchy_data.values()]}')
        return []
    
    current_site_name = site_data_to_search.get('site', floor_name)
    
    # Helper for processing data points
    def process_single_equipment_data_points(
        equipment_data: Dict[str, Any],
        equipment_key: str,
        path_system_name: str,
        path_system_asset_code: Optional[str]
    ):
        equipment_actual_name = equipment_data.get('equipmentName', equipment_key)
        equipment_id = equipment_data.get('equipmentId', equipment_key)
        equipment_asset_code = equipment_data.get('assetCode')
        
        for current_search_tag_array in search_tag_groups:
            if not current_search_tag_array or len(current_search_tag_array) == 0:
                continue
            
            for dp in equipment_data.get('dataPoints', []):
                dp_tags = dp.get('haystackTags', [])
                dp_tags_set = set(dp_tags)
                match_score = sum(1 for tag in current_search_tag_array if tag in dp_tags_set)
                
                if match_score == len(current_search_tag_array) and len(dp_tags) == len(current_search_tag_array):
                    results.append({
                        'site': current_site_name,
                        'system_name': path_system_name,
                        'system_assetCode': path_system_asset_code,
                        'subsystem_name': equipment_actual_name,
                        'subsystem_assetCode': equipment_asset_code,
                        'dataPointName': dp.get('dataPointName'),
                        'matched_datapoint_tags': dp_tags,
                        'systemId': dp.get('systemId'),
                        'equipmentId': equipment_id,
                        'matchScore': match_score,
                        'queryTags': current_search_tag_array
                    })
    
    # Recursive helper to find specific equipment
    def find_and_process_specific_equipment_recursive(
        current_level_data: Dict[str, Any],
        top_level_system_name: str,
        top_level_system_asset_code: Optional[str]
    ) -> bool:
        for key_in_level, item_data in current_level_data.items():
            if key_in_level in ['site', 'system', 'subsystem', 'assetCode', 'equipmentName']:
                continue
            
            if isinstance(item_data, dict):
                if key_in_level == target_equipment_id and isinstance(item_data.get('dataPoints'), list):
                    process_single_equipment_data_points(
                        item_data, key_in_level, top_level_system_name, top_level_system_asset_code
                    )
                    return True
                
                if find_and_process_specific_equipment_recursive(
                    item_data, top_level_system_name, top_level_system_asset_code
                ):
                    return True
        return False
    
    # Recursive helper for all equipment under system type
    def process_all_equipment_under_system_recursive(
        current_level_data: Dict[str, Any],
        top_level_system_name: str,
        top_level_system_asset_code: Optional[str]
    ):
        for key_in_level, item_data in current_level_data.items():
            if key_in_level in ['site', 'system', 'subsystem', 'assetCode']:
                continue
            
            if isinstance(item_data, dict):
                if (isinstance(item_data.get('dataPoints'), list) and 
                    (item_data.get('equipmentName') or item_data.get('equipmentId') or 'assetCode' in item_data)):
                    process_single_equipment_data_points(
                        item_data, key_in_level, top_level_system_name, top_level_system_asset_code
                    )
                else:
                    process_all_equipment_under_system_recursive(
                        item_data, top_level_system_name, top_level_system_asset_code
                    )
    
    # Recursive helper for all equipment on floor
    def process_all_equipment_on_floor_recursive(
        current_level_data: Dict[str, Any],
        parent_system_name: Optional[str],
        parent_system_asset_code: Optional[str]
    ):
        for key_in_level, item_data in current_level_data.items():
            if key_in_level == 'site':
                continue
            
            if isinstance(item_data, dict):
                system_name = item_data.get('system', key_in_level or parent_system_name)
                system_asset_code = item_data.get('assetCode', parent_system_asset_code)
                
                if (isinstance(item_data.get('dataPoints'), list) and 
                    (item_data.get('equipmentName') or item_data.get('equipmentId') or 'assetCode' in item_data)):
                    process_single_equipment_data_points(
                        item_data, key_in_level, parent_system_name or system_name, 
                        parent_system_asset_code or system_asset_code
                    )
                else:
                    process_all_equipment_on_floor_recursive(
                        item_data, system_name, system_asset_code
                    )
    
    if target_equipment_id and target_equipment_id.strip():
        for system_category_key, system_category_data in site_data_to_search.items():
            if system_category_key == 'site':
                continue
            
            if isinstance(system_category_data, dict):
                actual_system_category_name = system_category_data.get('system', system_category_key)
                system_category_asset_code = system_category_data.get('assetCode')
                
                find_and_process_specific_equipment_recursive(
                    system_category_data, actual_system_category_name, system_category_asset_code
                )
    
    elif system_type and system_type.strip():
        for system_category_key, system_category_data in site_data_to_search.items():
            if system_category_key == 'site':
                continue
            
            if isinstance(system_category_data, dict):
                actual_system_category_name = system_category_data.get('system', system_category_key)
                system_category_asset_code = system_category_data.get('assetCode')
                
                if actual_system_category_name == system_type:
                    process_all_equipment_under_system_recursive(
                        system_category_data, actual_system_category_name, system_category_asset_code
                    )
    
    else:
        process_all_equipment_on_floor_recursive(site_data_to_search, None, None)
    
    seen = set()
    unique_results = []
    
    for result in results:
        key = (
            result['dataPointName'],
            result['site'],
            result['system_name'],
            result['subsystem_name'],
            result['subsystem_assetCode'],
            tuple(result['queryTags'])
        )
        if key not in seen:
            seen.add(key)
            unique_results.append(result)
    
    unique_results.sort(key=lambda x: (-x['matchScore'], x['dataPointName']))
    
    return unique_results


def fetch_and_find_data_points(
    building_id: str,
    floor_id: Optional[str],
    equipment_id: str,
    search_tag_groups: List[List[str]],
    ticket: str,
    software_id: str,
    account_id: str,
    system_type: Optional[str] = None,
    env: Optional[str] = "dev"
) -> List[Dict[str, Any]]:
    """
    Wrapper function to fetch hierarchy data and find data points.
    
    Args:
        building_id: The ID of the building
        floor_id: The ID of the floor to search within (if None, searches all floors)
        equipment_id: The ID/key of the specific equipment
        search_tag_groups: Array of arrays of Haystack tags
        ticket: Authentication ticket
        software_id: Software ID
        account_id: Account ID
        system_type: Optional system type
        
    Returns:
        List of matched data points
    """
    try:
        process_variable_filter = {'buildingId': building_id} if building_id else None
        
        building_associated_data = get_my_instances_v2(
            ticket=ticket,
            process_name="Building To Bacnet Association",
            software_id=software_id,
            account_id=account_id,
            predefined_filters={'taskName': 'Building Association'},
            process_variable_filters=process_variable_filter,
            env=env
        )
        
        log.debug(f"Building association data: {building_associated_data}")
        
        if not building_associated_data or len(building_associated_data) == 0:
            log.warning("No building association data found")
            return []
        
        first_instance_data = building_associated_data[0].get('data', {})
        bacnet_association_data = first_instance_data.get('floorAssociations', [])
        
        log.debug(f"Floor associations data: {bacnet_association_data}")
        
        floor_names_to_search = []
        
        if floor_id:
            floor_association = None
            for association in bacnet_association_data:
                if association.get('floorId') == floor_id:
                    floor_association = association
                    break
            
            if not floor_association or not floor_association.get('bacnetSite'):
                log.warning(f"No association or bacnetSite found for floorId: {floor_id}")
                log.warning(f"Available floor IDs: {[a.get('floorId') for a in bacnet_association_data]}")
                return []
            
            floor_names_to_search = [floor_association['bacnetSite']]
            log.info(f"Resolved floor name: {floor_names_to_search[0]} for floorId: {floor_id}")
        else:
            floor_names_to_search = [
                assoc.get('bacnetSite') for assoc in bacnet_association_data 
                if assoc.get('bacnetSite')
            ]
            log.info(f"No floor_id provided. Searching across all {len(floor_names_to_search)} floors: {floor_names_to_search}")
        
        tree_data = get_building_hierarchy_data(
            building_id=building_id,
            ticket=ticket,
            software_id=software_id,
            account_id=account_id,
            env=env
        )
        
        hierarchy_data = tree_data.get('hierarchyData') if tree_data else None
        
        if not hierarchy_data:
            log.warning("Hierarchy data is empty or invalid after fetching.")
            return []
        
        log.debug(f"Hierarchy data keys: {list(hierarchy_data.keys()) if isinstance(hierarchy_data, dict) else 'Not a dict'}")
        log.info(f"Searching for floor(s): {floor_names_to_search}, equipment: {equipment_id}, system_type: {system_type}")
        log.info(f"Search tag groups: {search_tag_groups}")
        
        all_results = []
        for floor_name in floor_names_to_search:
            floor_results = find_data_points_by_floor_and_tags(
                hierarchy_data,
                floor_name,
                equipment_id,
                search_tag_groups,
                system_type
            )
            all_results.extend(floor_results)
        
        log.info(f"Found {len(all_results)} matching data points across {len(floor_names_to_search)} floor(s)")
        return all_results
    
    except Exception as error:
        log.error(f"Error fetching or processing hierarchy data: {error}", exc_info=True)
        return []