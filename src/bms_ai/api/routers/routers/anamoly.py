import time
import warnings
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends
from cassandra.cluster import Session
from dotenv import load_dotenv
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.api.dependencies import get_cassandra_session
from src.bms_ai.components.anamoly_functions import *

load_dotenv()
log = setup_logger(__name__)
warnings.filterwarnings('ignore')

router = APIRouter(prefix="/anomalies",tags=["Anomalies"])

class DataQueryRequest(BaseModel):
    """Schema for querying historical or anomaly data."""
    feature: Optional[List[str]] = Field(None, description="List of datapoints (features) to query.")
    site: Optional[str] = Field(None, description="Site/Zone of the equipment.")
    equipment_id: Optional[str] = Field(None, description="Equipment ID.")
    system_type: Optional[str] = Field(None, description="System type.")
    building_id: str = Field(..., description="Building ID used to derive the table name.")
    limit: Optional[int] = Field(1000, description="Maximum number of rows to return.")
    start_date: Optional[str] = Field(None, description="Start timestamp for range filtering.")
    end_date: Optional[str] = Field(None, description="End timestamp for range filtering.")
    floor_id: Optional[str] = Field(None, description="Building ID (optional)")
    search_tag_groups: Optional[List[List[str]]] = Field(None, description="Search Tags for datapoint fetching")
    ticket: Optional[str] = Field(None, description="Ticket ID for Ikon API lookup (required if feature is None).")
    account_id: Optional[str] = Field(None, description="Account ID for Ikon API lookup (required if feature is None).")
    software_id: Optional[str] = Field(None, description="Software ID for Ikon API lookup (required if feature is None).")

class DatapointFetchingRequest(BaseModel):
    ticket: str = Field(..., description="Ticket ID for datapoint fetching")
    floor_id: Optional[str] = Field(None, description="Building ID (optional)")
    account_id: str = Field(..., description="Account ID  for datapoint fetching")
    software_id: str = Field(..., description="Software ID for datapoint fetching")
    building_id: Optional[str] = Field(None, description="Building ID (optional)")
    system_type: str = Field(..., description="System type (e.g., 'AHU', 'RTU')")
    equipment_id: str = Field("Ahu1", description="Equipment ID for datapoint fetching")
    search_tag_groups: List[List[str]] = Field(..., description="Search Tags for datapoint fetching")

def common_response_handler(request: DataQueryRequest, table_suffix: str, session: Session) -> Dict[str, Any]:
    """Handles query execution and response formatting for both historical and anomaly fetch endpoints."""
    request_params = request.model_dump()
    
    data_records = fetch_data_from_cassandra(request_params, table_suffix, session)
    
    limit = request.limit 
    formatted_output = format_cassandra_output(data_records, limit) 
    
    metadata = {
        "count": len(data_records)
    }
    
    final_response = formatted_output
    final_response.update(metadata) 
    
    return final_response

@router.post('/anomaly_detection_all_ahu')
def anomaly_detection_all_ahu(request: DatapointFetchingRequest) -> Dict[str, Any]:
    """Endpoint to trigger the anomaly detection process for all AHU assets."""
    start_time = time.time()

    building_id = request.building_id
    floor_id = request.floor_id
    equipment_id = request.equipment_id
    search_tags = request.search_tag_groups
    ticket_id = request.ticket
    software_id = request.software_id
    account_id = request.account_id
    system_type = request.system_type

    if not building_id:
        building_id = DEFAULT_BUILDING_ID

    result = anamoly_evaluation(building_id=building_id, floor_id=floor_id, equipment_id=equipment_id, search_tags=search_tags, ticket_id=ticket_id, software_id=software_id, account_id=account_id, system_type=system_type) 
    log.info(f"Anomaly detection completed in {time.time() - start_time:.2f} seconds.")
    return result
    
@router.post("/store_anamolies")
def store_anamolies_endpoint(request: DatapointFetchingRequest, session: Session = Depends(get_cassandra_session)):
    """
    Triggers the process: calls anamoly_evaluation to get results and inserts them 
    into Cassandra's historical and anomaly tables, asset by asset.
    """
    start_time = time.time()

    building_id = request.building_id
    floor_id = request.floor_id
    equipment_id = request.equipment_id
    search_tags = request.search_tag_groups
    ticket_id = request.ticket
    software_id = request.software_id
    account_id = request.account_id
    system_type = request.system_type

    if not building_id:
        building_id = DEFAULT_BUILDING_ID

    result = save_data_to_cassandra(building_id=building_id, floor_id=floor_id, equipment_id=equipment_id, search_tags=search_tags, ticket_id=ticket_id, software_id=software_id, account_id=account_id, system_type=system_type,session=session) 
    log.info(f"Anomaly detection completed in {time.time() - start_time:.2f} seconds.")
    return result
    
@router.post("/fetch_historical_data")
def fetch_historical_data_endpoint(
    request: DataQueryRequest, 
    session: Session = Depends(get_cassandra_session)
) -> Dict[str, Any]:
    """Fetches historical (all) data from the Cassandra history table."""
    return common_response_handler(request, HISTORY_TABLE_SUFFIX, session)

@router.post("/fetch_anomaly_data")
def fetch_anomaly_data_endpoint(
    request: DataQueryRequest, 
    session: Session = Depends(get_cassandra_session)
) -> Dict[str, Any]:
    """Fetches only anomaly data from the Cassandra anomaly table."""
    return common_response_handler(request, ANAMOLY_TABLE_SUFFIX, session)