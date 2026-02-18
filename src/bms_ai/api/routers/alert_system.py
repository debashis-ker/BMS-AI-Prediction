from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, Dict, List, Any
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.components.alert_functions import *
import warnings
from src.bms_ai.api.dependencies import get_cassandra_session
from cassandra.cluster import Session
import time
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

log = setup_logger(__name__)
 
warnings.filterwarnings('ignore')

router = APIRouter(prefix="/alarm", tags=["Prescriptive Optimization"])

class FetchingAlarmDataRequest(BaseModel):
    site: Optional[str] = Field(default='OS04')
    equipment_id: Optional[str] = Field(default='Ahu13', description="Equipment ID for which to fetch data.")
    building_id: Optional[str] = Field(default="36c27828-d0b4-4f1e-8a94-d962d342e7c2", description="Building ID for which to fetch data.")
    from_date: Optional[str] = Field(default=None, description="Start date for filtering (e.g., 'YYYY-MM-DD').")
    to_date: Optional[str] = Field(default=None, description="End date for filtering (e.g., 'YYYY-MM-DD').")
    alarm_names: Optional[List[str]] = Field(default=["Alarm_Freeze", "Alarm_Oscillation", "Alarm_Tracking", "Alarm_Heat_Stress","Alarm_Return_Air_Temp"], description="List of alarm names to count.")
    state: List[str] = Field(default=['warning', 'critical'], description="Filter by states: 'warning', 'critical', or both.")

class StoringAlarmDataRequest(BaseModel):
    building_id: Optional[str] = Field(default="36c27828-d0b4-4f1e-8a94-d962d342e7c2", description="Building ID (optional)")
    system_type: Optional[str] = Field(default="AHU", description="System type (e.g., 'AHU', 'MAHU')")
    equipment_id: Optional[str] = Field(default="Ahu13", description="Equipment ID for datapoint fetching")
    ticket: str = Field(default="", description="Ticket identifier for data processing (optional)")
    ticket_type: Optional[str] = Field(default=None, description="Type of ticket for data processing (optional)")
    software_id: Optional[str] = Field(default=None, description="Software ID for data processing (optional)")
    account_id: Optional[str] = Field(default=None, description="Account ID for data processing (optional)")

class TestAlarmRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of raw BMS data records to test.")
    ticket: Optional[str] = Field(default="")
    ticket_type: Optional[str] = Field(default=None)

def common_response_handler(request: FetchingAlarmDataRequest, session: Session) -> Dict[str, Any]:
    """Handles query execution and response formatting for both historical and anomaly fetch endpoints."""
    log.info(f"Initiating common response handler for site: {request.site}, equipment: {request.equipment_id}")
    try:
        request_params = request.model_dump()
        
        if session is None:
            log.error("Cassandra session dependency returned None.")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Cassandra session is not available.")

        data_records = fetch_alarm_data_from_cassandra(request_params, session)
        
        if data_records is None:
            log.error(f"fetch_alarm_data_from_cassandra returned None for equipment {request.equipment_id}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve records from Cassandra.")

        log.debug(f"Retrieved {len(data_records)} records from database.")
        formatted_output = format_cassandra_output(data_records) 
        
        if not isinstance(formatted_output, dict):
            log.error("format_cassandra_output failed to return a dictionary.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Data formatting error.")

        metadata = {
            "count": len(data_records)
        }
        
        final_response = formatted_output
        final_response.update(metadata) 
        log.info("Common response handler execution successful.")
        return final_response

    except HTTPException as he:
        raise he
    except Exception as e:
        log.error(f"Unexpected error in common_response_handler: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred: {str(e)}")

'''Cassandra Database Endpoints for storing and fetching data of alarm'''
@router.post("/evaluate_alarm")
def store_alarms_endpoint(request: StoringAlarmDataRequest, session: Session = Depends(get_cassandra_session)):
    """
    Triggers the process: calls store_alarm_data to get results and inserts them 
    into Cassandra's alarm tables.
    """
    log.info(f"POST /store_alarms received for equipment: {request.equipment_id}")
    start_time = time.time()

    if not request.ticket:
        log.warning("Storage request rejected: Missing ticket identifier.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="ticket is required for fetching occupancy data and scheduling data."
        )

    building_id = request.building_id
    equipment_id = request.equipment_id
    system_type = request.system_type
    ticket = request.ticket
    ticket_type = request.ticket_type
    software_id = request.software_id
    account_id = request.account_id

    log.debug(f"Request Context - Building: {building_id}, System: {system_type}, Ticket: {ticket}")

    try:
        if session is None:
            log.error("Database session unavailable for storage.")
            raise HTTPException(status_code=503, detail="Database session not found.")

        result = save_data_to_cassandra(
            building_id=building_id, 
            equipment_id=equipment_id,  
            system_type=system_type,  
            ticket=ticket, 
            ticket_type=ticket_type,  
            software_id=software_id,  
            account_id=account_id,  
            session=session
        )

        duration = time.time() - start_time
        log.info(f"Alarm data storage completed for {equipment_id} in {duration:.2f}s.")
        return result

    except HTTPException as he:
        raise he
    except ConnectionError as ce:
        log.error(f"Database connection error during storage: {str(ce)}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database connection failed during storage.")
    except Exception as e:
        log.error(f"Critical error in store_alarms_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Storage operation failed: {str(e)}")
    
@router.post("/fetch_alarm_data")
def fetch_alarm_data(
    request: FetchingAlarmDataRequest, 
    session: Session = Depends(get_cassandra_session)
) -> Dict[str, Any]:
    """Fetch alarm data from Cassandra."""
    log.info(f"POST /fetch_alarm_data for site: {request.site}")
    return common_response_handler(request, session)

@router.post("/count_alarms_db", response_model=Dict[str, Dict[str, int]])
def count_alarms_db_endpoint(
    request: FetchingAlarmDataRequest, 
    session: Session = Depends(get_cassandra_session)
):
    """
    Returns the total count of specific alarms grouped by state (warning/critical).
    """
    log.info(f"POST /count_alarms_db for equipment: {request.equipment_id}")
    try:
        if session is None:
            log.error("Database session missing in count endpoint.")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database session not established.")

        params = request.model_dump()
        result = count_alarms_from_db(params, session)
        
        if result is None:
            log.error(f"Alarm count calculation returned None for {request.equipment_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Could not calculate counts.")

        log.info(f"DB Alarm Count Request for {request.equipment_id} successful.")
        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        log.error(f"Error in count_alarms_db_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while counting alarms.")
       
@router.post("/test_alarm")
def test_alarm_endpoint(request: TestAlarmRequest, session: Session = Depends(get_cassandra_session)):
    '''
    Testing endpoint for alarm logic
    '''
    log.info("POST /test_alarm initiated.")
    try:
        if not request.data:
            log.warning("Test request received with empty data list.")
            raise HTTPException(status_code=400, detail="Data list cannot be empty for testing.")

        results = test_alarm_logic(
            records=request.data, 
            ticket=request.ticket,  
            ticket_type=request.ticket_type,
            session = session
        )

        log.info(f"Test alarm logic processed. Results found: {len(results)}")
        
        content = jsonable_encoder({
            "status": "SUCCESS",
            "alarm_count": len(results),
            "results": results
        })
        
        return JSONResponse(content=content)

    except HTTPException as he:
        raise he
    except Exception as e:
        log.error(f"Test Alarm execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Testing failed: {str(e)}")