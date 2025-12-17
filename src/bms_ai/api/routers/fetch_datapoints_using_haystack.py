from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional

from src.bms_ai.utils.ikon_apis import fetch_and_find_data_points
from src.bms_ai.logger_config import setup_logger

log = setup_logger(__name__)
router = APIRouter(prefix="/fetch-datapoints", tags=["BMS Data Points"])


class FetchDataPointsRequest(BaseModel):
    building_id: str = Field(..., description="Building ID")
    floor_id: Optional[str] = Field(None, description="Floor ID (if not provided, searches all floors)")
    equipment_id: str = Field(..., description="Equipment ID (can be empty string)")
    search_tag_groups: List[List[str]] = Field(..., description="Array of Haystack tag arrays to search")
    ticket: str = Field(..., description="Authentication ticket")
    ticket_type: Optional[str] = Field(None, description="Type of ticket (e.g., 'jobUser')")
    software_id: str = Field(..., description="Software ID")
    account_id: str = Field(..., description="Account ID")
    system_type: Optional[str] = Field(None, description="Optional system type (e.g., 'FCU', 'AHU')")


class FetchDataPointsResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]]
    count: int


@router.post("/", response_model=FetchDataPointsResponse)
def fetch_data_points_endpoint(req: FetchDataPointsRequest) -> FetchDataPointsResponse:
    """
    Fetch and find data points by floor, equipment, and Haystack tags.
    
    - Fetches building hierarchy data
    - Resolves floor names from floor IDs
    - Searches for data points matching Haystack tags
    - Returns matched data points with metadata
    """
    try:
        log.info(f"Fetching data points for building: {req.building_id}, floor: {req.floor_id}")
        
        results = fetch_and_find_data_points(
            building_id=req.building_id,
            floor_id=req.floor_id,
            equipment_id=req.equipment_id,
            search_tag_groups=req.search_tag_groups,
            ticket=req.ticket,
            software_id=req.software_id,
            account_id=req.account_id,
            system_type=req.system_type,
            env="prod",
            ticket_type=req.ticket_type
        )
        
        log.info(f"Successfully fetched {len(results)} data points")
        
        return FetchDataPointsResponse(
            success=True,
            data=results,
            count=len(results)
        )
    
    except Exception as e:
        log.error(f"Error in fetch_data_points_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))