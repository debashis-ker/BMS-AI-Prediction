from typing import Optional
from pydantic import BaseModel, Field
from src.bms_ai.components.savings_dashboard_functions import dashboard_savings_data, occupancy_dashboard 
import warnings
from src.bms_ai.logger_config import setup_logger
from fastapi import APIRouter, HTTPException, Depends
from cassandra.cluster import Session
from src.bms_ai.api.dependencies import get_cassandra_session

class dashboard_request(BaseModel):
    """Request model for MPC optimization endpoint."""
    building_id: str = Field("36c27828-d0b4-4f1e-8a94-d962d342e7c2", description="Building ID for querying live data")
    from_date : Optional[str] = Field(None, description="Start date for data fetching in ISO format: YYYY-MM-DD HH:MM:SS")
    to_date : Optional[str] = Field(None, description="End date for data fetching in ISO format: YYYY-MM-DD HH:MM:SS")
    summary_needed : Optional[bool] = Field(False, description="Ai Summary Overview of the data for the given period")

warnings.filterwarnings("ignore")

log = setup_logger(__name__)

router = APIRouter(prefix="/savings_dashboard", tags=["Saving Dashboard"])

@router.post("/fetch_dashboard_data")
def fetch_dashboard_data(
    request: dashboard_request, 
    session: Session = Depends(get_cassandra_session)
):
    try:
        savings_data = dashboard_savings_data(
            building_id=request.building_id, 
            from_date=request.from_date, 
            to_date=request.to_date,
            session=session
        )
        
        if not savings_data:
            return {"message": "No setpoint optimization differences found for the given period."}

        log.info(f"Fetched {len(savings_data)} records for building {request.building_id}")
        return savings_data

    except Exception as e:
        log.error(f"Error fetching dashboard data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")

@router.post("/fetch_occupancy_dashboard")
def fetch_occupancy_dashboard(
    request: dashboard_request, 
    session: Session = Depends(get_cassandra_session)
):
    try:
        occupancy_data = occupancy_dashboard(
            building_id=request.building_id, 
            from_date=request.from_date, 
            to_date=request.to_date,
            session=session
        )
        
        if not occupancy_data:
            return {
                "success": False,
                "message": "No occupancy data found for the given period."
            }
        
        log.info(f"Fetched occupancy dashboard data for building {request.building_id}")
        return occupancy_data
    
    except Exception as e:
        log.error(f"Error fetching occupancy dashboard data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")