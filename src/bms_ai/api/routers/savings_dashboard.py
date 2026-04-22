from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from src.bms_ai.components.savings_dashboard_functions import dashboard_savings_data, occupancy_dashboard , get_energy_comparison_data
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

class period_request(BaseModel):
    """Request model for MPC optimization endpoint."""
    from_date : str = Field(..., description="Start date for data fetching in ISO format: YYYY-MM-DD HH:MM:SS")
    to_date : str = Field(..., description="End date for data fetching in ISO format: YYYY-MM-DD HH:MM:SS")

class comparison_request(BaseModel):
    """Request model for MPC optimization endpoint."""
    building_id: str = Field("36c27828-d0b4-4f1e-8a94-d962d342e7c2", description="Building ID for querying live data")
    period_a : period_request = Field(..., description="Period A for comparison")
    period_b : period_request = Field(..., description="Period B for comparison")
    frequency: str = Field("D", description="Frequency: 'D' for Daily, 'H' for Hourly")

class EnergyComparisonResponse(BaseModel):
    consumption_comparison_chart_data: Dict[str, Any]
    delta_t_chart_data: Dict[str, Any]
    flow_vs_consumption_chart_data: Dict[str, Any]
    total_consumption_data: Dict[str, float]
    consumption_change_data: float
    cost_change_data: float
    avg_delta_t_data: Dict[str, float]
    rth_delta_data: Dict[str, float]
    avg_flow_data: Dict[str, float]

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
    
@router.post("/compare_periods", response_model=EnergyComparisonResponse)
async def compare_periods(
    request: comparison_request,
    session: Session = Depends(get_cassandra_session)
):
    try:
        comparison_data = await get_energy_comparison_data(
            building_id=request.building_id,
            period_a={"from_date": request.period_a.from_date, "to_date": request.period_a.to_date},
            period_b={"from_date": request.period_b.from_date, "to_date": request.period_b.to_date},
            frequency=request.frequency,
            session=session
        )
        
        if not comparison_data:
            return {
                "success": False,
                "message": "No data found for the given periods."
            }
        
        log.info(f"Fetched energy comparison data for building {request.building_id}")
        return EnergyComparisonResponse(**comparison_data)
    
    except Exception as e:
        log.error(f"Error comparing periods: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")