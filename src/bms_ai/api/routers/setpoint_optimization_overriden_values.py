from typing import Optional
from pydantic import BaseModel, Field
from src.bms_ai.components.setpoint_optimization_overriden_function import calculate_setpoint_diffs, fetch_setpoint_diffs_averages, fetch_setpoint_diffs
from src.bms_ai.utils.save_cassandra_data import save_optimized_setpoint_difference_data
import warnings
from src.bms_ai.logger_config import setup_logger
from fastapi import APIRouter, HTTPException, Depends
from cassandra.cluster import Session
from src.bms_ai.api.dependencies import get_cassandra_session

class OptimizationRequest(BaseModel):
    """Request model for MPC optimization endpoint."""
    building_id: str = Field("36c27828-d0b4-4f1e-8a94-d962d342e7c2", description="Building ID for querying live data")
    equipment_id: str = Field("Ahu13", description="Equipment ID (e.g., 'Ahu13')")
    from_date : Optional[str] = Field(None, description="Start date for data fetching in ISO format: YYYY-MM-DD HH:MM:SS")
    to_date : Optional[str] = Field(None, description="End date for data fetching in ISO format: YYYY-MM-DD HH:MM:SS")

warnings.filterwarnings("ignore")

log = setup_logger(__name__)

router = APIRouter(prefix="/setpoint_optimization_diff", tags=["Saving Setpoint Optimization Differences"])

@router.post("/save_setpoint_optimization_diff")
async def save_setpoint_optimization_diff(
    request: OptimizationRequest, 
    session: Session = Depends(get_cassandra_session)
):
    try:
        setpoint_diffs = await calculate_setpoint_diffs(
            equipment_id=request.equipment_id, 
            from_date=request.from_date, 
            to_date=request.to_date,
            session=session
        )
        
        if not setpoint_diffs:
            return {"message": "No unusual setpoint activity found for the given period. Nothing to save."}

        save_optimized_setpoint_difference_data(
            data_chunk=setpoint_diffs, 
            building_id=request.building_id, 
            session=session
        )
        
        log.info(f"Saved {len(setpoint_diffs)} records for {request.equipment_id}")
        return {
            "success": True,
            "message": f"Successfully saved {len(setpoint_diffs)} setpoint optimization differences",
            "count": len(setpoint_diffs)
        }

    except Exception as e:
        log.error(f"Error saving setpoint optimization differences: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")

@router.post("/fetch_setpoint_optimization_differences")
def fetch_setpoint_optimization_differences(
    request: OptimizationRequest, 
    session: Session = Depends(get_cassandra_session)
):
    result = fetch_setpoint_diffs(building_id=request.building_id, equipment_id=request.equipment_id, session=session, start_date=request.from_date, end_date=request.to_date)
    return result

@router.post("/fetch_setpoint_optimization_average")
def fetch_setpoint_optimization_average(
    request: OptimizationRequest, 
    session: Session = Depends(get_cassandra_session)
):
    result = fetch_setpoint_diffs_averages(building_id=request.building_id, equipment_id=request.equipment_id, session=session, start_date=request.from_date, end_date=request.to_date)
    return result

