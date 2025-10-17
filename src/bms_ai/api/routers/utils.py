from pydantic import BaseModel, Field
from sklearn import pipeline
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
import pandas as pd
import os

from urllib3 import request
from src.bms_ai.utils.main_utils import clean_actual_v_predicted_fan_power_data
from src.bms_ai.utils.ikon_apis import get_my_instances_v2

from src.bms_ai.api.dependencies import get_prescriptive_pipeline
from src.bms_ai.pipelines.prescriptive_pipeline import PrescriptivePipeline
from src.bms_ai.logger_config import setup_logger

log = setup_logger(__name__)
router = APIRouter(prefix="/utils", tags=["Utilities"])


class FetchInstancesRequest(BaseModel):
    """Request model for fetching instances from IKON service."""
    ticket: str = Field(..., description="Authentication ticket")
    account_id: str = Field(..., description="Account ID")
    software_id: Optional[str] = Field(None, description="Software ID (optional, uses env var if not provided)")
    process_name: str = Field(..., description="Process name")
    predefined_filters: Optional[Dict] = Field(None, description="Predefined filters")
    process_variable_filters: Optional[Dict] = Field(None, description="Process variable filters")
    task_variable_filters: Optional[Dict] = Field(None, description="Task variable filters")
    mongo_where_clause: Optional[str] = Field(None, description="MongoDB where clause")
    projections: Optional[List[str]] = Field(None, description="List of fields to project")
    all_instances: bool = Field(False, description="Whether to fetch all instances")


class FanPowerRequest(BaseModel):
    """Request model for fetching actual vs predicted fan power data."""
    resampled : bool = Field(True, description="Whether to use resampled data")
    ticket: str = Field(..., description="Authentication ticket")
    account_id: str = Field(..., description="Account ID")
    software_id: Optional[str] = Field(None, description="Software ID (optional, uses env var if not provided)")
    


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify the service is running.
    
    Returns:
        A simple JSON indicating the service status.
    """
    return {"status": "healthy"}

@router.post("/fetch-instances")
async def fetch_instances_endpoint(request: FetchInstancesRequest) -> Dict[str, Any] | List[Any]:
    """
    Args:
        request: FetchInstancesRequest model containing all required parameters
        
    Returns:
        JSON response from IKON service (can be dict or list)
        
    Raises:
        HTTPException: If request fails or environment variables are missing
    """
    try:
        log.info(f"Received request to fetch instances for process: {request.process_name}")
        log.debug(f"Request details: ticket={request.ticket[:8]}..., account_id={request.account_id}")
        
        result = get_my_instances_v2(
            ticket=request.ticket,
            process_name=request.process_name,
            software_id=request.software_id or os.getenv("SOFTWARE_ID", ""),
            account_id=request.account_id,
            predefined_filters=request.predefined_filters,
            process_variable_filters=request.process_variable_filters,
            task_variable_filters=request.task_variable_filters,
            mongo_where_clause=request.mongo_where_clause,
            projections=request.projections,
            all_instances=request.all_instances
        )
        
        if result is None:
            log.error("get_my_instances_v2 returned None")
            raise HTTPException(status_code=500, detail="Failed to fetch instances from IKON service")
        
        log.info("Successfully processed fetch instances request")
        log.info(f"Fetched {len(result) if isinstance(result, list) else '1'} instances")
        log.debug(f"Fetched instances type: {type(result)}")
        log.debug(f"Fetched instances content: {result}")
        log.debug(f"Fetched data: {result[0]['data']}")
        
        return result
    except EnvironmentError as e:
        log.error(f"Environment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error in fetch_instances_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.get("/get_actual_v_predicted_fan_power")
def get_actual_v_predicted_fan_power(req: FanPowerRequest) -> List[Dict[str, Any]]:
    try:
        log.info("Received request to fetch actual vs predicted fan power data")
        log.debug(f"Request details: ticket={req.ticket[:8]}..., account_id={req.account_id}, resampled={req.resampled}")

        projections = ["Data"]

        result = get_my_instances_v2(
            ticket=req.ticket,
            process_name="Predict and Store setpoint data updated",
            software_id=req.software_id or os.getenv("SOFTWARE_ID", ""),
            account_id=req.account_id,
            predefined_filters={"taskNames": ["Data Node"]},
            process_variable_filters=None,
            task_variable_filters=None,
            mongo_where_clause=None,
            all_instances=False,
            projections=projections
        )

        if result is None:
            log.error("get_my_instances_v2 returned None")
            raise HTTPException(status_code=500, detail="Failed to fetch fan power data from IKON service")
        #print(type(result))
        if isinstance(result, list):
            data = result[0].get('data') if result and isinstance(result[0], dict) else None
            print(data['predictedData'])
            if data and data['predictedData']:
                return clean_actual_v_predicted_fan_power_data(data['predictedData'], resampled=False) if not req.resampled else clean_actual_v_predicted_fan_power_data(data['predictedData'], resampled=True)
            else:
                log.warning("No 'data' field found in the first result item; returning blank array.")
                return []
      

        return result

    except EnvironmentError as e:
        log.error(f"Environment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error in get_actual_v_predicted_fan_power: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
