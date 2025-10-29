from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends,Request
from typing import Optional,Dict, List, Any
from src.bms_ai.pipelines.prescriptive_pipeline import PrescriptivePipeline
from src.bms_ai.api.dependencies import get_prescriptive_pipeline
from src.bms_ai.logger_config import setup_logger


log = setup_logger(__name__)


class OptimizeRequest(BaseModel):
    current_conditions: Dict[str, Any] = Field(..., description="Current system state")
    search_space: Optional[Dict[str, List[float]]] = Field(None, description="Setpoint ranges to search (optional)")
    optimization_method: Optional[str] = Field("hybrid", description="Optimization method: 'grid', 'random', or 'hybrid'")
    n_iterations: Optional[int] = Field(2000, description="Number of iterations for random/hybrid search")

class OptimizeResponse(BaseModel):
    best_setpoints: Dict[str, float]
    min_fan_power_kw: float
    total_combinations_tested: int
    optimization_method: str
    optimization_time_seconds: float

router = APIRouter(prefix="/optimize", tags=["Prescriptive Optimization"])

# @router.post("/", response_model=OptimizeResponse)
# def optimize(req: OptimizeRequest, pipeline: PrescriptivePipeline = Depends(get_pipeline()))-> OptimizeResponse:
#     '''Endpoint to perform optimization based on current conditions and search space.
#      - current_conditions: Dict with current sensor readings and setpoints.
#      - search_space: Dict defining ranges for setpoints to explore. If empty, defaults are used.
     
#      Returns the best setpoints found, minimum fan power, and optimization details.'''
#     try:
#         search_space = req.search_space
#         if not search_space:
#             import numpy as np
#             search_space = {
#                 'RA  temperature setpoint': list(np.arange(20.0, 27.5, 0.5)),
#                 'RA CO2 setpoint': list(np.arange(500.0, 825.0, 25.0)),
#                 'SA Pressure setpoint': list(np.arange(500.0, 1250.0, 50.0))
#             }
#             log.info("No search_space provided, using default search space.")
#         result = pipeline.run_optimization(req.current_conditions, search_space)
#         log.info(f"Optimization result: {result}")
#         if "message" in result:
#             raise HTTPException(status_code=400, detail=result["message"])
#         return OptimizeResponse(**result)
#     except HTTPException:
#         raise
#     except Exception as e:
#         log.error(f"Optimization request failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))



@router.post("/", response_model=OptimizeResponse)
def optimize(
    req: OptimizeRequest,
    prescriptive_pipeline: PrescriptivePipeline = Depends(get_prescriptive_pipeline)
) -> OptimizeResponse:
    """Endpoint to perform optimization based on current conditions and search space.

    - **current_conditions**: Dict with current sensor readings and setpoints.
    - **search_space**: Dict defining ranges for setpoints to explore. If empty, defaults are used.
    - **optimization_method**: 'grid' (exhaustive), 'random' (Monte Carlo), or 'hybrid' (grid + refinement)
    - **n_iterations**: Number of iterations for random/hybrid methods

    Returns the best setpoints found, minimum fan power, and optimization details.
    """
    
    try:
        search_space = req.search_space
        current_conditions = req.current_conditions
        optimization_method = req.optimization_method or "random"
        n_iterations = req.n_iterations or 1000
        
        log.info(f"Received optimization request with method='{optimization_method}', current_conditions: {current_conditions}")
        print(f"Optimization method: {optimization_method}, iterations: {n_iterations}")
        
        if not search_space:
            import numpy as np
            # Define default search space based on optimization method
            if optimization_method == "grid":
                search_space = {
                    'RA  temperature setpoint': list(np.arange(20.0, 27.5, 1.0)),      # 8 values
                    'RA CO2 setpoint': list(np.arange(500.0, 825.0, 50.0)),            # 7 values
                    'SA Pressure setpoint': list(np.arange(500.0, 1250.0, 50.0))       # 15 values
                }
            else:
                search_space = {
                    'RA  temperature setpoint': [20.0, 27.0],  
                    'RA CO2 setpoint': [500.0, 800.0],
                    'SA Pressure setpoint': [500.0, 1200.0]
                }
            log.info(f"No search_space provided, using default '{optimization_method}' search space.")
        
        result = prescriptive_pipeline.run_optimization(
            current_conditions, 
            search_space,
            optimization_method=optimization_method,
            n_iterations=n_iterations
        )
        
        log.info(f"Optimization result: {result}")
        if "message" in result:
            raise HTTPException(status_code=400, detail=result["message"])
        return OptimizeResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Optimization request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))