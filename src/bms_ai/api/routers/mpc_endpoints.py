"""
MPC Endpoints for Cinema AHU Optimization
==========================================

FastAPI endpoints for MPC-based setpoint optimization.
Provides real-time optimization for cinema AHU systems.

Author: BMS-AI Team
Date: January 2026
"""

from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from cassandra.cluster import Session
from src.bms_ai.api.dependencies import get_cassandra_session
from src.bms_ai.logger_config import setup_logger
from pathlib import Path
import json

from src.bms_ai.utils.save_cassandra_data import save_data_to_cassandraV2
from src.bms_ai.mpc.mpc_inference_pipeline import (
    MPCInferencePipeline,
    InferenceConfig
)

import requests
import pandas as pd
import joblib
import warnings
import math
import time
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

log = setup_logger(__name__)

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/mpc", tags=["MPC Optimization"])


# =============================================================================
# MODEL LOADING AT STARTUP
# =============================================================================

# Load MPC model at module import (startup)
_mpc_model_loaded = False

try:
    _mpc_model_loaded = MPCInferencePipeline.load_model_on_startup(equipment_id="Ahu13")
    if _mpc_model_loaded:
        log.info("MPC Model for Ahu13 loaded successfully at startup")
    else:
        log.warning("Failed to load MPC Model for Ahu13 at startup")
except Exception as e:
    log.error(f"Error loading MPC Model at startup: {e}")


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class MPCOptimizeRequest(BaseModel):
    """Request model for MPC optimization endpoint."""
    ticket: str = Field(..., description="Ticket ID for movie schedule API")
    ticket_type: Optional[str] = Field(None, description="Ticket type (e.g., 'software', 'model', etc.)")
    account_id: str = Field(..., description="Account ID associated with the request")
    software_id: str = Field(..., description="Software ID for versioning")
    building_id: str = Field("36c27828-d0b4-4f1e-8a94-d962d342e7c2", description="Building ID for querying live data")
    site: str = Field("OS04", description="Site name where the equipment is located")
    system_type: str = Field("AHU", description="System type")
    equipment_id: str = Field("Ahu13", description="Equipment ID (e.g., 'Ahu13')")
    screen_id: str = Field("Screen 13", description="Screen ID for occupancy lookup (e.g., 'Screen 13')")


class MPCOptimizeResponse(BaseModel):
    """Response model for MPC optimization endpoint."""
    success: bool = Field(..., description="Whether the optimization was successful")
    equipment_id: Optional[str] = Field(None, description="Equipment ID")
    timestamp_utc: Optional[str] = Field(None, description="UTC timestamp of optimization")
    timestamp_sharjah: Optional[str] = Field(None, description="Sharjah local time")
    optimized_setpoint: Optional[float] = Field(None, description="MPC-optimized SpTREff setpoint")
    actual_sptreff: Optional[float] = Field(None, description="Current/actual SpTREff from BMS")
    actual_tempsp1: Optional[float] = Field(None, description="Current room temperature")
    target_temperature: Optional[float] = Field(None, description="Target comfort temperature")
    setpoint_difference: Optional[float] = Field(None, description="Difference: optimized - actual")
    occupied: Optional[int] = Field(None, description="Occupancy status (0 or 1)")
    movie_name: Optional[str] = Field(None, description="Movie name or [PRE-COOLING] prefix")
    mode: Optional[str] = Field(None, description="Mode: occupied, pre_cooling, unoccupied")
    is_precooling: Optional[bool] = Field(None, description="True if in pre-cooling period")
    time_until_next_movie: Optional[int] = Field(None, description="Minutes until next movie")
    outside_temp: Optional[float] = Field(None, description="Outdoor temperature (°C)")
    outside_humidity: Optional[float] = Field(None, description="Outdoor humidity (%)")
    fb_vfd: Optional[float] = Field(None, description="Fan speed (%)")
    fb_fad: Optional[float] = Field(None, description="Fresh air damper (%)")
    co2_ra: Optional[float] = Field(None, description="CO2 level (ppm)")
    co2_load_category: Optional[str] = Field(None, description="CO2 load category")
    co2_load_factor: Optional[float] = Field(None, description="CO2 load factor (0.0-1.5+)")
    optimization_status: Optional[str] = Field(None, description="Optimization status")
    objective_value: Optional[float] = Field(None, description="Optimizer objective value")
    used_features: Optional[Dict[str, Any]] = Field(None, description="Features used for prediction")
    next_tempsp1_lag: Optional[float] = Field(None, description="TempSp1 for next prediction lag")
    next_sptreff_lag: Optional[float] = Field(None, description="SpTREff for next prediction lag")
    previous_setpoint: Optional[float] = Field(None, description="Previous setpoint for rate limiting")
    screen_id: Optional[str] = Field(None, description="Screen ID")
    saved_to_cassandra: Optional[bool] = Field(None, description="Whether result was saved to Cassandra")
    error: Optional[str] = Field(None, description="Error message if failed")
    error_type: Optional[str] = Field(None, description="Error type if failed")


class MPCStatusResponse(BaseModel):
    """Response model for MPC status endpoint."""
    mpc_model_loaded: bool
    equipment_id: str
    model_path: str
    status: str


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/status", response_model=MPCStatusResponse)
async def get_mpc_status():
    """
    Get MPC system status.
    
    Returns information about the loaded MPC model and system readiness.
    """
    return MPCStatusResponse(
        mpc_model_loaded=_mpc_model_loaded,
        equipment_id="Ahu13",
        model_path="artifacts/ahu13/mpc_model.joblib",
        status="ready" if _mpc_model_loaded else "model_not_loaded"
    )


@router.post("/optimize", response_model=MPCOptimizeResponse)
async def optimize_setpoint(
    request: MPCOptimizeRequest,
    session: Session = Depends(get_cassandra_session)
):
    """
    Run MPC optimization for cinema AHU.
    
    This endpoint:
    1. Fetches live sensor data from BMS (last 10 minutes, resampled)
    2. Fetches lag values from Cassandra (previous optimization)
    3. Fetches current weather from Open-Meteo
    4. Fetches occupancy from movie schedule API
    5. Runs MPC optimization
    6. Saves results to Cassandra
    7. Returns optimized setpoint and all relevant data
    
    The endpoint is designed to be called at 10-minute intervals.
    """
    log.info(f"[MPC Optimize] Request for {request.equipment_id} / {request.screen_id}")
    
    # Check if model is loaded
    if not _mpc_model_loaded:
        log.error("[MPC Optimize] MPC model not loaded")
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": "MPC model not loaded. Please contact administrator.",
                "error_type": "MODEL_NOT_LOADED"
            }
        )
    
    try:
        # Create inference pipeline
        config = InferenceConfig(
            equipment_id=request.equipment_id,
            screen_id=request.screen_id,
            building_id=request.building_id
        )
        
        pipeline = MPCInferencePipeline(
            cassandra_session=session,
            config=config
        )
        
        # Run inference
        result = pipeline.run_inference(
            equipment_id=request.equipment_id,
            screen_id=request.screen_id,
            ticket=request.ticket,
            building_id=request.building_id
        )
        
        if not result.get('success'):
            log.error(f"[MPC Optimize] Optimization failed: {result.get('error')}")
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": result.get('error'),
                    "error_type": result.get('error_type')
                }
            )
        
        log.info(f"[MPC Optimize] Success: optimal_setpoint={result.get('optimized_setpoint')}")
        return MPCOptimizeResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[MPC Optimize] Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "error_type": "INTERNAL_ERROR"
            }
        )


@router.get("/last_optimization/{equipment_id}", response_model=MPCOptimizeResponse)
async def get_last_optimization(
    equipment_id: str = "Ahu13",
    session: Session = Depends(get_cassandra_session)
):
    """
    Get the most recent optimization result for an equipment.
    
    Args:
        equipment_id: Equipment identifier (e.g., 'Ahu13')
        
    Returns:
        The last optimization result from Cassandra
    """
    log.info(f"[MPC Last] Fetching last optimization for {equipment_id}")
    
    config = InferenceConfig(equipment_id=equipment_id)
    table_name = config.table_name
    
    query = f"""
        SELECT * FROM {table_name}
        WHERE equipment_id = '{equipment_id}'
        LIMIT 1;
    """
    
    try:
        rows = session.execute(query)
        row = rows.one()
        
        if not row:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "error": f"No optimization found for {equipment_id}",
                    "error_type": "NOT_FOUND"
                }
            )
        
        # Convert row to dict
        result = {col: getattr(row, col, None) for col in row._fields}
        
        # Parse used_features JSON
        if result.get('used_features'):
            try:
                result['used_features'] = json.loads(result['used_features'])
            except:
                pass
        
        # Format timestamps
        if result.get('timestamp_utc'):
            result['timestamp_utc'] = result['timestamp_utc'].isoformat()
        
        result['success'] = True
        
        return MPCOptimizeResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[MPC Last] Error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "error_type": "DATABASE_ERROR"
            }
        )


@router.get("/history/{equipment_id}")
async def get_optimization_history(
    equipment_id: str = "Ahu13",
    limit: int = 100,
    session: Session = Depends(get_cassandra_session)
):
    """
    Get optimization history for an equipment.
    
    Args:
        equipment_id: Equipment identifier
        limit: Maximum number of records to return (default 100)
        
    Returns:
        List of optimization results
    """
    log.info(f"[MPC History] Fetching history for {equipment_id}, limit={limit}")
    
    config = InferenceConfig(equipment_id=equipment_id)
    table_name = config.table_name
    
    query = f"""
        SELECT * FROM {table_name}
        WHERE equipment_id = '{equipment_id}'
        LIMIT {limit};
    """
    
    try:
        rows = session.execute(query)
        
        results = []
        for row in rows:
            record = {col: getattr(row, col, None) for col in row._fields}
            
            # Parse used_features JSON
            if record.get('used_features'):
                try:
                    record['used_features'] = json.loads(record['used_features'])
                except:
                    pass
            
            # Format timestamps
            if record.get('timestamp_utc'):
                record['timestamp_utc'] = record['timestamp_utc'].isoformat()
            
            results.append(record)
        
        return {
            "success": True,
            "equipment_id": equipment_id,
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        log.error(f"[MPC History] Error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "error_type": "DATABASE_ERROR"
            }
        )


