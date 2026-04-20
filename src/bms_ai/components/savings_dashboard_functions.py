from typing import Any, Dict, Optional, List
import os
import pandas as pd
from dateutil import parser
from datetime import datetime, timezone
import warnings
from src.bms_ai.logger_config import setup_logger
from fastapi import HTTPException

log = setup_logger(__name__)

warnings.filterwarnings('ignore')

def fetch_historical_setpoint_diff(
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2", 
    session: Any = None, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetches setpoint override data for ALL equipment in a building.
    """
    
    table_suffix = building_id.replace('-', '').lower()
    table_name = f"setpoint_overriden_data_{table_suffix}"
    keyspace = os.getenv("CASSANDRA_KEYSPACE", "user_keyspace")
    
    query = f'SELECT * FROM {keyspace}."{table_name}"'
    params = []
    where_clauses = []

    if start_date:
        where_clauses.append("timestamp >= ?")
        params.append(parser.parse(start_date).replace(tzinfo=timezone.utc))
    
    if end_date:
        where_clauses.append("timestamp <= ?")
        params.append(parser.parse(end_date).replace(tzinfo=timezone.utc))

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += " ALLOW FILTERING"

    try:
        log.info(f"Fetching raw records from {table_name} for all AHUs")
        
        if params:
            stmt = session.prepare(query)
            rows = session.execute(stmt, params)
        else:
            rows = session.execute(query)

        records = []
        for row in rows:
            records.append({
                "equipment_id": getattr(row, 'equipment_id', None),
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "timestamp_utc": getattr(row, 'timestamp_utc', None),
                "mode": getattr(row, 'mode', None),
                "chwfb_diff": round(float(getattr(row, 'chwfb_diff', 0) or 0), 4),
                "sptreff_diff": round(float(getattr(row, 'sptreff_diff', 0) or 0), 4),
                "tempsp1_diff": round(float(getattr(row, 'tempsp1_diff', 0) or 0), 4)
            })

        if not records:
            return {
                "success": False,
                "message": "No data found for the specified criteria.",
                "total_records": 0,
                "data": []
            }

        return {
            "success": True,
            "total_records": len(records),
            "data": records
        }

    except Exception as e:
        log.error(f"Fetch Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database fetch failed: {str(e)}")

def fetch_building_optimization_history(
    building_id: str, 
    session: Any, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetches MPC optimization history for ALL equipment in a building.
    Uses the optimization results table to get actual sensor values (actual_tempsp1).
    """
    table_suffix = building_id.replace('-', '').lower()
    table_name = f"mpc_optimization_results_{table_suffix}"
    keyspace = os.getenv("CASSANDRA_KEYSPACE", "user_keyspace")
    
    query = f'SELECT equipment_id, timestamp_utc, mode, actual_tempsp1 FROM {keyspace}."{table_name}"'
    params = []
    where_clauses = []

    if start_date:
        where_clauses.append("timestamp_utc >= ?")
        params.append(parser.parse(start_date).replace(tzinfo=timezone.utc))
    else:
        where_clauses.append("timestamp_utc >= ?")
        start_date = datetime.utcnow() - pd.Timedelta(days=1) 
        params.append(start_date)

    if end_date:
        where_clauses.append("timestamp_utc <= ?")
        params.append(parser.parse(end_date).replace(tzinfo=timezone.utc))
    else:
        where_clauses.append("timestamp_utc <= ?")
        end_date = datetime.utcnow()
        params.append(end_date)

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += " ALLOW FILTERING"

    try:
        log.info(f"Fetching building-wide optimization history from {table_name}")
        rows = session.execute(session.prepare(query), params) if params else session.execute(query)

        records = []
        for row in rows:
            records.append({
                "equipment_id": getattr(row, 'equipment_id', None),
                "mode": (getattr(row, 'mode', "") or "").lower(),
                "actual_temp": getattr(row, 'actual_tempsp1', None)
            })

        return {"success": True, "data": records} if records else {"success": False, "data": []}

    except Exception as e:
        log.error(f"History Fetch Error: {str(e)}")
        return {"success": False, "error": str(e), "data": []}

def dashboard_savings_data(
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2", 
    session: Any = None, 
    from_date: Optional[str] = None, 
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetches and processes setpoint override data for dashboard display.
    """
    try:
        data = fetch_historical_setpoint_diff(
            building_id=building_id, 
            session=session, 
            start_date=from_date, 
            end_date=to_date
        )

        if not from_date and not to_date:
            from_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            to_date = datetime.now()
            log.info(f"No date range provided. Defaulting to current month: {from_date} to {to_date}")

        if not data["success"]:
            {
                "success": False,
                "total_optimizations": 0,
                "average_temperature_diff": 0.0
            }
        
        total_optimizations = data["total_records"]
        average_temperature_diff = round(sum(record["tempsp1_diff"] for record in data["data"]) / total_optimizations, 4) if total_optimizations > 0 else 0.0 
        
        return {
            "success": True,
            "total_optimizations": total_optimizations, 
            "average_temperature_diff": average_temperature_diff
        }

    except Exception as e:
        log.error(f"Dashboard Data Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing dashboard data: {str(e)}")

def occupancy_dashboard(
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2", 
    session: Any = None, 
    from_date: Optional[str] = None, 
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculates occupancy counts and average ACTUAL temperatures for all equipment.
    """
    try:
        historical_data = fetch_building_optimization_history(building_id, session, from_date, to_date)
        setpoint_diff_data = fetch_historical_setpoint_diff(
            building_id=building_id, 
            session=session, 
            start_date=from_date, 
            end_date=to_date
        )

        if not from_date and not to_date:
            from_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            to_date = datetime.now()
            log.info(f"No date range provided. Defaulting to current month: {from_date} to {to_date}")
        
        if not historical_data["success"] or not historical_data["data"]:
            return {"success": False, "message": "No history data found."}

        occ_modes = ["occupied", "pre_cooling", "inter_show"]

        occupied_chwfb_diffs = [r["chwfb_diff"] for r in setpoint_diff_data["data"] if r["mode"] in occ_modes and r["chwfb_diff"] is not None]
        unoccupied_chwfb_diffs = [r["chwfb_diff"] for r in setpoint_diff_data["data"] if r["mode"] == "unoccupied" and r["chwfb_diff"] is not None]
        avg_occ_chwfb_diff = round(sum(occupied_chwfb_diffs) / len(occupied_chwfb_diffs), 4) if occupied_chwfb_diffs else 0.0
        avg_unocc_chwfb_diff = round(sum(unoccupied_chwfb_diffs) / len(unoccupied_chwfb_diffs), 4) if unoccupied_chwfb_diffs else 0.0
        
        occ_temps = [r["actual_temp"] for r in historical_data["data"] if r["mode"] in occ_modes and r["actual_temp"] is not None]
        unocc_temps = [r["actual_temp"] for r in historical_data["data"] if r["mode"] == "unoccupied" and r["actual_temp"] is not None]

        return {
            "success": True,
            "occupied_stats": {
                "count": len([r for r in setpoint_diff_data["data"] if r["mode"] in occ_modes]),
                "max_temp": max(occ_temps) if occ_temps else None,
                "min_temp": min(occ_temps) if occ_temps else None,
                "average_chwfb_diff": avg_occ_chwfb_diff
            },
            "unoccupied_stats": {
                "count": len([r for r in setpoint_diff_data["data"] if r["mode"] == "unoccupied"]),
                "max_temp": max(unocc_temps) if unocc_temps else None,
                "min_temp": min(unocc_temps) if unocc_temps else None,
                "average_chwfb_diff": avg_unocc_chwfb_diff
            }
        }

    except Exception as e:
        log.error(f"Occupancy Dashboard Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
