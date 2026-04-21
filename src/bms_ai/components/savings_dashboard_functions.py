from typing import Any, Dict, Optional, List
import os
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta, timezone
import warnings
from src.bms_ai.api.routers.energy_consumption import GetEnergyDataRequest, get_energy_data
from dateutil.relativedelta import relativedelta
from src.bms_ai.logger_config import setup_logger
from fastapi import HTTPException
import requests

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

async def get_energy_comparison_data(
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2",
    period_a: Dict[str, Any] = None,
    period_b: Dict[str, Any] = None,
    frequency: str = "D",
    session: Any = None, 
) -> Dict[str, Any]:
    """
    Performs two database calls and aligns the data for comparison analysis.
    """
    request_a = GetEnergyDataRequest(
        building_id=building_id,
        from_date=period_a.get("from_date"),
        to_date=period_a.get("to_date"),
        frequency=frequency
    )
    
    request_b = GetEnergyDataRequest(
        building_id=building_id,
        from_date=period_b.get("from_date"),
        to_date=period_b.get("to_date"),
        frequency=frequency
    )

    data_a = await get_energy_data(body=request_a, session=session)
    data_b = await get_energy_data(body=request_b, session=session)

    log.info(f"Period A status: {data_a.get('status')}, Records: {len(data_a.get('data', []))}")
    log.info(f"Period B status: {data_b.get('status')}, Records: {len(data_b.get('data', []))}")

    list_a = data_a.get("data", [])
    list_b = data_b.get("data", [])

    if not list_a and not list_b:
        raise HTTPException(
            status_code=404, 
            detail="No energy data records found for either comparison period. Please ensure data is processed for these dates."
        )
    
    key_fmt = '%Y-%m-%d %H:00' if frequency == "H" else '%Y-%m-%d'
    
    map_a = {parser.parse(r['period_start']).strftime(key_fmt): r for r in data_a.get("data", [])}
    map_b = {parser.parse(r['period_start']).strftime(key_fmt): r for r in data_b.get("data", [])}

    freq_map = {
        "H": relativedelta(hours=1),
        "D": relativedelta(days=1),
        "W": relativedelta(weeks=1),
        "M": relativedelta(months=1)
    }
    step = freq_map.get(frequency.upper(), relativedelta(days=1))

    start_a = parser.parse(period_a.get("from_date"))
    start_b = parser.parse(period_b.get("from_date"))
    
    total_steps = max(len(data_a.get("data", [])), len(data_b.get("data", [])))

    consumption_comparison_chart_data = {}
    delta_t_chart_data = {}
    flow_vs_consumption_chart_data = {}

    for i in range(total_steps):
        step_label = f"period_{i+1}"
        
        curr_a_key = (start_a + (step * i)).strftime(key_fmt)
        curr_b_key = (start_b + (step * i)).strftime(key_fmt)
        
        item_a = map_a.get(curr_a_key)
        item_b = map_b.get(curr_b_key)

        consumption_comparison_chart_data[step_label] = {
            "period_a_cost": item_a.get("estimated_cost_aed") if item_a else None,
            "period_b_cost": item_b.get("estimated_cost_aed") if item_b else None,
            "period_a_rth": item_a.get("rth_value") if item_a else None,
            "period_b_rth": item_b.get("rth_value") if item_b else None,
            "period_a_key": item_a.get("period_start") if item_a else None,
            "period_b_key": item_b.get("period_start") if item_b else None,
        }
        delta_t_chart_data[step_label] = {
            "period_a": item_a.get("delta_t") if item_a else None,
            "period_b": item_b.get("delta_t") if item_b else None,
            "period_a_key": item_a.get("period_start") if item_a else None,
            "period_b_key": item_b.get("period_start") if item_b else None,
        }

        flow_vs_consumption_chart_data[step_label] = {
            "period_a": item_a.get("avg_flow") if item_a else None,
            "period_b": item_b.get("avg_flow") if item_b else None,
            "period_a_key": item_a.get("period_start") if item_a else None,
            "period_b_key": item_b.get("period_start") if item_b else None,
        }

    total_a = data_a.get("total_rth", 0)
    total_b = data_b.get("total_rth", 0)
    total_cost_aed_a = data_a.get("total_cost_aed", 0)
    total_cost_aed_b = data_b.get("total_cost_aed", 0)
    cost_change_data = ((total_cost_aed_b - total_cost_aed_a) / total_cost_aed_a) if total_cost_aed_a != 0 else 0
    consumption_change_data = ((total_b - total_a) / total_a) if total_a != 0 else 0

    return {
        "consumption_comparison_chart_data": consumption_comparison_chart_data,
        "delta_t_chart_data": delta_t_chart_data,
        "flow_vs_consumption_chart_data": flow_vs_consumption_chart_data,
        "total_consumption_data": {
            "period_a": data_a.get("total_rth") or 0.0,
            "period_b": data_b.get("total_rth") or 0.0
        },
        "consumption_change_data": round(consumption_change_data, 2),
        "cost_change_data": round(cost_change_data, 2),
        "avg_delta_t_data": {
            "period_a": data_a.get("avg_delta_t") or 0.0,
            "period_b": data_b.get("avg_delta_t") or 0.0
        },
        "rth_delta_data": {
            "period_a": (total_a) if total_a != 0 else 0.0,
            "period_b": (total_b) if total_b != 0 else 0.0
        },
        "avg_flow_data": {
            "period_a": data_a.get("avg_flow") or 0.0,
            "period_b": data_b.get("avg_flow") or 0.0
        }
    }