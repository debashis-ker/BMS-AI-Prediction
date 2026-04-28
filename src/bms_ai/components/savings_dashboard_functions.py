from typing import Any, Dict, Optional
import os
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta, timezone
import warnings
from src.bms_ai.api.routers.energy_consumption import GetEnergyDataRequest, get_energy_data
from src.bms_ai.components.setpoint_optimization_summarizer_functions import generate_optimization_summary_response
from dateutil.relativedelta import relativedelta
from src.bms_ai.logger_config import setup_logger
from fastapi import HTTPException

log = setup_logger(__name__)

warnings.filterwarnings('ignore')

prompts = {
    'consumption_comparison_summary': """Role: Energy Financial Controller. Task: Compare two specific time periods. State the percentage change in consumption and cost. Identify the period with the highest peak cost from the chart data. Constraints: Objective tone, use AED and RTh units in every sentence, state 'Value is unavailable' for nulls, keep the answer concise in 2-3 sentences only.""",
    'efficiency_metrics_summary': """"Role: Thermal Efficiency Engine. Task: Correlate Flow, Delta T, and RTh. Explain the relationship between Average Flow and Average Delta T across both periods. Determine if a higher flow resulted in a proportional increase in RTh delta. Constraints: Strictly technical data correlations, no introductory filler, replace underscores with spaces, , keep the answer concise in 2-3 sentences only.""",
    'situation_analysis_summary': """Role: Building Environment Auditor. Task: Analyze thermal conditions based on occupancy. Compare maximum and minimum temperatures as well as count between occupied and unoccupied states. Explicitly mention the average chilled water feedback (chwfb diff) for both states to show the shift in cooling demand. Constraints: Clinical tone, no bolding, every sentence must contain a temperature value, keep the answer concise in 2-3 sentences only.""",
    'AHU_performance_summary': """"Role: HVAC Operational Analyst. Task: Summarize AHU optimization activity. Identify the average count of number of optimization across all AHUs mentioned in the dictionary of ahu_optimization_counts. Identify the top AHU with the highest optimization counts. Constraints: Neutral tone, replace underscores with spaces, mention average of optimization every time, , keep the answer concise in 2-3 sentences only.""",
    'main_progress_dashboard_summary': """Role: Executive Facilities Performance Analyst. 
        Task: Summarize the building's optimization progress with a focus on operational excellence. 
        Analysis: 
        1. Use 'total_optimizations' to highlight active system engagement (e.g., "Optimization actions (X instances)"). 
        2. Correlate 'average_chwfb_diff' as the primary mechanical driver for efficiency (e.g., "Achieving a X% Chilled Valve Difference"). 
        3. Link 'average_temperature_diff' and 'comfort_level' to demonstrate thermal stability and tenant satisfaction (e.g., "Maintaining a X°C Space Temp Difference with X% comfort compliance").
        Operational Context: If comfort_level is above 95%, describe it as "perfect compliance" or "high precision." If average_chwfb_diff is positive, frame it as "dynamic load-matching."
        Constraints: Start with a brief positive opening. Use professional business terminology. Replace all underscores with spaces. Limit to 3 concise, impactful sentences."""
}

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

def comfort_savings_dashboard_data(
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2", 
    session: Any = None, 
    from_date: Optional[str] = None, 
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetches and processes comfort-related data for dashboard display.
    Currently a placeholder for future comfort metric implementations.
    """
    try:
        data = fetch_building_optimization_history(building_id, session, from_date, to_date)

        if not data["success"] or not data["data"]:
            return {"success": False, "message": "No history data found."}

        occ_temps = [r["actual_temp"] for r in data["data"] if r["mode"] == "occupied" and r["actual_temp"] is not None]

        comfort_percentage = (sum(1 for temp in occ_temps if 20 <= temp <= 26) / len(occ_temps)) * 100 if occ_temps else None

        return {
            "comfort_percentage": round(comfort_percentage, 2) if comfort_percentage is not None else None
        }              
    
    except Exception as e:
        log.error(f"Comfort Dashboard Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing comfort dashboard data: {str(e)}")

def dashboard_savings_data(
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2", 
    session: Any = None, 
    from_date: Optional[str] = None, 
    to_date: Optional[str] = None,
    summary_needed : bool = False
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
                "average_temperature_diff": 0.0,
                "ahu_optimization_counts": {}
            }
        
        total_optimizations = data["total_records"]
        average_temperature_diff = round(sum(record["tempsp1_diff"] for record in data["data"]) / total_optimizations, 4) if total_optimizations > 0 else 0.0 
        comfort_percentage = comfort_savings_dashboard_data(building_id=building_id, session=session, from_date=from_date, to_date=to_date).get("comfort_percentage")
        chwfb_diffs = [r["chwfb_diff"] for r in data["data"] if r["chwfb_diff"] is not None]
        average_chwfb_diff = round(sum(r["chwfb_diff"] for r in data["data"] if r["chwfb_diff"] is not None) / len(chwfb_diffs), 4) if chwfb_diffs else None

        if not data.get("success") or not data.get("data"):
            log.warning(f"No optimization history found")
            return {}

        ahu_optimization_counts = {}
        for record in data["data"]:
            equipment_id = record.get("equipment_id")
            if equipment_id:
                if equipment_id not in ahu_optimization_counts:
                    ahu_optimization_counts[equipment_id] = {"optimization": 0}
                
                ahu_optimization_counts[equipment_id]["optimization"] += 1
        
        response = {
            "success": True,
            "total_optimizations": total_optimizations, 
            "average_temperature_diff": average_temperature_diff,
            "ahu_optimization_counts": ahu_optimization_counts,
            "average_chwfb_diff": average_chwfb_diff
        }

        if comfort_percentage is not None:
            response["comfort_level"] = comfort_percentage

        if summary_needed:
            AHU_performance_data = generate_optimization_summary_response(
                data={"ahu_optimization_counts": response["ahu_optimization_counts"]}, 
                prompt=prompts['AHU_performance_summary']
            )
            main_progress_dashboard_summary_data = generate_optimization_summary_response(
                data={k: v for k, v in response.items() if k in ['total_optimizations', 'average_temperature_diff', 'comfort_level', 'average_chwfb_diff']}, 
                prompt=prompts['main_progress_dashboard_summary']
            )
            response["AHU_performance_summary"] = AHU_performance_data
            response["main_progress_dashboard_summary"] = main_progress_dashboard_summary_data

        return response

    except Exception as e:
        log.error(f"Dashboard Data Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing dashboard data: {str(e)}")

def occupancy_dashboard(
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2", 
    session: Any = None, 
    from_date: Optional[str] = None, 
    to_date: Optional[str] = None,
    summary_needed : bool = False
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
        avg_occ_chwfb_diff = round(sum(occupied_chwfb_diffs) / len(occupied_chwfb_diffs), 4) if occupied_chwfb_diffs else None
        avg_unocc_chwfb_diff = round(sum(unoccupied_chwfb_diffs) / len(unoccupied_chwfb_diffs), 4) if unoccupied_chwfb_diffs else None
        
        occ_temps = [r["actual_temp"] for r in historical_data["data"] if r["mode"] in occ_modes and r["actual_temp"] is not None]
        unocc_temps = [r["actual_temp"] for r in historical_data["data"] if r["mode"] == "unoccupied" and r["actual_temp"] is not None]
        occ_count = len([r for r in setpoint_diff_data["data"] if r["mode"] in occ_modes])
        unocc_count = len([r for r in setpoint_diff_data["data"] if r["mode"] == "unoccupied"])

        response = {
            "success": True,
            "occupied_stats": {
                "count": occ_count,
                "max_temp": max(occ_temps) if occ_temps and occ_count > 0 else None,
                "min_temp": min(occ_temps) if occ_temps and occ_count > 0 else None,
                "average_chwfb_diff": avg_occ_chwfb_diff
            },
            "unoccupied_stats": {
                "count": unocc_count,
                "max_temp": max(unocc_temps) if unocc_temps and unocc_count > 0 else None,
                "min_temp": min(unocc_temps) if unocc_temps and unocc_count > 0  else None,
                "average_chwfb_diff": avg_unocc_chwfb_diff
            }
        }

        if summary_needed:
            situation_analysis_data = generate_optimization_summary_response(data={k: response[k] for k in ['occupied_stats','unoccupied_stats']}, prompt=prompts['situation_analysis_summary'])
            response["situation_analysis_summary"] = situation_analysis_data

        return response

    except Exception as e:
        log.error(f"Occupancy Dashboard Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_energy_comparison_data(
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2",
    period_a: Dict[str, Any] = None,
    period_b: Dict[str, Any] = None,
    frequency: str = "D",
    summary_needed: bool = False,
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

    start_a = parser.parse(period_a.get("from_date")).replace(tzinfo=timezone.utc)
    end_a = parser.parse(period_a.get("to_date")).replace(tzinfo=timezone.utc)
    start_b = parser.parse(period_b.get("from_date")).replace(tzinfo=timezone.utc)
    end_b = parser.parse(period_b.get("to_date")).replace(tzinfo=timezone.utc)

    freq_config = {
        "H": {"step": relativedelta(hours=1), "base_attr": {"minute":0, "second":0, "microsecond":0}},
        "D": {"step": relativedelta(days=1), "base_attr": {"hour":0, "minute":0, "second":0, "microsecond":0}},
        "W": {"step": relativedelta(days=7), "base_attr": {"hour":0, "minute":0, "second":0, "microsecond":0}},
        "M": {"step": relativedelta(months=1), "base_attr": {"day":1, "hour":0, "minute":0, "second":0, "microsecond":0}},
        "Q": {"step": relativedelta(months=3), "base_attr": {"day":1, "hour":0, "minute":0, "second":0, "microsecond":0}},
    }

    f = frequency.upper()
    config = freq_config.get(f, freq_config["D"])
    step = config["step"]

    base_a = start_a.replace(**config["base_attr"])
    base_b = start_b.replace(**config["base_attr"])
    
    if f == "H":
        total_steps = max(int((end_a - base_a).total_seconds() // 3600), int((end_b - base_b).total_seconds() // 3600)) + 1
    elif f == "D":
        total_steps = max((end_a - base_a).days, (end_b - base_b).days) + 1
    elif f == "W":
        base_a -= timedelta(days=base_a.weekday())
        base_b -= timedelta(days=base_b.weekday())
        total_steps = max((end_a - base_a).days // 7, (end_b - base_b).days // 7) + 1
    elif f == "M":
        total_steps = max((end_a.year - base_a.year) * 12 + end_a.month - base_a.month, 
                          (end_b.year - base_b.year) * 12 + end_b.month - base_b.month) + 1
    elif f == "Q":
        base_a = base_a.replace(month=((base_a.month - 1) // 3) * 3 + 1)
        base_b = base_b.replace(month=((base_b.month - 1) // 3) * 3 + 1)
        total_steps = max(((end_a.year - base_a.year) * 12 + end_a.month - base_a.month) // 3, 
                          ((end_b.year - base_b.year) * 12 + end_b.month - base_b.month) // 3) + 1

    map_a = {parser.parse(r['period_start']): r for r in list_a}
    map_b = {parser.parse(r['period_start']): r for r in list_b}

    consumption_comparison_chart_data = {}
    delta_t_chart_data = {}
    flow_vs_consumption_chart_data = {}

    for i in range(total_steps):
        step_label = f"period_{i+1}"
        curr_a = base_a + (step * i)
        curr_b = base_b + (step * i)

        valid_a = start_a <= curr_a <= end_a
        valid_b = start_b <= curr_b <= end_b

        item_a = map_a.get(curr_a) if valid_a else None
        item_b = map_b.get(curr_b) if valid_b else None

        period_a_key = curr_a.isoformat() if (f in ["W", "M", "Q","D"] or valid_a) else None
        period_b_key = curr_b.isoformat() if (f in ["W", "M", "Q","D"] or valid_b) else None


        consumption_comparison_chart_data[step_label] = {
            "period_a_cost": item_a.get("estimated_cost_aed") if item_a else None,
            "period_b_cost": item_b.get("estimated_cost_aed") if item_b else None,
            "period_a_rth": item_a.get("rth_value") if item_a else None,
            "period_b_rth": item_b.get("rth_value") if item_b else None,
            "period_a_key": period_a_key,
            "period_b_key": period_b_key,
        }
        delta_t_chart_data[step_label] = {
            "period_a": item_a.get("delta_t") if item_a else None,
            "period_b": item_b.get("delta_t") if item_b else None,
            "period_a_key": period_a_key,
            "period_b_key": period_b_key,
        }

        flow_vs_consumption_chart_data[step_label] = {
            "period_a": item_a.get("avg_flow") if item_a else None,
            "period_b": item_b.get("avg_flow") if item_b else None,
            "period_a_key": period_a_key,
            "period_b_key": period_b_key,
        }

    total_a = data_a.get("total_rth", 0)
    total_b = data_b.get("total_rth", 0)
    total_cost_aed_a = data_a.get("total_cost_aed", 0)
    total_cost_aed_b = data_b.get("total_cost_aed", 0)
    cost_change_data = ((total_cost_aed_b - total_cost_aed_a) / total_cost_aed_a) * 100 if total_cost_aed_a != 0 else 0
    consumption_change_data = ((total_b - total_a) / total_a) * 100 if total_a != 0 else 0

    response = {
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

    if summary_needed:
            comparison_keys = ["consumption_comparison_chart_data", "total_consumption_data", "consumption_change_data", "cost_change_data"]
            comp_data = {k: response.get(k) for k in comparison_keys}
            response["consumption_comparison_summary"] = generate_optimization_summary_response(data=comp_data, prompt=prompts['consumption_comparison_summary'])

            efficiency_keys = ["avg_delta_t_data", "rth_delta_data", "avg_flow_data", "delta_t_chart_data", "flow_vs_consumption_chart_data"]
            eff_data = {k: response.get(k) for k in efficiency_keys}
            response["efficiency_metrics_summary"] = generate_optimization_summary_response(data=eff_data, prompt=prompts['efficiency_metrics_summary'])

    return response
