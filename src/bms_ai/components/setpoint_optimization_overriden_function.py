from typing import Dict, Any, List, Optional
import pandas as pd
import os
import warnings
from openai import OpenAI
from dotenv import load_dotenv
import json
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.cassandra_utils import fetch_cassandra_data
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
from dateutil import parser
from cassandra.cluster import Session
from src.bms_ai.api.dependencies import get_cassandra_session
from fastapi import HTTPException, Depends
from src.bms_ai.mpc.mpc_inference_pipeline import InferenceConfig

load_dotenv()

log = setup_logger(__name__)

warnings.filterwarnings('ignore')

try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        log.warning("OPENAI_API_KEY not found in environment variables. ChatGPT features will be limited.")
        client = None
    else:
        client = OpenAI(api_key=openai_api_key)
        log.info("OpenAI client initialized successfully")
except Exception as e:
    log.error(f"Failed to initialize OpenAI client: {e}")
    client = None

async def get_optimization_history(
    equipment_id: str = "Ahu13",
    status: str = "success",
    summary_needed : bool = False,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    session: Session = Depends(get_cassandra_session)
):
    """
    Get optimization history for an equipment.
    
    By default, fetches records from the last 24 hours.
    
    Args:
        equipment_id: Equipment identifier
        status: Filter by optimization status - 'success' (default), 'all' for all records
        from_date: Start date in UTC ISO format (e.g., '2026-02-03T10:00:00'). Defaults to 24 hours ago.
        to_date: End date in UTC ISO format (e.g., '2026-02-03T22:00:00'). Defaults to now.
        
    Returns:
        List of optimization results
    """

    # Calculate default date range (last 24 hours)
    if not to_date:
        end_time = datetime.utcnow()
    else:
        try:
            end_time = datetime.fromisoformat(to_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": f"Invalid to_date format: {to_date}. Use ISO format (e.g., '2026-02-03T22:00:00')",
                    "error_type": "INVALID_DATE_FORMAT"
                }
            )
    
    if not from_date:
        start_time = end_time - timedelta(hours=24)
    else:
        try:
            start_time = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": f"Invalid from_date format: {from_date}. Use ISO format (e.g., '2026-02-03T10:00:00')",
                    "error_type": "INVALID_DATE_FORMAT"
                }
            )
    
    log.info(f"[MPC History] Fetching history for {equipment_id}, status={status}, from={start_time}, to={end_time}")
    
    config = InferenceConfig(equipment_id=equipment_id)
    table_name = config.table_name
    
    status_filter = "AND optimization_status IN ('success', 'bypassed')" if status != "all" else ""
    
    start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    
    query = f"""
        SELECT * FROM {table_name}
        WHERE equipment_id = '{equipment_id}'
        AND timestamp_utc >= '{start_str}'
        AND timestamp_utc <= '{end_str}'
        {status_filter}
        ALLOW FILTERING;
    """
    
    try:
        rows = session.execute(query)
        
        results = []
        for row in rows:
            record = {col: getattr(row, col, None) for col in row._fields}
            
            if record.get('used_features'):
                try:
                    record['used_features'] = json.loads(record['used_features'])
                except:
                    pass
            
            if record.get('timestamp_utc'):
                record['timestamp_utc'] = record['timestamp_utc'].isoformat()
            
            results.append(record)

        return {
            "success": True,
            "equipment_id": equipment_id,
            "start_date": start_time.strftime('%Y-%m-%d %H:%M:%S'), 
            "end_date": end_time.strftime('%Y-%m-%d %H:%M:%S'),     
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

'''Setpoint Unusal Activity of Overriding Optimization Functions'''

def data_pipeline(records: List[Dict[str, Any]], STANDARD_DATE_COLUMN: str = "data_received_on") -> pd.DataFrame:
    """Preprocesses raw data: Fixes leakage and adds thermal lags."""
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    
    if 'monitoring_data' in df.columns:
        mapping = {'inactive': 0.0, 'active': 1.0}
        df['monitoring_data'] = df['monitoring_data'].replace(mapping, regex=False)
        df['monitoring_data'] = pd.to_numeric(df['monitoring_data'], errors='coerce')
    
    df[STANDARD_DATE_COLUMN] = pd.to_datetime(df[STANDARD_DATE_COLUMN], errors='coerce')
    if df[STANDARD_DATE_COLUMN].dt.tz is not None:
        df[STANDARD_DATE_COLUMN] = df[STANDARD_DATE_COLUMN].dt.tz_localize(None)

    aggregated = df.groupby(['equipment_id',STANDARD_DATE_COLUMN, 'datapoint'])['monitoring_data'].agg('first')
    result_df = aggregated.unstack(level='datapoint').reset_index()

    result_df = result_df.sort_values(STANDARD_DATE_COLUMN).set_index(STANDARD_DATE_COLUMN)
    return result_df

def get_optimization_analysis(data):
    unusual_timestamps_utc = []

    result = []

    for records in data['results']:
        if ((records['used_features']["occupied"] == 1) and records['used_features']["occupied_setpoint"] != records["actual_sptreff"] and records['actual_sptreff'] != records['optimized_setpoint']):
            log.info('All conditions matched within occupied session of setpoint optimization : Screen is Occupied, Occupied Setpoint not equals to actual setpoint, actual setpoint not equals to optimized setpoint')
            log.info(f"timestamp : {records['timestamp_utc']}")
            log.info(f"timestamp_sharjah : {records['timestamp_sharjah']}")
            result.append({"timestamp": records['timestamp_utc'], "timestamp_sharjah" : records['timestamp_sharjah'], "mode" : records['mode'] , "optimized_setpoint": records['optimized_setpoint'], "actual_sptreff": records['actual_sptreff'], "equipment_id": records['equipment_id'], "actual_tempsp1": records['actual_tempsp1']})

        elif ((records['used_features']["occupied"] == 0) and records['used_features']["unoccupied_setpoint"] != records["actual_sptreff"] and records['actual_sptreff'] != records['optimized_setpoint']):
            log.info('All conditions matched within unoccupied session of setpoint optimization : Screen is unoccupied, unoccupied Setpoint not equals to actual setpoint, actual setpoint not equals to optimized setpoint')
            log.info(f"timestamp : {records['timestamp_utc']}")
            log.info(f"timestamp_sharjah : {records['timestamp_sharjah']}")
            result.append({"timestamp": records['timestamp_utc'],  "timestamp_sharjah" : records['timestamp_sharjah'], "mode" : records['mode'] , "optimized_setpoint": records['optimized_setpoint'], "actual_sptreff": records['actual_sptreff'], "equipment_id": records['equipment_id'], "actual_tempsp1": records['actual_tempsp1']})

    if len(result) > 0:
        for records in result:
            unusual_timestamps_utc.append(records['timestamp'])
        log.info(f"timestamps of occupied sessions with setpoint optimization issues: {unusual_timestamps_utc}")
    return result

async def calculate_setpoint_diffs(equipment_id="Ahu1", from_date=None, to_date=None, session=None):
    """
    Analyzes setpoint issues by comparing telemetry data before and after an issue occurs.
    The window extension logic has been removed to ensure stable performance.
    """

    if not (from_date and to_date):
        now_utc = datetime.now(timezone.utc)
        start_dt = now_utc - timedelta(minutes=30)
        from_date = start_dt.strftime('%Y-%m-%dT%H:%M:%S')
        to_date = now_utc.strftime('%Y-%m-%dT%H:%M:%S')
        log.info(f"Using default UTC range: {from_date} to {to_date}")

    data = await get_optimization_history(equipment_id=equipment_id,from_date=from_date, to_date=to_date, session=session)
    
    unusual_setpoint_data = get_optimization_analysis(data)

    final_dict = []
    opt_lookup = {item['timestamp']: item['optimized_setpoint'] for item in unusual_setpoint_data}

    for record in unusual_setpoint_data:
        timestamp = record['timestamp']
        timestamp_sharjah = record.get('timestamp_sharjah')
        mode = record.get('mode')
        equip_id = record.get('equipment_id')
        
        dt_obj = datetime.fromisoformat(timestamp)
        current_from_date = (dt_obj - timedelta(minutes=20)).isoformat()
        current_to_date = (dt_obj + timedelta(minutes=10)).isoformat()

        raw_data = fetch_cassandra_data(
            equipment_id=equip_id, 
            from_date=current_from_date, 
            to_date=current_to_date
        )
        
        if not raw_data:
            log.info(f"No telemetry data found for {timestamp}")
            continue

        processed_df = data_pipeline(raw_data).reset_index()
        current_val = opt_lookup.get(timestamp)

        issue_data_df = processed_df[processed_df['SpTREff'] != current_val]

        if issue_data_df.empty:
            continue

        for idx in issue_data_df.index:
            if idx + 1 < len(processed_df):
                row_curr = processed_df.iloc[idx]
                row_next = processed_df.iloc[idx + 1]
                
                
                diff_entry = {
                    "timestamp": timestamp,
                    "timestamp_sharjah": timestamp_sharjah,
                    "mode": mode,
                    "equipment_id": equip_id,
                    "SpTREff_diff": abs(row_next.get('SpTREff', 0) - row_curr.get('SpTREff', 0)),
                    "TempSp1_diff": abs(row_next.get('TempSp1', 0) - row_curr.get('TempSp1', 0)),
                    "ChwFb_diff": abs(row_next.get('ChwFb', 0) - row_curr.get('ChwFb', 0))
                }
                final_dict.append(diff_entry)

    return final_dict

def fetch_setpoint_diffs_averages(
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2", 
    equipment_id: str = "Ahu1", 
    session: Any = None, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    
    table_suffix = building_id.replace('-', '').lower()
    table_name = f"setpoint_overriden_data_{table_suffix}"
    keyspace = os.getenv("CASSANDRA_KEYSPACE","user_keyspace")

    select_clause = """
        AVG(chwfb_diff) as avg_chwfb, 
        AVG(sptreff_diff) as avg_sptreff, 
        AVG(tempsp1_diff) as avg_tempsp1,
        MIN(timestamp) as actual_from,
        MAX(timestamp) as actual_to,
        COUNT(*) as record_count
    """
    
    query = f"SELECT {select_clause} FROM {keyspace}.\"{table_name}\" WHERE equipment_id = ?"
    params = [equipment_id]

    if start_date:
        query += " AND timestamp >= ?"
        params.append(parser.parse(start_date).replace(tzinfo=timezone.utc)) #type: ignore
    if end_date:
        query += " AND timestamp <= ?"
        params.append(parser.parse(end_date).replace(tzinfo=timezone.utc)) #type: ignore

    try:
        log.info(f"Executing Aggregate Query on {table_name}")
        
        stmt = session.prepare(query)
        result = session.execute(stmt, params).one()

        if not result or result.record_count == 0:
            return {
                "success": False,
                "message": "No data found for the specified criteria.",
                "averages": None,
                "total_records": 0
            }

        return {
            "from_date": result.actual_from.isoformat() if result.actual_from else None,
            "to_date": result.actual_to.isoformat() if result.actual_to else None,
            "total_records": result.record_count,
            "avg_chwfb_diff": round(float(result.avg_chwfb or 0), 4),
            "avg_sptreff_diff": round(float(result.avg_sptreff or 0), 4),
            "avg_tempsp1_diff": round(float(result.avg_tempsp1 or 0), 4)
        }

    except Exception as e:
        log.error(f"Aggregation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database aggregation failed: {str(e)}")

def fetch_setpoint_diffs(
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2", 
    equipment_id: str = "Ahu1", 
    session: Any = None, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    
    table_suffix = building_id.replace('-', '').lower()
    table_name = f"setpoint_overriden_data_{table_suffix}"
    keyspace = os.getenv("CASSANDRA_KEYSPACE", "user_keyspace")

    select_clause = """
        equipment_id, 
        timestamp, 
        chwfb_diff, 
        mode, 
        sptreff_diff, 
        tempsp1_diff, 
        timestamp_sharjah
    """
    
    query = f"SELECT {select_clause} FROM {keyspace}.\"{table_name}\" WHERE equipment_id = ?"
    params = [equipment_id]

    if start_date:
        query += " AND timestamp >= ?"
        params.append(parser.parse(start_date).replace(tzinfo=timezone.utc)) #type: ignore
    if end_date:
        query += " AND timestamp <= ?"
        params.append(parser.parse(end_date).replace(tzinfo=timezone.utc)) #type: ignore

    try:
        log.info(f"Fetching raw records from {table_name}")
        
        stmt = session.prepare(query)
        rows = session.execute(stmt, params)

        records = []
        for row in rows:
            records.append({
                "equipment_id": row.equipment_id,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "timestamp_sharjah": row.timestamp_sharjah,
                "mode": row.mode,
                "chwfb_diff": round(float(row.chwfb_diff or 0), 4),
                "sptreff_diff": round(float(row.sptreff_diff or 0), 4),
                "tempsp1_diff": round(float(row.tempsp1_diff or 0), 4)
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
