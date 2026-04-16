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
import numpy as np
import requests

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

async def calculate_setpoint_diffs(equipment_id="Ahu1", from_date=None, to_date=None, session=None):
    if not (from_date and to_date):
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        to_date_dt = now_utc
        from_date_dt = now_utc - timedelta(minutes=50)
        eval_start_time = now_utc - timedelta(minutes=30)
        from_date = from_date_dt.strftime('%Y-%m-%dT%H:%M:%S')
        to_date = to_date_dt.strftime('%Y-%m-%dT%H:%M:%S')
        log.info(f"[{equipment_id}] No dates: Fetching 50m, evaluating last 30m.")
    else:
        from_date_dt = datetime.fromisoformat(from_date.replace('Z', '')).replace(tzinfo=None)
        to_date_dt = datetime.fromisoformat(to_date.replace('Z', '')).replace(tzinfo=None)    
        eval_start_time = from_date_dt + timedelta(minutes=20) 
        log.info(f"[{equipment_id}] Dates provided: Fetching 50m range, skipping first 20m priming.")

    log.info(f"[{equipment_id}] Time Specs: FetchStart={from_date_dt}, EvalStart={eval_start_time}, End={to_date_dt}")

    mpc_data = await get_optimization_history(from_date=from_date, to_date=to_date, equipment_id=equipment_id, session=session)

    if mpc_data.get('count', 0) == 0:
        log.warning(f"[{equipment_id}] Aborting: No MPC records found in this period.")
        raise HTTPException(
            status_code=500, 
            detail="No MPC Setpoint records found within this time period.")
    
    raw_data = fetch_cassandra_data(
        equipment_id=equipment_id, from_date=from_date, to_date=to_date, 
        datapoints=["SpTREff","ChwFb","TempSp1","TempSp","AvgTmp"]
    )
    if not raw_data:
        log.warning(f"[{equipment_id}] Aborting: No Cassandra data found within this period.")
        raise HTTPException(
            status_code=500, 
            detail="No Cassandra data found within this time period.")

    log.info(f"[{equipment_id}] Step 3: Processing DataFrames...")
    raw_data_df = data_pipeline(raw_data).reset_index().rename(columns={"data_received_on": "timestamp"})
    raw_data_df['timestamp'] = pd.to_datetime(raw_data_df['timestamp']).dt.tz_localize(None)
    
    if 'TempSp1' not in raw_data_df.columns or raw_data_df['TempSp1'].isna().all():
        if 'AvgTmp' in raw_data_df.columns:
            log.info(f"[{equipment_id}] TempSp1 not found or empty. Falling back to 'AvgTmp'.")
            raw_data_df['TempSp1'] = raw_data_df['AvgTmp']
        elif 'TempSp' in raw_data_df.columns:
            log.info(f"[{equipment_id}] TempSp1 not found or empty. Filling with 'TempSp' values.")
            raw_data_df['TempSp1'] = raw_data_df['TempSp']
        else:
            log.warning(f"[{equipment_id}] Neither TempSp1 nor AvgTmp found in data.")

    mpc_results_df = pd.DataFrame(mpc_data['results'])
    mpc_results_df['timestamp'] = (pd.to_datetime(mpc_results_df['timestamp_utc']) + timedelta(minutes=1)).dt.tz_localize(None)
    mpc_results_df = mpc_results_df.rename(columns={'timestamp_utc': 'mpc_run_time'})
    mpc_results_df = mpc_results_df[['timestamp', 'optimized_setpoint', 'mode', 'mpc_run_time']]

    combined_df = pd.merge(raw_data_df, mpc_results_df, on='timestamp', how='outer').sort_values('timestamp')
    
    combined_df[['optimized_setpoint', 'mode', 'mpc_run_time']] = combined_df[['optimized_setpoint', 'mode', 'mpc_run_time']].ffill()
    combined_df = combined_df.dropna(subset=['SpTREff'])

    combined_df['SpTREff'] = combined_df['SpTREff'].round(1)
    combined_df['optimized_setpoint'] = combined_df['optimized_setpoint'].round(1)

    diff_df = combined_df[combined_df['SpTREff'] != combined_df['optimized_setpoint']].dropna(subset=['SpTREff', 'optimized_setpoint'])

    log.info(f"[{equipment_id}] Data synchronized. Rows with differences: \n{diff_df[['SpTREff', 'optimized_setpoint']].to_string() if not diff_df.empty else 'None'}")

    temp_results = []

    unusual_mask = (combined_df['SpTREff'] != combined_df['optimized_setpoint']) & (combined_df['timestamp'] >= eval_start_time)
    affected_mpc_times = combined_df[unusual_mask]['mpc_run_time'].dropna().unique()

    for mpc_time in affected_mpc_times:
        group = combined_df[combined_df['mpc_run_time'] == mpc_time]
        
        if len(group) < 2:
            continue

        idx_min = group['ChwFb'].idxmin()
        idx_max = group['ChwFb'].idxmax()

        row_min = group.loc[idx_min]
        row_max = group.loc[idx_max]

        chwfb_swing = round(abs(row_max['ChwFb'] - row_min['ChwFb']),2)
        
        sptreff_offset = round(abs(row_max['SpTREff'] - row_max['optimized_setpoint']),2)
        
        tempsp1_swing = round(abs(row_max['TempSp1'] - row_min['TempSp1']),2)

        if sptreff_offset > 0.5:
            temp_results.append({
                "timestamp": row_max['timestamp'].isoformat(),
                "timestamp_utc": mpc_time,                   
                "equipment_id": equipment_id,
                "mode": row_max['mode'],
                "SpTREff_diff": sptreff_offset,
                "TempSp1_diff": tempsp1_swing,
                "ChwFb_diff": chwfb_swing
            })

    if not temp_results:
        log.warning(f"[{equipment_id}] Result: No manual overrides detected with valid fluctuations.")
        return []
        
    final_df = pd.DataFrame(temp_results)
    final_dict = final_df.sort_values('timestamp').replace({np.nan: None}).to_dict(orient='records')
    
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
        MIN(timestamp_utc) as actual_from,
        MAX(timestamp_utc) as actual_to,
        COUNT(*) as record_count
    """
    
    query = f"SELECT {select_clause} FROM {keyspace}.\"{table_name}\" WHERE equipment_id = ?"
    params = [equipment_id]

    if start_date:
        query += " AND timestamp >= ?"
        params.append(parser.parse(start_date).replace(tzinfo=timezone.utc))
    if end_date:
        query += " AND timestamp <= ?"
        params.append(parser.parse(end_date).replace(tzinfo=timezone.utc))

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
            "from_date": result.actual_from if result.actual_from else None,
            "to_date": result.actual_to if result.actual_to else None,
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
        timestamp_utc
    """
    
    query = f"SELECT {select_clause} FROM {keyspace}.\"{table_name}\" WHERE equipment_id = ?"
    params = [equipment_id]

    if start_date:
        query += " AND timestamp >= ?"
        params.append(parser.parse(start_date).replace(tzinfo=timezone.utc))
    if end_date:
        query += " AND timestamp <= ?"
        params.append(parser.parse(end_date).replace(tzinfo=timezone.utc))

    try:
        log.info(f"Fetching raw records from {table_name}")
        
        stmt = session.prepare(query)
        rows = session.execute(stmt, params)

        records = []
        for row in rows:
            records.append({
                "equipment_id": row.equipment_id,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "timestamp_utc": row.timestamp_utc,
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
