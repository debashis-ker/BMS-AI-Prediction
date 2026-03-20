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

    log.info(f"[{equipment_id}] Step 1: Fetching MPC Optimization history...")
    mpc_data = await get_optimization_history(from_date=from_date, to_date=to_date, equipment_id=equipment_id, session=session)
    if mpc_data.get('count', 0) == 0:
        log.warning(f"[{equipment_id}] Aborting: No MPC records found in this period.")
        raise HTTPException(
            status_code=500, 
            detail="No MPC Setpoint records found within this time period.")
    
    log.info(f"[{equipment_id}] Step 2: Fetching raw telemetry from Cassandra...")
    raw_data = fetch_cassandra_data(
        equipment_id=equipment_id, from_date=from_date, to_date=to_date, 
        datapoints=["SpTREff", "TempSp1", "ChwFb"]
    )
    if not raw_data:
        log.warning(f"[{equipment_id}] Aborting: No Cassandra data found within this period.")
        raise HTTPException(
            status_code=500, 
            detail="No Cassandra data found within this time period.")

    log.info(f"[{equipment_id}] Step 3: Processing DataFrames...")
    raw_data_df = data_pipeline(raw_data).reset_index().rename(columns={"data_received_on": "timestamp"})
    raw_data_df['timestamp'] = pd.to_datetime(raw_data_df['timestamp']).dt.tz_localize(None)
    
    mpc_results_df = pd.DataFrame(mpc_data['results'])
    mpc_results_df['timestamp'] = (pd.to_datetime(mpc_results_df['timestamp_utc']) + timedelta(minutes=1)).dt.tz_localize(None)
    mpc_results_df = mpc_results_df.rename(columns={'timestamp_utc': 'mpc_run_time'})
    mpc_results_df = mpc_results_df[['timestamp', 'optimized_setpoint', 'mode', 'mpc_run_time']]

    combined_df = pd.merge(raw_data_df, mpc_results_df, on='timestamp', how='outer').sort_values('timestamp')
    
    combined_df[['optimized_setpoint', 'mode', 'mpc_run_time']] = combined_df[['optimized_setpoint', 'mode', 'mpc_run_time']].ffill()
    combined_df = combined_df.dropna(subset=['SpTREff'])
    
    log.info(f"[{equipment_id}] Data synchronized. Merged rows: {len(combined_df)}")

    log.info(f"[{equipment_id}] Step 4: Finding manual overrides...")
    temp_results = []
    global_last_stable_row = None

    for i in range(len(combined_df)):
        row = combined_df.iloc[i]
        curr_ts = row['timestamp']
        curr_sp = row['SpTREff']
        ai_sp = row['optimized_setpoint']
        
        if curr_sp == ai_sp:
            global_last_stable_row = row
            log.debug(f"[{equipment_id}] Stable Baseline Updated at {curr_ts}: Sp={curr_sp}")

        if curr_sp != ai_sp:
            if curr_ts < eval_start_time:
                log.debug(f"[{equipment_id}] IGNORE (Priming Window): Override at {curr_ts} (Sp:{curr_sp} != AI:{ai_sp})")
                continue

            current_stable_row = None
            for j in range(i - 1, -1, -1):
                if combined_df.iloc[j]['SpTREff'] == combined_df.iloc[j]['optimized_setpoint']:
                    current_stable_row = combined_df.iloc[j]
                    break
            
            effective_stable_row = current_stable_row if current_stable_row is not None else global_last_stable_row
            
            if effective_stable_row is not None:
                stable_sp = effective_stable_row['SpTREff']
                sptreff_diff = round(abs(curr_sp - stable_sp), 4)
                
                if sptreff_diff > 0:
                    log.info(f"[{equipment_id}] VALID OVERRIDE: Time={curr_ts} | Actual={curr_sp} | Baseline={stable_sp} | Diff={sptreff_diff} | AI_Cycle={row['mpc_run_time']}")
                    temp_results.append({
                        "timestamp": curr_ts.isoformat(),
                        "timestamp_utc": row['mpc_run_time'], 
                        "equipment_id": equipment_id,
                        "mode": row['mode'],
                        "SpTREff_diff": sptreff_diff,
                        "TempSp1_diff": round(abs(row['TempSp1'] - effective_stable_row['TempSp1']), 4),
                        "ChwFb_diff": round(abs(row['ChwFb'] - effective_stable_row['ChwFb']), 4) if 'ChwFb' in row else 0
                    })
            else:
                log.warning(f"[{equipment_id}] POTENTIAL ISSUE at {curr_ts} (Sp:{curr_sp} != AI:{ai_sp}) but NO STABLE BASELINE found in the last 40m.")


    if not temp_results:
        log.warning(f"[{equipment_id}] Result: No manual overrides detected in the 30-minute evaluation window.")
        return []

    log.info(f"[{equipment_id}] Step 5: Deduplicating {len(temp_results)} override rows...")
    final_df = pd.DataFrame(temp_results)
    final_df = final_df.loc[final_df.groupby('timestamp_utc')['SpTREff_diff'].idxmax()]
    
    final_dict = final_df.sort_values('timestamp').replace({np.nan: None}).to_dict(orient='records')
    log.info(f"[{equipment_id}] Final Result: Successfully extracted {len(final_dict)} deduplicated override events.")
    
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
