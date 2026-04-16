from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import HTTPException, status
from dateutil import parser
from cassandra.cluster import Session
from src.bms_ai.utils.setpoint_optimization_utils import fetch_movie_schedule, get_occupancy_status_for_timestamp
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.cassandra_utils import fetch_cassandra_data
from datetime import datetime, timezone, timedelta
import os
import numpy as np
import json
from pathlib import Path
from cassandra.concurrent import execute_concurrent

log = setup_logger(__name__)

STATE_FILE = "artifacts/ahu13_alarm_timestamp_track/last_processed_state.json"

FIXED_SYSTEM_TYPE = "AHU"
KEYSPACE_NAME = os.getenv('CASSANDRA_KEYSPACE')
TABLE_SUFFIX = "alarm_data"

if not KEYSPACE_NAME:
    log.error("CASSANDRA_KEYSPACE environment variable is not set. Database operations will fail.")

def check_freeze_alarm(df: pd.DataFrame, config: Dict[str, Any] = None, hist_states: pd.Series = None) -> pd.Series:
    """
    Logic: Evaluates if the Supply Air Temperature (TempSu) drops below a specified threshold, indicating a potential freeze risk.
    Trigger:
    - A "hit" occurs when 'TempSu' < `temp_threshold` (default 5.0).
    - Warning (1): Triggered if the number of hits is >= `warning_hits` (default 1).
    - Critical (-1): Triggered if the number of hits is >= `critical_hits` (default 3).
    Note: Requires the 'TempSu' column. Skips evaluation gracefully if the column is missing or entirely empty.
    """
    config = config or {}
    temp_threshold = config.get('supply_air_temp_threshold', 5.0)
    warn_hits = config.get('warning_hits', 1)
    crit_hits = config.get('critical_hits', 3)

    log.info("Evaluating Freeze Alarm logic.")
    required_cols = ['TempSu']
    missing = [c for c in required_cols if c not in df.columns or df[c].isna().all()]
    if missing: 
        log.warning(f"Required columns {missing} missing/empty. Skipping Freeze Alarm.")
        return pd.Series(0, index=df.index)
    
    raw_hit = (df['TempSu'] < temp_threshold).astype(int)

    if hist_states is not None and not hist_states.empty:
        hist_raw_hits = (hist_states != 0).astype(int)
        combined_hits = pd.concat([hist_raw_hits, raw_hit]).reset_index(drop=True)
    else:
        combined_hits = raw_hit
    reset_blocks = (combined_hits == 0).cumsum()
    consecutive_count = combined_hits.groupby(reset_blocks).cumsum()
    
    if hist_states is not None and not hist_states.empty:
        consecutive_count = pd.Series(consecutive_count.iloc[len(hist_states):].values, index=df.index)
    
    status_series = pd.Series(0, index=df.index)
    status_series[consecutive_count >= warn_hits] = 1  
    status_series[consecutive_count >= crit_hits] = -1
    log.debug(f"Freeze Alarm evaluation complete. Detected hits: {status_series.any()}")
    return status_series.astype(int)

def check_oscillation_alarm(df: pd.DataFrame, config: Dict[str, Any] = None, hist_states: pd.Series = None) -> pd.Series:
    """
    Logic: Detects rapid, unstable hunting/switching in the VFD Command (CmdVFD).
    Trigger:
    - A "hit" occurs when the absolute change in CmdVFD is exactly 1 for two consecutive samples (e.g., 0 -> 1 -> 0, 1 -> 0 -> 1).
    - Warning (1): Triggered if the number of hits is >= `warning_hits` (default 1).
    - Critical (-1): Triggered if the number of hits is >= `critical_hits` (default 2).
    Note: Requires the 'CmdVFD' column. Skips evaluation gracefully if the column is missing or entirely empty.
    """
    config = config or {}
    warn_hits = config.get('warning_hits', 1)
    crit_hits = config.get('critical_hits', 2)

    log.info("Evaluating Oscillation Alarm logic.")
    required_cols = ['CmdVFD']
    missing = [c for c in required_cols if c not in df.columns or df[c].isna().all()]
    if missing: 
        log.warning(f"Required columns {missing} missing/empty. Skipping Oscillation Alarm.")
        return pd.Series(0, index=df.index)
    
    changes = df['CmdVFD'].diff().abs().fillna(0)
    raw_hit = ((changes == 1) & (changes.shift(1) == 1)).astype(int)
    
    if hist_states is not None and not hist_states.empty:
        hist_raw_hits = (hist_states != 0).astype(int)
        combined_hits = pd.concat([hist_raw_hits, raw_hit]).reset_index(drop=True)
    else:
        combined_hits = raw_hit
    reset_blocks = (combined_hits == 0).cumsum()
    consecutive_count = combined_hits.groupby(reset_blocks).cumsum()
    
    if hist_states is not None and not hist_states.empty:
        consecutive_count = pd.Series(consecutive_count.iloc[len(hist_states):].values, index=df.index)
    
    status_series = pd.Series(0, index=df.index)
    status_series[consecutive_count >= warn_hits] = 1
    status_series[consecutive_count >= crit_hits] = -1
    log.debug(f"Oscillation Alarm evaluation complete. Detected hits: {status_series.any()}")
    return status_series.astype(int)

def check_tracking_alarm(df: pd.DataFrame, config: Dict[str, Any] = None, hist_states: pd.Series = None) -> pd.Series:
    """
    Logic: Monitors the percentage deviation between the VFD Speed Command (CMDSpdVFD) and actual Speed Feedback (FbVFD).
    Trigger:
    - Suppression: Only evaluates when 'CMDSpdVFD' > `fan_speed_command_run_threshold` (default 1.0) to ignore systems that are turned off.
    - Calculation: Relative Error = |CMDSpdVFD - FbVFD| / (CMDSpdVFD).
    - A "hit" occurs when Error > `command_feedback_deviation_threshold` (default 0.03 or 3%).
    - Warning (1): Triggered if the number of hits is >= `warning_hits` (default 1).
    - Critical (-1): Triggered if the number of hits is >= `critical_hits` (default 3).
    Note: Requires 'CMDSpdVFD' and 'FbVFD' columns. Skips evaluation gracefully if missing or entirely empty.
    """
    config = config or {}
    cmd_running_threshold = config.get('fan_speed_command_run_threshold', 1.0)
    error_threshold = config.get('command_feedback_deviation_threshold', 0.03)
    log.info(f"Tracking Alarm Config - fan_speed_command_run_threshold: {cmd_running_threshold}, command_feedback_deviation_threshold: {error_threshold}")
    warn_hits = config.get('warning_hits', 1)
    crit_hits = config.get('critical_hits', 3)

    log.info("Evaluating Tracking Alarm logic with 0-command suppression.")
    required_cols = ['CMDSpdVFD', 'FbVFD']
    missing = [c for c in required_cols if c not in df.columns or df[c].isna().all()]
    if missing: 
        log.warning(f"Required columns {missing} missing/empty. Skipping Tracking Alarm.")
        return pd.Series(0, index=df.index)
    
    is_running = df['CMDSpdVFD'] > cmd_running_threshold
    
    divisor = df['CMDSpdVFD'].replace(0, 1)
    error = (df['CMDSpdVFD'] - df['FbVFD']).abs() / divisor
    log.info(f"Tracking Alarm - Error: {error.max()}")
    
    raw_hit = ((error > error_threshold) & is_running).astype(int)
    
    if hist_states is not None and not hist_states.empty:
        hist_raw_hits = (hist_states != 0).astype(int)
        combined_hits = pd.concat([hist_raw_hits, raw_hit]).reset_index(drop=True)
    else:
        combined_hits = raw_hit
    reset_blocks = (combined_hits == 0).cumsum()
    consecutive_count = combined_hits.groupby(reset_blocks).cumsum()
    
    if hist_states is not None and not hist_states.empty:
        consecutive_count = pd.Series(consecutive_count.iloc[len(hist_states):].values, index=df.index)
    
    status_series = pd.Series(0, index=df.index)
    status_series[consecutive_count >= warn_hits] = 1
    status_series[consecutive_count >= crit_hits] = -1
    
    return status_series.astype(int)

def check_return_air_temp_alarm(df: pd.DataFrame, config: Dict[str, Any] = None, hist_states: pd.Series = None) -> pd.Series:
    """
    Logic: Identifies poor cooling/heating performance when the Fresh Air Damper is fully open.
    Trigger:
    - A "hit" occurs when the absolute difference |TRe - TempSu| > `supply_and_return_air_temp_diff_threshold` (default 5.0) AND Space is occupied AND AHU is ON (CmdVFD == 1.0).
    - Warning (1): Triggered if the number of hits is >= `warning_hits` (default 1).
    - Critical (-1): Triggered if the number of hits is >= `critical_hits` (default 3).
    Note: Requires 'TRe', 'TempSu', 'Occupied_Flag', and 'CmdVFD' columns. Skips evaluation gracefully if missing or entirely empty.
    """
    config = config or {}
    temp_diff_threshold = config.get('supply_and_return_air_temp_diff_threshold', 5.0)
    warn_hits = config.get('warning_hits', 1)
    crit_hits = config.get('critical_hits', 3)

    log.info("Evaluating Return Air Temp Alarm logic.")
    required_cols = ['TRe', 'Occupied_Flag', 'TempSu', 'CmdVFD']
    missing = [c for c in required_cols if c not in df.columns or df[c].isna().all()]
    if missing: 
        log.warning(f"Required columns {missing} missing/empty. Skipping Return Air Temp Alarm.")
        return pd.Series(0, index=df.index)
    
    raw_hit = (((df['TRe'] - df['TempSu']) > temp_diff_threshold) & (df['CmdVFD'] == 1.0) & (df['Occupied_Flag'] == 'Occupied')).astype(int)
    if hist_states is not None and not hist_states.empty:
        hist_raw_hits = (hist_states != 0).astype(int)
        combined_hits = pd.concat([hist_raw_hits, raw_hit]).reset_index(drop=True)
    else:
        combined_hits = raw_hit
    reset_blocks = (combined_hits == 0).cumsum()
    consecutive_count = combined_hits.groupby(reset_blocks).cumsum()
    
    if hist_states is not None and not hist_states.empty:
        consecutive_count = pd.Series(consecutive_count.iloc[len(hist_states):].values, index=df.index)
    
    status_series = pd.Series(0, index=df.index)
    status_series[consecutive_count >= warn_hits] = 1
    status_series[consecutive_count >= crit_hits] = -1
    return status_series.astype(int)

def check_heat_stress_alarm(df: pd.DataFrame, config: Dict[str, Any] = None, hist_states: pd.Series = None) -> pd.Series:
    """
    Logic: Detects uncomfortable temperatures for occupants during active hours, specifically when cooling fails to meet the setpoint despite high valve demand.
    Trigger:
    - A "hit" occurs when all of the following are true:
        1. 'ChwFb' > `max_cooling_valve_feedback_threshold` (default 100.0)
        2. |TempSp1 - SpTREff| > `setpoint_space_air_temp_diff_threshold` (default 5.0)
        3. 'Occupied_Flag' == 'Occupied'
        4. 'CmdVFD' == 1.0 (AHU is ON)
    - Warning (1): Triggered if the number of hits is >= `warning_hits` (default 1).
    - Critical (-1): Triggered if the number of hits is >= `critical_hits` (default 3).
    - Escalation: If currently in a Warning state (1) AND 'TempSp1' > `space_air_temp_uncomfortable_level_threshold` (default 26.0), it immediately upgrades to Critical (-1).
    Note: Requires 'TempSp1', 'Occupied_Flag', 'ChwFb', and 'SpTREff', 'CmdVFD' columns. Skips evaluation gracefully if missing or entirely empty.
    """
    config = config or {}
    diff_threshold = config.get('setpoint_space_air_temp_diff_threshold', 5.0) 
    temp_sp1_threshold = config.get('space_air_temp_uncomfortable_level_threshold', 26.0)
    chwfb_threshold = config.get('max_cooling_valve_feedback_threshold', 100.0)
    
    warn_hits = config.get('warning_hits', 1)
    crit_hits = config.get('critical_hits', 3)

    log.info("Evaluating Heat Stress Alarm logic.")
    required_cols = ['TempSp1', 'Occupied_Flag', 'ChwFb', 'SpTREff','CmdVFD']
    missing = [c for c in required_cols if c not in df.columns or df[c].isna().all()]
    if missing: 
        log.warning(f"Required columns {missing} missing/empty. Skipping Heat Stress Alarm.")
        return pd.Series(0, index=df.index)
    
    raw_hit = ((df['ChwFb'] > chwfb_threshold) & 
               ((df['TempSp1'] - df['SpTREff']).abs() > diff_threshold) & 
               (df['Occupied_Flag'] == 'Occupied') & (df['CmdVFD'] == 1.0)).astype(int)
               
    if hist_states is not None and not hist_states.empty:
        hist_raw_hits = (hist_states != 0).astype(int)
        combined_hits = pd.concat([hist_raw_hits, raw_hit]).reset_index(drop=True)
    else:
        combined_hits = raw_hit
    reset_blocks = (combined_hits == 0).cumsum()
    consecutive_count = combined_hits.groupby(reset_blocks).cumsum()
    
    if hist_states is not None and not hist_states.empty:
        consecutive_count = pd.Series(consecutive_count.iloc[len(hist_states):].values, index=df.index)
    
    status_series = pd.Series(0, index=df.index)
    status_series[consecutive_count >= warn_hits] = 1
    status_series[consecutive_count >= crit_hits] = -1
    
    upgrade_mask = (df['TempSp1'] > temp_sp1_threshold) & (df['Occupied_Flag'] == "Occupied") & (df['CmdVFD'] == 1.0) & (status_series == 1)
    status_series[upgrade_mask] = -1
    
    return status_series.astype(int)

ALARM_FUNCTIONS = {
    "Alarm_Freeze": check_freeze_alarm,
    "Alarm_Oscillation": check_oscillation_alarm,
    "Alarm_Tracking": check_tracking_alarm,
    "Alarm_Heat_Stress": check_heat_stress_alarm,
    "Alarm_Return_Air_Temp": check_return_air_temp_alarm
}

def get_last_timestamp(equipment_id: str) -> Optional[str]:
    """
    Reads the last successfully processed data timestamp for a specific equipment from a local JSON state file.
    Returns None if the file is missing, empty, or the equipment ID is not found.
    """
    state_path = Path(STATE_FILE)
    if not state_path.exists():
        try:
            log.info(f"State file path {STATE_FILE} not found. Creating directory structure.")
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump({}, f)
            return None
        except Exception as e:
            log.error(f"Failed to initialize state file: {e}")
            return None

    if state_path.stat().st_size == 0:
        log.info(f"State file {STATE_FILE} is empty.")
        return None

    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
            return data.get(equipment_id, {}).get("timestamp")
    except (json.JSONDecodeError, Exception) as e:
        log.warning(f"Failed to read {STATE_FILE} (likely corrupt): {e}")
        return None
    
def update_last_timestamp(equipment_id: str, timestamp: str, repeat_count: int = 0):
    """
    Persists the most recent data timestamp and a repeat_count to the local JSON state file.
    Used to maintain continuity and detect stale data streams across processing cycles.
    """
    state_path = Path(STATE_FILE)
    data = {}
    try:
        if not state_path.parent.exists():
            state_path.parent.mkdir(parents=True, exist_ok=True)

        if state_path.exists() and state_path.stat().st_size > 0:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
        
        data[equipment_id] = {"timestamp": timestamp, "repeat_count": repeat_count}
        
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=4)
        log.info(f"Updated last processed state for {equipment_id} to {timestamp}")
    except Exception as e:
        log.error(f"Failed to update state file {STATE_FILE}: {e}")

def data_pipeline(records: List[Dict[str, Any]], STANDARD_DATE_COLUMN: str = "data_received_on", ticket: str = "", ticket_type = None) -> pd.DataFrame:
    """
    Orchestrates raw BMS record transformation.
    Logic:
    1. Converts monitoring data to numeric values.
    2. Pivots data from long format to wide format (grouped by site, equipment, and time).
    3. Fetches movie schedules via external API.
    4. Maps screen occupancy status to specific equipment timestamps to generate the 'Occupied_Flag'.
    """
    log.info(f"Starting data pipeline with {len(records)} records.")
    if not records: 
        log.warning("Empty records list received in data_pipeline.")
        return pd.DataFrame()
        
    try:
        df = pd.DataFrame(records)
        prev_data = df.copy()
        
        if 'monitoring_data' in df.columns:
            df['monitoring_data'] = pd.to_numeric(df['monitoring_data'].replace({'inactive': 0.0, 'active': 1.0}), errors='coerce')
        
        df[STANDARD_DATE_COLUMN] = pd.to_datetime(df[STANDARD_DATE_COLUMN], errors='coerce',utc=True)
        if df[STANDARD_DATE_COLUMN].isnull().any():
             log.error("Found unparseable timestamps in data records.")
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="One or more timestamps could not be parsed.")

        result_df = df.groupby(['site', 'equipment_id', STANDARD_DATE_COLUMN, 'datapoint'])['monitoring_data'].agg('first').unstack(level='datapoint').reset_index()

        fallbacks = {
            'Ahu4': {'CMDSpdVFD': 'CmdSpsvfd'},
            'Ahu14': {'TempSp1': 'AvgTmp'},
            'Ahu16': {'TempSp1': 'TempSp'},
        }
        
        for equip, mapping in fallbacks.items():
            mask = result_df['equipment_id'] == equip
            if mask.any():
                for target_col, fallback_col in mapping.items():
                    if fallback_col in result_df.columns:
                        if target_col not in result_df.columns:
                            result_df[target_col] = np.nan
                        result_df.loc[mask, target_col] = result_df.loc[mask, target_col].fillna(result_df.loc[mask, fallback_col])

        meta_cols = ['asset_code','device_id','device_ip']
        for col in meta_cols:
            if col in prev_data.columns:
                result_df[col] = prev_data[col].iloc[:len(result_df)].values
        
        try:
            print("Fetching movie schedule...")
            log.info(f"Fetching movie schedule for ticket: {ticket}")
            if not ticket_type:
                schedule_data = fetch_movie_schedule(ticket=ticket)
            else:
                schedule_data = fetch_movie_schedule(ticket=ticket, ticket_type=ticket_type)
        except Exception as e:
            log.error(f"External API fetch_movie_schedule failed: {e}")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to fetch movie schedule from external service.")
        
        if not schedule_data:
            log.error(f"No schedule data found for ticket {ticket}.")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"No schedule data found for ticket {ticket}.")

        def map_equipment_to_screen(equip):
            return equip.replace('Ahu', 'Screen ')

        unique_timestamps = result_df[STANDARD_DATE_COLUMN].unique()
        occupancy_map = {}

        for ts in unique_timestamps:
            ts_dt = pd.to_datetime(ts).to_pydatetime()
            mask = result_df[STANDARD_DATE_COLUMN] == ts
            
            if mask.any():
                equip_id = result_df[mask]['equipment_id'].iloc[0]
                screen_name = map_equipment_to_screen(equip_id)

                occ_result = get_occupancy_status_for_timestamp(
                    schedule_data=schedule_data,
                    screen=screen_name,
                    timestamp_utc=ts_dt
                )
                
                if not occ_result:
                    log.error(f"Occupancy service returned null for {screen_name} at {ts_dt}")
                    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to fetch occupancy status from external service.")

                occupancy_map[ts] = 'Occupied' if occ_result.get('status') == 1 else 'Unoccupied'

        result_df['Occupied_Flag'] = result_df[STANDARD_DATE_COLUMN].map(occupancy_map)
        log.info(f"Pipeline completed successfully. Result size: {len(result_df)} rows.")
        return result_df

    except HTTPException as he:
        raise he
    except Exception as e:
        log.error(f"Critical error in data_pipeline: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data transformation failed: {str(e)}")
    
def alarm_evaluation(
    data: pd.DataFrame, 
    freeze_alarm_config: Dict[str, Any] = None,
    oscillation_alarm_config: Dict[str, Any] = None,
    tracking_alarm_config: Dict[str, Any] = None,
    heat_stress_alarm_config: Dict[str, Any] = None,
    return_air_temp_alarm_config: Dict[str, Any] = None,
    historical_alarms_df: pd.DataFrame = None
) -> pd.DataFrame:
    if data.empty or 'data_received_on' not in data.columns: return pd.DataFrame()

    data = data.sort_values('data_received_on')
    alarm_cols_detected = []
    
    config_mapping = {
        "Alarm_Freeze": freeze_alarm_config or {},
        "Alarm_Oscillation": oscillation_alarm_config or {},
        "Alarm_Tracking": tracking_alarm_config or {},
        "Alarm_Heat_Stress": heat_stress_alarm_config or {},
        "Alarm_Return_Air_Temp": return_air_temp_alarm_config or {}
    }
    
    for alarm_name, func in ALARM_FUNCTIONS.items():
        try:
            specific_config = config_mapping[alarm_name]
            alarm_df = data.copy()
            if 'last_n_minutes' in specific_config:
                cutoff = alarm_df['data_received_on'].max() - pd.Timedelta(minutes=specific_config['last_n_minutes'])
                alarm_df = alarm_df[alarm_df['data_received_on'] >= cutoff]
            elif 'from_date' in specific_config and 'to_date' in specific_config:
                from_dt = pd.to_datetime(specific_config['from_date'], utc=True)
                to_dt = pd.to_datetime(specific_config['to_date'], utc=True)
                alarm_df = alarm_df[(alarm_df['data_received_on'] >= from_dt) & (alarm_df['data_received_on'] <= to_dt)]

            hist_states = None
            if historical_alarms_df is not None and not historical_alarms_df.empty:
                if alarm_name in historical_alarms_df.columns:
                    hist_states = historical_alarms_df[alarm_name]

            if alarm_df.empty:
                data[alarm_name] = 0
            else:
                res_series = func(alarm_df, config=specific_config, hist_states=hist_states)
                data[alarm_name] = res_series
                data[alarm_name] = data[alarm_name].fillna(0).astype(int)
                
            alarm_cols_detected.append(alarm_name)
        except Exception as e:
            log.error(f"Error during {alarm_name} evaluation: {e}")

    filtered_df = data[data[alarm_cols_detected].any(axis=1)].copy()
    return filtered_df

def format_cassandra_output(data_records: List[Dict], limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Logic: Sanitizes and standardizes database records for JSON serialization and API response.
    Parameters:
    - data_records: A list of dictionaries returned from a Cassandra query.
    - limit: Optional integer to truncate the result list.
    Logic Flow:
    1. Iterates through records to ensure the 'data_received_on' (timestamp) field is timezone-aware.
    2. Force-converts all timestamps to UTC using 'astimezone(timezone.utc)' to prevent client-side offset errors.
    3. Applies a slice to the list if a 'limit' is specified.
    Returns: A dictionary containing a standardized "data" list.
    """
    formatted_list = []
    try:
        for record in data_records:
            if 'data_received_on' in record and record['data_received_on']:
                if hasattr(record['data_received_on'], 'astimezone'):
                    record['data_received_on'] = record['data_received_on'].astimezone(timezone.utc)
            formatted_list.append(record)

        if limit is not None:
            formatted_list = formatted_list[:limit]

        return {"data": formatted_list}
    except Exception as e:
        log.error(f"Error while formatting Cassandra output: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error formatting database output.")

def store_data(data_chunk: List[Dict], building_id: str, session: Session) -> int:
    """
    Handles bulk insertion of alarm records into Cassandra.
    Logic:
    1. Dynamically creates the building-specific table if it does not exist (matching the required schema).
    2. Sanitizes records by removing Null/NaN values.
    3. Filters out unmapped fallback columns to prevent database schema crashes.
    4. Performs strict type-casting (NumPy to Python native) to ensure driver compatibility.
    5. Uses parameterized SQL with tuple-based execution to prevent formatting errors.
    """
    if not data_chunk: 
        log.info("No active alarm records to persist.")
        return 0
    
    table_name = f"{TABLE_SUFFIX}_{building_id.replace('-', '').lower()}"
    
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {KEYSPACE_NAME}."{table_name}" (
        site text,
        equipment_id text,
        timestamp timestamp,
        "Alarm_Freeze" int,
        "Alarm_Heat_Stress" int,
        "Alarm_Oscillation" int,
        "Alarm_Return_Air_Temp" int,
        "Alarm_Tracking" int,
        "CMDSpdVFD" double,
        "CmdVFD" double,
        "FbFAD" double,
        "FbVFD" double,
        "Occupied_Flag" text,
        "TRe" double,
        "TempSp1" double,
        "TempSu" double,
        "ChwFb" double,
        "SpTREff" double,
        asset_code text,
        device_id text,
        device_ip text,
        PRIMARY KEY ((site, equipment_id), timestamp)
    ) WITH CLUSTERING ORDER BY (timestamp DESC);
    """
    try:
        session.execute(create_table_query)
        log.debug(f"Ensured table {table_name} exists.")
    except Exception as e:
        log.error(f"Failed to initialize table schema for {table_name}: {e}")
        raise HTTPException(status_code=500, detail="Database schema initialization failed.")

    allowed_columns = {
        'site', 'equipment_id', 'timestamp', 'CMDSpdVFD', 'CmdVFD', 'FbFAD', 
        'FbVFD', 'Occupied_Flag', 'TRe', 'TempSp1', 'TempSu', 'ChwFb', 'SpTREff', 
        'asset_code', 'device_id', 'device_ip'
    }

    statements_and_params = []

    for record in data_chunk:
        try:
            clean_record = {k: v for k, v in record.items() if pd.notna(v)}
            
            if 'data_received_on' in clean_record:
                clean_record['timestamp'] = clean_record.pop('data_received_on')

            if not all(k in clean_record for k in ['site', 'equipment_id', 'timestamp']):
                log.warning(f"Skipping record: Missing primary keys. Have: {list(clean_record.keys())}")
                continue

            final_data = {}
            for k, v in clean_record.items():
                if k in allowed_columns or str(k).startswith("Alarm_"):
                    if "Alarm_" in k:
                        final_data[k] = int(v)
                    elif isinstance(v, (np.integer, int)):
                        final_data[k] = int(v)
                    elif isinstance(v, (np.floating, float)):
                        final_data[k] = float(v)
                    elif isinstance(v, (pd.Timestamp, datetime)):
                        final_data[k] = v.to_pydatetime() if hasattr(v, 'to_pydatetime') else v
                    else:
                        final_data[k] = str(v)

            if not final_data:
                continue

            columns = list(final_data.keys())
            placeholders = ", ".join(["%s" for _ in columns])
            column_names = ", ".join([f'"{c}"' for c in columns])
            
            query = f'INSERT INTO {KEYSPACE_NAME}."{table_name}" ({column_names}) VALUES ({placeholders})'
            params = tuple(final_data[c] for c in columns)
            
            statements_and_params.append((query, params))
            
        except Exception as e:
            error_ts = record.get('data_received_on') or record.get('timestamp')
            log.error(f"Data formatting failed for record {error_ts}: {e}")

    rows_affected = 0
    if statements_and_params:
        try:
            results = execute_concurrent(session, statements_and_params, concurrency=100)
            for success, result_or_exc in results:
                if success:
                    rows_affected += 1
                else:
                    log.error(f"Cassandra concurrent INSERT failed: {result_or_exc}")
                    
        except Exception as e:
            log.error(f"Concurrent batch execution failed: {e}")

    log.info(f"Successfully stored {rows_affected} records concurrently.")
    return rows_affected

def save_data_to_cassandra(
    session: Session, building_id: str, equipment_id: str, system_type: str = "AHU", 
    ticket: str = "", ticket_type: str = "", freeze_alarm_config=None, oscillation_alarm_config=None, tracking_alarm_config=None, 
    heat_stress_alarm_config=None, return_air_temp_alarm_config=None
) -> Dict[str, Any]:
    try:
        log.info(f"Starting real-time sync for {equipment_id} in building {building_id}")
        now_utc = datetime.now(timezone.utc)
        
        def format_bms_time(dt_obj):
            return dt_obj.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"

        is_explicit_range = False
        all_configs = [freeze_alarm_config, oscillation_alarm_config, tracking_alarm_config, heat_stress_alarm_config, return_air_temp_alarm_config]
        
        earliest_from = now_utc - timedelta(minutes=30)
        latest_to = now_utc
        max_rows_needed = 3 
        conf_from = None
        conf_to = None

        for conf in all_configs:
            if conf and isinstance(conf, dict):
                req_rows = max(conf.get('warning_hits', 1), conf.get('critical_hits', 3))
                max_rows_needed = max(max_rows_needed, req_rows)
                
                if 'from_date' in conf and 'to_date' in conf:
                    is_explicit_range = True
                    dt_from = pd.to_datetime(conf['from_date'], utc=True)
                    dt_to = pd.to_datetime(conf['to_date'], utc=True)
                    conf_from = min(conf_from, dt_from) if conf_from else dt_from
                    conf_to = max(conf_to, dt_to) if conf_to else dt_to
                    
                elif 'last_n_minutes' in conf:
                    dt_from = now_utc - timedelta(minutes=conf['last_n_minutes'])
                    conf_from = min(conf_from, dt_from) if conf_from else dt_from
                    conf_to = max(conf_to, now_utc) if conf_to else now_utc

        if(conf_from and conf_to):
            earliest_from = conf_from
            latest_to = conf_to

        from_date_str = format_bms_time(earliest_from)
        to_date_str = format_bms_time(latest_to)
        
        raw_data = fetch_cassandra_data(
            system_type=system_type, equipment_id=equipment_id, 
            datapoints=['CMDSpdVFD', 'FbVFD', 'TRe', 'TempSp1', 'TempSu', 'CmdVFD', 'FbFAD','SpTREff', 'ChwFb','AvgTmp', 'CmdSpsvfd', 'TempSp', 'ChwTemp'],
            from_date=from_date_str, to_date=to_date_str
        )

        unique_ts_count = len(set(r.get('data_received_on') for r in raw_data if r.get('data_received_on')))

        historical_alarms_df = pd.DataFrame()
        if is_explicit_range and unique_ts_count >= max_rows_needed:
            log.info(f"Explicit range provided with sufficient data ({unique_ts_count}). Evaluating whole dataset.")
        elif unique_ts_count < max_rows_needed and raw_data:
            missing_count = max_rows_needed - unique_ts_count
            log.warning(f"Insufficient data ({unique_ts_count} < {max_rows_needed}). Fetching {missing_count} previous states from Alarm Table.")
            
            table_name = f"{TABLE_SUFFIX}_{building_id.replace('-', '').lower()}"
            
            site_val = raw_data[0].get('site') 
            
            query = f"""SELECT "timestamp", "Alarm_Freeze", "Alarm_Oscillation", "Alarm_Tracking", 
                               "Alarm_Return_Air_Temp", "Alarm_Heat_Stress"
                        FROM {KEYSPACE_NAME}."{table_name}" 
                        WHERE site = %s AND equipment_id = %s 
                        ORDER BY timestamp DESC LIMIT %s ALLOW FILTERING"""
            try:
                hist_rows = session.execute(query, (site_val, equipment_id, missing_count))
                hist_data = [dict(r._asdict()) for r in hist_rows]
                
                if hist_data:
                    historical_alarms_df = pd.DataFrame(hist_data).sort_values('timestamp').reset_index(drop=True)
                    latest_hist_ts = pd.to_datetime(historical_alarms_df['timestamp'].max(), utc=True)
                    earliest_raw_ts = pd.to_datetime(min([r.get('data_received_on') for r in raw_data if r.get('data_received_on')]), utc=True)
                    
                    time_gap = earliest_raw_ts - latest_hist_ts
                    
                    if time_gap > timedelta(days=1):
                        log.error(f"Time gap rejected for {equipment_id}. History is {time_gap} old.")
                        raise HTTPException(
                            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                            detail="No Previous Alarms were recorded today to evaluate the current alarm due to insufficient data within the given timeframe"
                        )
            except HTTPException as he:
                raise he
            except Exception as e:
                log.error(f"Failed to fetch historical alarm states: {e}")

        if not raw_data:
            log.info("No raw sensor data found after fetch attempts.")
            return {"status": "SUCCESS", "records": 0, "message": "No data found."}
            
        latest_pull_ts = max([r.get('data_received_on') for r in raw_data if r.get('data_received_on')])
            
        state_data = {}
        if Path(STATE_FILE).exists():
            with open(STATE_FILE, "r") as f:
                try:
                    state_data = json.load(f).get(equipment_id, {"timestamp": None, "repeat_count": 0})
                except json.JSONDecodeError:
                    state_data = {"timestamp": None, "repeat_count": 0}
            
        last_ts = state_data.get("timestamp")
        repeat_count = state_data.get("repeat_count", 0)

        if str(latest_pull_ts) == str(last_ts):
            repeat_count += 1
            log.info(f"Stale data alert for {equipment_id}. Count: {repeat_count}")
            if repeat_count >= 2:
                update_last_timestamp(equipment_id, str(latest_pull_ts), repeat_count)
                return {"status": "SKIPPED", "message": "Stale data threshold reached (2 repeats)."}
            else:
                repeat_count = 0
            
        update_last_timestamp(equipment_id, str(latest_pull_ts), repeat_count)

        df_processed = data_pipeline(raw_data, ticket=ticket, ticket_type=ticket_type)
        if df_processed.empty: 
            return {"status": "SUCCESS", "message": "Pipeline result empty."}

        df_processed = df_processed.sort_values('data_received_on')

        log.info("Evaluating all alarms dynamically via evaluation wrapper.")
        df_to_store = alarm_evaluation(
            df_processed, 
            freeze_alarm_config=freeze_alarm_config,
            oscillation_alarm_config=oscillation_alarm_config,
            tracking_alarm_config=tracking_alarm_config,
            heat_stress_alarm_config=heat_stress_alarm_config,
            return_air_temp_alarm_config=return_air_temp_alarm_config,
            historical_alarms_df=historical_alarms_df
        )
            
        total_stored = store_data(df_to_store.to_dict(orient='records'), building_id, session)

        return {"status": "SUCCESS", "records_stored": total_stored}

    except Exception as e:
        log.error(f"Critical failure in save_data_to_cassandra: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
def fetch_alarm_data_from_cassandra(params: Dict[str, Any], session: Session) -> List[Dict]:
    """
    Logic: Queries historical alarm events from the building-specific Cassandra table.
    Parameters:
    - params: Dictionary containing 'building_id', 'site', 'equipment_id', 'from_date', 'to_date', 'alarm_names', and 'state'.
    Logic Flow:
    1. Dynamically constructs a CQL query targeting the table 'alarm_data_<building_id>'.
    2. Applies time-range filters (defaulting to the last 24 hours if dates are missing).
    3. Executes a prepared statement with 'ALLOW FILTERING'.
    4. Normalizes column casing to handle driver outputs, then post-filters results.
    """
    log.info(f"Fetching alarm records for parameters: {params}")
    building_id = params.get('building_id')
    if not building_id:
        log.error("Missing required parameter: building_id")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="building_id is required.")

    if not KEYSPACE_NAME:
        log.error("Cassandra keyspace environment variable missing.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Cassandra configuration missing.")

    safe_building_id = building_id.replace('-', '').lower()
    table_name = f"{TABLE_SUFFIX}_{safe_building_id}"

    query_base = f'SELECT * FROM {KEYSPACE_NAME}."{table_name}" WHERE '
    conditions = []
    values = []

    if params.get('site'):
        conditions.append("site = ?")
        values.append(params.get('site'))

    if params.get('equipment_id'):
        conditions.append("equipment_id = ?")
        values.append(params.get('equipment_id'))

    try:
        from_date = params.get('from_date')
        to_date = params.get('to_date')

        if not from_date or not to_date:
            from_date = datetime.now(timezone.utc) - timedelta(days=1)
            to_date = datetime.now(timezone.utc)
            log.info("Date range missing. Defaulting to last 24 hours.")
        else:
            if isinstance(from_date, str): from_date = parser.parse(from_date)
            if isinstance(to_date, str): to_date = parser.parse(to_date)
    except Exception as e:
        log.error(f"Date parsing failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid date format: {e}")

    conditions.append("timestamp >= ?")
    values.append(from_date)
    conditions.append("timestamp <= ?")
    values.append(to_date)

    full_query = query_base + " AND ".join(conditions) + " ALLOW FILTERING;"

    try:
        stmt = session.prepare(full_query)
        rows = session.execute(stmt, values)
        
        if not rows:
            log.info("No records found in Cassandra for the given query.")
            return []

        requested_alarms = params.get('alarm_names') or list(ALARM_FUNCTIONS.keys())
        input_states = params.get('state', [])
        if isinstance(input_states, str): input_states = [input_states]
            
        target_vals = []
        for s in input_states:
            s_lower = s.lower()
            if s_lower == 'warning': target_vals.append(1)
            elif s_lower == 'critical': target_vals.append(-1)
        
        if not target_vals or 'all' in [s.lower() for s in input_states]:
            target_vals = [1, -1]

        filtered_results = []
        
        lower_to_exact = {a.lower(): a for a in requested_alarms}

        for row in rows:
            raw_dict = dict(row._asdict())
            normalized_dict = {}
            
            for k, v in raw_dict.items():
                k_lower = k.lower()
                if k_lower in lower_to_exact:
                    normalized_dict[lower_to_exact[k_lower]] = v
                else:
                    normalized_dict[k] = v

            hit = False
            for a in requested_alarms:
                try:
                    val = int(float(normalized_dict.get(a) or 0))
                    if val in target_vals:
                        hit = True
                        break
                except (ValueError, TypeError):
                    pass

            if hit:
                if 'timestamp' in normalized_dict: 
                    normalized_dict['data_received_on'] = normalized_dict.pop('timestamp')
                filtered_results.append(normalized_dict)

        log.info(f"Fetch completed. Returned {len(filtered_results)} records.")
        return filtered_results
    
    except Exception as e:
        log.error(f"Database read operation failed: {e}")
        raise HTTPException(status_code=503, detail="Database read error.")

def count_alarms_from_db(params: Dict[str, Any], session: Session) -> Dict[str, Any]:
    """
    Logic: Aggregates alarm occurrences into a structured count summary grouped by severity state.
    """
    log.info("Calculating alarm counts from database.")
    try:
        records = fetch_alarm_data_from_cassandra(params, session)
        requested_alarms = params.get('alarm_names') or list(ALARM_FUNCTIONS.keys())
        
        state_map = {"warning": 1, "critical": -1}
        input_states = params.get('state', [])
        if isinstance(input_states, str): input_states = [input_states]

        final_counts = {}
        if not records:
            for state in input_states:
                final_counts[state.lower()] = {name: 0 for name in requested_alarms}
            return final_counts
        
        df = pd.DataFrame(records)
        for state_str in input_states:
            state_key = state_str.lower()
            target_value = state_map.get(state_key)
            state_dict = {}
            for alarm in requested_alarms:
                if alarm in df.columns:
                    series = pd.to_numeric(df[alarm], errors='coerce').fillna(0)
                    state_dict[alarm] = int((series == target_value).sum())
                else:
                    state_dict[alarm] = 0
            final_counts[state_key] = state_dict
                
        return final_counts
    except HTTPException as he:
        raise he
    except Exception as e:
        log.error(f"Error in count_alarms_from_db: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate grouped alarm counts.")
     
def test_alarm_logic(
    records: List[Dict[str, Any]], building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2",
    ticket: str = "", ticket_type: Optional[str] = None, session: Session = None,
    freeze_alarm_config=None, oscillation_alarm_config=None, tracking_alarm_config=None, 
    heat_stress_alarm_config=None, return_air_temp_alarm_config=None
) -> List[Dict[str, Any]]:
    """
    A diagnostic function for manual verification of alarm logic using provided JSON data.
    Logic:
    Runs the standard pipeline and evaluation without requiring a live BMS feed. 
    If a Cassandra session is provided, it persists the test results to the database.
    """
    log.info(f"Executing manual test for alarm logic with {len(records)} records.")
    
    df_processed = data_pipeline(records, ticket=ticket, ticket_type=ticket_type)
    if df_processed.empty:
        log.warning("Pipeline returned empty dataframe; nothing to evaluate.")
        return []

    log.info("Evaluating all alarms dynamically via evaluation wrapper for test data.")
    df_alarms = alarm_evaluation(
        df_processed,
        freeze_alarm_config=freeze_alarm_config,
        oscillation_alarm_config=oscillation_alarm_config,
        tracking_alarm_config=tracking_alarm_config,
        heat_stress_alarm_config=heat_stress_alarm_config,
        return_air_temp_alarm_config=return_air_temp_alarm_config
    )

    if session is not None:
        if not df_alarms.empty:
            log.info(f"Saving {len(df_alarms)} test records to Cassandra for building {building_id}...")
            store_data(df_alarms.to_dict(orient='records'), building_id, session)
        else:
            log.info("No active alarms detected in test data; skipping Cassandra storage.")

    df_alarms = df_alarms.replace([np.inf, -np.inf], None)
    df_alarms = df_alarms.astype(object).where(pd.notnull(df_alarms), None)
    
    return df_alarms.to_dict(orient='records')