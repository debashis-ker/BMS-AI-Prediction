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

log = setup_logger(__name__)

STATE_FILE = "artifacts/ahu13_alarm_timestamp_track/last_processed_state.json"

FIXED_SYSTEM_TYPE = "AHU"
KEYSPACE_NAME = os.getenv('CASSANDRA_KEYSPACE')
TABLE_SUFFIX = "alarm_data"

if not KEYSPACE_NAME:
    log.error("CASSANDRA_KEYSPACE environment variable is not set. Database operations will fail.")

def check_freeze_alarm(df: pd.DataFrame) -> pd.Series:
    """
    Logic: Evaluates if the Supply Air Temperature (TempSu) is dangerously low, posing a risk to the coil.
    Trigger:
    - Warning (1): Triggered if 'TempSu' < 5.0 for at least 1 sample in a rolling window of 3.
    - Critical (-1): Triggered if 'TempSu' < 5.0 for 3 consecutive samples.
    Note: Requires 'TempSu' column.
    """

    log.info("Evaluating Freeze Alarm logic.")
    if 'TempSu' not in df.columns: 
        log.warning("Column 'TempSu' missing for Freeze Alarm evaluation.")
        return pd.Series(0, index=df.index)
    
    raw_hit = (df['TempSu'] < 5).astype(int)
    consecutive_count = raw_hit.rolling(window=3, min_periods=1).sum()
    
    status_series = pd.Series(0, index=df.index)
    status_series[consecutive_count >= 1] = 1  
    status_series[consecutive_count >= 3] = -1
    log.debug(f"Freeze Alarm evaluation complete. Detected hits: {status_series.any()}")
    return status_series.astype(int)

def check_oscillation_alarm(df: pd.DataFrame) -> pd.Series:
    """
    Logic: Detects rapid, unstable switching in the VFD Command (CmdVFD).
    Trigger:
    - Warning (1): Triggered if at least one flip (0->1 or 1->0) is detected within 3 samples.
    - Critical (-1): Triggered if 2 or more flips are detected within a rolling window of 3 samples.
    Note: Uses absolute difference of shifted values to detect state changes.
    """

    log.info("Evaluating Oscillation Alarm logic.")
    if 'CmdVFD' not in df.columns: 
        log.warning("Column 'CmdVFD' missing for Oscillation Alarm evaluation.")
        return pd.Series(0, index=df.index)
    
    changes = df['CmdVFD'].diff().abs().fillna(0)
    raw_hit = ((changes == 1) & (changes.shift(1) == 1)).astype(int)
    
    consecutive_count = raw_hit.rolling(window=3, min_periods=1).sum()
    
    status_series = pd.Series(0, index=df.index)
    status_series[consecutive_count >= 1] = 1
    status_series[consecutive_count >= 2] = -1
    log.debug(f"Oscillation Alarm evaluation complete. Detected hits: {status_series.any()}")
    return status_series.astype(int)

def check_tracking_alarm(df: pd.DataFrame) -> pd.Series:
    """
    Logic: Monitors the deviation between the VFD Speed Command and actual Speed Feedback.
    Trigger:
    - Suppression: Logic only executes if 'CMDSpdVFD' > 1.0 (Unit must be commanded ON).
    - Calculation: Error = |CMDSpdVFD - FbVFD| / CMDSpdVFD.
    - Warning (1): Triggered if Error > 3% for at least 1 sample in a window of 3.
    - Critical (-1): Triggered if Error > 3% for 3 consecutive samples.
    """

    log.info("Evaluating Tracking Alarm logic with 0-command suppression.")
    if not all(c in df.columns for c in ['CMDSpdVFD', 'FbVFD']): 
        log.warning("Required columns ['CMDSpdVFD', 'FbVFD'] missing for Tracking Alarm.")
        return pd.Series(0, index=df.index)
    
    is_running = df['CMDSpdVFD'] > 1.0
    
    divisor = df['CMDSpdVFD'].replace(0, 1)
    error = (df['CMDSpdVFD'] - df['FbVFD']).abs() / divisor
    
    raw_hit = ((error > 0.03) & is_running).astype(int)
    
    consecutive_count = raw_hit.rolling(window=3, min_periods=1).sum()
    
    status_series = pd.Series(0, index=df.index)
    status_series[consecutive_count >= 1] = 1
    status_series[consecutive_count >= 3] = -1
    
    return status_series.astype(int)

def check_return_air_temp_alarm(df: pd.DataFrame) -> pd.Series:
    """
    Logic: Identifies poor cooling/heating performance when the Fresh Air Damper is fully open.
    Trigger:
    - Warning (1): Triggered if |TRe - TempSp1| > 10.0 AND 'FbFAD' >= 95.0 for at least 1 sample.
    - Critical (-1): Triggered if these conditions persist for 3 consecutive samples.
    Note: Indicates the unit cannot reach setpoint despite maximum fresh air intake.
    """

    log.info("Evaluating Return Air Temp Alarm logic.")
    if not all(c in df.columns for c in ['TRe', 'TempSp1', 'FbFAD']): 
        log.warning("Required columns ['TRe', 'TempSp1', 'FbFAD'] missing.")
        return pd.Series(0, index=df.index)
    
    raw_hit = (((df['TRe'] - df['TempSp1']).abs() > 10) & (df['FbFAD'] >= 95.0)).astype(int)
    consecutive_count = raw_hit.rolling(window=3, min_periods=1).sum()
    
    status_series = pd.Series(0, index=df.index)
    status_series[consecutive_count >= 1] = 1
    status_series[consecutive_count >= 3] = -1
    return status_series.astype(int)

def check_heat_stress_alarm(df: pd.DataFrame) -> pd.Series:
    """
    Logic: Detects uncomfortable temperatures for occupants during active hours.
    Trigger:
    - Warning (1): Triggered if 'TempSp1' > 26.0 AND 'Occupied_Flag' is 'Occupied' for at least 1 sample.
    - Critical (-1): Triggered if high temperature persists for 3 consecutive samples during occupancy.
    """

    log.info("Evaluating Heat Stress Alarm logic.")
    if not all(c in df.columns for c in ['TempSp1', 'Occupied_Flag']): 
        log.warning("Required columns ['TempSp1', 'Occupied_Flag'] missing.")
        return pd.Series(0, index=df.index)
    
    raw_hit = ((df['TempSp1'] > 26.0) & (df['Occupied_Flag'] == 'Occupied')).astype(int)
    consecutive_count = raw_hit.rolling(window=3, min_periods=1).sum()
    
    status_series = pd.Series(0, index=df.index)
    status_series[consecutive_count >= 1] = 1
    status_series[consecutive_count >= 3] = -1
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
        
        # df[STANDARD_DATE_COLUMN] = pd.to_datetime(df[STANDARD_DATE_COLUMN], errors='coerce')
        df[STANDARD_DATE_COLUMN] = pd.to_datetime(df[STANDARD_DATE_COLUMN], errors='coerce',utc=True)
        if df[STANDARD_DATE_COLUMN].isnull().any():
             log.error("Found unparseable timestamps in data records.")
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="One or more timestamps could not be parsed.")

        # if df[STANDARD_DATE_COLUMN].dt.tz is not None: 
        #     df[STANDARD_DATE_COLUMN] = df[STANDARD_DATE_COLUMN].dt.tz_localize(None)

        result_df = df.groupby(['site', 'equipment_id', STANDARD_DATE_COLUMN, 'datapoint'])['monitoring_data'].agg('first').unstack(level='datapoint').reset_index()

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
            log.warning(f"No schedule data found for ticket {ticket}. Defaulting to Unoccupied.")
            result_df['Occupied_Flag'] = 'Unoccupied'
            return result_df

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
        log.error(f"Critical error in data_pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data transformation failed: {str(e)}")
    
def alarm_evaluation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the processed dataset to isolate significant events.
    Logic:
    Identifies and returns only the rows where at least one alarm function (Freeze, Oscillation, etc.) 
    has returned a non-zero status (Warning or Critical).
    """
    if data.empty: 
        log.info("Empty dataframe passed to alarm_evaluation. Skipping.")
        return data
    
    if 'data_received_on' not in data.columns:
        log.error("Column 'data_received_on' missing. Cannot evaluate alarms.")
        return pd.DataFrame()

    data = data.sort_values('data_received_on')
    alarm_cols_detected = []
    
    for alarm_name, func in ALARM_FUNCTIONS.items():
        try:
            log.debug(f"Running evaluation function for: {alarm_name}")
            data[alarm_name] = func(data)
            alarm_cols_detected.append(alarm_name)
        except Exception as e:
            log.error(f"Error during {alarm_name} evaluation: {e}")

    if not alarm_cols_detected:
        log.warning("No alarms were successfully evaluated.")
        return pd.DataFrame()

    filtered_df = data[data[alarm_cols_detected].any(axis=1)].copy()
    if filtered_df.empty:
        log.info("No active alarms detected in this cycle. Skipping storage.")
        
    log.info(f"Alarm evaluation complete. Found {len(filtered_df)} alarm events.")
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
    1. Sanitizes records by removing Null/NaN values.
    2. Performs strict type-casting (NumPy to Python native) to ensure driver compatibility.
    3. Uses parameterized SQL with tuple-based execution to prevent formatting errors.
    4. Renames 'data_received_on' to 'timestamp' for DB schema alignment.
    """
    if not data_chunk: 
        log.info("No active alarm records to persist.")
        return 0
    
    table_name = f"{TABLE_SUFFIX}_{building_id.replace('-', '').lower()}"
    rows_affected = 0

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

            columns = list(final_data.keys())
            placeholders = ", ".join(["%s" for _ in columns])
            column_names = ", ".join([f'"{c}"' for c in columns])
            
            query = f'INSERT INTO {KEYSPACE_NAME}."{table_name}" ({column_names}) VALUES ({placeholders})'
            
            session.execute(query, tuple(final_data[c] for c in columns))
            rows_affected += 1
            
        except Exception as e:
            error_ts = record.get('data_received_on') or record.get('timestamp')
            log.error(f"Cassandra INSERT failed for record {error_ts}: {e}")

    log.info(f"Successfully stored {rows_affected} records.")
    return rows_affected

def save_data_to_cassandra(session: Session, building_id: str, equipment_id: str, system_type: str = "AHU", ticket: str = "", ticket_type: str = "", software_id: str = "", account_id : str = "") -> Dict[str, Any]:
    """
    The primary synchronization routine for real-time alarm detection.
    Logic:
    1. Fetches raw sensor data for the last 10 minutes.
    2. Density Check: If 'CmdVFD' records < 3, retries fetch with a 30-minute lookback.
    3. Stale Check: Skips processing if data hasn't updated after 2 consecutive cycles.
    4. Executes the pipeline, evaluates all 5 alarm types, and persists results to Cassandra.
    """
    log.info(f"Starting real-time sync for {equipment_id} in building {building_id}")
    now_utc = datetime.now(timezone.utc)
    
    def format_bms_time(dt_obj):
        return dt_obj.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"

    try:
        polling_interval = 10 
        max_lookback_minutes = 30
        log.debug(f"Using static polling interval: {polling_interval} minutes.")

        to_date = format_bms_time(now_utc)
        from_date = format_bms_time(now_utc - timedelta(minutes=polling_interval))

        log.info(f"Initial fetch (Window: {polling_interval}m) from {from_date} to {to_date}")
        raw_data = fetch_cassandra_data(
            system_type=system_type, equipment_id=equipment_id, 
            datapoints=['CMDSpdVFD', 'FbVFD', 'TRe', 'TempSp1', 'TempSu', 'CmdVFD', 'FbFAD'],
            from_date=from_date, to_date=to_date
        )

        cmd_vfd_count = len([r for r in raw_data if r.get('datapoint') == 'CmdVFD'])
        log.info(f"CmdVFD records found: {cmd_vfd_count}")

        if cmd_vfd_count < 3:
            log.warning(f"Insufficient CmdVFD data ({cmd_vfd_count} records). Retrying with max lookback: {max_lookback_minutes}m")
            from_date = format_bms_time(now_utc - timedelta(minutes=max_lookback_minutes))
            
            raw_data = fetch_cassandra_data(
                system_type=system_type, equipment_id=equipment_id, 
                datapoints=['CMDSpdVFD', 'FbVFD', 'TRe', 'TempSp1', 'TempSu', 'CmdVFD', 'FbFAD'],
                from_date=from_date, to_date=to_date
            )

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

        log.info("Evaluating all alarms (Freeze, Tracking, Heat Stress, Return Air, Oscillation).")
        df_processed['Alarm_Freeze'] = check_freeze_alarm(df_processed)
        df_processed['Alarm_Tracking'] = check_tracking_alarm(df_processed)
        df_processed['Alarm_Heat_Stress'] = check_heat_stress_alarm(df_processed)
        df_processed['Alarm_Return_Air_Temp'] = check_return_air_temp_alarm(df_processed)
        df_processed['Alarm_Oscillation'] = check_oscillation_alarm(df_processed)

        df_to_store = alarm_evaluation(df_processed)
        
        total_stored = store_data(df_to_store.to_dict(orient='records'), building_id, session)

        return {"status": "SUCCESS", "records_stored": total_stored}

    except HTTPException as he:
        raise he
    except Exception as e:
        log.error(f"Critical failure in save_data_to_cassandra: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
def fetch_alarm_data_from_cassandra(params: Dict[str, Any], session: Session) -> List[Dict]:
    """
    Logic: Queries historical alarm events from the building-specific Cassandra table.
    Parameters:
    - params: Dictionary containing 'building_id', 'site', 'equipment_id', 'from_date', 'to_date', 'alarm_names', and 'state'.
    Logic Flow:
    1. Dynamically constructs a CQL query targeting the table 'alarm_data_<building_id>'.
    2. Applies time-range filters (defaulting to the last 24 hours if dates are missing).
    3. Executes a prepared statement with 'ALLOW FILTERING' for non-primary key attributes.
    4. Post-filters results in memory based on requested alarm names and states ('warning': 1, 'critical': -1).
    Returns: A list of filtered dictionary records.
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
        results = [dict(row._asdict()) for row in rows]
        
        if not results:
            log.info("No records found in Cassandra for the given query.")
            return []

        requested_alarms = params.get('alarm_names') or [k for k in results[0].keys() if "Alarm_" in k]
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
        for r in results:
            if any(int(r.get(a, 0) or 0) in target_vals for a in requested_alarms):
                if 'timestamp' in r: r['data_received_on'] = r.pop('timestamp')
                filtered_results.append(r)

        log.info(f"Fetch completed. Returned {len(filtered_results)} records.")
        return filtered_results
    
    except Exception as e:
        log.error(f"Database read operation failed: {e}")
        raise HTTPException(status_code=503, detail="Database read error.")
    
def count_alarms_from_db(params: Dict[str, Any], session: Session) -> Dict[str, Any]:
    """
    Logic: Aggregates alarm occurrences into a structured count summary grouped by severity state.
    Parameters:
    - params: Dictionary containing the filter criteria for 'fetch_alarm_data_from_cassandra'.
    - session: An active Cassandra Session.
    Logic Flow:
    1. Calls 'fetch_alarm_data_from_cassandra' to retrieve relevant historical records.
    2. Converts the result set into a Pandas DataFrame for high-performance aggregation.
    3. Iterates through the requested states (e.g., 'warning', 'critical') and counts non-zero entries for each alarm column.
    4. Fills missing alarm categories with 0 to ensure a consistent response schema.
    Returns: A nested dictionary in the format { "state": { "alarm_name": count } }.
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
    
def test_alarm_logic(records: List[Dict[str, Any]], ticket: str = "", ticket_type = None, session: Session = None) -> List[Dict[str, Any]]:
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

    df_processed['Alarm_Freeze'] = check_freeze_alarm(df_processed)
    df_processed['Alarm_Tracking'] = check_tracking_alarm(df_processed)
    df_processed['Alarm_Heat_Stress'] = check_heat_stress_alarm(df_processed)
    df_processed['Alarm_Return_Air_Temp'] = check_return_air_temp_alarm(df_processed)
    df_processed['Alarm_Oscillation'] = check_oscillation_alarm(df_processed)

    df_alarms = alarm_evaluation(df_processed)

    if session is not None:
        if not df_alarms.empty:
            b_id = records[0].get('building_id', "36c27828-d0b4-4f1e-8a94-d962d342e7c2")
            log.info(f"Saving {len(df_alarms)} test records to Cassandra for building {b_id}...")
            store_data(df_alarms.to_dict(orient='records'), b_id, session)
        else:
            log.info("No active alarms detected in test data; skipping Cassandra storage.")

    df_alarms = df_alarms.replace([np.inf, -np.inf], None)
    df_alarms = df_alarms.astype(object).where(pd.notnull(df_alarms), None)
    
    return df_alarms.to_dict(orient='records')