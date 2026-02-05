import pandas as pd
from typing import Any, Dict, List, Optional
from fastapi import HTTPException
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.cassandra_utils import fetch_cassandra_data

log = setup_logger(__name__)

EMISSION_FACTOR = 0.4041

try:
    STATIC_PEAK_DATA = pd.read_json('src/bms_ai/utils/peak_demand/peak_demand_results.json', orient='index')
except Exception as e:
    log.debug(f"Error loading data: {e}. STATIC_PEAK_DATA initialized as empty DataFrame.")
    STATIC_PEAK_DATA = pd.DataFrame()

def transform_to_dataframe(json_input_data: Dict[str, Any]) -> pd.DataFrame:
    GROUPING_COLS: List[str] = ['data_received_on', 'equipment_id', 'site']
    empty_df = pd.DataFrame(columns=GROUPING_COLS + ['Eg'])
    
    try:
        records: List[Dict[str, Any]] = (
            json_input_data.get("data", {}).get("queryResponse", []) or 
            json_input_data.get("queryResponse", []) or 
            json_input_data.get("data", [])
        )
        if not records:
            log.warning("No records found in the input JSON structure.")
            return empty_df

        df = pd.DataFrame(records)
        
        if 'system_type' not in df.columns:
            log.error("'system_type' column missing in raw data.")
            return empty_df
            
        mtr_df = df[df["system_type"] == "MtrEMU"]
        if mtr_df.empty:
            log.warning("No records found for 'MtrEMU' system type.")
            return empty_df

        mtr_df["data_received_on"] = pd.to_datetime(mtr_df["data_received_on"], errors='coerce')
        if mtr_df["data_received_on"].dt.tz is not None: #type: ignore
             mtr_df["data_received_on"] = mtr_df["data_received_on"].dt.tz_localize(None) #type: ignore
        
        mtr_df.dropna(subset=["data_received_on"], inplace=True)
        
        if 'monitoring_data' in mtr_df.columns:
            mtr_df['monitoring_data'] = mtr_df['monitoring_data'].astype(str).str.strip()
        else:
            log.error("'monitoring_data' column missing.")
            return empty_df

        pivoted_df = mtr_df.pivot_table(
            index=['data_received_on', 'equipment_id', 'site'],
            columns='datapoint',
            values='monitoring_data',
            aggfunc='first' #type:ignore
        ).reset_index()

        if isinstance(pivoted_df.columns, pd.MultiIndex):
             pivoted_df.columns = [
                col[-1] if col[-1] else str(col[0]) for col in pivoted_df.columns
             ]
        
        if 'Eg' not in pivoted_df.columns:
            log.error("The required 'Eg' datapoint is missing after pivoting.")
            return empty_df

        pivoted_df['Eg'] = pd.to_numeric(pivoted_df['Eg'], errors='coerce')
        pivoted_df.dropna(subset=['Eg'], inplace=True)
        
        if pivoted_df.empty:
            log.warning("All records were dropped after final 'Eg' cleanup.")
            return empty_df
        
        required_cols = ['data_received_on', 'equipment_id', 'site', 'Eg']
        missing_cols = [col for col in required_cols if col not in pivoted_df.columns]
        
        if missing_cols:
            log.error(f"Final DataFrame is missing required columns: {missing_cols}")
            return empty_df

        log.info(f"Data processing successful. {len(pivoted_df)} rows ready for aggregation.")
        return pivoted_df[required_cols]
        
    except Exception as e:
        log.error(f"Critical error in transform_to_dataframe: {e}", exc_info=True)
        raise RuntimeError(f"Data transformation failed: {e}")

def calculate_aggregate_emissions(df_input: pd.DataFrame) -> pd.DataFrame:
    """Calculates min/max energy and resulting carbon emissions."""
    if df_input.empty:
        return pd.DataFrame(columns=['equipment_id', 'site', 'min_Eg', 'max_Eg', 'count', 
                                     'Energy_Range_kWh', 'carbon_emission_kg'])
    
    try:
        df = df_input
        df['Eg'] = pd.to_numeric(df['Eg'], errors='coerce')
        df = df.dropna(subset=['Eg'])
        
        emission_summary = df.groupby(['equipment_id', 'site']).agg(
            min_Eg=('Eg', 'min'),
            max_Eg=('Eg', 'max'),
            count=('Eg', 'size')
        ).reset_index()

        emission_summary['Energy_Range_kWh'] = emission_summary['max_Eg'] - emission_summary['min_Eg']
        
        emission_summary = emission_summary[emission_summary['Energy_Range_kWh'] > 0]
        
        emission_summary['carbon_emission_kg'] = emission_summary['Energy_Range_kWh'] * EMISSION_FACTOR
        
        return emission_summary

    except Exception as e:
        log.error(f"Error during emission calculation: {e}", exc_info=True)
        raise RuntimeError(f"Aggregation failed: {str(e)}")

def get_emission_report(equipment_id: List[str], zone: Optional[str], request_data: Any) -> Dict[str, Any]:
    try:
        data = fetch_cassandra_data(
            datapoints=['Eg'], 
            system_type='MtrEMU',
            from_date=request_data.from_date, 
            to_date=request_data.to_date
        )

        json_input_data = {"queryResponse": data}
        df_processed = transform_to_dataframe(json_input_data)
        
        if df_processed.empty:
            return {
                "Request_Parameters": {"carbon_emission_kg": 0, "energy_consumed_kwh": 0, "breakdown_by_equipment_and_zone": []}
            }

        df_filtered = calculate_aggregate_emissions(df_processed)

        if equipment_id:
            if isinstance(equipment_id, list):
                df_filtered = df_filtered[df_filtered['equipment_id'].isin(equipment_id)]
            else:
                df_filtered = df_filtered[df_filtered['equipment_id'] == equipment_id]
            
        if zone:
            df_filtered = df_filtered[df_filtered['site'] == zone] 

        specific_emission_kg = df_filtered['carbon_emission_kg'].sum()
        specific_energy_kwh = df_filtered['Energy_Range_kWh'].sum()

        df_filtered['Energy_Range_kWh'] = df_filtered['Energy_Range_kWh'].round(2)
        df_filtered['carbon_emission_kg'] = df_filtered['carbon_emission_kg'].round(2)
        
        return {
                "carbon_emission_kg": round(float(specific_emission_kg), 2),
                "energy_consumed_kwh": round(float(specific_energy_kwh), 2),
                "breakdown_by_equipment_and_zone": df_filtered[['equipment_id', 'site', 'Energy_Range_kWh', 'carbon_emission_kg']].to_dict('records')
            }
    
    except Exception as e:
        log.error(f"Error in get_emission_report: {e}", exc_info=True)
        raise RuntimeError(str(e))
    
def get_emission_report_from_json(equipment_id: Optional[str] = None, zone: Optional[str] = None) -> Dict[str, Any]:
    try:
        df_emissions = pd.read_json('src/bms_ai/utils/carbon_emission/carbon_emission.json', orient='index')
        
        cleaned_index = (
            df_emissions.index.to_series()
            .str.strip("()")  
            .str.replace("'", "", regex=False) 
            .str.replace(" ", "", regex=False)
        )
        
        df_emissions[['equipment_id', 'site']] = cleaned_index.str.split(',', expand=True)
        df_emissions = df_emissions.reset_index(drop=True)
        df_emissions.rename(columns={'diff': 'Energy_Range_kWh'}, inplace=True)
        
        df_emissions['carbon_emission_kg'] = df_emissions['Energy_Range_kWh'] * EMISSION_FACTOR
            
        if df_emissions.empty:
            return {
                "carbon_emission_kg": 0,
                "energy_consumed_kwh": 0,
                "breakdown_by_equipment_and_zone": []
            }

        mdb_filtered = df_emissions[(df_emissions['site'] == "OS01") & (df_emissions['equipment_id'] == "EMU03")]
        smdb_filtered = df_emissions[(df_emissions['site'] == "OS01") & (df_emissions['equipment_id'] == "EMU06")]
        mdb_carbon_emission_kg = mdb_filtered["carbon_emission_kg"].sum()
        mdb_energy_kwh = mdb_filtered["Energy_Range_kWh"].sum()
        smdb_carbon_emission_kg = smdb_filtered["carbon_emission_kg"].sum()
        smdb_energy_kwh = smdb_filtered["Energy_Range_kWh"].sum()
        
        df_emissions.rename(columns={'Energy_Range_kWh': 'energy_range_kwh'}, inplace=True)
        
        final_minimal_report = {
            "carbon_emission_kg": round((mdb_carbon_emission_kg + smdb_carbon_emission_kg),2),
            "energy_consumed_kwh": round((mdb_energy_kwh + smdb_energy_kwh),2),
            "breakdown_by_equipment_and_zone": df_emissions[['equipment_id', 'site', 'energy_range_kwh', 'carbon_emission_kg']].round(2).to_dict('records')
        }
        
        return final_minimal_report
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"Configuration Error: The static data file was not"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Critical processing error: {e}"
        )

def dynamic_transform_to_dataframe(
    json_input_data: Dict[str, Any], 
    system_type: str = "",
    target_datapoints: List[str] = [],
    grouping_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Dynamically transforms JSON input into a pivoted DataFrame with multiple target columns.
    
    :param json_input_data: Raw dictionary from API/JSON source.
    :param system_type: System filter (e.g., 'MtrEMU').
    :param target_datapoints: List of metrics to extract (e.g., ['Eg', 'PowT', 'PowB']).
    :param grouping_cols: Columns to keep as indices.
    """
    
    if grouping_cols is None:
        grouping_cols = ['data_received_on', 'equipment_id', 'site']
        
    empty_df = pd.DataFrame(columns=grouping_cols + target_datapoints)
    
    try:
        records: List[Dict[str, Any]] = (
            json_input_data.get("data", {}).get("queryResponse", []) or 
            json_input_data.get("queryResponse", []) or 
            json_input_data.get("data", [])
        )
        
        if not records:
            log.warning("No records found in the input JSON structure.")
            return empty_df

        df = pd.DataFrame(records)
        
        if 'system_type' not in df.columns:
            log.error("'system_type' column missing in raw data.")
            return empty_df
            
        filtered_df = df[df["system_type"] == system_type]
        if filtered_df.empty:
            log.warning(f"No records found for '{system_type}'.")
            return empty_df

        if 'data_received_on' in filtered_df.columns:
            filtered_df["data_received_on"] = pd.to_datetime(filtered_df["data_received_on"], errors='coerce')
            if filtered_df["data_received_on"].dt.tz is not None:
                filtered_df["data_received_on"] = filtered_df["data_received_on"].dt.tz_localize(None)
            filtered_df.dropna(subset=["data_received_on"], inplace=True)
        
        pivoted_df = filtered_df.pivot_table(
            index=grouping_cols,
            columns='datapoint',
            values='monitoring_data',
            aggfunc='first'
        ).reset_index()

        if isinstance(pivoted_df.columns, pd.MultiIndex):
             pivoted_df.columns = [col[-1] if col[-1] else str(col[0]) for col in pivoted_df.columns]
        
        existing_targets = [col for col in target_datapoints if col in pivoted_df.columns]
        
        if not existing_targets:
            log.error(f"None of the requested datapoints {target_datapoints} found after pivoting.")
            return empty_df

        for col in existing_targets:
            pivoted_df[col] = pd.to_numeric(pivoted_df[col], errors='coerce')
        
        final_cols = grouping_cols + existing_targets
        log.info(f"Transformation successful. Extracted: {existing_targets}")
        
        return pivoted_df[final_cols]
        
    except Exception as e:
        log.error(f"Critical error in transform_to_dataframe: {e}", exc_info=True)
        raise RuntimeError(f"Data transformation failed: {e}")

def calculate_peak_demand(df_input: pd.DataFrame, equipment_id: str) -> Dict[str, Any]:
    """Internal logic to process peak demand for a specific equipment ID."""
    df = df_input[df_input["equipment_id"] == equipment_id]

    if df.empty:
        return {'peak_demand_power': None, 'peak_demand_date': None, 'error': 'No data found'}

    target_col = None
    if "PowT" in df.columns and df["PowT"].notna().any():
        target_col = "PowT"
    elif "PowB" in df.columns and df["PowB"].notna().any():
        target_col = "PowB"

    if not target_col:
        return {'peak_demand_power': None, 'peak_demand_date': None, 'error': 'No valid numeric PowT or PowB data found'}

    df = df.sort_values("data_received_on")
    
    valid_df = df.dropna(subset=[target_col])
    
    if valid_df.empty:
        return {'peak_demand_power': None, 'peak_demand_date': None, 'error': 'All values are NaN'}

    max_idx = valid_df[target_col].idxmax()
    peak_row = valid_df.loc[max_idx]

    return {
        'peak_demand_power': float(peak_row[target_col]),
        'peak_demand_date': peak_row["data_received_on"].strftime('%Y-%m-%d %H:%M:%S')
    }

def dynamic_peak_demand(json_input_data: Dict[str, Any], equipment_id: Optional[str], zone: Optional[str]) -> Dict[str, Any]:
    try:
        df_processed = transform_to_dataframe(json_input_data)

        if df_processed.empty:
            log.info("Returning empty report due to no processed data.")
            return {
                "Request_Parameters": {"equipment_id": equipment_id, "zone": zone},
                "Peak_Demand_Report": {"equipment_id": equipment_id, "zone": zone, "peak_demand_power": None, "peak_demand_date": None},
                "Processing_Status": "No Data Processed"
            }
        
        peak_demand_data = calculate_peak_demand(df_processed, equipment_id)

        final_report = {
            "Peak_Demand_Report": {
                equipment_id : {
                    "peak_demand_power": peak_demand_data.get('peak_demand_power'),
                    "peak_demand_date": peak_demand_data.get('peak_demand_date')
                },
            }
        }

        return final_report
    
    except RuntimeError as e:
        log.error(f"API processing failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        log.critical(f"Unexpected critical error in get_peak_demand: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during peak demand generation."
        )
