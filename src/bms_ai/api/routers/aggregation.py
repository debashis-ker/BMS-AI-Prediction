from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, List, Any, Callable
import pandas as pd
from src.bms_ai.logger_config import setup_logger
import warnings
import time
from src.bms_ai.utils import statistics_utils

log = setup_logger(__name__)

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/aggregation", tags=["Prescriptive Optimization"])

class request_structure(BaseModel):
    data: Dict[str, Any] = Field(default={}, description="Data Input")
    columns: List[str] = Field(default=["Fan Power meter (KW)"], description="Columns Required for Statistics")
    date_columns: str = Field(default="data_received_on", description="Date Column Required for Statistics")
    from_date : Optional[str] = Field(None, description="Start date for range-based calculations (e.g., '2024-01-01')")
    to_date : Optional[str] = Field(None, description="End date for range-based calculations (e.g., '2024-01-31')")
    system_type : str = Field(default="AHU", description="System type filter (e.g., 'AHU')")
    site : str = Field(default="Ground Floor", description="Site filter (e.g., 'Ground Floor')")
    stats: List[str] = Field(default=["min","max","mean","standard deviation","variance","find_last_min","find_last_max","min_date_hits", "max_data_hits"],description="Statistics Required for Output")

class response_structure(BaseModel):
    stats_val: Dict[str, Dict[str, Any]] = Field(..., description="Stats Values grouped by column")

def process_api_dict_to_dataframe(api_data_dict: dict) -> pd.DataFrame:
    try:
        if "queryResponse" not in api_data_dict:
            log.error("Input dictionary is missing 'queryResponse' key.")
            
        records = api_data_dict["queryResponse"]
        if not records:
             return pd.DataFrame()
        df = pd.DataFrame(records)
        return df

    except Exception as e:
        log.error(f"Error processing JSON data: {e}")
        raise HTTPException(status_code=400, detail=f"An unexpected error occurred in JSON Data: {e}")

def data_pipeline(data: Dict[str, Any], date_column: str, site : str, system_type : str, requested_columns: List[str], from_date: Optional[str], to_date: Optional[str]) -> pd.DataFrame:
    try:
        try:
            df_ahu = process_api_dict_to_dataframe(data)
            if df_ahu.empty:
                log.error("DataFrame is empty after processing.")
        except Exception as e:
            log.error(f"Error in initial DataFrame processing: {e}")
            raise HTTPException(status_code=400, detail="Problem in the initial data processing.")

        try:
            df_ahu = df_ahu[(df_ahu['system_type'] == system_type) & (df_ahu['site'] == site)].copy()
            if df_ahu.empty:
                log.error(f"Data for system_type='{system_type}' and site='{site}' not found after filtering.")
                
        except Exception as e:
            log.error(f"Error applying equipment filters: {e}")
            raise HTTPException(status_code=400, detail="Data with desired system_type and site not available.")
        
        try:
            if date_column not in df_ahu.columns:
                log.error(f"Date column '{date_column}' not found.")
                 
            df_ahu[date_column] = pd.to_datetime(df_ahu[date_column])
            df_ahu[date_column] = df_ahu[date_column].dt.tz_localize(None)
            df_ahu.set_index(date_column, inplace=True)
            df_ahu.sort_index(ascending=True, inplace=True)

        except Exception as e:
            log.error(f"Error processing date column: {e}")
            raise HTTPException(status_code=400, detail="Date Column Input Not Found or processing failed.")

        for col_name, mapping in [
            ('monitoring_data', {'inactive': 0.0, 'active': 1.0})
        ]:
            if col_name in df_ahu.columns:
                try:
                    df_ahu[col_name] = df_ahu[col_name].replace(mapping, regex=False).astype('float64', errors='ignore')
                except Exception as e:
                    log.warning(f"Failed to convert/replace for column '{col_name}': {e}")
            else:
                 if col_name in ['monitoring_data']:
                    log.error(f"Required column '{col_name}' not found.")
                 
        Required_Columns = ['datapoint', 'monitoring_data'] 
        
        cols_to_keep = [col for col in Required_Columns if col in df_ahu.columns]
        df_ahu = df_ahu[cols_to_keep] 
        
        if from_date or to_date:
            try:
                df_ahu = df_ahu.loc[from_date:to_date]
                log.info(f"DataFrame trimmed by date range: {from_date} to {to_date}")
            except Exception as e:
                log.warning(f"Error trimming data by date range: {e}. Proceeding without trimming.")
        try:
            temp_df = df_ahu.reset_index() 
            aggregated_scores = temp_df.groupby([date_column, 'datapoint'])['monitoring_data'].agg('first')
            result_df = aggregated_scores.unstack(level='datapoint')
            
        except Exception as e:
            log.error(f"Error in aggregation: {e}")
            raise HTTPException(status_code=400, detail="Aggregation failed, check 'monitoring_data' or 'datapoint' columns.")
        
        if result_df.isna().sum().sum() > 0:
            log.info("Dropping rows with NaN values in the result DataFrame.")
            result_df.dropna(inplace=True)
            
        return result_df
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error in data_pipeline: {e}")
        raise HTTPException(status_code=400, detail="An internal error occurred during data processing.")

def stats_checker(request_data: request_structure) -> Dict[str, Dict[str, Any]]:
    input_data = request_data.data
    columns = request_data.columns
    stats_columns = request_data.stats
    date = request_data.date_columns
    from_date = request_data.from_date
    to_date = request_data.to_date
    system_type = request_data.system_type
    site = request_data.site

    data_df = data_pipeline(data=input_data, date_column=date, system_type=system_type, site=site, requested_columns=columns, from_date=from_date, to_date=to_date)
    final_return: Dict[str, Dict[str, Any]] = {}

    response_table: Dict[str, Callable] = {
        "min":                 lambda column, dataFrame: statistics_utils.find_min(column, dataFrame),
        "max":                 lambda column, dataFrame: statistics_utils.find_max(column, dataFrame),
        "mode":                lambda column, dataFrame: statistics_utils.find_mode(column, dataFrame),
        "mean":                lambda column, dataFrame: statistics_utils.find_mean(column, dataFrame),
        "standard deviation":  lambda column, dataFrame: statistics_utils.find_std(column, dataFrame),
        "variance":            lambda column, dataFrame: statistics_utils.find_var(column, dataFrame),
        "find_last_min":       lambda column, dataFrame: statistics_utils.find_last_min_date(column, dataFrame),
        "find_last_max":       lambda column, dataFrame: statistics_utils.find_last_max_date(column, dataFrame),
        "count_min_hits":      lambda column, dataFrame: statistics_utils.count_min_feature_occurence(column, dataFrame),
        "count_max_hits":      lambda column, dataFrame: statistics_utils.count_max_feature_occurence(column, dataFrame),
        "min_date_hits":       lambda column, dataFrame: statistics_utils.find_min_dates(column, dataFrame),
        "max_data_hits":       lambda column, dataFrame: statistics_utils.find_max_dates(column, dataFrame),
    }

    for column in columns:
        if column not in data_df.columns:
            log.warning(f"Column '{column}' not found in processed data. Skipping.")
            continue
            
        column_stats = {}
        for item in stats_columns:
            if item in response_table:
                column_stats[item] = response_table[item](column, data_df)
            
        final_return[column] = column_stats

    return final_return
    
@router.post('/data_stats_checker', response_model=response_structure)
def data_stats_checker(
    request_data: request_structure
):
    start = time.time()
    # log.info(f"Input Data: {request_data.dict()}") 
    result = stats_checker(request_data)
    end = time.time()
    log.info(f"Stastics completed in {end - start:.2f} seconds") 
    return {"stats_val": result}