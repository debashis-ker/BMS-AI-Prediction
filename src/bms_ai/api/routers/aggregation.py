from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, List, Any, Callable
import pandas as pd
from src.bms_ai.logger_config import setup_logger
import warnings
import time

log = setup_logger(__name__)

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/aggregation", tags=["Prescriptive Optimization"])

ColumnName = str
DateColumn = str
Dataframe = pd.DataFrame

class request_structure(BaseModel):
    data: Dict[str, Any] = Field(default={}, description="Data Input")
    columns: List[str] = Field(default=["Fan Power meter (KW)"], description="Columns Required for Statistics")
    date_columns: str = Field(default="data_received_on", description="Date Column Required for Statistics")
    from_date_columns : Optional[str] = Field(None, description="Start date for range-based calculations (e.g., '2024-01-01')")
    to_date_columns : Optional[str] = Field(None, description="End date for range-based calculations (e.g., '2024-01-31')")
    stats: List[str] = Field(default=["min","max","mean","standard deviation","variance","find_last_min","find_last_max","min_date_hits", "max_data_hits", "min_date_hits_in_range", "max_date_hits_in_range"],description="Statistics Required for Output")

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

def data_pipeline(data: Dict[str, Any], date_column: str) -> pd.DataFrame:
    try:
        try:
            df_ahu = process_api_dict_to_dataframe(data)
            if df_ahu.empty:
                log.error("DataFrame is empty after processing.")
        except Exception as e:
            log.error(f"Error in initial DataFrame processing: {e}")
            raise HTTPException(status_code=400, detail="Problem in the initial data processing.")

        try:
            df_ahu = df_ahu[(df_ahu['system_type'] == "AHU") & (df_ahu['equipment_name'] == "AHU-GF-01")].copy()
            if df_ahu.empty:
                log.error("AHU Ground Floor Data Not Found after filtering.")
                
        except Exception as e:
            log.error(f"Error applying equipment filters: {e}")
            raise HTTPException(status_code=400, detail="AHU Ground Floor Data Not Found.")

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
            ('monitoring_data', {'inactive': 0.0, 'active': 1.0}),
            ('site', {'Ground Floor': 0.0, 'Rooftop': 1.0})
        ]:
            if col_name in df_ahu.columns:
                try:
                    df_ahu[col_name] = df_ahu[col_name].replace(mapping).astype('float64')
                except Exception as e:
                    log.warning(f"Failed to convert/replace for column '{col_name}': {e}")
            else:
                 if col_name in ['monitoring_data', 'site']:
                    log.error(f"Required column '{col_name}' not found.")
        
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
    
    except Exception as e:
        log.error(f"Unexpected error in data_pipeline: {e}")
        raise HTTPException(status_code=400, detail="An internal error occurred during data processing.")

def find_min(datapoint : ColumnName, data : Dataframe):
    return data[datapoint].min().item()

def find_max(datapoint : ColumnName, data : Dataframe):
    return data[datapoint].max().item()

def find_mean(datapoint : ColumnName, data : Dataframe):
    return data[datapoint].mean()

def find_mode(datapoint : ColumnName, data : Dataframe):
    mode_val = data[datapoint].mode()
    if not mode_val.empty:
        return mode_val.iloc[0].item()
    return None

def find_std(datapoint : ColumnName , data : Dataframe):
    return data[datapoint].std()

def find_var(datapoint : ColumnName , data : Dataframe):
    return data[datapoint].var()

def count_min_feature_occurence(datapoint : ColumnName, data : Dataframe, date : DateColumn) -> int:
    min_val = find_min(datapoint, data)
    return int((data[datapoint] == min_val).sum()) 

def count_max_feature_occurence(datapoint : ColumnName, data : Dataframe, date : DateColumn) -> int:
    max_val = find_max(datapoint, data)
    return int((data[datapoint] == max_val).sum())

def find_min_dates(datapoint : ColumnName, data : Dataframe, date : DateColumn) -> List:
    return data.index[data[datapoint] == find_min(datapoint, data)].tolist()

def find_last_min_date(datapoint : ColumnName, data : Dataframe, date : DateColumn) -> str:
    min_indices = data.index[data[datapoint] == find_min(datapoint,data)]
    if not min_indices.empty:
        return str(min_indices[-1])
    return "N/A"

def find_max_dates(datapoint : ColumnName, data : Dataframe, date: DateColumn) -> List:
    return data.index[data[datapoint] == find_max(datapoint, data)].tolist()

def find_last_max_date(datapoint : ColumnName, data : Dataframe, date: DateColumn) -> str:
    max_indices = data.index[data[datapoint] == find_max(datapoint,data)]
    if not max_indices.empty:
        return str(max_indices[-1])
    return "N/A"

def find_min_dates_in_range(datapoint: ColumnName, data: Dataframe, date: DateColumn, from_date: Optional[str], to_date: Optional[str]) -> List[str]:
    min_val = find_min(datapoint, data)
    mask = (data[datapoint] == min_val)
    
    if from_date:
        try:
            start_date = pd.to_datetime(from_date)
            mask = mask & (data.index >= start_date)
        except Exception as e:
            log.warning(f"Invalid from_date format: {from_date}. Skipping start range filter. Error: {e}")
            
    if to_date:
        try:
            end_date = pd.to_datetime(to_date)
            mask = mask & (data.index <= end_date)
        except Exception as e:
            log.warning(f"Invalid to_date format: {to_date}. Skipping end range filter. Error: {e}")
            
    return data.index[mask].tolist()

def find_max_dates_in_range(datapoint: ColumnName, data: Dataframe, date: DateColumn, from_date: Optional[str], to_date: Optional[str]) -> List[str]:
    max_val = find_max(datapoint, data)
    mask = (data[datapoint] == max_val)
    
    if from_date:
        try:
            start_date = pd.to_datetime(from_date)
            mask = mask & (data.index >= start_date)
        except Exception as e:
            log.warning(f"Invalid from_date format: {from_date}. Skipping start range filter. Error: {e}")
            
    if to_date:
        try:
            end_date = pd.to_datetime(to_date)
            mask = mask & (data.index <= end_date)
        except Exception as e:
            log.warning(f"Invalid to_date format: {to_date}. Skipping end range filter. Error: {e}")
            
    return data.index[mask].tolist()


def stats_checker(request_data: request_structure) -> Dict[str, Dict[str, Any]]:
    input_data = request_data.data
    columns = request_data.columns
    stats_columns = request_data.stats
    date = request_data.date_columns
    from_date = request_data.from_date_columns
    to_date = request_data.to_date_columns

    data_df = data_pipeline(data=input_data, date_column=date)
    
    final_return: Dict[str, Dict[str, Any]] = {}

    response_table: Dict[str, Callable] = {
        "min":                 lambda c, d, dt, fdt, tdt: find_min(c, d),
        "max":                 lambda c, d, dt, fdt, tdt: find_max(c, d),
        "mode":                lambda c, d, dt, fdt, tdt: find_mode(c, d),
        "mean":                lambda c, d, dt, fdt, tdt: find_mean(c, d),
        "standard deviation":  lambda c, d, dt, fdt, tdt: find_std(c, d),
        "variance":            lambda c, d, dt, fdt, tdt: find_var(c, d),
        "find_last_min":       lambda c, d, dt, fdt, tdt: find_last_min_date(c, d, dt),
        "find_last_max":       lambda c, d, dt, fdt, tdt: find_last_max_date(c, d, dt),
        "count_min_hits":      lambda c, d, dt, fdt, tdt: count_min_feature_occurence(c, d, dt),
        "count_max_hits":      lambda c, d, dt, fdt, tdt: count_max_feature_occurence(c, d, dt),
        "min_date_hits":       lambda c, d, dt, fdt, tdt: find_min_dates(c, d, dt),
        "max_data_hits":       lambda c, d, dt, fdt, tdt: find_max_dates(c, d, dt),
        "min_date_hits_in_range": lambda c, d, dt, fdt, tdt: find_min_dates_in_range(c, d, dt, fdt, tdt),
        "max_date_hits_in_range": lambda c, d, dt, fdt, tdt: find_max_dates_in_range(c, d, dt, fdt, tdt),
    }

    for column in columns:
        if column not in data_df.columns:
            log.warning(f"Column '{column}' not found in processed data. Skipping.")
            continue
            
        column_stats = {}
        for item in stats_columns:
            if item in response_table:
                column_stats[item] = response_table[item](column, data_df, date, from_date, to_date)
            
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