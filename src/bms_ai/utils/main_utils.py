import os
import sys
import joblib
from src.bms_ai.exception import CustomException
from src.bms_ai.logger_config import setup_logger
from typing import Any, Dict, List, Union

import pandas as pd



log = setup_logger(__name__)

def save_object(file_path, obj):
    try:
        log.info(f"Saving object to {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)
        log.info("Object saved successfully.")

    except Exception as e:
        log.error(f"Exception occurred in save_object: {e}")
        raise CustomException(e, sys)
    


def get_average_monitoring_data(
    df: pd.DataFrame, 
    resampling_frequency: str, 
    datetime_column: str, 
    monitoring_columns: List[str]
) -> pd.DataFrame:
    """
    Calculates the average of monitoring data over a specified resampling frequency for multiple columns.

    Args:
        df: The input DataFrame.
        resampling_frequency: The frequency string for resampling (e.g., 'D', 'W', 'M').
        datetime_column: The name of the column containing datetime information.
        monitoring_columns: A list of column names to be resampled.

    Returns:
        A DataFrame with the resampled and averaged data.
    """
    return df.set_index(datetime_column)[monitoring_columns].resample(resampling_frequency).mean()

def resample_data_dynamically(
    df: pd.DataFrame,
    datetime_column: str = "data_received_on",
    monitoring_columns: Union[str, List[str]] = "monitoring_data"
) -> pd.DataFrame:
    """
    Cleans, sorts, and dynamically resamples time-series data for one or more columns.

    This function automatically determines the most appropriate resampling frequency 
    (from hourly to yearly) based on the total time span of the data. It handles
    data type conversion, missing values, and sorts the data before resampling.

    Args:
        df: The input DataFrame.
        datetime_column: The name of the datetime column. Defaults to "data_received_on".
        monitoring_columns: A single column name (str) or a list of column names (List[str]) 
                            to resample. Defaults to "monitoring_data".

    Returns:
        A pandas DataFrame containing the resampled data, or an empty DataFrame if an error occurs.
    """
    try:
        df_copy = df.copy()

        if isinstance(monitoring_columns, str):
            monitoring_columns = [monitoring_columns]

        df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column], errors="coerce")
        for col in monitoring_columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

        columns_to_check = [datetime_column] + monitoring_columns
        df_copy.dropna(subset=columns_to_check, inplace=True)

        if df_copy.empty:
            return pd.DataFrame()

        df_copy.sort_values(by=datetime_column, ascending=True, inplace=True)

        time_span = df_copy[datetime_column].max() - df_copy[datetime_column].min()

        if time_span < pd.Timedelta(days=2):
            resampling_frequency = "H" 
        elif time_span < pd.Timedelta(weeks=2):
            resampling_frequency = "D" 
        elif time_span < pd.Timedelta(days=100):
            resampling_frequency = "W"  
        elif time_span < pd.Timedelta(days=365 * 2):
            resampling_frequency = "M" 
        elif time_span < pd.Timedelta(days=365 * 5):
            resampling_frequency = "Q"  
        else:
            resampling_frequency = "Y"  

        resampled_data = get_average_monitoring_data(
            df_copy, resampling_frequency, datetime_column, monitoring_columns
        )
        
        resampled_data.dropna(how='all', inplace=True)

    except KeyError as e:
        print(f"Error: Column not found -> {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during resampling: {e}")
        return pd.DataFrame()
        
    return resampled_data 


def clean_actual_v_predicted_fan_power_data(data: List[Any], resampled: bool = False) -> List[Dict[str, Any]]:
    
    """
    Cleans and prepares actual vs predicted fan power data for analysis.

    Args:
        data: The input list containing fan power data.
        resampled: Boolean indicating if the data should be resampled.

    Returns:
        A list of dictionaries containing cleaned and processed fan power data.
    """
    
    if not data:
        return []
    
    log.info("In clean_actual_v_predicted_fan_power_data")
    rows = []

    for entry in data:
        try:
            rows.append({
                "timestamp": entry['timeStamp'],
                "dm_pred_min_fan_power_kw": entry['dmPred']['min_fan_power_kw'],
                "as_pred_min_fan_power_kw": entry['asPred']['min_fan_power_kw'],
                "actual_predictions": entry['actualPred']['prediction']['0']['3']
            })
        except (KeyError, TypeError) as e:
            log.warning(f"Skipping malformed entry: {e}")
            continue
    
    if not rows:
        return []
    
    if not resampled:
        return rows
    
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = resample_data_dynamically(
        df=df, 
        datetime_column="timestamp", 
        monitoring_columns=["dm_pred_min_fan_power_kw", "as_pred_min_fan_power_kw", "actual_predictions"]
    )
    
    if df.empty:
        log.warning("Resampling resulted in empty DataFrame")
        return []
    
    log.info("Resampled actual vs predicted fan power data successfully.")
    
    df_reset = df.reset_index()
    return df_reset.to_dict(orient='records')
        
        