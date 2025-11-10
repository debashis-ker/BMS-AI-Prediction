import os
import sys
import joblib
from src.bms_ai.exception import CustomException
from src.bms_ai.logger_config import setup_logger
from typing import Any, Dict, List, Union

import pandas as pd
import numpy as np



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

        if time_span < pd.Timedelta(days=5):
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
        
        
        print(f"resampling_frequency determined: {resampling_frequency} :: time_span: {time_span}")
        
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



def print_dataframe_info(df, name="DataFrame", max_unique=20):
    """
    Print comprehensive information about a pandas DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze
    name : str, optional
        Name to identify the DataFrame (default: "DataFrame")
    max_unique : int, optional
        Maximum number of unique values to display for categorical columns (default: 20)
    """
    print(f"\n{'='*70}")
    print(f"Comprehensive Analysis for: {name}")
    print(f"{'='*70}\n")
    
    print("1. BASIC INFORMATION")
    print("-" * 70)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Duplicate Rows: {df.duplicated().sum()}")
    print()
    
    print("2. DATAFRAME INFO")
    print("-" * 70)
    df.info()
    print()
    
    print("3. COLUMN TYPES BREAKDOWN")
    print("-" * 70)
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} column(s)")
    print()
    
    print("4. DETAILED COLUMN INFORMATION")
    print("-" * 70)
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        print(f"\nColumn: '{col}'")
        print(f"  Data Type: {dtype}")
        print(f"  Non-Null: {non_null} | Null: {null_count} ({null_pct:.1f}%)")
        print(f"  Unique Values: {unique_count}")
        
        if dtype == 'object' or dtype.name == 'category' or unique_count <= max_unique:
            value_counts = df[col].value_counts()
            print(f"  Value Counts (top {min(10, len(value_counts))}):")
            for val, count in value_counts.head(10).items():
                pct = (count / len(df)) * 100
                print(f"    {val}: {count} ({pct:.1f}%)")
    
    print()
    
    # ============== MISSING VALUES SUMMARY ==============
    print("5. MISSING VALUES SUMMARY")
    print("-" * 70)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_df = pd.DataFrame({
            'Column': missing[missing > 0].index,
            'Missing Count': missing[missing > 0].values,
            'Missing %': (missing[missing > 0].values / len(df) * 100).round(2)
        })
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found!")
    print()
    
    # ============== NUMERIC COLUMNS ANALYSIS ==============
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("6. NUMERIC COLUMNS - DESCRIPTIVE STATISTICS")
        print("-" * 70)
        print(df[numeric_cols].describe())
        print()
        
        print("Additional Numeric Statistics:")
        print("-" * 70)
        for col in numeric_cols:
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Median: {df[col].median():.2f}")
            print(f"  Mode: {df[col].mode().values[0] if len(df[col].mode()) > 0 else 'N/A'}")
            print(f"  Std Dev: {df[col].std():.2f}")
            print(f"  Variance: {df[col].var():.2f}")
            print(f"  Range: {df[col].min():.2f} to {df[col].max():.2f}")
            print(f"  IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")
        print()
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print("7. CATEGORICAL COLUMNS - UNIQUE VALUES")
        print("-" * 70)
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"\n{col}: {unique_count} unique value(s)")
            
            if unique_count <= max_unique:
                print(f"  All unique values and counts:")
                value_counts = df[col].value_counts()
                for val, count in value_counts.items():
                    pct = (count / len(df)) * 100
                    print(f"    '{val}': {count} ({pct:.1f}%)")
            else:
                print(f"  (Too many unique values to display. Showing top 10)")
                value_counts = df[col].value_counts().head(10)
                for val, count in value_counts.items():
                    pct = (count / len(df)) * 100
                    print(f"    '{val}': {count} ({pct:.1f}%)")
        print()
    
    print("8. DATA PREVIEW")
    print("-" * 70)
    print("First 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print()
    
    print(f"{'='*70}\n")
        
        