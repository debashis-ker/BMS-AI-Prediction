import pandas as pd
from typing import Optional, Dict, List, Any, Union
from fastapi import HTTPException
import numpy as np
from src.bms_ai.logger_config import setup_logger

log = setup_logger(__name__)

def find_min(datapoint : str, data : pd.DataFrame):
    if data.empty:
        return "N/A"
    if pd.api.types.is_numeric_dtype(data[datapoint].dtype):
        non_na_min = data[datapoint].min()
        try:
            return non_na_min.item()
        except AttributeError:
            return non_na_min 
    return data[datapoint].min()

def find_max(datapoint : str, data : pd.DataFrame):
    if data.empty:
        return "N/A"
    if pd.api.types.is_numeric_dtype(data[datapoint].dtype):
        non_na_max = data[datapoint].max()
        try:
            return non_na_max.item()
        except AttributeError:
            return non_na_max
    return data[datapoint].max()

def find_mean(datapoint : str, data : pd.DataFrame):
    if data.empty:
        return "N/A"
    return data[datapoint].mean()

def find_mode(datapoint : str, data : pd.DataFrame):
    if data.empty:
        return "N/A"
    
    mode_val = data[datapoint].mode()
    if not mode_val.empty:
        val = mode_val.iloc[0]
        try:
            return val.item()
        except AttributeError:
            return val
    return None

def find_std(datapoint : str , data : pd.DataFrame):
    if data.empty or len(data) < 2:
        return "N/A"
    return data[datapoint].std()

def find_var(datapoint : str , data : pd.DataFrame):
    if data.empty or len(data) < 2:
        return "N/A"
    return data[datapoint].var()

def count_min_feature_occurence(datapoint : str, data : pd.DataFrame) -> int:
    if data.empty:
        return 0
        
    min_val = find_min(datapoint, data)
    
    if isinstance(min_val, str) and min_val == "N/A":
        return 0
        
    return int((data[datapoint] == min_val).sum()) 

def count_max_feature_occurence(datapoint : str, data : pd.DataFrame) -> int:
    if data.empty:
        return 0
        
    max_val = find_max(datapoint, data)

    if isinstance(max_val, str) and max_val == "N/A":
        return 0
        
    return int((data[datapoint] == max_val).sum())

def format_datetime(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    formatted_time = ts.strftime('%Y-%m-%dT%H:%M:%S.%f')
    formatted_time = formatted_time[:-3]
    offset = ts.strftime('%z')
    if ':' in offset:
        offset = offset.replace(':', '')
    return f"{formatted_time}{offset}"

def format_timestamps_list(timestamps: pd.DatetimeIndex) -> List[str]:
    return [format_datetime(ts) for ts in timestamps]

def find_min_dates(datapoint : str, data : pd.DataFrame) -> List:
    min_val = find_min(datapoint, data) 
    
    if isinstance(min_val, (str, type(None))):
        return []

    min_indices = pd.DatetimeIndex(data.index[data[datapoint] == min_val])
    
    return format_timestamps_list(min_indices)

def find_max_dates(datapoint : str, data : pd.DataFrame) -> List:
    max_val = find_max(datapoint, data)
    
    if isinstance(max_val, (str, type(None))):
        return []
        
    max_indices = pd.DatetimeIndex(data.index[data[datapoint] == max_val])
    
    return format_timestamps_list(max_indices)

def find_last_min_date(datapoint : str, data : pd.DataFrame) -> str:
    if data.empty:
        return "N/A"

    min_val = find_min(datapoint, data)
    if isinstance(min_val, (str, type(None))):
        return "N/A"

    min_indices = pd.DatetimeIndex(data.index[data[datapoint] == min_val])
    
    if not min_indices.empty:
        return format_datetime(min_indices[-1])
    return "N/A"

def find_last_max_date(datapoint : str, data : pd.DataFrame) -> str:
    if data.empty:
        return "N/A"
        
    max_val = find_max(datapoint, data)
    if isinstance(max_val, (str, type(None))):
        return "N/A"
        
    max_indices = pd.DatetimeIndex(data.index[data[datapoint] == max_val])

    if not max_indices.empty:
        return format_datetime(max_indices[-1])
    return "N/A"

DEFAULT_HIERARCHY = ["site", "equipment_id"]
DEFAULT_CATEGORICAL_STATS = ["count", "mode"]
DEFAULT_NUMERIC_STATS = [
    "min", "max", "mean", "mode", "variance", "standard deviation", "count_max_hits", "count_min_hits", 
    "dates_of_min_value", "dates_of_max_value", "last_date_of_min_value", "last_date_of_max_value"
]

def resampling_data_pipeline(
    data: Dict[str, Any],
    date_column: str,
    requested_columns: List[str],
    from_date: Optional[str],
    to_date: Optional[str]
) -> pd.DataFrame:
    try:
        df = process_api_dict_to_dataframe(data)

        if df.empty:
            raise HTTPException(status_code=400, detail='Pipeline received an empty DataFrame.')

        if date_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f'Date Column "{date_column}" not found in data.'
            )
        
        null_strings = ["null", "NaN", "NA"]
        df["monitoring_data"].replace(null_strings, np.nan, inplace=True)
        
        df.dropna(subset=["monitoring_data"], inplace=True)

        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df[date_column] = df[date_column].dt.tz_localize(None)
        df.set_index(date_column, inplace=True)
        df.sort_index(ascending=True, inplace=True)

        required_cols = ['datapoint', 'monitoring_data']
        df = df[[c for c in required_cols if c in df.columns]]

        if from_date or to_date:
            try:
                df = df.loc[from_date:to_date]
                # print(df.head())
            except Exception:
                pass

        temp = df.reset_index() 
        
        aggregated = temp.groupby([date_column, "datapoint"])["monitoring_data"].agg("first")
        result = aggregated.unstack(level="datapoint") 

        for col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='ignore') #type:ignore

        if requested_columns:
            existing_cols = [c for c in requested_columns if c in result.columns]
            result = result[existing_cols]

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline error: {e}")

def resample_by_bins(df: pd.DataFrame, n_points: int) -> pd.DataFrame:
    total_len = len(df) 
    bins = np.linspace(0, total_len, n_points + 1) 
    bin_index = np.digitize(np.arange(total_len), bins) - 1 

    print(bin_index)

    unix_timestamps_ns = df.index.to_series().view(np.int64)
    
    average_unix_timestamp_ns = unix_timestamps_ns.groupby(bin_index).mean()
    
    average_time_index = pd.to_datetime(average_unix_timestamp_ns, unit='ns')
    
    df_num = df.select_dtypes(include=[np.number]) 
    df_cat = df.select_dtypes(exclude=[np.number]) 

    df_num = df_num.apply(pd.to_numeric, errors='coerce')

    df_num_resampled = df_num.groupby(bin_index).mean().iloc[:n_points]

    df_cat_resampled = df_cat.groupby(bin_index).agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    ).iloc[:n_points]

    df_resampled = pd.concat([df_num_resampled, df_cat_resampled], axis=1)

    df_resampled.index = average_time_index.iloc[:n_points] #type:ignore
    
    return df_resampled

def multi_col_resample_data_dynamically(
    df: pd.DataFrame,
    datetime_column: str = "data_received_on", 
    monitoring_columns: Union[str, List[str]] = "monitoring_data" 
) -> pd.DataFrame:
    """
    Resamples data based on the total time span to an appropriate frequency (e.g., 'H', 'D', 'W').
    This function is now type-aware and handles multiple columns by splitting 
    into numeric (mean aggregation) and categorical (mode aggregation) data.
    """
    df_copy = df.copy() 

    if isinstance(monitoring_columns, str):
        monitoring_columns = [monitoring_columns] 

    df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column], errors="coerce")

    for col in monitoring_columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(df_copy[col])


    df_copy.dropna(subset=[datetime_column], inplace=True)
    df_copy.sort_values(by=datetime_column, inplace=True)

    if df_copy.empty:
        return pd.DataFrame()

    time_span = df_copy[datetime_column].max() - df_copy[datetime_column].min() 

    if time_span < pd.Timedelta(days=5):
        freq = "H"
    elif time_span < pd.Timedelta(weeks=2):
        freq = "D"
    elif time_span < pd.Timedelta(days=100):
        freq = "W"
    elif time_span < pd.Timedelta(days=365 * 2):
        freq = "M"
    elif time_span < pd.Timedelta(days=365 * 5):
        freq = "Q"
    else:
        freq = "Y"

    df_indexed = df_copy.set_index(datetime_column)
    
    numeric_cols = [col for col in monitoring_columns if pd.api.types.is_numeric_dtype(df_indexed[col])]
    cat_cols = [col for col in monitoring_columns if not pd.api.types.is_numeric_dtype(df_indexed[col])]

    resampled_parts = []
    
    if numeric_cols:
        resampled_num = df_indexed[numeric_cols].resample(freq).mean()
        resampled_parts.append(resampled_num)
        
    if cat_cols:
        resampled_cat = df_indexed[cat_cols].resample(freq).agg(
             lambda x: x.mode()[0] if not x.mode().empty else None #type:ignore
        )
        resampled_parts.append(resampled_cat)

    if not resampled_parts:
        return pd.DataFrame()

    resampled = pd.concat(resampled_parts, axis=1)
    resampled.dropna(how="all", inplace=True)

    return resampled


# Aggregation Functions

def process_api_dict_to_dataframe(api_data_dict: dict) -> pd.DataFrame:
    try:
        records = api_data_dict.get("queryResponse", [])
        return pd.DataFrame(records)
    except Exception as e:
        log.error(f"Error processing JSON data: {e}")
        raise HTTPException(status_code=400, detail=f"An unexpected error occurred in JSON Data: {e}")

def try_numeric_conversion(series: pd.Series) -> pd.Series:
    s_clean = series.replace(["null", ""], np.nan) 
    converted = pd.to_numeric(s_clean, errors="coerce")
    
    if converted.notna().sum() > 0:
        return converted.replace({np.nan: None})
    else:
        return series.astype(str).replace("nan", None).replace("null", None).replace("", None).dropna()

MASTER_CANONICAL_STATS = [
    "min", "max", "mean", "mode", "variance", "standard deviation", 
    "count_max_hits", "count_min_hits", "dates_of_min_value", "dates_of_max_value", 
    "last_date_of_min_value", "last_date_of_max_value", "count", "top" 
]

def compute_numeric_stats(df_series: pd.DataFrame, stats_list: List[str]) -> Dict[str, Any]:
    datapoint_name = df_series.columns[0]
    
    valid_numeric_stats = [
        stat for stat in stats_list if stat in MASTER_CANONICAL_STATS and stat not in ["count"]
    ]
    
    out = {st: None for st in valid_numeric_stats}
    
    non_null_count = int(df_series[datapoint_name].count())
    
    if df_series.empty or non_null_count == 0:
        return out

    if "min" in out: out["min"] = find_min(datapoint_name, df_series)
    if "max" in out: out["max"] = find_max(datapoint_name, df_series) 
    if "mean" in out: out["mean"] = find_mean(datapoint_name, df_series) 
    if "mode" in out: out["mode"] = find_mode(datapoint_name, df_series) #type:ignore
    if "standard deviation" in out: out["standard deviation"] = find_std(datapoint_name, df_series) #type:ignore
    if "variance" in out: out["variance"] = find_var(datapoint_name, df_series) #type:ignore

    for k, v in out.items():
        if v == "N/A": out[k] = None
        
    if "count_min_hits" in out: out["count_min_hits"] = count_min_feature_occurence(datapoint_name, df_series) #type:ignore
    if "count_max_hits" in out: out["count_max_hits"] = count_max_feature_occurence(datapoint_name, df_series) #type:ignore
        
    if "dates_of_min_value" in out: out["dates_of_min_value"] = find_min_dates(datapoint_name, df_series) #type:ignore
    if "dates_of_max_value" in out: out["dates_of_max_value"] = find_max_dates(datapoint_name, df_series) #type:ignore

    if "last_date_of_min_value" in out: out["last_date_of_min_value"] = find_last_min_date(datapoint_name, df_series) #type:ignore
    if "last_date_of_max_value" in out: out["last_date_of_max_value"] = find_last_max_date(datapoint_name, df_series) #type:ignore

    return out

def compute_categorical_stats(df_series: pd.DataFrame, stats_list: List[str]) -> Dict[str, Any]:
    datapoint_name = df_series.columns[0]
    out: Dict[str, Any] = {}
    
    categorical_keys_requested = [key for key in stats_list if key in ["count", "mode"]]

    total_record_count = len(df_series)
    
    non_null_count = int(df_series[datapoint_name].count())

    if non_null_count == 0:
        if "count" in categorical_keys_requested:
            out["count"] = total_record_count
        if "mode" in categorical_keys_requested:
            out["mode"] = None
        return out
    
    if "count" in categorical_keys_requested:
        out["count"] = total_record_count 
        
    mode_val = find_mode(datapoint_name, df_series)
    
    if "mode" in categorical_keys_requested:
        out["mode"] = mode_val
            
    return out

def build_hierarchy_skeleton(raw_df: pd.DataFrame, hierarchy: List[str]) -> Dict[str, Any]:
    if raw_df is None or raw_df.empty or not hierarchy:
        return {}
        
    valid_hierarchy = [h for h in hierarchy if h in raw_df.columns]
    if not valid_hierarchy:
        return {}

    combos = raw_df[valid_hierarchy].drop_duplicates().fillna("unknown")
    skeleton: Dict[str, Any] = {}
    
    for _, row in combos.iterrows():
        cur = skeleton
        for h in valid_hierarchy:
            val = row[h] 
            key = str(val)
            if key not in cur:
                cur[key] = {}
            cur = cur[key]
            
    return skeleton

def set_nested_value(dct: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur = dct
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    
    cur[keys[-1]] = value
    
def data_pipeline(
    data: Dict[str, Any],
    date_column: str,
    hierarchy: List[str],
    from_date: Optional[str],
    to_date: Optional[str],
) -> Dict[str, Any]:
    
    try:
        df = process_api_dict_to_dataframe(data)
        if df.empty: return {"raw_df": pd.DataFrame()}

        if date_column not in df.columns: return {"raw_df": df}

        null_strings = ["null", "NaN", "NA"]
        df["monitoring_data"].replace(null_strings, np.nan, inplace=True)
        
        df.dropna(subset=["monitoring_data"], inplace=True)

        df[date_column] = pd.to_datetime(df[date_column], errors="coerce", utc=True)
        df = df.dropna(subset=[date_column]).set_index(date_column)
         
        if df.empty: return {"raw_df": pd.DataFrame()}

        if from_date:
            try: df = df[df.index >= pd.to_datetime(from_date, utc=True)]
            except Exception: pass
        if to_date:
            try: df = df[df.index <= pd.to_datetime(to_date, utc=True)]
            except Exception: pass
                
        if df.empty: return {"raw_df": pd.DataFrame()}

        required_cols = list(set(hierarchy + ['datapoint', 'monitoring_data']))
        present_cols = [c for c in required_cols if c in df.columns]
        
        return {"raw_df": df[present_cols].copy()}

    except Exception:
        log.exception("data_pipeline failed")
        return {"raw_df": pd.DataFrame()}
    