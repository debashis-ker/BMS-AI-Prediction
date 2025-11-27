import pandas as pd
from typing import List

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
