from pydantic import BaseModel, Field, RootModel
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, List, Any, Union
import pandas as pd
from src.bms_ai.logger_config import setup_logger
import warnings
from src.bms_ai.utils.statistics_utils import *
import numpy as np

log = setup_logger(__name__)

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/aggregation", tags=["Prescriptive Optimization"])

DEFAULT_HIERARCHY = ["site", "equipment_id"]
DEFAULT_CATEGORICAL_STATS = ["count", "top", "mode"]
DEFAULT_NUMERIC_STATS = [
    "min", "max", "mean", "mode", "variance", "standard deviation", 
    "find_last_min", "find_last_max", "count_max_hits", "count_min_hits", 
    "min_date_hits", "max_data_hits", "min_date_hits_in_range", "max_date_hits_in_range"
]

class ResampleRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="The API response containing data under 'queryResponse'.")
    date_column: str = Field(default="data_received_on", description="The name of the column containing datetime values.")
    columns: Optional[List[str]] = Field(None, description="Optional list of specific data point columns to keep after pivoting.")
    n_points: Optional[int] = Field(None, description="If provided, resample data to this fixed number of points using binning.")
    from_date: Optional[str] = Field(None, description="Optional start date for filtering (e.g., 'YYYY-MM-DD').")
    to_date: Optional[str] = Field(None, description="Optional end date for filtering (e.g., 'YYYY-MM-DD').")

class ResampleResponse(RootModel):
    root: Dict[str, Dict[str, Any]]

class HierStatsRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Raw JSON. Must contain key 'queryResponse' -> list[dict]")
    date_column: str = Field(default="data_received_on", description="Name of datetime column in the records (e.g. 'data_received_on')")
    columns: Optional[List[str]] = Field(None, description="Columns to compute stats for. if omitted, compute for all.")
    hierarchy: Optional[List[str]] = Field(None, description="Hierarchy columns, top->child (e.g. ['site','equipment_id'])")
    stats: Optional[List[str]] = Field(None, description="Numeric stats to compute, e.g. ['min','max','mean','std','count']")
    from_date: Optional[str] = Field(None, description="Trim start date (inclusive)")
    to_date: Optional[str] = Field(None, description="Trim end date (inclusive)")

class HierStatsResponse(BaseModel):
    stats_val: Dict[str, Any]

# Resampling Functions

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

        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df[date_column] = df[date_column].dt.tz_localize(None)
        df.set_index(date_column, inplace=True)
        df.sort_index(ascending=True, inplace=True)

        required_cols = ['datapoint', 'monitoring_data']
        df = df[[c for c in required_cols if c in df.columns]]

        if from_date or to_date:
            try:
                df = df.loc[from_date:to_date]
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

    first_time_index = df.index.to_series().groupby(bin_index).first()
    
    df_num = df.select_dtypes(include=[np.number]) 
    df_cat = df.select_dtypes(exclude=[np.number]) 

    df_num_resampled = df_num.groupby(bin_index).mean()

    df_cat_resampled = df_cat.groupby(bin_index).agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )

    df_resampled = pd.concat([df_num_resampled, df_cat_resampled], axis=1)

    df_resampled.index = first_time_index # type: ignore
    
    return df_resampled.iloc[:n_points]

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

def standardize_stat_key(key: str) -> str:
    """Maps common variations of stat names to the required output keys."""
    key = key.lower()
    if key in ("std", "standard_deviation", "standard deviation"):
        return "standard deviation"
    elif key in ("mean", "avg", "average"):
        return "mean"
    elif key in ("min", "minimum"):
        return "min"
    elif key in ("max", "maximum"):
        return "max"
    elif key in ("count", "n"):
        return "count"
    elif key in ("mode",):
        return "mode"
    elif key in ("top",): 
        return "top"
    elif key in ("variance",):
        return "variance"
    elif key in ("find_last_min",):
        return "find_last_min"
    elif key in ("find_last_max",):
        return "find_last_max"
    elif key in ("count_min_hits",):
        return "count_min_hits"
    elif key in ("count_max_hits",):
        return "count_max_hits"
    elif key in ("min_date_hits",):
        return "min_date_hits"
    elif key in ("max_data_hits",):
        return "max_data_hits"
    elif key in ("min_date_hits_in_range",):
        return "min_date_hits_in_range"
    elif key in ("max_date_hits_in_range",):
        return "max_date_hits_in_range"
    return key

def compute_numeric_stats(df_series: pd.DataFrame, stats_list: List[str]) -> Dict[str, Any]:
    datapoint_name = df_series.columns[0]
    
    filtered_stats_list = [
        stat for stat in stats_list if standardize_stat_key(stat) not in ["count", "top"]
    ]
    out = {standardize_stat_key(st): None for st in filtered_stats_list}
    
    non_null_count = int(df_series[datapoint_name].count())
    
    if df_series.empty or non_null_count == 0:
        return out

    if "min" in out:
        out["min"] = find_min(datapoint_name, df_series) #type:ignore
    if "max" in out:
        out["max"] = find_max(datapoint_name, df_series) #type:ignore
    if "mean" in out:
        out["mean"] = find_mean(datapoint_name, df_series) #type:ignore
    if "mode" in out:
        out["mode"] = find_mode(datapoint_name, df_series) #type:ignore
    if "standard deviation" in out:
        out["standard deviation"] = find_std(datapoint_name, df_series) #type:ignore
    if "variance" in out:
        out["variance"] = find_var(datapoint_name, df_series) #type:ignore

    for k, v in out.items():
        if v == "N/A":
            out[k] = None

    if "find_last_min" in out:
        out["find_last_min"] = find_min(datapoint_name, df_series)  #type:ignore

    if "find_last_max" in out:
        out["find_last_max"] = find_max(datapoint_name, df_series) #type:ignore
        
    if "count_min_hits" in out:
        out["count_min_hits"] = count_min_feature_occurence(datapoint_name, df_series) #type:ignore
        
    if "count_max_hits" in out:
        out["count_max_hits"] = count_max_feature_occurence(datapoint_name, df_series) #type:ignore
        
    if "min_date_hits" in out:
        min_dates = find_min_dates(datapoint_name, df_series)
        out["min_date_hits"] = min_dates[0] if min_dates else None

    if "max_data_hits" in out:
        max_dates = find_max_dates(datapoint_name, df_series)
        out["max_data_hits"] = max_dates[0] if max_dates else None

    if "min_date_hits_in_range" in out:
        min_dates = find_min_dates(datapoint_name, df_series)
        out["min_date_hits_in_range"] = min_dates[0] if min_dates else None

    if "max_date_hits_in_range" in out:
        out["max_date_hits_in_range"] = find_last_max_date(datapoint_name, df_series) #type:ignore

    return out

def compute_categorical_stats(df_series: pd.DataFrame, stats_list: List[str]) -> Dict[str, Any]:
    datapoint_name = df_series.columns[0]
    out: Dict[str, Any] = {}
    
    categorical_keys_requested = [key for key in stats_list if key in ["count", "top", "mode"]]

    total_record_count = len(df_series)
    
    non_null_count = int(df_series[datapoint_name].count())

    if non_null_count == 0:
        if "count" in categorical_keys_requested:
            out["count"] = total_record_count
        if "mode" in categorical_keys_requested:
            out["mode"] = None
        if "top" in categorical_keys_requested:
            out["top"] = None
        return out
    
    if "count" in categorical_keys_requested:
        out["count"] = total_record_count 
        
    mode_val = find_mode(datapoint_name, df_series)
    
    if "mode" in categorical_keys_requested:
        out["mode"] = mode_val
        
    if "top" in categorical_keys_requested:
        out["top"] = mode_val
            
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
    
@router.post("/resample", response_model=ResampleResponse)
def resample_endpoint(req: ResampleRequest):
    df = resampling_data_pipeline(
        data=req.data,
        date_column=req.date_column,
        requested_columns=req.columns or [],
        from_date=req.from_date,
        to_date=req.to_date
    )

    if df.empty:
        raise HTTPException(status_code=400, detail="Pipeline returned empty DataFrame.")

    if req.columns:
        df = df[[c for c in req.columns if c in df.columns]]

    if req.n_points:
        df_resampled = resample_by_bins(df, req.n_points)
    else:
        df_reset = df.reset_index(names=[req.date_column])
        df_resampled = multi_col_resample_data_dynamically(df_reset, req.date_column, df.columns.tolist())

    if isinstance(df_resampled.index, pd.DatetimeIndex):
         df_resampled.index = df_resampled.index.astype(str)
    
    elif df_resampled.index.dtype in (np.int32, np.int64):
        df_resampled.index = df_resampled.index.astype(str)

    response_dict = {
        col: df_resampled[col].to_dict()
        for col in df_resampled.columns
    }

    return ResampleResponse(root=response_dict)

    
@router.post("/hierarchical_stats", response_model=HierStatsResponse)
def hier_stats_endpoint(request: HierStatsRequest):
    try:
        hierarchy = request.hierarchy if request.hierarchy is not None else DEFAULT_HIERARCHY
        
        stats_list_raw = request.stats if request.stats is not None else DEFAULT_NUMERIC_STATS
        stats_list = [standardize_stat_key(s) for s in stats_list_raw]
        
        for cat_stat in DEFAULT_CATEGORICAL_STATS:
             if cat_stat not in stats_list:
                stats_list.append(cat_stat)
            
        pipeline_out = data_pipeline(
            data=request.data,
            date_column=request.date_column,
            hierarchy=hierarchy,
            from_date=request.from_date,
            to_date=request.to_date,
        )

        raw_df = pipeline_out.get("raw_df", pd.DataFrame())
        
        group_cols = [h for h in hierarchy if h in raw_df.columns]
        if 'datapoint' in raw_df.columns:
            group_cols.append('datapoint')
            
        final_stats_val = build_hierarchy_skeleton(raw_df, hierarchy)

        if raw_df.empty:
            return HierStatsResponse(stats_val=final_stats_val)
            
        if request.columns is None and 'datapoint' in raw_df.columns:
            columns_to_compute = raw_df['datapoint'].dropna().unique().tolist()
        else:
            columns_to_compute = request.columns or []
        
        if not columns_to_compute:
             return HierStatsResponse(stats_val=final_stats_val)


        if "monitoring_data" in raw_df.columns:
            grouped = raw_df.groupby(group_cols)
            
            for name, group in grouped:
                datapoint_name = str(name[-1])
                hier_keys = [str(k) for k in name[:-1]] 

                if datapoint_name in columns_to_compute:
                    df_series = group[["monitoring_data"]].rename(columns={"monitoring_data": datapoint_name}).copy()

                    series_converted = try_numeric_conversion(df_series[datapoint_name])
                    df_series[datapoint_name] = series_converted

                    is_numeric = pd.api.types.is_numeric_dtype(series_converted.dtype) and series_converted.count() > 0

                    if is_numeric:
                        stats_result = compute_numeric_stats(df_series, stats_list)
                    else:
                        stats_result = compute_categorical_stats(df_series, DEFAULT_CATEGORICAL_STATS)
                        
                    set_nested_value(final_stats_val, hier_keys + [datapoint_name], stats_result)
        
        return HierStatsResponse(stats_val=final_stats_val)
        
    except HTTPException as e:
        raise e
    except Exception as e:
        log.error(f"Endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during statistics computation.")