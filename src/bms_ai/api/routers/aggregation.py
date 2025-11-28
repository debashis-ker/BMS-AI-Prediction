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
        
        stats_list = request.stats if request.stats is not None else DEFAULT_NUMERIC_STATS
        
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