"""
FastAPI router for Chilled-Water Energy Consumption.

Endpoints:
    POST /energy_consumption/process   -> process_hourly_energy  (fetch + calc + store)
    POST /energy_consumption/history   -> get_energy_data        (retrieve stored data)
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from cassandra.cluster import Session

from src.bms_ai.api.dependencies import get_cassandra_session
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.energy_consumption_utils import (
    DEFAULT_BUILDING_ID,
    process_and_store_hourly_energy,
    fetch_energy_data,
    aggregate_energy_data,
)

import warnings

warnings.filterwarnings("ignore")

log = setup_logger(__name__)

router = APIRouter(prefix="/energy_consumption", tags=["Energy Consumption"])




class ProcessEnergyRequest(BaseModel):
    building_id: str = Field(
        default=DEFAULT_BUILDING_ID,
        description="Building UUID (with or without hyphens).",
    )
    from_date: Optional[str] = Field(
        default=None,
        description="ISO-8601 start datetime. Defaults to 1 hour ago.",
        examples=["2026-03-04T16:00:00Z"],
    )
    to_date: Optional[str] = Field(
        default=None,
        description="ISO-8601 end datetime. Defaults to now.",
        examples=["2026-03-04T17:00:00Z"],
    )


class GetEnergyDataRequest(BaseModel):
    building_id: str = Field(
        default=DEFAULT_BUILDING_ID,
        description="Building ID.",
    )
    from_date: Optional[str] = Field(
        default=None,
        description="ISO-8601 start datetime. Defaults to 24 hours ago.",
        examples=["2026-03-03T00:00:00Z"],
    )
    to_date: Optional[str] = Field(
        default=None,
        description="ISO-8601 end datetime. Defaults to now.",
        examples=["2026-03-04T00:00:00Z"],
    )
    frequency: Optional[str] = Field(
        default="H",
        description=(
            "Aggregation frequency for the returned data. "
            "H = Hourly (default), D = Daily, W = Weekly, M = Monthly, Q = Quarterly."
        ),
    )


SUMMATION_KEYS: Dict[str, str] = {
    "total_kwh": "kwh_value",
    "total_rth": "rth_value",
    "total_cost_aed": "estimated_cost_aed",
    "avg_delta_t": "delta_t",
    "avg_flow": "avg_flow",
    "avg_chws": "avg_chws",
    "avg_chwr": "avg_chwr",
}


class MonthlyTotalRequest(BaseModel):
    building_id: str = Field(
        default=DEFAULT_BUILDING_ID,
        description="Building ID.",
    )
    from_date: Optional[str] = Field(
        default=None,
        description="ISO-8601 start datetime (UTC). Defaults to start of the current month.",
        examples=["2026-03-01T00:00:00Z"],
    )
    to_date: Optional[str] = Field(
        default=None,
        description="ISO-8601 end datetime (UTC). Defaults to current datetime.",
        examples=["2026-03-06T00:00:00Z"],
    )
    keys: Optional[List[str]] = Field(
        default=["total_kwh"],
        description=(
            "List of keys to compute summations for. "
            "Allowed keys: total_kwh, total_rth, total_cost_aed, "
            "avg_delta_t, avg_flow, avg_chws, avg_chwr."
        ),
    )




def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO string into a timezone-aware datetime, or return None."""
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid datetime format: '{value}'. Use ISO-8601 (e.g. 2026-03-04T16:00:00Z).",
        )




@router.post("/process", summary="Process & store chilled-water energy + billing")
async def process_hourly_energy(
    body: ProcessEnergyRequest,
    session: Session = Depends(get_cassandra_session),
):
    """
    1. Fetch raw CHW / FLOWMTR data for the requested window (default: last 1 hour).
    2. Calculate energy: ``kWh = (Avg_ChwR - Avg_ChwS) * Avg_Flow * 4.186``.
    3. Calculate billing: ``RTh = kWh / 3.517``, ``Cost = RTh * 0.568 AED``.
    4. Persist results into ``chilled_water_energy_consumption_<buildingId>``.

    Returns a summary including the computed record.
    """
    from_dt = _parse_dt(body.from_date)
    to_dt = _parse_dt(body.to_date)

    try:
        result = process_and_store_hourly_energy(
            session=session,
            building_id=body.building_id,
            from_dt=from_dt,
            to_dt=to_dt,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[EnergyAPI] process_hourly_energy failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/history", summary="Retrieve stored chilled-water energy & billing data")
async def get_energy_data(
    body: GetEnergyDataRequest,
    session: Session = Depends(get_cassandra_session),
):
    """
    Retrieve previously computed energy consumption and billing data.

    * Defaults:
        - Last 24 hours, split hourly.
    * Custom:*
        - Provide ``from_date`` / ``to_date`` plus ``frequency``
          (H = hourly, D = daily, W = weekly, M = monthly, Q = quarterly).
    """
    now = datetime.now(timezone.utc)
    from_dt = _parse_dt(body.from_date) or (now - timedelta(hours=24))
    to_dt = _parse_dt(body.to_date) or now
    freq = (body.frequency or "H").upper()

    if freq not in ("H", "D", "W", "M", "Q"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid frequency '{freq}'. Must be one of H, D, W, M, Q.",
        )

    try:
        records = fetch_energy_data(
            session=session,
            building_id=body.building_id,
            from_dt=from_dt,
            to_dt=to_dt,
        )

        if not records:
            return {
                "status": "no_data",
                "message": "No energy consumption records found for the given parameters.",
                "data": [],
            }

        aggregated = aggregate_energy_data(records, freq=freq)

        def _safe(v: float) -> float:
            return 0.0 if (math.isnan(v) or math.isinf(v)) else v

        for rec in aggregated:
            for k, v in rec.items():
                if isinstance(v, float):
                    rec[k] = _safe(v)

        n = len(aggregated)
        return {
            "status": "success",
            "frequency": freq,
            "from_date": from_dt.isoformat(),
            "to_date": to_dt.isoformat(),
            "total_records": n,
            "avg_delta_t": round(_safe(sum(r.get("delta_t", 0) for r in aggregated) / n), 4),
            "avg_flow": round(_safe(sum(r.get("avg_flow", 0) for r in aggregated) / n), 4),
            "total_kwh": round(_safe(sum(r.get("kwh_value", 0) for r in aggregated)), 4),
            "total_rth": round(_safe(sum(r.get("rth_value", 0) for r in aggregated)), 4),
            "total_cost_aed": round(_safe(sum(r.get("estimated_cost_aed", 0) for r in aggregated)), 4),
            "data": aggregated,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[EnergyAPI] get_energy_data failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/get_aggregated_values", summary="Get monthly total / aggregated values for selected keys")
async def monthly_total(
    body: MonthlyTotalRequest,
    session: Session = Depends(get_cassandra_session),
):
    """
    Return aggregated totals (or averages) for the requested period.
        * Defaults to current month-to-date if no dates provided.
        * Default key is ``total_kwh`` but you can specify any combination of:
            - total_kwh
    """
    now = datetime.now(timezone.utc)
    from_dt = _parse_dt(body.from_date) or now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    to_dt = _parse_dt(body.to_date) or now

    requested_keys = body.keys or ["total_kwh"]
    invalid_keys = [k for k in requested_keys if k not in SUMMATION_KEYS]
    if invalid_keys:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid key(s): {invalid_keys}. "
                f"Allowed keys: {list(SUMMATION_KEYS.keys())}"
            ),
        )

    try:
        records = fetch_energy_data(
            session=session,
            building_id=body.building_id,
            from_dt=from_dt,
            to_dt=to_dt,
        )

        if not records:
            return {
                "status": "no_data",
                "message": "No energy consumption records found for the given period.",
                "from_date": from_dt.isoformat(),
                "to_date": to_dt.isoformat(),
                "totals": {k: 0.0 for k in requested_keys},
            }

        def _safe(v: float) -> float:
            return 0.0 if (math.isnan(v) or math.isinf(v)) else v

        totals: Dict[str, float] = {}
        for key in requested_keys:
            field = SUMMATION_KEYS[key]
            values = [r.get(field, 0) or 0 for r in records]
            if key.startswith("avg_"):
                totals[key] = round(_safe(sum(values) / len(values)), 4)
            else:
                totals[key] = round(_safe(sum(values)), 4)

        return {
            "status": "success",
            "from_date": from_dt.isoformat(),
            "to_date": to_dt.isoformat(),
            "total_records": len(records),
            "totals": totals,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[EnergyAPI] monthly_total failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))