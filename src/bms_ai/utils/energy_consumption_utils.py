"""
Utility / service layer for Chilled-Water Energy Consumption calculations.

Formula:  Energy (kWh) = (Avg_ChwR - Avg_ChwS) * Avg_Flow * 4.186
    - Temperature in degC, Flow in L/s
    - The constant 4.186 kJ/(kg.degC) converts the thermal power to kJ/s (~ kW).
    - For a 1-hour window the average kW IS the kWh value for that hour.

Billing:
    RTh  = kWh / 3.517
    Cost = RTh * CONSUMPTION_RATE_PER_RTH (AED)

CQL Table (per building):
    chilled_water_energy_consumption_<buildingId>
"""

import os
import requests
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from fastapi import HTTPException
from cassandra.cluster import Session
from dotenv import load_dotenv

from src.bms_ai.logger_config import setup_logger

load_dotenv()

log = setup_logger(__name__)

# ---------------------------- constants ----------------------------
IKON_DATA_URL: str = f"{os.getenv('IKON_BASE_URL_PROD')}/bms-express-server/data"
SPECIFIC_HEAT_WATER = 4.186          # kJ/(kg.degC)  ~  kJ/(L.degC) for water
KWH_PER_RTH = 3.517                  # 1 RTh = 3.517 kWh
CONSUMPTION_RATE_PER_RTH = 0.568     # AED per RTh  (change this value as needed)
KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "user_keyspace")
DEFAULT_BUILDING_ID = "36c27828d0b44f1e8a94d962d342e7c2" 

# Required datapoints
REQUIRED_DATAPOINTS = ["ChwS", "ChwR", "Flow"]
SYSTEM_TYPES = ["CHW", "FLOWMTR"]


def get_table_name(building_id: str) -> str:
    """Return sanitised table name for a building."""
    cleaned = building_id.replace("-", "").lower()
    return f"chilled_water_energy_consumption_{cleaned}"


CREATE_TABLE_CQL = """
CREATE TABLE IF NOT EXISTS {keyspace}.{table_name} (
    recorded_at             timestamp,
    avg_chws                double,
    avg_chwr                double,
    avg_flow                double,
    delta_t                 double,
    kwh_value               double,
    rth_value               double,
    consumption_rate_per_rth double,
    estimated_cost_aed      double,
    created_at              timestamp,
    PRIMARY KEY (recorded_at)
);
"""


def _build_fetch_query(
    building_id: str,
    from_dt: datetime,
    to_dt: Optional[datetime] = None,
) -> str:
    cleaned = building_id.replace("-", "").lower()
    table = f"datapoint_live_monitoring_{cleaned}"
    from_str = from_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + " UTC"

    if to_dt is None:
        to_dt = datetime.now(timezone.utc)
    to_str = to_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + " UTC"

    dp_in = ", ".join([f"'{dp}'" for dp in REQUIRED_DATAPOINTS])
    sys_in = ", ".join([f"'{s}'" for s in SYSTEM_TYPES])

    query = (
        f"select * from {table} where "
        f"system_type in ({sys_in}) and "
        f"datapoint in ({dp_in}) and "
        f"data_received_on >= '{from_str}' and "
        f"data_received_on <= '{to_str}' "
        f"allow filtering;"
    )
    return query


def fetch_chw_data(
    building_id: str,
    from_dt: datetime,
    to_dt: Optional[datetime] = None,
    url: str = IKON_DATA_URL,
) -> List[Dict[str, Any]]:
    
    query = _build_fetch_query(building_id, from_dt, to_dt)
    log.info(f"[EnergyUtil] Fetching CHW data -> query length={len(query)}")
    log.debug(f"[EnergyUtil] Query: {query}")

    payload = {"query": query}
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()

        if isinstance(raw, list):
            data_list = raw
        elif isinstance(raw, dict):
            data_list = raw.get("queryResponse", [])
        else:
            data_list = []

        if not isinstance(data_list, list):
            raise ValueError("Unexpected API response format.")

        log.info(f"[EnergyUtil] Fetched {len(data_list)} records.")
        return data_list

    except requests.exceptions.RequestException as e:
        log.error(f"[EnergyUtil] HTTP error during CHW fetch: {e}")
        raise HTTPException(status_code=502, detail=f"External API error: {e}")
    except Exception as e:
        log.error(f"[EnergyUtil] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def calculate_hourly_energy(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given raw CHW/FLOWMTR records for the last hour, compute a single energy
    record whose ``recorded_at`` is the current time (when the function is called).

    Returns a list with one dict ready for Cassandra INSERT:
        [{recorded_at, avg_chws, avg_chwr, avg_flow, delta_t, kwh_value,
          rth_value, consumption_rate_per_rth, estimated_cost_aed, created_at}]
    """
    if not raw_data:
        log.warning("[EnergyUtil] No raw data provided for energy calculation.")
        return []

    df = pd.DataFrame(raw_data)

    df["data_received_on"] = pd.to_datetime(df["data_received_on"], utc=True, errors="coerce")
    df["monitoring_data"] = pd.to_numeric(df["monitoring_data"], errors="coerce")
    df.dropna(subset=["data_received_on", "monitoring_data"], inplace=True)

    if df.empty:
        log.warning("[EnergyUtil] DataFrame empty after cleaning.")
        return []

    chws_vals = df.loc[df["datapoint"] == "ChwS", "monitoring_data"]
    chwr_vals = df.loc[df["datapoint"] == "ChwR", "monitoring_data"]
    flow_vals = df.loc[df["datapoint"] == "Flow", "monitoring_data"]

    if chws_vals.empty or chwr_vals.empty or flow_vals.empty:
        log.warning(
            f"[EnergyUtil] Missing datapoint(s) - "
            f"ChwS={len(chws_vals)}, ChwR={len(chwr_vals)}, Flow={len(flow_vals)}"
        )
        return []

    now = datetime.now(timezone.utc)

    avg_chws = float(chws_vals.mean())
    avg_chwr = float(chwr_vals.mean())
    avg_flow = float(flow_vals.mean())
    delta_t = avg_chwr - avg_chws

    kwh = delta_t * avg_flow * SPECIFIC_HEAT_WATER

    rth = kwh / KWH_PER_RTH
    cost = rth * CONSUMPTION_RATE_PER_RTH

    record = {
        "recorded_at": now,                                 
        "avg_chws": round(avg_chws, 4),
        "avg_chwr": round(avg_chwr, 4),
        "avg_flow": round(avg_flow, 4),
        "delta_t": round(delta_t, 4),
        "kwh_value": round(kwh, 4),
        "rth_value": round(rth, 4),
        "consumption_rate_per_rth": CONSUMPTION_RATE_PER_RTH,
        "estimated_cost_aed": round(cost, 4),
        "created_at": now,
    }

    log.info(f"[EnergyUtil] Calculated energy: kWh={record['kwh_value']}, RTh={record['rth_value']}, cost={record['estimated_cost_aed']} AED")
    return [record]


def ensure_table(session: Session, building_id: str) -> str:
    """Create the energy consumption table if it doesn't exist. Returns table name."""
    table_name = get_table_name(building_id)
    cql = CREATE_TABLE_CQL.format(keyspace=KEYSPACE, table_name=table_name)
    try:
        session.execute(cql)
        log.info(f"[EnergyUtil] Ensured table {table_name} exists.")
    except Exception as e:
        log.error(f"[EnergyUtil] Failed to create table {table_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Table creation error: {e}")
    return table_name


def store_energy_records(
    session: Session,
    building_id: str,
    records: List[Dict[str, Any]],
) -> int:
    """Insert energy records into Cassandra. Returns count of inserted rows."""
    if not records:
        return 0

    table_name = ensure_table(session, building_id)

    insert_cql = f"""
    INSERT INTO {KEYSPACE}.{table_name} (
        recorded_at, avg_chws, avg_chwr, avg_flow,
        delta_t, kwh_value, rth_value, consumption_rate_per_rth,
        estimated_cost_aed, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    try:
        prepared = session.prepare(insert_cql)
    except Exception as e:
        log.error(f"[EnergyUtil] Prepare failed: {e}")
        raise HTTPException(status_code=500, detail=f"CQL prepare error: {e}")

    inserted = 0
    for rec in records:
        try:
            session.execute(prepared, (
                rec["recorded_at"],
                rec["avg_chws"],
                rec["avg_chwr"],
                rec["avg_flow"],
                rec["delta_t"],
                rec["kwh_value"],
                rec["rth_value"],
                rec["consumption_rate_per_rth"],
                rec["estimated_cost_aed"],
                rec["created_at"],
            ))
            inserted += 1
        except Exception as e:
            log.error(f"[EnergyUtil] Insert failed for {rec['recorded_at']}: {e}")

    log.info(f"[EnergyUtil] Inserted {inserted}/{len(records)} records into {table_name}.")
    return inserted


def fetch_energy_data(
    session: Session,
    building_id: str,
    from_dt: datetime,
    to_dt: datetime,
) -> List[Dict[str, Any]]:
    table_name = ensure_table(session, building_id)

    from_str = from_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
    to_str = to_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"

    query = (
        f"SELECT * FROM {KEYSPACE}.{table_name} "
        f"WHERE recorded_at >= '{from_str}' "
        f"AND recorded_at <= '{to_str}' "
        f"ALLOW FILTERING;"
    )
    log.debug(f"[EnergyUtil] Fetch query: {query}")

    try:
        rows = session.execute(query)
        results = []
        for row in rows:
            results.append({
                "recorded_at": row.recorded_at.isoformat() if row.recorded_at else None,
                "avg_chws": row.avg_chws,
                "avg_chwr": row.avg_chwr,
                "avg_flow": row.avg_flow,
                "delta_t": row.delta_t,
                "kwh_value": row.kwh_value,
                "rth_value": row.rth_value,
                "consumption_rate_per_rth": row.consumption_rate_per_rth,
                "estimated_cost_aed": row.estimated_cost_aed,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            })
        log.info(f"[EnergyUtil] Fetched {len(results)} stored records.")
        return results
    except Exception as e:
        log.error(f"[EnergyUtil] Fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {e}")


def aggregate_energy_data(
    records: List[Dict[str, Any]],
    freq: str = "H",
) -> List[Dict[str, Any]]:
    """
    Re-aggregate energy records into coarser buckets.

    freq:
        H  -> hourly   (passthrough / group by hour)
        D  -> daily
        W  -> weekly
        M  -> monthly
        Q  -> quarterly
    """
    if not records:
        return []

    df = pd.DataFrame(records)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    df.sort_values("recorded_at", inplace=True)

    freq_map = {"H": "h", "D": "D", "W": "W-MON", "M": "MS", "Q": "QS"}
    pd_freq = freq_map.get(freq.upper(), "h")

    df.set_index("recorded_at", inplace=True)

    agg = df.resample(pd_freq).agg(
        avg_chws=("avg_chws", "mean"),
        avg_chwr=("avg_chwr", "mean"),
        avg_flow=("avg_flow", "mean"),
        delta_t=("delta_t", "mean"),
        kwh_value=("kwh_value", "sum"),
        rth_value=("rth_value", "sum"),
        estimated_cost_aed=("estimated_cost_aed", "sum"),
        consumption_rate_per_rth=("consumption_rate_per_rth", "first"),
    ).dropna(how="all")

    agg = agg.round(4)
    agg.reset_index(inplace=True)
    agg.rename(columns={"recorded_at": "period_start"}, inplace=True)
    agg["period_start"] = agg["period_start"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    return agg.to_dict(orient="records")


def process_and_store_hourly_energy(
    session: Session,
    building_id: str,
    from_dt: Optional[datetime] = None,
    to_dt: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    End-to-end: fetch raw CHW data for the last 1 hour (or custom window),
    calculate energy, and persist to Cassandra.
    """
    if from_dt is None:
        from_dt = datetime.now(timezone.utc) - timedelta(hours=1)
    if to_dt is None:
        to_dt = datetime.now(timezone.utc)

    raw_data = fetch_chw_data(building_id, from_dt, to_dt)

    if not raw_data:
        return {
            "status": "no_data",
            "message": "No CHW/Flow data found for the requested time window.",
            "records_inserted": 0,
        }

    energy_records = calculate_hourly_energy(raw_data)

    if not energy_records:
        return {
            "status": "insufficient_data",
            "message": "Data found but one or more required datapoints (ChwS, ChwR, Flow) were missing.",
            "records_inserted": 0,
        }

    inserted = store_energy_records(session, building_id, energy_records)

    return {
        "status": "success",
        "message": f"Processed and inserted {inserted} record(s).",
        "records_inserted": inserted,
        "energy_records": [
            {
                "recorded_at": r["recorded_at"].isoformat() if isinstance(r["recorded_at"], datetime) else str(r["recorded_at"]),
                "avg_chws": r["avg_chws"],
                "avg_chwr": r["avg_chwr"],
                "avg_flow": r["avg_flow"],
                "delta_t": r["delta_t"],
                "kwh_value": r["kwh_value"],
                "rth_value": r["rth_value"],
                "consumption_rate_per_rth": r["consumption_rate_per_rth"],
                "estimated_cost_aed": r["estimated_cost_aed"],
            }
            for r in energy_records
        ],
    }