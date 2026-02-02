
from datetime import datetime, timedelta, timezone
import threading
from contextlib import asynccontextmanager
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
import os

import paho.mqtt.client as mqtt
import json
import threading
from datetime import datetime

from src.bms_ai.components.mqtt_ttn.mqtt_client import is_connected, start_mqtt, stop_mqtt
from src.bms_ai.components.mqtt_ttn.mqtt_client import data_lock, latest_data
from src.bms_ai.api.dependencies import get_cassandra_session
from src.bms_ai.logger_config import setup_logger

from src.bms_ai.logger_config import setup_logger


# Request body model
class MQTTDataRequest(BaseModel):
    buildingId: str
    hours: Optional[int] = None
    sensor_id: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None



log = setup_logger(__name__)


router = APIRouter(prefix="/mqtt", tags=["fetch mqtt data"])

mqtt_thread = None

#------------------------mqtt-ttn application code starts here--------------------------------

@asynccontextmanager
async def lifespan(app):
    # Startup
    global mqtt_thread
    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()
    print("FastAPI startup - MQTT client started")
    
    yield
    
    # Shutdown
    stop_mqtt()
    print("FastAPI shutdown - MQTT client stopped")


@router.get("/health")
def mqtt_health_check():
    """Endpoint to check MQTT connection health."""
    connected = is_connected()
    print(f"MQTT connection status: {'connected' if connected else 'disconnected'}")
    status = "connected" if connected else "disconnected"
    return {"mqtt_status": status, "timestamp": datetime.now().isoformat()}

@router.post("/data")
def mqtt_get_latest_data(request: MQTTDataRequest):
    """Endpoint to fetch MQTT data from Cassandra with flexible time filtering.
    
    Request body parameters:
    - buildingId (required): Table name building id
    - hours: Number of hours to look back (if provided, overrides default)
    - sensor_id: Optional sensor_id to filter by
    - start_time: Custom start time (ISO format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD)
    - end_time: Custom end time (ISO format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD)
    
    Priority:
    1. If start_time and end_time provided, use those
    2. If hours provided, use last N hours
    3. Default: UTC 00:00 today to current time
    
    Returns: List of sensor readings within the time range
    """
    try:
        session = get_cassandra_session()
        
        # Determine time range
        now = datetime.now(timezone.utc)
        
        if request.start_time and request.end_time:
            # Parse custom time range
            try:
                # Handle both ISO formats: YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS
                if 'T' in request.start_time:
                    time_start = datetime.fromisoformat(request.start_time)
                else:
                    time_start = datetime.fromisoformat(request.start_time + "T00:00:00")
                
                if 'T' in request.end_time:
                    time_end = datetime.fromisoformat(request.end_time)
                else:
                    time_end = datetime.fromisoformat(request.end_time + "T23:59:59")
                
                log.info(f"Using custom time range: {time_start} to {time_end}")
            except ValueError as ve:
                return {
                    "status": "error",
                    "message": f"Invalid time format. Use ISO format (YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD): {str(ve)}",
                    "data": None,
                }
        elif request.hours:
            # Use hours parameter
            time_start = now - timedelta(hours=request.hours)
            time_end = now
            log.info(f"Using hours-based range: last {request.hours} hours")
        else:
            # Default: UTC 00:00 today to current time
            time_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            time_end = now
            log.info(f"Using default range: UTC 00:00 to now ({time_start} to {time_end})")
        
        if not request.buildingId:
            return {
                "status": "error",
                "message": "buildingId is required",
                "data": None,
            }
        
        if request.sensor_id:
            # Query with sensor_id filter
            query = f"""
                SELECT * FROM bms_live_monitoring_mqtt_{request.buildingId}
                WHERE sensor_id = %s 
                AND event_timestamp >= %s
                AND event_timestamp <= %s
                ALLOW FILTERING
            """
            rows = list(session.execute(query, (request.sensor_id, time_start, time_end)))
            log.info(f"Fetched {len(rows)} records for sensor_id {request.sensor_id}")
        else:
            # Query all sensors
            query = f"""
                SELECT * FROM bms_live_monitoring_mqtt_{request.buildingId}
                WHERE event_timestamp >= %s
                AND event_timestamp <= %s
                ALLOW FILTERING
            """
            rows = list(session.execute(query, (time_start, time_end)))
            log.info(f"Fetched {len(rows)} records from all sensors")
        
        # Convert rows to list of dictionaries
        data_list = []
        for row in rows:
            data_list.append({
                'sensor_id': row.sensor_id,
                'corporate_id': row.corporate_id,
                'created_at': row.created_at,
                'sensor_type': row.sensor_type,
                'device_id': row.device_id,
                'room_name': row.room_name,
                'temperature': row.temperature,
                'total_in': row.total_in,
                'total_out': row.total_out,
                'people_count': row.people_count,
                'period_in': row.period_in,
                'period_out': row.period_out,
                'periodic_people_count': row.periodic_people_count,
                'battery': row.battery,
                'rssi': row.rssi,
                'snr': row.snr,
                'sf': row.sf,
                'event_timestamp': row.event_timestamp.isoformat() if row.event_timestamp else None,
            })
        
        return {
            "status": "success",
            "count": len(data_list),
            "time_range": {
                "start": time_start.isoformat(),
                "end": time_end.isoformat(),
            },
            "sensor_id_filter": request.sensor_id,
            "data": data_list,
        }
        
    except Exception as e:
        log.error(f"Error fetching data from Cassandra: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "data": None,
        }