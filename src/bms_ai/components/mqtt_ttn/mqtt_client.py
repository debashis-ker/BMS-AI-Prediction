import paho.mqtt.client as mqtt
import json
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
import os
from dotenv import load_dotenv
from cassandra.cluster import Session
import pytz
from src.bms_ai.api.dependencies import data_lock, latest_data, mqtt_client
from src.bms_ai.api.dependencies import get_cassandra_session
from src.bms_ai.logger_config import setup_logger

load_dotenv()


MQTT_HOST = os.getenv('MQTT_HOST')
MQTT_PORT = int(os.getenv('MQTT_PORT'))
MQTT_USER = os.getenv('MQTT_USER')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
MQTT_TOPIC = os.getenv('MQTT_TOPIC')

log = setup_logger(__name__)

# Connection status tracking
mqtt_connection_status = {
    'is_connected': False,
    'last_connected': None,
    'lock': threading.Lock()
}

# # Shared state
# latest_data = {}
# data_lock = threading.Lock()

# # Initialize MQTT client
# mqtt_client = mqtt.Client()


# Cassandra session
# session = get_cassandra_session()

# ---------- MQTT Callbacks ----------

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info(f"Connected to MQTT broker at {MQTT_HOST}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
        log.info(f"Subscribed to topic: {MQTT_TOPIC}")
        # Update connection status
        with mqtt_connection_status['lock']:
            mqtt_connection_status['is_connected'] = True
            mqtt_connection_status['last_connected'] = datetime.now().isoformat()
            if mqtt_connection_status['is_connected']:
                try:
                    session = get_cassandra_session()
                    session.execute("""
                                      CREATE TABLE IF NOT EXISTS bms_live_monitoring_mqtt_36c27828d0b44f1e8a94d962d342e7c2 (
                                                sensor_id int,
                                                corporate_id int,
                                                created_at timestamp,
                                                sensor_type text,
                                                device_id text,
                                                room_name text,
                                                temperature float,
                                                total_in int,
                                                total_out int,
                                                people_count int,
                                                period_in int,
                                                period_out int,
                                                periodic_people_count int,
                                                battery int,
                                                rssi int,
                                                snr float,
                                                sf int,
                                              
                                                PRIMARY KEY (sensor_id,created_at)
                                            )
                                            WITH CLUSTERING ORDER BY (created_at DESC);


                                      """)
                    log.info("Table created/verified successfully")
                except Exception as e:
                    log.error(f"Error creating table: {e}", exc_info=True)
    else:
        log.info(f"Failed to connect, return code {rc}")
        with mqtt_connection_status['lock']:
            mqtt_connection_status['is_connected'] = False


def on_disconnect(client, userdata, rc):
    if rc != 0:
        log.info(f"Unexpected disconnection. Return code: {rc}")
    else:
        log.info("Disconnected from MQTT broker")
    
    # Update connection status
    with mqtt_connection_status['lock']:
        mqtt_connection_status['is_connected'] = False


def on_message(client, userdata, msg):
    try:
        log.info(f"Message received on topic: {msg.topic}")
        payload_str = msg.payload.decode("utf-8")
        
        try:
            data = json.loads(payload_str)
        except json.JSONDecodeError:
            log.warning("Payload is not JSON, skipping.")
            return

        
        # . FIX FOR DOUBLE INSERTION:
        # If your Primary Key includes event_timestamp, and you use datetime.now(),
        # a retry from MQTT will have a different millisecond, creating a duplicate.
        # Ideally, use the sensor's own 'created_at' for the timestamp if it's unique.
        
        session = get_cassandra_session()
        # 2. Define the insert query
        
       
        
        query = """
            INSERT INTO bms_live_monitoring_mqtt_36c27828d0b44f1e8a94d962d342e7c2 (
                sensor_id, corporate_id, created_at, sensor_type, device_id, room_name, 
                temperature, total_in, total_out, people_count, period_in, period_out, 
                periodic_people_count, battery, rssi, snr, sf
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        prepared = session.prepare(query)

        values = (
            int(data.get('sensor_id')) if data.get('sensor_id') is not None else None,
            int(data.get('corporate_id')) if data.get('corporate_id') is not None else None,
            # If created_at is a timestamp in Cassandra, convert the JSON value to a datetime object
            # If it's an int, keep it as int.
            data.get('created_at'), 
            str(data.get('sensor_type')) if data.get('sensor_type') else None,
            str(data.get('device_id')) if data.get('device_id') else None,
            str(data.get('room_name')) if data.get('room_name') else None,
            float(data.get('temperature')) if data.get('temperature') is not None else None,
            int(data.get('total_in')) if data.get('total_in') is not None else None,
            int(data.get('total_out')) if data.get('total_out') is not None else None,
            int(data.get('people_count')) if data.get('people_count') is not None else None,
            int(data.get('period_in')) if data.get('period_in') is not None else None,
            int(data.get('period_out')) if data.get('period_out') is not None else None,
            int(data.get('periodic_people_count')) if data.get('periodic_people_count') is not None else None,
            int(data.get('battery')) if data.get('battery') is not None else None,
            int(data.get('rssi')) if data.get('rssi') is not None else None,
            float(data.get('snr')) if data.get('snr') is not None else None,
            int(data.get('sf')) if data.get('sf') is not None else None,
         
        )

        session.execute(prepared, values)
        log.info(f"Successfully inserted sensor_id {data.get('sensor_id')}")
        # Print with localized formatting
        print(f"Data inserted at {data.get('created_at')} for sensor_id {data.get('sensor_id')}")

    except Exception as e:
        log.error(f"Error processing message: {e}", exc_info=True)

    except Exception as e:
        log.error(f"Error processing message: {e}", exc_info=True)
        print(f"Error inserting data: {e}")


# ---------- MQTT Setup ----------

mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message = on_message
# mqtt_client.on_log = on_log


def start_mqtt():
    try:
        log.info("Connecting to MQTT broker...")
        mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        mqtt_client.loop_forever()
    except Exception as e:
        log.info(f"MQTT connection error: {e}")
        with mqtt_connection_status['lock']:
            mqtt_connection_status['is_connected'] = False


def stop_mqtt():
    mqtt_client.disconnect()


def is_connected():
    """
    Returns the cached connection status instead of relying on client.is_connected()
    which can be flaky during rapid status checks.
    """
    with mqtt_connection_status['lock']:
        return mqtt_connection_status['is_connected']
