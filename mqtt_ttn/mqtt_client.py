import paho.mqtt.client as mqtt
import json
import threading
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()


from src.bms_ai.logger_config import setup_logger

MQTT_HOST = os.getenv('MQTT_HOST')
MQTT_PORT = int(os.getenv('MQTT_PORT'))
MQTT_USER = os.getenv('MQTT_USER')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
MQTT_TOPIC = os.getenv('MQTT_TOPIC')

log = setup_logger(__name__)

# Shared state
latest_data = {}
data_lock = threading.Lock()

# Initialize MQTT client
mqtt_client = mqtt.Client()


# ---------- MQTT Callbacks ----------

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info(f"Connected to MQTT broker at {MQTT_HOST}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
        log.info(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        log.info(f"Failed to connect, return code {rc}")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        log.info(f"Unexpected disconnection. Return code: {rc}")
    else:
        log.info("Disconnected from MQTT broker")


def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8")
        print(f"\nNew message on {msg.topic}: {payload}")

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {"raw_payload": payload}

        global latest_data
        with data_lock:
            latest_data.clear()
            latest_data.update({
                "timestamp": datetime.now().isoformat(),
                "topic": msg.topic,
                "qos": msg.qos,
                "data": data,
            })
        log.info(f"Message stored in latest_data")

    except Exception as e:
        log.info(f"Error processing message: {e}")


def on_log(client, userdata, level, buf):
    if level == mqtt.MQTT_LOG_ERR:
        log.info(f"MQTT Error: {buf}")


# ---------- MQTT Setup ----------

mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message = on_message
mqtt_client.on_log = on_log


def start_mqtt():
    try:
        log.info("Connecting to MQTT broker...")
        mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        mqtt_client.loop_forever()
    except Exception as e:
        log.info(f"MQTT connection error: {e}")


def stop_mqtt():
    mqtt_client.disconnect()


def is_connected():
    return mqtt_client.is_connected()
