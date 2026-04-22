import requests
import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler

API_BASE_URL = "http://127.0.0.1:8000/mpc/optimize"
SETPOINT_API_URL = "http://127.0.0.1:8000/setpoint_optimization_diff/save_setpoint_optimization_diff"

EQUIPMENT_LIST = [
    {"id": "Ahu13", "screen": "Screen 13"},
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def trigger_optimization():
    logger.info("Starting scheduled optimization cycle...")
    for ahu in EQUIPMENT_LIST:
        payload = {
            "ticket": "72d69b3d-a67a-4a6b-bb18-abf5a909c51f",
            "ticket_type": "jobUser",
            "account_id": "7a66effc-2da2-44c2-84c6-23061ae62671",
            "software_id": "6e1c5d34-3711-4614-8f19-39a730463dc8",
            "building_id": "36c27828-d0b4-4f1e-8a94-d962d342e7c2",
            "system_type": "AHU",
            "equipment_id": ahu["id"],
            "screen_id": ahu["screen"],
        }
        try:
            response = requests.post(API_BASE_URL, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"SUCCESS | {ahu['id']} | New Setpoint: {data.get('optimized_setpoint')}")
            else:
                logger.error(f"FAILED | {ahu['id']} | Status: {response.status_code}")
        except Exception as e:
            logger.error(f"ERROR | {ahu['id']} | Could not reach API: {str(e)}")

def check_average_unusual_activity():
    logger.info("Starting setpoint unusual average difference cycle...")
    
    for ahu in EQUIPMENT_LIST:
        payload = {
            "equipment_id": ahu["id"],
            "building_id": "36c27828-d0b4-4f1e-8a94-d962d342e7c2"
        }
        try:
            response = requests.post(SETPOINT_API_URL, json=payload, timeout=60)
            if response.status_code == 200:
                res_data = response.json()
                logger.info(f"Unusual Setpoint Check: {res_data.get('message', 'Processed')}")
            else:
                logger.warning(f"Check failed for {ahu['id']} with status {response.status_code}")
        except Exception as e:
            logger.error(f"ERROR | {ahu['id']} | Unusual Activity API error: {str(e)}")

def store_anamolies_data():
    logger.info("Starting anomalies data storage cycle...")
    
    for ahu in EQUIPMENT_LIST:
        payload = {
            "building_id": "36c27828-d0b4-4f1e-8a94-d962d342e7c2",
            "search_tag_groups": [
                ["point", "ahu", "airHandlingEquip", "sensor", "status", "fan"],
                ["point", "ahu", "airHandlingEquip", "sensor", "temp", "supply", "air"],
                ["point","ahu","airHandlingEquip","sensor","fan","vfd","supply"],
                ["point","ahu","airHandlingEquip","sensor","humidity","return","air"],
                ["point","ahu","airHandlingEquip","sensor","humidity","supply","air"],
                ["point", "humidity", "sensor", "space", "air"],
                ["point", "sp", "temp"],
                ["point", "sensor", "humidity", "space", "air", "average"],
                ["point", "sensor", "humidity", "supply", "air"],
                ["point", "sensor", "temp", "supply", "air"],
                ["point", "sp", "humidity"],
                ["point", "sensor", "damper", "outside", "air"],
                ["point", "humidity", "space", "air", "sensor"],
                ["point", "sp", "co2", "co"],
                ["point", "co2", "co", "sensor", "average", "space", "air"],
                ["point", "sensor", "speed", "vfd"],
                ["point", "fan", "alarm", "fault", "vfd"],
                ["point", "sensor", "humidity", "return", "air"],
                ["point", "sensor", "humidity", "supply", "air", "ahu", "airHandlingEquip"],
                ["point", "ahu", "airHandlingEquip", "sensor", "fan", "vfd", "supply"],
                ["point", "ahu", "airHandlingEquip"],
                ["point", "co2", "co", "space", "air", "sensor"],
                ["point", "co2", "co", "sensor", "space", "air"],
                ["point", "sensor", "humidity"],
                ["point", "fan", "sensor", "flow", "status"],
                ["point", "sensor", "temp", "chilled", "water"],
                ["point", "sensor", "vfd"],
                ["point", "cmd", "co", "valve", "chilled", "water"],
                ["point", "ahu", "airHandlingEquip", "sensor", "temp", "co", "coil"],
                ["point", "sensor", "temp", "co"],
                ["point", "sensor", "vfd", "speed"],
                ["point", "co2", "co", "air", "sensor", "space"],
                ["point", "sp", "temp", "occupied"],
                ["point", "co2", "co"],
                ["point", "fan", "sensor", "status", "vfd"],
                ["point", "sensor", "temp", "supply", "air", "ahu", "airHandlingEquip"],
                ["point", "ahu", "airHandlingEquip", "sensor", "humidity", "return", "air"],
                ["point", "sp", "co2", "co", "alarm"],
                ["point", "co2", "co", "sensor", "space", "air", "average"],
                ["point", "alarm", "filter"],
                ["point", "cmd", "co", "vfd"]
            ],
            "equipment_id": ahu["id"],  # FIXED: Now uses the loop variable
            "ticket": "cafbc489-d8b7-425f-927f-0dc887852e5d",
            "software_id": "6e1c5d34-3711-4614-8f19-39a730463dc8",
            "account_id": "7a66effc-2da2-44c2-84c6-23061ae62671",
            "system_type": "AHU",
            "ticket_type": "jobUser"
        }
        
        try:
            response = requests.post("http://127.0.0.1:8000/prod/anomalies/store_anamolies", json=payload, timeout=60)
            if response.status_code == 200:
                res_data = response.json()
                logger.info(f"Anomalies Data Stored for {ahu['id']}: {res_data.get('message', 'Processed')}")
            else:
                logger.warning(f"Anomalies check failed for {ahu['id']} with status {response.status_code}")
        except Exception as e:
            logger.error(f"ERROR | {ahu['id']} | Anomalies API error: {str(e)}")

def alarm_check():
    API_URL = "http://127.0.0.1:8000/alarm/evaluate_alarm"
    logger.info("Starting alarm checking...")
    
    for ahu in EQUIPMENT_LIST:
        payload = {
            "building_id" : "36c27828-d0b4-4f1e-8a94-d962d342e7c2",
            "system_type" : "AHU",
            "equipment_id" : ahu["id"],
            "ticket" : "6d973fc7-ebe6-483c-aeaf-e09915751813",
            "ticket_type" : "jobUser",
            "heat_stress_alarm_config": {
                "warning_hits": 1,
                "critical_hits": 3
            },
            "freeze_alarm_config": {
                "warning_hits": 1,
                "critical_hits": 3
            },
            "oscillation_alarm_config": {
                "warning_hits": 1,
                "critical_hits": 3
            },
            "tracking_alarm_config": {
                "warning_hits": 1,
                "critical_hits": 3
            },
            "return_air_temp_alarm_config": {
                "warning_hits": 1,
                "critical_hits": 3
            }
        }
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            if response.status_code == 200:
                res_data = response.json()
                # FIXED: /evaluate_alarm returns 'records_stored', not 'alarm_count'
                logger.info(f"Alarm check for {ahu['id']} | Records Stored: {res_data.get('records_stored', 0)} | Status: {res_data.get('status')} | Msg: {res_data.get('message', 'N/A')}")
            else:
                logger.warning(f"Alarm check failed for {ahu['id']} with status {response.status_code}")
        except Exception as e:
            logger.error(f"ERROR | {ahu['id']} | Alarm API error: {str(e)}")

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    # Stagger jobs slightly to prevent simultaneous DB hits
    # scheduler.add_job(trigger_optimization, 'interval', minutes=4, next_run_time=datetime.now())
    # scheduler.add_job(check_average_unusual_activity, 'interval', minutes=4, next_run_time=datetime.now())
    # scheduler.add_job(store_anamolies_data, 'interval', minutes=4, next_run_time=datetime.now())
    scheduler.add_job(alarm_check, 'interval', minutes=30, next_run_time=datetime.now())
    
    logger.info("MPC Independent Scheduler Started [Interval: 10 Minutes]")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shut down.")