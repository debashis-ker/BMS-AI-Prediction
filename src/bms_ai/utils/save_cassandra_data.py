import requests
import time
import ast
import re
import json
from cassandra.cluster import Cluster,Session
from cassandra.auth import PlainTextAuthProvider
from dateutil import parser
from typing import Iterator, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.bms_ai.logger_config import setup_logger

log = setup_logger(__name__)

CASSANDRA_HOST = ['127.0.0.1']
CASSANDRA_PORT = 9042
#KEYSPACE_NAME = 'anamoly'
KEYSPACE_NAME = 'user_keyspace'
CHUNK_SIZE = 100

HISTORY_TABLE_SUFFIX = "historical_data"
ANAMOLY_TABLE_SUFFIX = "anamoly_data"
ANAMOLY_TTL_SECONDS = 30 * 24 * 60 * 60

API_URL = "http://127.0.0.1:8000/prod/anomaly_detection_all_ahu"

BUILDING_IDS = ["36c27828-d0b4-4f1e-8a94-d962d342e7c2"]

DEFAULT_METADATA = {
    "equipment_id": "",
    "system_type": "AHU",
    "site": ""
}

REQUEST_BODY = {}

app = FastAPI(title="Cassandra Batch Ingestion API")

class IngestionResponse(BaseModel):
    status: str = Field(...)
    total_assets_processed: int
    total_records_inserted: int
    duration_seconds: float
    message: str


def fetch_data_in_chunks(url: str, chunk_size: int) -> Iterator[Tuple[str, Dict[str, str], Iterator[List[Dict]]]]:
    print(f"Fetching data from batch endpoint {url}...")
    
    response = requests.post(url, json=REQUEST_BODY)
    response.raise_for_status()
    api_data: Dict[str, Any] = response.json()
    
    all_assets_results = api_data.get('data', {}).get('all_anomalies_by_asset', {})

    if not all_assets_results:
        raise ValueError("Error: 'all_anomalies_by_asset' key not found or empty in API response.")

    for asset_code, asset_details in all_assets_results.items():
        
        asset_metadata = {
            "site": asset_details.get('site', DEFAULT_METADATA['site']),
            "equipment_id": asset_details.get('equipment_id', DEFAULT_METADATA['equipment_id']),
            "system_type": asset_details.get('system_type', DEFAULT_METADATA['system_type']),
        }
        
        building_id = BUILDING_IDS[0] 
        
        historical_data_map = asset_details.get('data', {}).get('historical_data', {})
        
        if not historical_data_map:
            print(f"Warning: No historical data found for asset {asset_code}. Skipping.")
            continue
            
        all_readings = []
        for feature_name, readings in historical_data_map.items():
            for reading in readings:
                value = reading.get(feature_name)
                
                flat_record = {
                    'feature_name': feature_name, 
                    'timestamp': reading.get('timestamp'),
                    feature_name: value,
                    'Anamoly_Flag': reading.get('Anamoly_Flag')
                }
                all_readings.append(flat_record)
        
        chunk_iterator = (all_readings[i:i + chunk_size] 
                          for i in range(0, len(all_readings), chunk_size))
        
        yield building_id, asset_metadata, chunk_iterator



def save_data_to_cassandraV2(data_chunk : List[Dict], building_id: str, metadata: Dict, session: Session) -> int: #type:ignore
    """Saves data chunk to Cassandra. Returns the number of rows inserted."""
    cluster = None
    rows_inserted = 0
    Optimization_HISTORY_TABLE_SUFFIX = "historical_data_opt"
    Optimization_SETPOINT_TABLE_SUFFIX = "setpoint_data_opt"
    
    site_value = metadata["site"]
    equipment_id_value = metadata["equipment_id"]
    system_type_value = metadata["system_type"]

    history_table = f"{Optimization_HISTORY_TABLE_SUFFIX}_{building_id.replace('-', '').lower()}"
    setpoint_table = f"{Optimization_SETPOINT_TABLE_SUFFIX}_{building_id.replace('-', '').lower()}"
    
    CREATE_BASE_CQL = """
    CREATE TABLE IF NOT EXISTS {keyspace}."{table_name}" ( 
        timestamp timestamp, 
        actual_value text, 
        predicted_value text, 
        difference_actual_and_pred text, 
        current_setpoints text, 
        optimized_setpoints text, 
        site text,
        equipment_id text, 
        system_type text,
        PRIMARY KEY ((site, system_type, equipment_id), timestamp)) 
        WITH CLUSTERING ORDER BY (timestamp ASC) {ttl_option};
    """
    
    INSERT_BASE_CQL = f"""
    INSERT INTO {KEYSPACE_NAME}."{{table_name}}" 
    (timestamp, actual_value, predicted_value, difference_actual_and_pred, current_setpoints, optimized_setpoints, site, equipment_id, system_type)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        # username = 'admin'
        # password = 'admin'
        # auth_provider = PlainTextAuthProvider(username=username, password=password)
        # cluster = Cluster(CASSANDRA_HOST, port=CASSANDRA_PORT)
        # if auth_provider:
        #     cluster.auth_provider = auth_provider
        # session = cluster.connect(KEYSPACE_NAME)
        
        session.execute(CREATE_BASE_CQL.format(keyspace=KEYSPACE_NAME, table_name=history_table, ttl_option=""))
        session.execute(CREATE_BASE_CQL.format(keyspace=KEYSPACE_NAME, table_name=setpoint_table, ttl_option=f"AND default_time_to_live = {ANAMOLY_TTL_SECONDS}"))
        
        history_stmt = session.prepare(INSERT_BASE_CQL.format(table_name=history_table))
        anamoly_stmt = session.prepare(INSERT_BASE_CQL.format(table_name=setpoint_table))

        data_tuple = ()
        data_tuple = (
            parser.parse(data_chunk[0].get('timestamp')),#type:ignore
            str(data_chunk[0].get('actual_value')),
            str(data_chunk[0].get('predicted_value')),
            str(data_chunk[0].get('difference_actual_and_pred')),
            str(data_chunk[0].get('current_setpoints')),
            str(data_chunk[0].get('optimized_setpoints')),#type:ignore
            site_value,
            equipment_id_value,
            system_type_value
        )
        
        session.execute(history_stmt, data_tuple)
        session.execute(anamoly_stmt, data_tuple)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cassandra insert error: {e}")
    """  finally:
        if cluster:
            cluster.shutdown() """

def fetch_adjustment_hisoryData(building_id: str,site: str, system_type: str, equipment_id: str, start_date: str, end_date: str, limit: int, session: Session) -> List[Dict]:
    """
    Fetches adjustment history data for a given building and equipment from an external API.
    """
    ADJUSTMENT_API_URL = "https://ikoncloud.keross.com/bms-express-server/data"
    Optimization_HISTORY_TABLE_SUFFIX = "historical_data_opt"
    Optimization_SETPOINT_TABLE_SUFFIX = "setpoint_data_opt"

    Query_BASE_CQL = f"""
    Select * from {KEYSPACE_NAME}."{{table_name}}"  
    where site='{site}' and system_type='{system_type}' and equipment_id='{equipment_id}' and timestamp >= '{start_date}' and timestamp <= '{end_date}' 
    LIMIT {limit}
    ALLOW FILTERING;"""

    try:
        # username = 'admin'
        # password = 'admin'
        # auth_provider = PlainTextAuthProvider(username=username, password=password)
        # cluster = Cluster(CASSANDRA_HOST, port=CASSANDRA_PORT)
        # if auth_provider:
        #     cluster.auth_provider = auth_provider

        # session = cluster.connect(KEYSPACE_NAME)
        
        #history_table = f"{building_id.replace('-', '').lower()}_{HISTORY_TABLE_SUFFIX}"
        history_table = f"{Optimization_HISTORY_TABLE_SUFFIX}_{building_id.replace('-', '').lower()}"
        setpoint_table = f"{Optimization_SETPOINT_TABLE_SUFFIX}_{building_id.replace('-', '').lower()}"
        query_cql = Query_BASE_CQL.format(table_name=history_table)
        log.info(f"Executing Cassandra query: {query_cql}")
        print(query_cql)
        
        rows = session.execute(query_cql)
        log.info(f"Number of rows fetched: {rows.current_rows}")
        
        result = []
        for row in rows:
            record = {
                'timestamp': row.timestamp.isoformat() if row.timestamp else None,
                'actual_value': row.actual_value,
                'predicted_value': row.predicted_value,
                'difference_actual_and_pred': row.difference_actual_and_pred,
                'current_setpoints': row.current_setpoints,
                'optimized_setpoints': row.optimized_setpoints,
                "site": row.site,
                'equipment_id': row.equipment_id,
                'system_type': row.system_type
            }
            # Converting string representations of dictionaries back to actual dictionaries
            try:
                """ s1_fixed = record['optimized_setpoints'].replace("np.float64", "float")
                s2_fixed = record['current_setpoints'].replace("np.float64", "float")

                record['optimized_setpoints']= ast.literal_eval(s1_fixed)
                record["current_setpoints"] = ast.literal_eval(s2_fixed) """
                optimized_str = re.sub(r'np\.float64\(([\d.+-eE]+)\)', r'\1', record['optimized_setpoints'])
                current_str = re.sub(r'np\.float64\(([\d.+-eE]+)\)', r'\1', record['current_setpoints'])

                # Also handle any other numpy types
                optimized_str = re.sub(r'np\.\w+\(([\d.+-eE]+)\)', r'\1', optimized_str)
                current_str = re.sub(r'np\.\w+\(([\d.+-eE]+)\)', r'\1', current_str)

                # Replace single quotes with double quotes for JSON parsing
                optimized_str = optimized_str.replace("'", '"')
                current_str = current_str.replace("'", '"')

                record['optimized_setpoints'] = json.loads(optimized_str)
                record["current_setpoints"] = json.loads(current_str)
            except Exception as e:
                log.error(f"Error parsing setpoints: {e}")

            log.debug(f"Fetched record: {record}")
            result.append(record)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cassandra query error: {e}")
    """  finally:
            if cluster:
                cluster.shutdown() """