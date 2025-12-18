import requests
import time
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



def save_data_to_cassandraV2(data_chunk : List[Dict], building_id: str, metadata: Dict, session: Session) -> int:
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
        best_setpoints text, 
        timestamp timestamp, 
        best_target_value text, 
        optimization_direction text, 
        selected_features_used text, 
        total_combinations_tested int, 
        site text,
        optimization_method text,
        optimization_time_seconds int,
        equipment_id text, 
        system_type text,
        PRIMARY KEY ((site, system_type, equipment_id), timestamp)) 
        WITH CLUSTERING ORDER BY (timestamp ASC) {ttl_option};
    """
    
    INSERT_BASE_CQL = f"""
    INSERT INTO {KEYSPACE_NAME}."{{table_name}}" 
    (best_setpoints, timestamp, best_target_value, optimization_direction, selected_features_used, total_combinations_tested, site, optimization_method, optimization_time_seconds, equipment_id, system_type)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            str(data_chunk[0].get('best_setpoints')),
            parser.parse(data_chunk[0].get('timestamp')),
            str(data_chunk[0].get('best_target_value')),
            str(data_chunk[0].get('optimization_direction')),
            str(data_chunk[0].get('selected_features_used')),
            int(data_chunk[0].get('total_combinations_tested')),
            site_value,
            str(data_chunk[0].get('optimization_method')),
            int(data_chunk[0].get('optimization_time_seconds')),
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
                'best_setpoints': row.best_setpoints,
                'timestamp': row.timestamp.isoformat() if row.timestamp else None,
                'best_target_value': row.best_target_value,
                'optimization_direction': row.optimization_direction,
                'selected_features_used': row.selected_features_used,
                'total_combinations_tested': row.total_combinations_tested,
                'optimization_method': row.optimization_method,
                'optimization_time_seconds': row.optimization_time_seconds,
                "site": row.site,
                'equipment_id': row.equipment_id,
                'system_type': row.system_type
            }
            result.append(record)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cassandra query error: {e}")
    """  finally:
            if cluster:
                cluster.shutdown() """




@app.post("/store_anamolies", response_model=IngestionResponse)
def store_anamolies():
    """
    Triggers the full batch process: fetches anomaly detection results from the 
    external API and inserts them into Cassandra's historical and anomaly tables.
    """
    start_time = time.time()
    total_assets_processed = 0
    total_records_inserted = 0
    
    try:
        api_iterator = fetch_data_in_chunks(API_URL, CHUNK_SIZE)
        
        for building_id, api_metadata, api_chunks in api_iterator:
            total_assets_processed += 1
            
            print(f"Processing Asset {total_assets_processed}: {api_metadata['site']}/{api_metadata['equipment_id']}")
            
            for i, chunk in enumerate(api_chunks):
                inserted_count = save_data_to_cassandra(chunk, building_id, api_metadata)
                total_records_inserted += inserted_count
                
        duration = time.time() - start_time
        
        return {
            "status": "SUCCESS",
            "total_assets_processed": total_assets_processed,
            "total_records_inserted": total_records_inserted,
            "duration_seconds": round(duration, 3),
            "message": "Batch ingestion completed successfully."
        }
        
    except requests.exceptions.RequestException as req_err:
        raise HTTPException(status_code=500, detail=f"External API fetch failed: {req_err}")
    except ValueError as val_err:
        raise HTTPException(status_code=500, detail=f"Data processing error: {val_err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during batch execution: {e}")