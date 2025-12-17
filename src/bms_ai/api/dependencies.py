from fastapi import Request
from src.bms_ai.pipelines.prescriptive_pipeline import PrescriptivePipeline
from src.bms_ai.logger_config import setup_logger
from cassandra.cluster import Cluster, Session
from typing import Optional
import time
import threading
import os
from dotenv import load_dotenv
from cassandra.auth import PlainTextAuthProvider

# Load environment variables
load_dotenv()

log = setup_logger(__name__)

CASS_SESSION_LOCK = threading.Lock()
CASS_SESSION_DATA = {
    'cluster': None,
    'session': None,
    'last_used': None,
    'ttl_seconds': 3600,  # 1 hour
    'cleanup_timer': None
}


def _cleanup_cassandra_session():
    """Internal function to cleanup idle Cassandra session after TTL expires."""
    with CASS_SESSION_LOCK:
        if CASS_SESSION_DATA['session'] is not None:
            log.info("[Cassandra] TTL expired. Cleaning up idle session.")
            try:
                if CASS_SESSION_DATA['cluster'] is not None:
                    CASS_SESSION_DATA['cluster'].shutdown()
            except Exception as e:
                log.error(f"[Cassandra] Error during cleanup: {e}")
            finally:
                CASS_SESSION_DATA['cluster'] = None
                CASS_SESSION_DATA['session'] = None
                CASS_SESSION_DATA['last_used'] = None
                CASS_SESSION_DATA['cleanup_timer'] = None


def _reset_cleanup_timer():
    """Reset the cleanup timer to TTL seconds from now."""
    if CASS_SESSION_DATA['cleanup_timer'] is not None:
        CASS_SESSION_DATA['cleanup_timer'].cancel()
    
    timer = threading.Timer(CASS_SESSION_DATA['ttl_seconds'], _cleanup_cassandra_session)
    timer.daemon = True
    timer.start()
    CASS_SESSION_DATA['cleanup_timer'] = timer


def get_cassandra_session() -> Session:
    """FastAPI dependency to get or create a shared Cassandra session with 1-hour TTL.
    
    - Creates a new session if none exists
    - Reuses existing session if it exists
    - Automatically cleans up after 1 hour of inactivity
    - Thread-safe implementation
    
    Connection parameters are loaded from .env file
    """
    auth_provider = None
    CASSANDRA_HOST = "localhost"
    CASSANDRA_PORT = 9042
    KEYSPACE_NAME = 'user_keyspace'
    # Load connection parameters from environment variables
    if os.getenv("ENVIRONMENT") == "production":
        auth_provider = PlainTextAuthProvider(
            username=os.getenv('CASSANDRA_USERNAME', 'CASSANDRA_USERNAME'),
            password=os.getenv('CASSANDRA_PASSWORD', 'CASSANDRA_PASSWORD')
        )
        CASSANDRA_HOST = os.getenv('CASSANDRA_HOST', '192.168.2.32').split(',')
        CASSANDRA_PORT = int(os.getenv('CASSANDRA_PORT', '9042'))
        KEYSPACE_NAME = os.getenv('CASSANDRA_KEYSPACE', 'user_keyspace')
    
    with CASS_SESSION_LOCK:
        current_time = time.time()
        
        if CASS_SESSION_DATA['session'] is not None:
            CASS_SESSION_DATA['last_used'] = current_time
            _reset_cleanup_timer()
            log.debug("[Cassandra] Reusing existing session")
            return CASS_SESSION_DATA['session']
        
        log.info(f"[Cassandra] Creating new session to {CASSANDRA_HOST}:{CASSANDRA_PORT}")
        try:
            cluster = Cluster(CASSANDRA_HOST, port=CASSANDRA_PORT)
            if os.getenv("ENVIRONMENT") == "production" and auth_provider is not None:
                cluster.auth_provider = auth_provider
            session = cluster.connect(KEYSPACE_NAME)
            
            CASS_SESSION_DATA['cluster'] = cluster
            CASS_SESSION_DATA['session'] = session
            CASS_SESSION_DATA['last_used'] = current_time
            
            _reset_cleanup_timer()
            
            log.info("[Cassandra] Session created successfully with 1-hour TTL")
            return session
            
        except Exception as e:
            log.error(f"[Cassandra] Failed to create session: {e}")
            raise


def get_prescriptive_pipeline(request: Request) -> PrescriptivePipeline:
    """FastAPI dependency to fetch or lazily create the shared PrescriptivePipeline.

    - Reads request.app.state.pipeline set at startup.
    - If missing, logs and creates a new PrescriptivePipeline, stores it, and returns it.
    """
    prescriptive_pipeline = getattr(request.app.state, 'prescriptive_pipeline', None)
    if prescriptive_pipeline is None:
        log.error("Pipeline not initialized in app state. Creating new PrescriptivePipeline.")
        prescriptive_pipeline = PrescriptivePipeline()
        request.app.state.prescriptive_pipeline = prescriptive_pipeline
    return prescriptive_pipeline
