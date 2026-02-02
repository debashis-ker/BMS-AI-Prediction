
from datetime import datetime
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# from bms_ai.api.routers import production_endpoints
from src.bms_ai.pipelines.prescriptive_pipeline import PrescriptivePipeline
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.api.routers import optimize,predict,another_optimize,utils,heatlh_check,aggregation,chatbot,production_endpoints,chatbot_ollama,production_endpoints,fetch_datapoints_using_haystack,lstm_predictions
import os
from src.bms_ai.api.routers import mqtt
from dotenv import load_dotenv


# from src.bms_ai.components.mqtt_ttn.mqtt_client import start_mqtt, stop_mqtt, is_connected, data_lock, latest_data 


load_dotenv()

log = setup_logger(__name__)

app = FastAPI(title="BMS AI",
              description="APIs for BMS system AI",
              version="1.0.0",
              lifespan=mqtt.lifespan
              )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.on_event("startup")
# def startup_event():
#     try:
#         log.info("Initializing PrescriptivePipeline at startup")
#         app.state.prescriptive_pipeline = PrescriptivePipeline()
#         log.info("PrescriptivePipeline initialized successfully")
#     except Exception as e:
#         log.critical(f"Failed to initialize PrescriptivePipeline: {e}")
#
app.include_router(optimize.router)
app.include_router(predict.router)
app.include_router(another_optimize.router)
app.include_router(heatlh_check.router)
app.include_router(aggregation.router)
app.include_router(production_endpoints.router)
app.include_router(utils.router)
app.include_router(chatbot.router)  
app.include_router(chatbot_ollama.router) 
app.include_router(fetch_datapoints_using_haystack.router)
app.include_router(lstm_predictions.router)
app.include_router(mqtt.router)

@app.get("/")
def root():
    '''Basic health check endpoint.'''
    return {"status": "ok", "pipeline_loaded": hasattr(app.state, 'pipeline') and app.state.pipeline is not None}


#------------------------mqtt-ttn application code starts here--------------------------------


# mqtt_thread = None


# ---------- App Lifecycle ----------

# @app.on_event("startup")
# async def startup_event():
#     global mqtt_thread
#     mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
#     mqtt_thread.start()
#     print("FastAPI startup - MQTT client started")


# @app.on_event("shutdown")
# async def shutdown_event():
#     stop_mqtt()
#     print("FastAPI shutdown - MQTT client stopped")


# ---------- API Endpoints ----------

# @app.get("/")
# async def root():
#     return {
#         "message": "TTN MQTT Data API",
#         "endpoints": {
#             "latest_data": "/data",
#             "health": "/api/health",
#         },
#     }


# @app.get("/data")
# async def get_latest_data():
#     with data_lock:
#         if latest_data:
#             return {"status": "success", "data": latest_data}
#         return {
#             "status": "no_data",
#             "message": "No data received yet",
#             "data": None,
#         }


# @app.get("/health")
# async def health_check():
#     connected = is_connected()
#     return {
#         "status": "healthy" if connected else "disconnected",
#         "mqtt_connected": connected,
#         "timestamp": datetime.now().isoformat(),
#     }

#mqtt-ttn application code ends here

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 2))
    
    import sys
    is_production = "--prod" in sys.argv or os.getenv("ENVIRONMENT", "development") == "production"
    
    if is_production:
        uvicorn.run("src.bms_ai.api.server:app", host="0.0.0.0", port=port, workers=workers)
    else:
        uvicorn.run("src.bms_ai.api.server:app", host="0.0.0.0", port=port, reload=True)
