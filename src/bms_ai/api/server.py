
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.bms_ai.pipelines.prescriptive_pipeline import PrescriptivePipeline
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.api.routers import optimize,predict,another_optimize

log = setup_logger(__name__)

app = FastAPI(title="BMS AI",
              description="APIs for BMS system AI",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    try:
        log.info("Initializing PrescriptivePipeline at startup")
        app.state.prescriptive_pipeline = PrescriptivePipeline()
        log.info("PrescriptivePipeline initialized successfully")
    except Exception as e:
        log.critical(f"Failed to initialize PrescriptivePipeline: {e}")

app.include_router(optimize.router)
app.include_router(predict.router)
app.include_router(another_optimize.router)




@app.get("/")
def root():
    '''Basic health check endpoint.'''
    return {"status": "ok", "pipeline_loaded": hasattr(app.state, 'pipeline') and app.state.pipeline is not None}
