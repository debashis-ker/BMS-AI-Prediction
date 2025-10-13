from fastapi import Request
from src.bms_ai.pipelines.prescriptive_pipeline import PrescriptivePipeline
from src.bms_ai.logger_config import setup_logger

log = setup_logger(__name__)


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
