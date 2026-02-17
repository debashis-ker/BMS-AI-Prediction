import time
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.components.movie_chatbot_functions import cinema_query

log = setup_logger(__name__)
router = APIRouter(prefix="/movie_chat", tags=["Movie Chatbot"])

class CinemaChatRequest(BaseModel):
    purpose: str = Field(..., description="Query")
    equipment_id: str = Field(default="Ahu13", description="Equipment ID")
    ticket: str = Field(default="", description="Ticket")
    ticket_type: Optional[str] = Field(None, description="Ticket Type")

@router.post("/ask")   
def ask (request: CinemaChatRequest):
    current_time = time.time()
    result = cinema_query(purpose=request.purpose, ticket=request.ticket, ticket_type=request.ticket_type, equipment_id=request.equipment_id)
    end_time = time.time()
    log.info(f"Query completed in {end_time - current_time:.2f} seconds.")
    return result