import re
from datetime import datetime, timezone
from fastapi import HTTPException
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.setpoint_optimization_utils import (
    fetch_movie_schedule, 
    get_occupancy_status_for_timestamp, 
    get_current_movie_occupancy_status,
    parse_time_str, 
    SHARJAH_OFFSET
)

log = setup_logger(__name__)

def convert_equipment_id(eid: str) -> str:
    if eid.lower().startswith('ahu'):
        return f"Screen {eid[3:]}"
    return eid

def identify_purpose(purpose: str) -> str:
    p = purpose.lower()
    mapping = {
        "total_duration": ["duration", "how long", "runtime", "hours and minutes"],
        "first_show": ["first", "earliest", "opening", "start of day"],
        "last_show": ["last show", "final show", "latest show", "end of day"],
        "movie_at_time": ["current", "now", "particular time", "show at", "playing at", "status"],
        "movie_schedule": ["schedule", "timings", "timing", "shows", "list", "search", "next", "after", "at"]
    }
    
    if any(x in p for x in ["duration", "how long", "runtime", "hours and minutes"]):
        return "total_duration"

    for key, keywords in mapping.items():
        if any(x in p for x in keywords): return key
    return "general_info"

def cinema_query(purpose, ticket, ticket_type, equipment_id):
    now_utc = datetime.now(timezone.utc)
    query_text = purpose 
    screen_name = convert_equipment_id(equipment_id)
    intent_key = identify_purpose(purpose) 
    
    try:
        schedule_data = fetch_movie_schedule(ticket=ticket, ticket_type=ticket_type)
        if not schedule_data: raise HTTPException(status_code=404, detail="Schedule not found")

        target_dt = now_utc.astimezone(SHARJAH_OFFSET).replace(tzinfo=None)
        target_date, day_key = target_dt.date(), target_dt.strftime('%a').lower()

        def get_time_occupancy():
            time_match = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)', query_text.lower())
            if time_match:
                try:
                    q_utc = datetime.combine(target_date, parse_time_str(time_match.group(1).strip())).replace(tzinfo=SHARJAH_OFFSET).astimezone(timezone.utc)
                    raw = get_occupancy_status_for_timestamp(schedule_data, screen_name, q_utc)
                except: raw = {}
            else:
                raw = get_current_movie_occupancy_status(schedule_data, screens=[screen_name]).get(screen_name, {})
            
            return {
                "occupancy_status": "Occupied" if raw.get("status") == 1 else "Unoccupied",
                "movie_name": raw.get(("movie_name"),"No Movie is playing right now"),
                "time_remaining_in_minutes": raw.get(("time_remaining"), "No Movie is playing right now")
            }

        def get_schedule_data():
            sessions = []
            for inst in schedule_data:
                try:
                    if not (datetime.strptime(inst.get('start_date',''), "%d/%m/%Y").date() <= target_date <= datetime.strptime(inst.get('end_date',''), "%d/%m/%Y").date()): continue
                except: continue
                for s_val in inst.get('sessions', {}).values():
                    if s_val.get('screen', '').lower() == screen_name.lower():
                        for tr in s_val.get('sessions_by_day', {}).get(day_key, {}).values():
                            sessions.append({"movie_name": s_val.get("film_title"), "range": tr, "raw_start": parse_time_str(tr.split('-')[0])})
            
            if not sessions: return {"message": "No shows found for today."}
            sessions.sort(key=lambda x: x['raw_start'])

            if "next" in query_text.lower() or "after" in query_text.lower():
                target_index = -1
                for i, s in enumerate(sessions):
                    clean_movie = s['movie_name'].split('(')[0].strip().lower()
                    if clean_movie in query_text.lower() or s['movie_name'].lower() in query_text.lower():
                        target_index = i
                
                if target_index != -1 and target_index + 1 < len(sessions):
                    next_s = sessions[target_index + 1]
                    return {
                        "next_show": next_s['movie_name'],
                        "time": next_s['range'],
                        "after_movie": sessions[target_index]['movie_name']
                    }
                elif target_index != -1:
                    return {"message": f"{sessions[target_index]['movie_name']} is the last show of the day."}
                
            time_match = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)', query_text.lower())
            if "at" in query_text.lower() and time_match:
                asked_time = parse_time_str(time_match.group(1).strip())
                matches = [s for s in sessions if s['raw_start'] == asked_time]
                
                if matches:
                    return {
                        "movies_at_requested_time": [m['movie_name'] for m in matches],
                        "showtime": time_match.group(1).strip(),
                        "screen": screen_name
                    }
                else:
                    return {"message": f"No movies start exactly at {time_match.group(1).strip()} on this screen."}

            movie_filtered = [
                s for s in sessions 
                if s['movie_name'].lower() in query_text.lower()
            ] or sessions

            if intent_key == "first_show":
                s = movie_filtered[0]
                return {
                    "first_show_time": s['range'].split('-')[0].strip(), 
                    "movie_name": s['movie_name'], 
                    "date": str(target_date)
                }
            
            if intent_key == "last_show":
                s = movie_filtered[-1]
                return {
                    "last_show_time": s['range'].split('-')[0].strip(), 
                    "movie_name": s['movie_name'], 
                    "date": str(target_date)
                }
            
            filtered = [s for s in sessions if s['movie_name'].lower() in query_text.lower()] or sessions
            
            if intent_key == "total_duration":
                total = sum([((datetime.combine(target_date, parse_time_str(s['range'].split('-')[1])) - datetime.combine(target_date, parse_time_str(s['range'].split('-')[0]))).total_seconds()/60) % 1440 for s in filtered])
                return {"total_daily_duration_minutes": int(total), "screen": screen_name}
            
            return {"screen": screen_name, "schedule_list": [s['range'] for s in filtered], "movie_names": list(set([s['movie_name'] for s in filtered]))}

        handlers = {
            "movie_at_time": get_time_occupancy,
            "first_show": get_schedule_data,
            "last_show": get_schedule_data,
            "total_duration": get_schedule_data,
            "movie_schedule": get_schedule_data
        }

        return {
            "success" : True, "equipment_id" : equipment_id,
            "timestamp_utc" : now_utc.isoformat(), "purpose": query_text,
            "data": handlers.get(intent_key, lambda: {})()
        }

    except Exception as e:
        log.error(f"Error: {e}")
        return {"success":False, "equipment_id":equipment_id, "timestamp_utc":now_utc.isoformat(), "purpose": query_text, "data":{"error": str(e)}}
    