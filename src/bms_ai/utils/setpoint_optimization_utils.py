from typing import Dict, Any, List, Optional
import os
import json
import requests
import math
from dotenv import load_dotenv
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.ikon_apis import get_my_instances_v2
from datetime import datetime, timedelta, timezone

load_dotenv()
log = setup_logger(__name__)

SHARJAH_OFFSET = timezone(timedelta(hours=4))

def fetch_movie_schedule(ticket: str="5a3c56a1-d54e-47c0-a350-8f5fe7cdd243") -> Optional[Dict[str, Any]]:
    """
    Fetches movie schedule data from IKON service.
    Returns the schedule data.
    """
    
    try:
        instances = get_my_instances_v2(ticket=ticket,
                                        process_name="Schedule Archival",
                                        predefined_filters={"taskName" : "Utility Bill"},
                                        software_id='6e1c5d34-3711-4614-8f19-39a730463dc8',
                                        account_id="7a66effc-2da2-44c2-84c6-23061ae62671",
                                        env="prod"
                                        )
        
        if not instances:
            log.warning("No instances found for Schedule Archival with Utility Bill task.")
            return None
        
        print(f"Found {len(instances)} instances for Schedule Archival with Utility Bill task.")
        log.debug(f"Found {len(instances)} instances for Schedule Archival with Utility Bill task.")
        
        result_data = instances[0]['data'].get('result', {})
        
        return result_data

    except Exception as e:
        print(f"Error fetching instances: {e}")
        log.error(f"Error in fetch_movie_schedule: {e}")
        return None
    


def get_current_movie_occupancy_status(
    schedule_data: Dict[str, Any],
    screen: str = "Screen 13",
    for_which_time: int = 0  
) -> Dict[str, Any]:
    """
    Returns movie occupancy status for the specified screen in Sharjah time.
    If a movie is showing, returns status=1 with movie name and time remaining.
    If no movie is showing, returns status=0 with time until next movie.
    
    Args:
        schedule_data: The schedule data dict containing 'sessions' 
        screen: Screen name to check (e.g., "Screen 13")
        for_which_time: Minutes from now to check (default 0 = current time)
    
    Returns:
        Dict with status, and if showing: movie_name and time_remaining
        If not showing: time_until_next_movie
    """
    day_map = {
        0: "mon", 1: "tue", 2: "wed",
        3: "thu", 4: "fri", 5: "sat", 6: "sun"
    }

    now_sharjah = datetime.now(SHARJAH_OFFSET).replace(tzinfo=None)
    target_dt = now_sharjah + timedelta(minutes=for_which_time)

    def parse_time_str(t_str: str):
        """Parse time string handling single-digit hours and flexible formats"""
        t_str = t_str.strip().lower().replace(' ', '')
        
        if t_str.endswith('p'):
            t_str = t_str[:-1] + "PM"
        elif t_str.endswith('a'):
            t_str = t_str[:-1] + "AM"
        
        colon_pos = t_str.find(':')
        if colon_pos > 0:
            hour_part = t_str[:colon_pos]
            if hour_part == '0':
                t_str = '12' + t_str[colon_pos:]
            elif len(hour_part) == 1:  
                t_str = '0' + t_str
        
        return datetime.strptime(t_str, "%I:%M%p").time()

    if 'data' in schedule_data and 'sessions' in schedule_data.get('data', {}):
        sessions = schedule_data['data']['sessions']
    else:
        sessions = schedule_data.get('sessions', {})
    
    upcoming_shows = []
    
    for day_offset in (0, -1, 1):  
        schedule_date = target_dt.date() + timedelta(days=day_offset)
        day_key = day_map[schedule_date.weekday()]

        for session_key, session_data in sessions.items():
            if session_data.get("screen", "").lower() != screen.lower():
                continue

            day_sessions = session_data.get("sessions_by_day", {}).get(day_key, {})
            if not day_sessions:
                continue

            for time_range in day_sessions.values():
                try:
                    parts = time_range.split("-")
                    if len(parts) != 2:
                        log.warning(f"Invalid time range format: {time_range}")
                        continue
                    
                    start_str = parts[0].strip()
                    end_str = parts[1].strip()
                    
                    start_time = parse_time_str(start_str)
                    end_time = parse_time_str(end_str)

                    show_start = datetime.combine(schedule_date, start_time)
                    show_end = datetime.combine(schedule_date, end_time)

                    if show_end <= show_start:
                        show_end += timedelta(days=1)

                    if show_start <= target_dt < show_end:
                        time_remaining_seconds = (show_end - target_dt).total_seconds()
                        time_remaining_minutes = math.ceil(time_remaining_seconds / 60)
                        return {
                            "status": 1,
                            "movie_name": session_data.get("film_title", "Unknown"),
                            "time_remaining": f"{time_remaining_minutes} minutes"
                        }
                    
                    if show_start > target_dt:
                        upcoming_shows.append({
                            "start": show_start,
                            "movie_name": session_data.get("film_title", "Unknown")
                        })

                except Exception as e:
                    log.debug(f"Error parsing time range '{time_range}': {e}")
                    continue

    if upcoming_shows:
        upcoming_shows.sort(key=lambda x: x["start"])
        next_show = upcoming_shows[0]
        time_until_next_seconds = (next_show["start"] - target_dt).total_seconds()
        time_until_next_minutes = math.ceil(time_until_next_seconds / 60)
        
        return {
            "status": 0,
            "time_until_next_movie": f"{time_until_next_minutes} minutes",
            "next_movie_name": next_show["movie_name"]
        }
    
    return {
        "status": 0,
        "time_until_next_movie": "No upcoming shows"
    }

if __name__ == "__main__":
    print("Testing fetch_movie_schedule function...")
    schedule_data = fetch_movie_schedule()
    
    if schedule_data:
        print("\nTesting get_current_movie_occupancy_status for Screen 13...")
        status = get_current_movie_occupancy_status(schedule_data, screen="Screen 13")
        print(f"Current status: {json.dumps(status, indent=2)}")
        
        print("\nTesting get_current_movie_occupancy_status for Screen 16...")
        status_16 = get_current_movie_occupancy_status(schedule_data, screen="Screen 16")
        print(f"Current status: {json.dumps(status_16, indent=2)}")
    else:
        print("Failed to fetch schedule data")