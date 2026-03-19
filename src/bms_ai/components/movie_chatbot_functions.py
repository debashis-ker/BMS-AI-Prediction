from datetime import datetime, timezone, timedelta
from typing import Optional
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.setpoint_optimization_utils import (
    fetch_movie_schedule, 
    get_occupancy_status_for_timestamp, 
    get_current_movie_occupancy_status,
    parse_time_str, 
    get_screen_operating_window,
    SHARJAH_OFFSET
)

log = setup_logger(__name__)

def format_utc(dt_obj):
    return dt_obj.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"

def format_bms_time(dt_obj):
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")

def schedule_selector(schedule_data, current_date: Optional[datetime], instance_index: Optional[int] = None):
    selected_instance_index = None
    
    if instance_index is not None:
        if instance_index < len(schedule_data):
            schedule = schedule_data[instance_index]
            selected_instance_index = instance_index
            log.debug(f"Using provided instance_index={instance_index}")
        else:
            log.error(f"Invalid instance_index {instance_index} for schedule_data length {len(schedule_data)}")
            return {}
    else:
        for idx, sched in enumerate(schedule_data):
            end_date_str = sched.get('end_date', '')
            start_date_str = sched.get('start_date', '')
            
            if not end_date_str:
                log.debug(f"Instance {idx}: No end_date, skipping")
                continue
            
            try:
                end_date = datetime.strptime(end_date_str, "%d/%m/%Y").date()
                start_date = datetime.strptime(start_date_str, "%d/%m/%Y").date() if start_date_str else None
                
                if start_date and current_date >= start_date and current_date <= end_date: #type:ignore
                    schedule = sched
                    selected_instance_index = idx
                    log.info(f"Auto-selected instance {idx}: {sched.get('cinema_name', 'Unknown')} ({start_date_str} to {end_date_str})")
                    break
                elif not start_date and current_date <= end_date: #type:ignore
                    schedule = sched
                    selected_instance_index = idx
                    log.info(f"Auto-selected instance {idx}: {sched.get('cinema_name', 'Unknown')} (end: {end_date_str})")
                    break
                else:
                    log.debug(f"Instance {idx}: current_date {current_date} not in range {start_date_str} to {end_date_str}")
            except Exception as e:
                log.warning(f"Failed to parse dates for instance {idx}: {e}")
                continue
        
        if schedule is None:
            log.warning(f"No valid schedule instance found for current date {current_date}. Using latest schedule [0] as fallback.")
            schedule = schedule_data[0]
            selected_instance_index = 0
    
    log.debug(f"Using schedule instance {selected_instance_index} for date {current_date}")

    return schedule

def total_duration_movies(purpose, schedule_data, screen_name, target_date, day_key):
    if screen_name is None:
        screens_to_process = [f"Screen {i}" for i in range(1, 17)]
    else:
        screens_to_process = [screen_name]

    all_screens_data = []

    for current_screen in screens_to_process:
        sessions_today = []
        
        for s_val in schedule_data.get('sessions', {}).values():
            if s_val.get('screen', '').lower() == current_screen.lower():
                day_sessions = s_val.get('sessions_by_day', {}).get(day_key, {})
                for time_range in day_sessions.values():
                    sessions_today.append({"range": time_range})
        
        if sessions_today:
            total = sum([
                ((datetime.combine(target_date, parse_time_str(s['range'].split('-')[1])) - 
                  datetime.combine(target_date, parse_time_str(s['range'].split('-')[0]))
                 ).total_seconds() / 60) % 1440 
                for s in sessions_today
            ])
            
            all_screens_data.append({
                "all_movies_total_duration_minutes": int(total), 
                "screen": current_screen
            })

    return {
        "success": True,
        "equipment_id": screen_name if screen_name else "All Screens",
        "timestamp_utc": format_utc(datetime.now(timezone.utc)),
        "purpose": purpose,
        "data": all_screens_data
    }

def show_time(purpose, schedule_data, screen_name, target_date, day_key, keyword="first"):
    if screen_name is None:
        screens_to_process = [f"Screen {i}" for i in range(1, 17)]
    else:
        screens_to_process = [screen_name]

    all_matches = []

    for current_screen in screens_to_process:
        window = get_screen_operating_window(schedule_data, current_screen, target_date, day_key)
        
        if not window:
            continue 
        
        target_sessions = []
        day_rollover = 0
        previous_start_time = None
        for s_val in schedule_data.get('sessions', {}).values():
            if s_val.get('screen', '').lower() == current_screen.lower():
                day_sessions = s_val.get('sessions_by_day', {}).get(day_key, {})
                for time_range in day_sessions.values():
                    try:
                        parts = time_range.split("-")
                        start_time = parse_time_str(parts[0].strip())
                        end_time = parse_time_str(parts[1].strip())

                        if previous_start_time is not None and start_time < previous_start_time:
                            day_rollover += 1

                        show_date = target_date + timedelta(days=day_rollover)
                        s_start = datetime.combine(show_date, start_time)
                        s_end = datetime.combine(show_date, end_time)
                        if s_end <= s_start: 
                            s_end += timedelta(days=1)
                        
                        target_sessions.append({
                            "start": s_start, 
                            "end": s_end, 
                            "movie": s_val.get("film_title")
                        })
                        previous_start_time = start_time
                    except Exception:
                        continue

        if target_sessions:
            if keyword == "first":
                match = min(target_sessions, key=lambda x: x['start'])
            else:
                match = max(target_sessions, key=lambda x: x['end'])

            all_matches.append({
                "movie_name": match['movie'],
                "movie timing": f"{format_bms_time(match['start'])} - {format_bms_time(match['end'])}" if match['start'] and match['end'] else None,
                "screen": current_screen
            })

    if not all_matches:
        return {"success": "false", "message": "No shows found for any screen today.", "purpose": purpose}

    return {
        "success": True,
        "equipment_id": screen_name if screen_name else "All Screens",
        "timestamp_utc": format_utc(datetime.now(timezone.utc)),
        "purpose": purpose,
        "data": all_matches
    }

def total_cinema_timings(purpose, schedule_data, screen_name, day_key, target_date):
    """
    If screen_name is None, returns a list of timings for Screen 1 through Screen 16.
    Otherwise, returns grouped timings for the specified screen.
    """
    if screen_name is None:
        screens_to_process = [f"Screen {i}" for i in range(1, 17)]
    else:
        screens_to_process = [screen_name]

    all_screens_timings = []

    for current_screen in screens_to_process:
        sessions_by_movie = {}
        day_rollover = 0
        previous_start_time = None
        
        for s_val in schedule_data.get('sessions', {}).values():
            if s_val.get('screen', '').lower() == current_screen.lower():
                film_title = s_val.get("film_title")
                day_sessions = s_val.get('sessions_by_day', {}).get(day_key, {})
                
                if day_sessions:
                    if film_title not in sessions_by_movie:
                        sessions_by_movie[film_title] = []
                    
                    for time_range in day_sessions.values():
                        try:
                            parts = time_range.split('-')
                            if len(parts) != 2:
                                continue

                            start_time_obj = parse_time_str(parts[0].strip())
                            end_time_obj = parse_time_str(parts[1].strip())

                            if previous_start_time is not None and start_time_obj < previous_start_time:
                                day_rollover += 1

                            show_date = target_date + timedelta(days=day_rollover)
                            start_dt = datetime.combine(show_date, start_time_obj)
                            end_dt = datetime.combine(show_date, end_time_obj)
                            
                            if end_dt <= start_dt:
                                end_dt += timedelta(days=1)
                            
                            formatted_range = f"{format_bms_time(start_dt)} - {format_bms_time(end_dt)}"
                            sessions_by_movie[film_title].append(formatted_range)
                            previous_start_time = start_time_obj
                        except:
                            continue
        
        if sessions_by_movie:
            all_screens_timings.append(
                {
                    "screen": current_screen,
                    "movie_timing": sessions_by_movie
                })

    if screen_name is None:
        return {
            "success": True,
            "equipment_id": "All Screens",
            "timestamp_utc": format_utc(datetime.now(timezone.utc)),
            "purpose" : purpose,
            "data": all_screens_timings}
    
    return {
        "success": True,
        "equipment_id": screen_name,
        "timestamp_utc": format_utc(datetime.now(timezone.utc)),
        "purpose" : purpose,
        "data": [all_screens_timings[0]]} if all_screens_timings else {"message": "No shows found", "screen": screen_name}

def current_cinema_name(purpose, schedule_list, screen_name):
    now_utc = datetime.now(timezone.utc)
    
    sharjah_dt = now_utc.astimezone(SHARJAH_OFFSET)
    selected_schedule = schedule_selector(schedule_list, current_date=sharjah_dt.date()) #type:ignore
    
    active_schedule_list = [selected_schedule]
    
    screens_to_check = [screen_name] if screen_name else None
    
    occupancy_dict = get_current_movie_occupancy_status(active_schedule_list, screens=screens_to_check)
    
    all_screens_data = []
    for name, data in occupancy_dict.items():
        detailed_status = get_occupancy_status_for_timestamp(active_schedule_list, name, now_utc)
        
        curr_name = data.get("movie_name") or detailed_status.get("movie_name")
        next_name = data.get("next_movie_name") or detailed_status.get("next_movie_name")
        
        if not curr_name and not next_name:
            continue

        cinema_start_time = detailed_status.get("show_start_time")
        cinema_end_time = detailed_status.get("show_end_time")
        
        screen_info = {
            "screen": name,
            "status": "Occupied" if data.get("status") == 1 else "Unoccupied",
            "current_cinema_name": curr_name,
            "current_cinema_timing": f"{cinema_start_time} - {cinema_end_time}" if cinema_start_time and cinema_end_time else None,
            "time_until_next_show_in_minutes": data.get("time_until_next_movie"),
            "next_movie_name": next_name,
            "next_movie_time": detailed_status.get("next_movie_start")
        }
            
        all_screens_data.append(screen_info)

    return {
        "success": True,
        "equipment_id": screen_name if screen_name else "All Screens",
        "timestamp_utc": format_utc(now_utc),
        "purpose": purpose,
        "data": all_screens_data
    }

def cinema_name_by_time_and_screen(purpose, particular_time, schedule_list, screen_name):
    if isinstance(particular_time, str):
        try:
            parsed_dt = datetime.fromisoformat(particular_time.replace('Z', '+00:00'))
        except ValueError:
            try:
                now_sharjah = datetime.now(SHARJAH_OFFSET)
                parsed_dt = datetime.combine(now_sharjah.date(), parse_time_str(particular_time)).replace(tzinfo=SHARJAH_OFFSET)
            except: parsed_dt = datetime.now(timezone.utc)
        particular_time = parsed_dt
    
    particular_time = particular_time or datetime.now(timezone.utc)

    check_date = particular_time.astimezone(SHARJAH_OFFSET).date()
    selected_schedule = schedule_selector(schedule_list, current_date=check_date) #type:ignore
    
    active_schedule_list = [selected_schedule]

    screens_to_process = [f"Screen {i}" for i in range(1, 17)] if screen_name is None else [screen_name]
    results = []

    for current_screen in screens_to_process:
        cinema = get_occupancy_status_for_timestamp(active_schedule_list, current_screen, particular_time)
        
        curr_name = cinema.get("movie_name")
        next_name = cinema.get('next_movie_name')
        show_start = cinema.get("show_start_time")
        show_end = cinema.get("show_end_time")
        
        if not curr_name and not next_name:
            continue

        screen_data = {
            "screen": current_screen,
            "status": "Occupied" if cinema.get("status") == 1 else "Unoccupied",
            "current_cinema_name": curr_name,
            "movie_timing": f"{show_start} - {show_end}" if show_start and show_end else None,
            "next_movie_name": next_name,
            "next_movie_time": cinema.get("next_movie_start")
        }
        results.append(screen_data)

    return {
        "success": True,
        "equipment_id": screen_name if screen_name else "All Screens",
        "timestamp_utc": format_utc(particular_time),
        "purpose": purpose,
        "data": results
    }

def cinema_query(purpose: str, ticket: str, ticket_type: str, screen_name: Optional[str] = None, particular_time: Optional[str] = None):
    now_utc = datetime.now(timezone.utc)

    screen_name = f"screen {screen_name}" if screen_name else None
    
    schedule_list = fetch_movie_schedule(ticket=ticket, ticket_type=ticket_type)
    if not schedule_list:
        return {"error": "Schedule not found", "success": False}
    
    target_dt = now_utc.astimezone(SHARJAH_OFFSET).replace(tzinfo=None)
    target_date = target_dt.date()
    day_key = target_date.strftime('%a').lower()

    schedule_data = schedule_selector(schedule_list, current_date=target_date) #type:ignore

    if purpose == "total_duration_movies": 
        return total_duration_movies(purpose,schedule_data, screen_name, target_date, day_key)
        
    if purpose == "first_show_time": 
        return show_time(purpose,schedule_data, screen_name, target_date, day_key, keyword="first")
        
    elif purpose == "last_show_time": 
        return show_time(purpose,schedule_data, screen_name, target_date, day_key, keyword="last")
        
    elif purpose == "get_all_cinema_timings": 
        return total_cinema_timings(purpose,schedule_data, screen_name, day_key, target_date)
    
    elif purpose == "current_cinema_name": 
        return current_cinema_name(purpose,schedule_list, screen_name)
        
    elif purpose == "cinema_name_by_time_and_screen": 
        return cinema_name_by_time_and_screen(purpose,particular_time, schedule_list, screen_name)
    
    return {"error": "Invalid purpose", "success": False}