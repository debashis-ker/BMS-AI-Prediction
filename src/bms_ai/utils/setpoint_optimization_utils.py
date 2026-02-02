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


def parse_time_str(t_str: str):
    """
    Parse time string handling single-digit hours and flexible formats.
    Handles formats like: "11:30p", "11:30 PM", "1:00a", "12:00AM", etc.
    """
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


def fetch_movie_schedule(ticket: str="1b9362bd-6a70-4827-b7e8-ff0af1f06375") -> Optional[List[Dict[str, Any]]]:
    """
    Fetches movie schedule data from IKON service.
    Returns all instances data (list of schedule data from all cinemas).
    Most recent/active schedule is typically at index 0.
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
        
        # Extract all instances data
        all_instances = []
        for idx, instance in enumerate(instances):
            instance_data = instance.get('data', {})
            result_data = instance_data.get('result', {})
            if result_data:
                result_data['_instance_index'] = idx
                result_data['_cinema_name'] = result_data.get('cinema_name', 'Unknown')
                all_instances.append(result_data)
                log.info(f"Instance {idx}: {result_data.get('cinema_name', 'Unknown')} - {result_data.get('start_date', '')} to {result_data.get('end_date', '')}")
        
        print(f"Extracted {len(all_instances)} schedule instances.")
        return all_instances

    except Exception as e:
        print(f"Error fetching instances: {e}")
        log.error(f"Error in fetch_movie_schedule: {e}")
        return None
    


def get_current_movie_occupancy_status(
    schedule_data: List[Dict[str, Any]],
    screens: Optional[List[str]] = None,
    for_which_time: int = 0,
    instance_index: int = 0
) -> Dict[str, Any]:
    """
    Returns movie occupancy status for the specified screens in Sharjah time.
    If a movie is showing, returns status=1 with movie name and time remaining.
    If no movie is showing, returns status=0 with time until next movie.
    
    Args:
        schedule_data: List of schedule data from all instances
        screens: List of screen names to check (e.g., ["Screen 13", "Screen 16"]).
                 If None, checks all screens in the dataset.
        for_which_time: Minutes from now to check (default 0 = current time)
        instance_index: Which instance to use (default 0 = most recent)
    
    Returns:
        Dict of dicts where keys are screen names and values contain:
        - status: 1 if movie is showing, 0 otherwise
        - movie_name and time_remaining if showing
        - time_until_next_movie if not showing
    """
    # Use specified instance index (default 0 for most recent)
    if not schedule_data or instance_index >= len(schedule_data):
        log.error(f"Invalid instance_index {instance_index} for schedule_data length {len(schedule_data) if schedule_data else 0}")
        return {}
    
    schedule = schedule_data[instance_index]
    day_map = {
        0: "mon", 1: "tue", 2: "wed",
        3: "thu", 4: "fri", 5: "sat", 6: "sun"
    }

    now_sharjah = datetime.now(SHARJAH_OFFSET).replace(tzinfo=None)
    target_dt = now_sharjah + timedelta(minutes=for_which_time)

    sessions = schedule.get('sessions', {})
    
    if screens is None:
        screens = list(set(
            session_data.get("screen", "")
            for session_data in sessions.values()
            if session_data.get("screen")
        ))
    
    result = {}
    
    for screen in screens:
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
                            result[screen] = {
                                "status": 1,
                                "movie_name": session_data.get("film_title", "Unknown"),
                                "time_remaining": time_remaining_minutes
                            }
                            break
                        
                        if show_start > target_dt:
                            upcoming_shows.append({
                                "start": show_start,
                                "movie_name": session_data.get("film_title", "Unknown")
                            })

                    except Exception as e:
                        log.debug(f"Error parsing time range '{time_range}': {e}")
                        continue
                
                if screen in result:
                    break
            
            if screen in result:
                break
        
        if screen not in result:
            if upcoming_shows:
                upcoming_shows.sort(key=lambda x: x["start"])
                next_show = upcoming_shows[0]
                time_until_next_seconds = (next_show["start"] - target_dt).total_seconds()
                time_until_next_minutes = math.ceil(time_until_next_seconds / 60)
                
                result[screen] = {
                    "status": 0,
                    "time_until_next_movie": time_until_next_minutes,
                    "next_movie_name": next_show["movie_name"]
                }
            else:
                result[screen] = {
                    "status": 0,
                    "time_until_next_movie": "No upcoming shows"
                }
    
    return result


def get_occupancy_status_for_timestamp(
    schedule_data: List[Dict[str, Any]],
    screen: str,
    timestamp_utc: datetime
) -> Dict[str, Any]:
    """
    Get occupancy status for a specific screen at a specific UTC timestamp.
    Converts UTC to Sharjah timezone and checks if a movie is showing.
    Automatically detects which schedule instance to use based on the timestamp date.
    For overlapping dates (e.g., 22/01/2026 in both instances), checks all matching instances.
    
    Args:
        schedule_data: List of schedule data from all instances
        screen: Screen name (e.g., "Screen 13")
        timestamp_utc: UTC datetime to check occupancy for
    
    Returns:
        Dict with:
        - status: 1 if occupied (movie showing), 0 if unoccupied
        - movie_name: Name of the movie (if occupied)
        - time_remaining: Time remaining in movie (if occupied)
        - show_start_time: Movie start time in Sharjah (if occupied)
        - show_end_time: Movie end time in Sharjah (if occupied)
        - timestamp_sharjah: The timestamp converted to Sharjah timezone
        - timestamp_utc: The original UTC timestamp
        - instance_used: Which schedule instance was used
    """
    day_map = {
        0: "mon", 1: "tue", 2: "wed",
        3: "thu", 4: "fri", 5: "sat", 6: "sun"
    }
    
    # Validate inputs
    if not schedule_data:
        log.error("No schedule data provided")
        return {"status": 0, "error": "No schedule data"}
    
    # Convert UTC to Sharjah time
    if timestamp_utc.tzinfo is None:
        timestamp_utc = timestamp_utc.replace(tzinfo=timezone.utc)
    timestamp_sharjah = timestamp_utc.astimezone(SHARJAH_OFFSET).replace(tzinfo=None)
    target_date = timestamp_sharjah.date()
    
    # Find all instances that cover this date
    matching_instances = []
    for idx, schedule in enumerate(schedule_data):
        start_date_str = schedule.get('start_date', '')
        end_date_str = schedule.get('end_date', '')
        
        if not start_date_str or not end_date_str:
            continue
        
        try:
            # Parse dates (format: "DD/MM/YYYY")
            start_date = datetime.strptime(start_date_str, "%d/%m/%Y").date()
            end_date = datetime.strptime(end_date_str, "%d/%m/%Y").date()
            
            # Check if target_date falls within this schedule range
            if start_date <= target_date <= end_date:
                matching_instances.append((idx, schedule))
                log.debug(f"Instance {idx} matches date {target_date}: {start_date} to {end_date}")
        except Exception as e:
            log.warning(f"Failed to parse dates for instance {idx}: {e}")
            continue
    
    if not matching_instances:
        log.warning(f"No schedule instance found for date {target_date}")
        return {
            "status": 0,
            "movie_name": None,
            "timestamp_sharjah": timestamp_sharjah.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_utc": timestamp_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "error": f"No schedule available for {target_date}"
        }
    
    log.info(f"Found {len(matching_instances)} schedule instance(s) covering {target_date}")
    
    # Collect all upcoming shows across all instances for when no movie is currently playing
    all_upcoming_shows = []
    
    def check_schedule_for_movie(schedule, instance_idx, check_overnight_from_previous_day=False, collect_upcoming=True):
        """
        Check a schedule for a movie at the target timestamp.
        If check_overnight_from_previous_day=True, only check for movies that started previous day.
        If collect_upcoming=True, also collect upcoming shows for pre-cooling logic.
        """
        sessions = schedule.get('sessions', {})
        
        for session_key, session_data in sessions.items():
            if session_data.get("screen", "").lower() != screen.lower():
                continue
            
            # Determine which day offsets to check
            if check_overnight_from_previous_day:
                # Only check previous day's shows that extend past midnight
                day_offsets = [-1]
            else:
                # Check current day and adjacent days
                day_offsets = [0, -1, 1]
            
            for day_offset in day_offsets:
                schedule_date = timestamp_sharjah.date() + timedelta(days=day_offset)
                day_key = day_map[schedule_date.weekday()]
                
                day_sessions = session_data.get("sessions_by_day", {}).get(day_key, {})
                if not day_sessions:
                    continue
                
                for time_range in day_sessions.values():
                    try:
                        parts = time_range.split("-")
                        if len(parts) != 2:
                            continue
                        
                        start_str = parts[0].strip()
                        end_str = parts[1].strip()
                        
                        start_time = parse_time_str(start_str)
                        end_time = parse_time_str(end_str)
                        
                        show_start = datetime.combine(schedule_date, start_time)
                        show_end = datetime.combine(schedule_date, end_time)
                        
                        # Handle overnight shows
                        if show_end <= show_start:
                            show_end += timedelta(days=1)
                        
                        # For overnight check, only consider shows that extend past midnight
                        if check_overnight_from_previous_day:
                            if show_end.date() != timestamp_sharjah.date():
                                continue  # Show doesn't extend to target date
                        
                        # Check if timestamp falls within this show
                        if show_start <= timestamp_sharjah < show_end:
                            time_remaining_seconds = (show_end - timestamp_sharjah).total_seconds()
                            time_remaining_minutes = math.ceil(time_remaining_seconds / 60)
                            
                            cinema_name = schedule.get('cinema_name', 'Unknown')
                            date_range = f"{schedule.get('start_date', '')} to {schedule.get('end_date', '')}"
                            
                            return {
                                "status": 1,
                                "movie_name": session_data.get("film_title", "Unknown"),
                                "time_remaining": time_remaining_minutes,
                                "show_start_time": show_start.strftime("%Y-%m-%d %H:%M:%S"),
                                "show_end_time": show_end.strftime("%Y-%m-%d %H:%M:%S"),
                                "timestamp_sharjah": timestamp_sharjah.strftime("%Y-%m-%d %H:%M:%S"),
                                "timestamp_utc": timestamp_utc.strftime("%Y-%m-%d %H:%M:%S"),
                                "instance_used": instance_idx,
                                "cinema_name": cinema_name,
                                "schedule_range": date_range,
                                "is_overnight_from_previous_schedule": check_overnight_from_previous_day
                            }
                        
                        # Collect upcoming shows (for pre-cooling logic)
                        if collect_upcoming and show_start > timestamp_sharjah:
                            all_upcoming_shows.append({
                                "start": show_start,
                                "movie_name": session_data.get("film_title", "Unknown"),
                                "instance_idx": instance_idx
                            })
                    
                    except Exception as e:
                        log.debug(f"Error parsing time range '{time_range}': {e}")
                        continue
        
        return None
    
    # For overlapping dates (multiple instances), handle overnight shows from older schedule
    # E.g., Jan 22 is in both [0] (22/01-29/01) and [1] (15/01-22/01)
    # A movie from Tue 21st night (old schedule) would still be playing early Wed 22nd
    
    if len(matching_instances) > 1:
        log.info(f"Overlapping date detected: {target_date}")
        
        # First, check OLDER schedules for overnight movies from previous day
        # Sort by instance_idx descending (older schedules first for overnight check)
        older_instances = sorted(matching_instances, key=lambda x: x[0], reverse=True)
        
        for instance_idx, schedule in older_instances:
            result = check_schedule_for_movie(schedule, instance_idx, check_overnight_from_previous_day=True)
            if result:
                log.info(f"Found overnight movie from older schedule (instance {instance_idx})")
                return result
        
        # Then, check NEWER schedules for current day movies
        # Sort by instance_idx ascending (newer schedules first for same-day check)
        newer_instances = sorted(matching_instances, key=lambda x: x[0])
        
        for instance_idx, schedule in newer_instances:
            result = check_schedule_for_movie(schedule, instance_idx, check_overnight_from_previous_day=False)
            if result:
                log.info(f"Found movie from newer schedule (instance {instance_idx})")
                return result
    else:
        # Single matching instance - check normally
        instance_idx, schedule = matching_instances[0]
        result = check_schedule_for_movie(schedule, instance_idx, check_overnight_from_previous_day=False)
        if result:
            return result
    
    # No movie found in any matching instance - unoccupied
    # Check for upcoming shows (for pre-cooling logic)
    if all_upcoming_shows:
        all_upcoming_shows.sort(key=lambda x: x["start"])
        next_show = all_upcoming_shows[0]
        time_until_next_seconds = (next_show["start"] - timestamp_sharjah).total_seconds()
        time_until_next_minutes = math.ceil(time_until_next_seconds / 60)
        
        return {
            "status": 0,
            "movie_name": None,
            "time_until_next_movie": time_until_next_minutes,
            "next_movie_name": next_show["movie_name"],
            "next_movie_start": next_show["start"].strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_sharjah": timestamp_sharjah.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_utc": timestamp_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "instances_checked": [idx for idx, _ in matching_instances]
        }
    
    return {
        "status": 0,
        "movie_name": None,
        "time_until_next_movie": None,
        "next_movie_name": None,
        "timestamp_sharjah": timestamp_sharjah.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_utc": timestamp_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "instances_checked": [idx for idx, _ in matching_instances]
    }


if __name__ == "__main__":
    print("Testing fetch_movie_schedule function...")
    schedule_data = fetch_movie_schedule()
    
    if schedule_data:
        print(f"\n✓ Fetched {len(schedule_data)} schedule instances")
        for idx, sched in enumerate(schedule_data):
            print(f"  [{idx}] {sched.get('cinema_name', 'Unknown')} - {sched.get('start_date', '')} to {sched.get('end_date', '')}")
        
        print("\n" + "="*60)
        print("Testing get_current_movie_occupancy_status for Screen 13...")
        print("="*60)
        status = get_current_movie_occupancy_status(schedule_data, screens=["Screen 13"])
        print(f"Current status: {json.dumps(status, indent=2)}")
        
        print("\n" + "="*60)
        print("Testing get_occupancy_status_for_timestamp...")
        print("="*60)
        
        # Test 1: Date in instance [1] (15/01/2026)
        test1 = datetime(2026, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        print(f"\nTest 1 - Jan 15 (instance [1] range)")
        print(f"UTC: {test1} | Sharjah: {test1.astimezone(SHARJAH_OFFSET)}")
        status1 = get_occupancy_status_for_timestamp(schedule_data, "Screen 13", test1)
        print(f"Result: {json.dumps(status1, indent=2)}")
        
        # Test 2: Connecting date (22/01/2026) EARLY MORNING - check for overnight movie from old schedule
        # Screen 15 has "28 years later: the bone" 11:30p-1:40a on Tue in OLD schedule (instance[1])
        test2a = datetime(2026, 1, 21, 21, 30, 0, tzinfo=timezone.utc)  # 1:30 AM Sharjah on 22nd
        print(f"\nTest 2a - Screen 15, Jan 22 @ 01:30 AM (OVERNIGHT from old schedule)")
        print(f"UTC: {test2a} | Sharjah: {test2a.astimezone(SHARJAH_OFFSET)}")
        print(f"Expected: Should find '28 years later: the bone' from instance [1] (15/01-22/01)")
        status2a = get_occupancy_status_for_timestamp(schedule_data, "Screen 15", test2a)
        print(f"Result: {json.dumps(status2a, indent=2)}")
        
        # Test 2b: Screen 14 at 12:30 AM - "En ghab el kot" 11:00p-1:05a on Tue
        test2b = datetime(2026, 1, 21, 20, 30, 0, tzinfo=timezone.utc)  # 12:30 AM Sharjah on 22nd
        print(f"\nTest 2b - Screen 14, Jan 22 @ 00:30 AM (OVERNIGHT from old schedule)")
        print(f"UTC: {test2b} | Sharjah: {test2b.astimezone(SHARJAH_OFFSET)}")
        print(f"Expected: Should find 'En ghab el kot' from instance [1] (15/01-22/01)")
        status2b = get_occupancy_status_for_timestamp(schedule_data, "Screen 14", test2b)
        print(f"Result: {json.dumps(status2b, indent=2)}")
        
        # Test 2c: Screen 8, Jan 22 @ 01:00 AM - "Happy patel: khatarnak" 11:20p-1:50a (Tue only!)
        test2c = datetime(2026, 1, 21, 21, 0, 0, tzinfo=timezone.utc)  # 1:00 AM Sharjah on 22nd
        print(f"\nTest 2c - Screen 8, Jan 22 @ 01:00 AM (OVERNIGHT - Tue only movie)")
        print(f"UTC: {test2c} | Sharjah: {test2c.astimezone(SHARJAH_OFFSET)}")
        print(f"Expected: Should find 'Happy patel: khatarnak' from instance [1] (Tue only)")
        status2c = get_occupancy_status_for_timestamp(schedule_data, "Screen 8", test2c)
        print(f"Result: {json.dumps(status2c, indent=2)}")
        
        # Test 2d: Connecting date (22/01/2026) DAYTIME - should use new schedule
        test2d = datetime(2026, 1, 22, 14, 30, 0, tzinfo=timezone.utc)  # 6:30 PM Sharjah
        print(f"\nTest 2d - Screen 13, Jan 22 @ 06:30 PM (DAYTIME - new schedule)")
        print(f"UTC: {test2d} | Sharjah: {test2d.astimezone(SHARJAH_OFFSET)}")
        status2d = get_occupancy_status_for_timestamp(schedule_data, "Screen 13", test2d)
        print(f"Result: {json.dumps(status2d, indent=2)}")
        
        # Test 3: Date in instance [0] only (25/01/2026)
        test3 = datetime(2026, 1, 25, 14, 30, 0, tzinfo=timezone.utc)
        print(f"\nTest 3 - Jan 25 (instance [0] only)")
        print(f"UTC: {test3} | Sharjah: {test3.astimezone(SHARJAH_OFFSET)}")
        status3 = get_occupancy_status_for_timestamp(schedule_data, "Screen 13", test3)
        print(f"Result: {json.dumps(status3, indent=2)}")
        
        # Test 4: Current time
        print("\n" + "="*60)
        print("Test 4 - Current time")
        print("="*60)
        current_time = datetime.now(timezone.utc)
        print(f"UTC: {current_time} | Sharjah: {current_time.astimezone(SHARJAH_OFFSET)}")
        status_now = get_occupancy_status_for_timestamp(schedule_data, "Screen 13", current_time)
        print(f"Result: {json.dumps(status_now, indent=2)}")
    else:
        print("Failed to fetch schedule data")