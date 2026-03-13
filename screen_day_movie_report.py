import argparse
import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

DAY_KEYS = {
    "mon": "mon",
    "monday": "mon",
    "tue": "tue",
    "tuesday": "tue",
    "wed": "wed",
    "wednesday": "wed",
    "thu": "thu",
    "thursday": "thu",
    "fri": "fri",
    "friday": "fri",
    "sat": "sat",
    "saturday": "sat",
    "sun": "sun",
    "sunday": "sun",
}


def normalize_day(day_value: str) -> str:
    key = day_value.strip().lower()
    if key not in DAY_KEYS:
        raise ValueError(f"Invalid day '{day_value}'. Use mon/tue/... or full day name.")
    return DAY_KEYS[key]


def normalize_screen(screen_value: str) -> str:
    raw = screen_value.strip().lower()
    if raw.startswith("screen "):
        number = raw.replace("screen ", "").strip()
    else:
        number = raw
    if not number.isdigit():
        raise ValueError(f"Invalid screen '{screen_value}'. Use number like 11 or text like 'Screen 11'.")
    return f"Screen {int(number)}"


def parse_time_str(value: str) -> datetime.time:
    token = value.strip().lower().replace(" ", "")
    if token.endswith("p"):
        token = token[:-1] + "PM"
    elif token.endswith("a"):
        token = token[:-1] + "AM"

    hour = token.split(":", 1)[0]
    if hour == "0":
        token = "12" + token[token.find(":"):]
    elif len(hour) == 1:
        token = "0" + token

    return datetime.strptime(token, "%I:%M%p").time()


def extract_session_rows(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        sessions = data.get("sessions")
        if isinstance(sessions, dict):
            return list(sessions.values())

    raise ValueError("Unsupported schedule format. Expected list of sessions or dict with 'sessions'.")


def get_movies_for_screen_day(
    data: Any,
    screen: str,
    day: str,
    base_date: Optional[date] = None,
) -> Dict[str, Any]:
    screen_name = normalize_screen(screen)
    day_key = normalize_day(day)
    rows = extract_session_rows(data)

    if base_date is None:
        base_date = date.today()

    serial_sessions: List[Dict[str, str]] = []
    timeline_bounds: List[tuple[datetime, datetime]] = []

    day_rollover = 0
    previous_start_time: Optional[datetime.time] = None

    for row in rows:
        if row.get("screen", "").strip().lower() != screen_name.lower():
            continue

        day_sessions = row.get("sessions_by_day", {}).get(day_key, {})
        if not day_sessions:
            continue

        for time_range in day_sessions.values():
            if "-" not in time_range:
                continue

            start_raw, end_raw = [part.strip() for part in time_range.split("-", 1)]
            start_time = parse_time_str(start_raw)
            end_time = parse_time_str(end_raw)

            # Preserve source order and move into next day when order wraps (PM -> AM).
            if previous_start_time is not None and start_time < previous_start_time:
                day_rollover += 1

            show_date = base_date + timedelta(days=day_rollover)
            start_dt = datetime.combine(show_date, start_time)
            end_dt = datetime.combine(show_date, end_time)
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)

            serial_sessions.append(
                {
                    "movie": row.get("film_title", "Unknown"),
                    "time_range": time_range,
                    "start": start_dt.strftime("%Y-%m-%d %I:%M %p"),
                    "end": end_dt.strftime("%Y-%m-%d %I:%M %p"),
                }
            )
            timeline_bounds.append((start_dt, end_dt))
            previous_start_time = start_time

    if not serial_sessions:
        return {
            "screen": screen_name,
            "day": day_key,
            "sessions": [],
            "first_time": None,
            "last_time": None,
            "message": "No sessions found for this screen/day.",
        }

    first_start = timeline_bounds[0][0]
    last_end = timeline_bounds[-1][1]

    return {
        "screen": screen_name,
        "day": day_key,
        "sessions": serial_sessions,
        "first_time": first_start.strftime("%Y-%m-%d %I:%M %p"),
        "last_time": last_end.strftime("%Y-%m-%d %I:%M %p"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get serial movie timings and first/last window for a given screen/day."
    )
    parser.add_argument("--file", default="schedule_data.json", help="Path to schedule json file")
    parser.add_argument("--screen", required=True, help="Screen number or name (e.g., 11 or 'Screen 11')")
    parser.add_argument("--day", required=True, help="Day key/name (e.g., thu or thursday)")
    parser.add_argument(
        "--base-date",
        default=None,
        help="Base date as YYYY-MM-DD for the selected day (default: today)",
    )

    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    base_date = None
    if args.base_date:
        base_date = datetime.strptime(args.base_date, "%Y-%m-%d").date()

    report = get_movies_for_screen_day(
        data=payload,
        screen=args.screen,
        day=args.day,
        base_date=base_date,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
