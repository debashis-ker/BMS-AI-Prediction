from typing import Dict, Any, Optional
from src.bms_ai.components.setpoint_optimization_overriden_function import fetch_setpoint_diffs_averages, fetch_setpoint_diffs
import json
from src.bms_ai.logger_config import setup_logger

log = setup_logger(__name__)

'''Setpoint Optimization Summary Functions'''

def get_overall_setpoint_optimization_summary(data_input: Dict, session: Any = None) -> Dict:
    results = data_input.get("results", []) if isinstance(data_input, dict) else data_input
    if not results:
        return {"message": "No data available for the selected period."}

    sorted_data = sorted(results, key=lambda x: x.get('timestamp_utc', ''))

    raw_records = []
    if session:
        try:
            raw_res = fetch_setpoint_diffs(
                equipment_id=data_input.get("equipment_id", ""),
                session=session,
                start_date=data_input.get("start_date"),
                end_date=data_input.get("end_date")
            )
            if raw_res.get("success"):
                raw_records = raw_res.get("data", [])
        except Exception as e:
            log.error(f"Failed to fetch raw records for counts: {e}")

    occ_modes = ["occupied", "pre_cooling", "inter_show"]
    
    occ_count = len([r for r in raw_records if r.get("mode") in occ_modes])
    unocc_count = len([r for r in raw_records if r.get("mode") == "unoccupied"])
    occupied_temps = [r.get("actual_tempsp1") for r in results 
                      if (r.get("mode") or "").lower() in occ_modes and r.get("actual_tempsp1") is not None]
    unoccupied_temps = [r.get("actual_tempsp1") for r in results 
                        if (r.get("mode") or "").lower() == "unoccupied" and r.get("actual_tempsp1") is not None]

    occ_situation = {
        "total_write_count_occupied": occ_count,
        "maximum_occupied_temperature": round(max(occupied_temps), 2) if occupied_temps else None,
        "minimum_occupied_temperature": round(min(occupied_temps), 2) if occupied_temps else None,
    }
    if not occupied_temps:
        occ_situation["reason"] = "room is unoccupied for all the time"

    unocc_situation = {
        "total_write_count_unoccupied": unocc_count,
        "maximum_unoccupied_temperature": round(max(unoccupied_temps), 2) if unoccupied_temps else None,
        "minimum_unoccupied_temperature": round(min(unoccupied_temps), 2) if unoccupied_temps else None,
    }
    if not unoccupied_temps:
        unocc_situation["reason"] = "room is occupied for all the time"

    last_feat = sorted_data[-1].get('used_features', {}) if sorted_data else {}
    eval_map = {
        'Outside Temperature': last_feat.get("outside_temp"),
        'Outside Humidity': last_feat.get("outside_humidity"),
        'Occupied Temperature': last_feat.get("occupied_setpoint"),
        'Unoccupied Temperature': last_feat.get("unoccupied_setpoint"),
        'Fan Speed Feedback': last_feat.get("FbVFD"),
        'Fresh Air Damper Feedback': last_feat.get("FbFAD"),
        'Screen Occupancy': last_feat.get("occupied"),
        'Space Air Temperature': last_feat.get("TempSp1"),
        'Return Air Humidity': last_feat.get("HuR1") or last_feat.get("hur1"),
        'Time of day Patterns (is weekend, is night)': last_feat.get("hour_sin") or last_feat.get("day_sin")
    }
    active_params = [k for k, v in eval_map.items() if v is not None]

    averages_data = None
    if session:
        try:
            averages_data = fetch_setpoint_diffs_averages(
                equipment_id=data_input.get("equipment_id", ""),
                session=session,
                start_date=data_input.get("start_date"),
                end_date=data_input.get("end_date")
            )
        except Exception as e:
            log.error(f"Failed to fetch averages for summary: {e}")

    return {
        "data": {
            "optimized_count": averages_data['total_records'] if averages_data else 0,
            "write_counts": {
                "occupied_situation": occ_situation,
                "unoccupied_situation": unocc_situation, 
            },
            "optimization_averages": averages_data,
            "evaluation_parameters": active_params if active_params else None
        }
    }


# def generate_optimization_summary_response(optimization_data: Dict[str, Any]) -> str:
#     """
#     Specifically handles data from get_overall_setpoint_optimization_summary 
#     to provide a high-level executive summary with using the AI narrator.
#     """
#     if not client:
#         return "Optimization summary is available, but the AI narrator is not configured."

#     try:
#         system_prompt = """Role: Automated Data Analysis Engine.

# Task: Generate a non technical data summary based on the provided JSON input. 

# Operational Constraints:
# 1. Tone: Neutral, clinical, and strictly objective. 
# 2. Prohibited Terminology: Do not use "saved," "saving," "energy," "maintained," "achieved," "recorded," "comfort," "efficiency," "guest," or any words implying intent, feeling, or effort.
# 3. Content: Every sentence must include at least two numerical values or specific data keys.
# 4. Data Handling: If a value is null/None, state: "Value unavailable due to zero occupancy."
# 5. Formatting: Do not use bold (**) markdown. Replace all underscores (_) with spaces in the output.
# 6. Conciseness: Provide only raw data correlations. No introductory or concluding filler.

# Analysis Logic:
# - Correlate optimized_count with the specific from_date and to_date range.
# - Map the occupancy_percentage against the evaluation_parameters used for the logic trigger.
# - Report average_occupied_temperature and average_precooling_temperature as static setpoint states.
# - Don't mention external factors, skip the line of external factors.

# """
#         user_prompt = f"""Below is the Setpoint Optimization Report for the equipment:
# {json.dumps(optimization_data, indent=2)}

# Format as JSON:
# {{
#     "answer": "your executive summary here",
# }}
# """

#         response = client.chat.completions.create(
#             model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             response_format={"type": "json_object"}
#         )

#         gpt_res = json.loads(response.choices[0].message.content)
#         return gpt_res.get("answer")

#     except Exception as e:
#         return "Error generating summary."
    