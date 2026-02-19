"""
Regenerate ahu_sensor_configs.json from the authoritative ahu_configs.py
"""
import json
import sys
sys.path.insert(0, '.')

from src.bms_ai.mpc.ahu_configs import AHU_CONFIGS

output = {}
for ahu_id, cfg in AHU_CONFIGS.items():
    output[ahu_id] = {
        "equipment_id": cfg.equipment_id,
        "screen_id": cfg.screen_id,
        "controller": cfg.controller,
        "device_id": cfg.device_id,
        "has_dual_sensors": cfg.has_dual_sensors,
        "sensors": {
            "space_temp": cfg.space_temp_sensor,
            "co2": cfg.co2_sensor,
            "humidity": cfg.humidity_sensor,
            "setpoint_eff": cfg.setpoint_eff_sensor,
            "vfd_feedback": cfg.vfd_feedback_sensor,
            "fad_feedback": cfg.fad_feedback_sensor,
            "supply_temp": cfg.supply_temp_sensor,
            "occupancy_setpoint": cfg.occupancy_setpoint_sensor,
        },
        "individual_sensors": {
            "temp": cfg.individual_temp_sensors,
            "co2": cfg.individual_co2_sensors,
            "humidity": cfg.individual_humidity_sensors,
        },
        "rename_map": cfg.sensor_rename_map,
        "required_datapoints": cfg.required_datapoints,
        "model_features": cfg.model_features,
        "missing_sensors": cfg.missing_sensors,
        "optimization_enabled": cfg.optimization_enabled,
    }

with open('artifacts/ahu_sensor_configs.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Generated ahu_sensor_configs.json with {len(output)} AHUs")
for ahu_id, data in output.items():
    missing = data["missing_sensors"]
    flag = f"  *** MISSING: {missing}" if missing else ""
    print(f"  {ahu_id}: temp={data['sensors']['space_temp']}, co2={data['sensors']['co2']}, hu={data['sensors']['humidity']}{flag}")
