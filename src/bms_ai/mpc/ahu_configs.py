"""
AHU Sensor Configurations for MPC Training & Inference
========================================================

Maps each AHU (Ahu1-Ahu16) to its actual BACnet datapoint names and defines
how raw sensor names translate to the standardised feature names expected by
the MPC pipeline (TempSp1, Co2RA, HuR1, SpTREff, FbVFD, FbFAD, TempSu, etc.)

Key differences across AHUs:
    ─────────────────────────────────────────────────────────────────
    Group A  (Ahu1-Ahu5)   Dual space sensors  -> use averages (TrAvg, Co2Avg, HuAvg1)
    Group B  (Ahu6-Ahu13)  Single sensors       -> use directly  (TempSp1, Co2RA, HuR1)
    Group C  (Ahu14)       Unique naming         -> AvgTmp, AvgCo2, AvgHu
    Group D  (Ahu15)       Single sensors (avg-labelled) -> TempSp1, Co2RA, HuR1
    Group E  (Ahu16)       TempSp (no '1')       -> TempSp, Co2RA, HuR1
    ─────────────────────────────────────────────────────────────────

Controller mapping:
    OS01 (2122753)  -> Ahu1, Ahu2, Ahu3
    OS02 (2122754)  -> Ahu4, Ahu5, Ahu14, Ahu15, Ahu16
    OS04 (2122756)  -> Ahu6, Ahu7, Ahu8, Ahu13
    OS05 (2122757)  -> Ahu9, Ahu10, Ahu11, Ahu12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# PER-AHU SENSOR CONFIGURATION
# =============================================================================

@dataclass
class AHUSensorConfig:
    """
    Describes how a specific AHU's raw BACnet datapoint names map to the
    standardised feature names consumed by the MPC model.

    Attributes
    ----------
    equipment_id : str
        Equipment identifier used in the BMS API query (e.g. "Ahu1").
    screen_id : str
        Human-readable screen / zone name (e.g. "Screen 1").
    controller : str
        Desigo PX controller id (e.g. "OS01").
    device_id : int
        BACnet device id.

    # ── raw datapoint names as they appear in the BMS API ──────────
    space_temp_sensor : str
        Raw datapoint for space air temperature.
        Ahu1-Ahu5 -> "TrAvg" (average of dual sensors)
        Ahu6-Ahu13 -> "TempSp1"
        Ahu14      -> "AvgTmp"
        Ahu15      -> "TempSp1" (already averaged)
        Ahu16      -> "TempSp"
    co2_sensor : str
        Raw datapoint for CO₂ level.
        Ahu1-Ahu3,Ahu5 -> "Co2Avg"
        Ahu4            -> "Co2Avg" (raw sensor is Co2RA1)
        Ahu6-Ahu13      -> "Co2RA"
        Ahu14           -> "AvgCo2" (available from 2026-02-19)
        Ahu15-Ahu16     -> "Co2RA" (already averaged)
    humidity_sensor : str
        Raw datapoint for space air humidity.
        Ahu1-Ahu5  -> "HuAvg1"
        Ahu6-Ahu13 -> "HuR1"
        Ahu14      -> "AvgHu"
        Ahu15-Ahu16 -> "HuR1"

    # ── additional raw sensors (identical naming across AHUs) ──────
    setpoint_eff_sensor : str
        Effective room-temperature setpoint. Always "SpTREff".
    vfd_feedback_sensor : str
        VFD speed feedback. Always "FbVFD".
    fad_feedback_sensor : str
        Fresh-air damper feedback. Always "FbFAD".
    supply_temp_sensor : str
        Supply-air temperature. Always "TempSu".
    occupancy_setpoint_sensor : str
        Occupied setpoint. Always "SpTROcc".

    # ── sensor inventory flags ─────────────────────────────────────
    has_dual_sensors : bool
        True for Ahu1-Ahu5 which have TempSp1, TempSp2, TrAvg etc.
    individual_temp_sensors : list[str]
        If dual sensors exist, the individual names (e.g. ["TempSp1", "TempSp2"]).
    individual_co2_sensors : list[str]
        If dual sensors exist, the individual names (e.g. ["Co2RA", "Co2RA2"]).
    individual_humidity_sensors : list[str]
        If dual sensors exist, the individual names (e.g. ["HuR1", "HuR2"]).
    """

    equipment_id: str
    screen_id: str
    controller: str
    device_id: int

    
    space_temp_sensor: str          # maps to feature "TempSp1"
    co2_sensor: Optional[str]       # maps to feature "Co2RA"
    humidity_sensor: str            # maps to feature "HuR1"

    # Common sensors (same name on every AHU)
    setpoint_eff_sensor: str = "SpTREff"
    vfd_feedback_sensor: str = "FbVFD"
    fad_feedback_sensor: str = "FbFAD"
    supply_temp_sensor: str = "TempSu"
    occupancy_setpoint_sensor: str = "SpTROcc"

    has_dual_sensors: bool = False
    individual_temp_sensors: List[str] = field(default_factory=list)
    individual_co2_sensors: List[str] = field(default_factory=list)
    individual_humidity_sensors: List[str] = field(default_factory=list)

    supply_temp_missing: bool = False   
    co2_missing: bool = False           

    optimization_enabled: bool = True   


    @property
    def required_datapoints(self) -> List[str]:
        """
        Return the list of BACnet datapoint names that must be fetched from
        the BMS API for this AHU.  Skips sensors flagged as missing.
        """
        points = [
            self.setpoint_eff_sensor,  
            self.vfd_feedback_sensor,   
            self.fad_feedback_sensor,   
            self.occupancy_setpoint_sensor,  
        ]
        if not self.supply_temp_missing:
            points.append(self.supply_temp_sensor) 

        if self.has_dual_sensors and self.individual_temp_sensors:
            points.extend(self.individual_temp_sensors)
            points.append(self.space_temp_sensor)  
        else:
            points.append(self.space_temp_sensor)

        # CO2 (skip if missing / None)
        if self.co2_sensor and not self.co2_missing:
            if self.has_dual_sensors and self.individual_co2_sensors:
                points.extend(self.individual_co2_sensors)
                points.append(self.co2_sensor)
            else:
                points.append(self.co2_sensor)

        # Humidity
        if self.has_dual_sensors and self.individual_humidity_sensors:
            points.extend(self.individual_humidity_sensors)
            points.append(self.humidity_sensor)
        else:
            points.append(self.humidity_sensor)

        return list(dict.fromkeys(points))  

    @property
    def missing_sensors(self) -> List[str]:
        """
        Return a list of standardised feature names that are NOT available
        in BACnet for this AHU.  The pipeline should handle these gracefully
        (e.g. fill with NaN or exclude from the model).
        """
        missing = []
        if self.supply_temp_missing:
            missing.append("TempSu")
        if self.co2_missing or self.co2_sensor is None:
            missing.append("Co2RA")
        return missing

    @property
    def sensor_rename_map(self) -> Dict[str, str]:
        """
        Mapping from *raw BACnet datapoint name* -> *standardised feature name*
        expected by the MPC model.

        The pipeline should use this after pivoting so that every AHU's
        DataFrame has the same columns regardless of raw naming differences.

        Example:
            Ahu1:  {"TrAvg": "TempSp1", "Co2Avg": "Co2RA", "HuAvg1": "HuR1"}
            Ahu13: {}  (names already match)
            Ahu14: {"AvgTmp": "TempSp1", "AvgCo2": "Co2RA", "AvgHu": "HuR1"}
            Ahu16: {"TempSp": "TempSp1"}
        """
        rename = {}
        # Temperature
        if self.space_temp_sensor != "TempSp1":
            rename[self.space_temp_sensor] = "TempSp1"
        # Supply temperature
        if self.supply_temp_sensor != "TempSu" and not self.supply_temp_missing:
            rename[self.supply_temp_sensor] = "TempSu"
        # CO2
        if self.co2_sensor and self.co2_sensor != "Co2RA":
            rename[self.co2_sensor] = "Co2RA"
        # Humidity
        if self.humidity_sensor != "HuR1":
            rename[self.humidity_sensor] = "HuR1"
        return rename

    @property
    def model_features(self) -> List[str]:
        """
        Return the standardised feature names expected by the MPC model.
        These are the same for every AHU (after renaming).

        Features that are flagged as missing for this AHU are still listed
        — the pipeline / model must handle them (e.g. impute or exclude).
        """
        return [
            "TempSp1",
            "TempSp1_lag_10m",
            "SpTREff",
            "SpTREff_lag_10_min",
            "outside_temp",
            "outside_humidity",
            "occupied",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "FbVFD",
            "FbFAD",
            "Co2RA",
            "HuR1",
            "Target_Temp",
        ]


# =============================================================================
# AHU REGISTRY  (Ahu1 – Ahu16, skip Ahu17 corridor)
# =============================================================================

# Group A: Dual sensors + averages (Ahu1–Ahu5) 

AHU1 = AHUSensorConfig(
    equipment_id="Ahu1",
    screen_id="Screen 1",
    controller="OS01",
    device_id=2122753,
    space_temp_sensor="TrAvg",
    co2_sensor="Co2Avg",
    humidity_sensor="HuAvg1",
    has_dual_sensors=True,
    individual_temp_sensors=["TempSp1", "TempSp2"],
    individual_co2_sensors=["Co2RA", "Co2RA2"],
    individual_humidity_sensors=["HuR1", "HuR2"],
    supply_temp_sensor="TSu",  # Ahu1 uses 'TSu' instead of 'TempSu'
)

AHU2 = AHUSensorConfig(
    equipment_id="Ahu2",
    screen_id="Screen 2",
    controller="OS01",
    device_id=2122753,
    space_temp_sensor="TrAvg",
    co2_sensor="Co2Avg",
    humidity_sensor="HuAvg1",
    has_dual_sensors=True,
    individual_temp_sensors=["TempSp1", "TempSp2"],
    individual_co2_sensors=["Co2RA", "Co2RA2"],
    individual_humidity_sensors=["HuR1", "HuR2"],
)

AHU3 = AHUSensorConfig(
    equipment_id="Ahu3",
    screen_id="Screen 3",
    controller="OS01",
    device_id=2122753,
    space_temp_sensor="TrAvg",
    co2_sensor="Co2Avg",
    humidity_sensor="HuAvg1",
    has_dual_sensors=True,
    individual_temp_sensors=["TempSp1", "TempSp2"],
    individual_co2_sensors=["Co2RA", "Co2RA2"],
    individual_humidity_sensors=["HuR1", "HuR2"],
)

AHU4 = AHUSensorConfig(
    equipment_id="Ahu4",
    screen_id="Screen 4",
    controller="OS02",
    device_id=2122754,
    space_temp_sensor="TrAvg",
    co2_sensor="Co2Avg",
    humidity_sensor="HuAvg1",
    has_dual_sensors=True,
    individual_temp_sensors=["TempSp1", "TempSp2"],
    individual_co2_sensors=["Co2RA1", "Co2RA2"],  
    individual_humidity_sensors=["HuR2"],  
)

AHU5 = AHUSensorConfig(
    equipment_id="Ahu5",
    screen_id="Screen 5",
    controller="OS02",
    device_id=2122754,
    space_temp_sensor="TrAvg",
    co2_sensor="Co2Avg",
    humidity_sensor="HuAvg1",
    has_dual_sensors=True,
    individual_temp_sensors=["TempSp1", "TempSp2"],
    individual_co2_sensors=["Co2RA", "Co2RA2"],
    individual_humidity_sensors=["HuR1", "HuR2"],
)

# ── Group B: Single sensors (Ahu6–Ahu13) ──────────────────────────

AHU6 = AHUSensorConfig(
    equipment_id="Ahu6",
    screen_id="Screen 6",
    controller="OS04",
    device_id=2122756,
    space_temp_sensor="TempSp1",
    co2_sensor="Co2RA",
    humidity_sensor="HuR1",
)

AHU7 = AHUSensorConfig(
    equipment_id="Ahu7",
    screen_id="Screen 7",
    controller="OS04",
    device_id=2122756,
    space_temp_sensor="TempSp1",
    co2_sensor="Co2RA",
    humidity_sensor="HuR1",
)

AHU8 = AHUSensorConfig(
    equipment_id="Ahu8",
    screen_id="Screen 8",
    controller="OS04",
    device_id=2122756,
    space_temp_sensor="TempSp1",
    co2_sensor="Co2RA",
    humidity_sensor="HuR1",
)

AHU9 = AHUSensorConfig(
    equipment_id="Ahu9",
    screen_id="Screen 9",
    controller="OS05",
    device_id=2122757,
    space_temp_sensor="TempSp1",
    co2_sensor="Co2RA",
    humidity_sensor="HuR1",
)

AHU10 = AHUSensorConfig(
    equipment_id="Ahu10",
    screen_id="Screen 10",
    controller="OS05",
    device_id=2122757,
    space_temp_sensor="TempSp1",
    co2_sensor="Co2RA",
    humidity_sensor="HuR1",
)

AHU11 = AHUSensorConfig(
    equipment_id="Ahu11",
    screen_id="Screen 11",
    controller="OS05",
    device_id=2122757,
    space_temp_sensor="TempSp1",
    co2_sensor="Co2RA",
    humidity_sensor="HuR1",
)

AHU12 = AHUSensorConfig(
    equipment_id="Ahu12",
    screen_id="Screen 12",
    controller="OS05",
    device_id=2122757,
    space_temp_sensor="TempSp1",
    co2_sensor="Co2RA",
    humidity_sensor="HuR1",
)

AHU13 = AHUSensorConfig(
    equipment_id="Ahu13",
    screen_id="Screen 13",
    controller="OS04",
    device_id=2122756,
    space_temp_sensor="TempSp1",
    co2_sensor="Co2RA",
    humidity_sensor="HuR1",
)


AHU14 = AHUSensorConfig(
    equipment_id="Ahu14",
    screen_id="Screen 14",
    controller="OS02",
    device_id=2122754,
    space_temp_sensor="AvgTmp",        
    co2_sensor="AvgCo2",               # CO2 available from 2026-02-19
    humidity_sensor="AvgHu",            
)


AHU15 = AHUSensorConfig(
    equipment_id="Ahu15",
    screen_id="Screen 15",
    controller="OS02",
    device_id=2122754,
    space_temp_sensor="TempSp1",      
    co2_sensor="Co2RA",               
    humidity_sensor="HuR1",          
)

#  Group E: Ahu16 – uses TempSp (without '1')

AHU16 = AHUSensorConfig(
    equipment_id="Ahu16",
    screen_id="Screen 16",
    controller="OS02",
    device_id=2122754,
    space_temp_sensor="TempSp",       
    co2_sensor="Co2RA",               
    humidity_sensor="HuR1",
)


# =============================================================================
# LOOKUP REGISTRY
# =============================================================================

AHU_CONFIGS: Dict[str, AHUSensorConfig] = {
    "Ahu1":  AHU1,
    "Ahu2":  AHU2,
    "Ahu3":  AHU3,
    "Ahu4":  AHU4,
    "Ahu5":  AHU5,
    "Ahu6":  AHU6,
    "Ahu7":  AHU7,
    "Ahu8":  AHU8,
    "Ahu9":  AHU9,
    "Ahu10": AHU10,
    "Ahu11": AHU11,
    "Ahu12": AHU12,
    "Ahu13": AHU13,
    "Ahu14": AHU14,
    "Ahu15": AHU15,
    "Ahu16": AHU16,
}

ALL_AHU_IDS: List[str] = list(AHU_CONFIGS.keys())

OPTIMIZABLE_AHU_IDS: List[str] = [
    ahu_id for ahu_id, cfg in AHU_CONFIGS.items()
    if cfg.optimization_enabled
]


def get_ahu_config(equipment_id: str) -> AHUSensorConfig:
    """
    Look up the sensor configuration for a given AHU.

    Args:
        equipment_id: Equipment identifier, e.g. "Ahu1", "Ahu13".
                       Case-insensitive matching is attempted.

    Returns:
        AHUSensorConfig for the requested AHU.

    Raises:
        ValueError: If the AHU is not found in the registry.
    """
    cfg = AHU_CONFIGS.get(equipment_id)
    if cfg:
        return cfg

    key_lower = equipment_id.lower()
    for k, v in AHU_CONFIGS.items():
        if k.lower() == key_lower:
            return v

    available = ", ".join(AHU_CONFIGS.keys())
    raise ValueError(
        f"Unknown equipment_id '{equipment_id}'. "
        f"Available AHUs: {available}"
    )


def get_required_datapoints(equipment_id: str) -> List[str]:
    """
    Convenience: return the list of BACnet datapoint names to fetch for an AHU.
    """
    return get_ahu_config(equipment_id).required_datapoints


def get_sensor_rename_map(equipment_id: str) -> Dict[str, str]:
    """
    Convenience: return the raw->standardised column rename map for an AHU.
    """
    return get_ahu_config(equipment_id).sensor_rename_map


# =============================================================================
# QUICK REFERENCE (printed when running module directly)
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AHU SENSOR CONFIGURATION REFERENCE")
    print("=" * 80)
    for ahu_id, cfg in AHU_CONFIGS.items():
        rename = cfg.sensor_rename_map
        rename_str = str(rename) if rename else "(no renaming needed)"
        missing = cfg.missing_sensors
        missing_str = ", ".join(missing) if missing else "(none)"
        print(
            f"\n{ahu_id:6s}  │  controller={cfg.controller}  "
            f"device={cfg.device_id}"
        )
        print(
            f"        │  temp={cfg.space_temp_sensor:<10s}  "
            f"co2={str(cfg.co2_sensor):<10s}  "
            f"humidity={cfg.humidity_sensor:<10s}  "
            f"dual={cfg.has_dual_sensors}"
        )
        print(f"        │  rename:  {rename_str}")
        print(f"        │  missing: {missing_str}")
        print(f"        │  fetch:   {cfg.required_datapoints}")
    print()
