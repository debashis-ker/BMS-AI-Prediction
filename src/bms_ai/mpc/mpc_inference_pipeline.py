"""
MPC Inference Pipeline for Cinema AHU
======================================

This module provides real-time MPC inference for cinema AHU optimization.
It handles:
1. Fetching live sensor data from BMS API (last 10 minutes, resampled)
2. Fetching lag values from Cassandra (previous optimization)
3. Fetching current weather from Open-Meteo
4. Fetching occupancy from movie schedule API
5. Running MPC optimization
6. Saving results to Cassandra
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from cassandra.cluster import Session

from src.bms_ai.logger_config import setup_logger
from src.bms_ai.mpc.mpc_training_pipeline import MPCTrainingPipeline
from src.bms_ai.mpc.cinema_ahu_mpc import CinemaAHUMPCSystem, MPCConfig
from src.bms_ai.utils.setpoint_optimization_utils import (
    fetch_movie_schedule,
    get_current_movie_occupancy_status,
    SHARJAH_OFFSET
)

log = setup_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class InferenceConfig:
    """Configuration for MPC inference pipeline."""
    
    # AHU Configuration
    equipment_id: str = "Ahu13"
    screen_id: str = "Screen 13"
    
   
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2"
    
    # API Configuration
    bms_api_url: str = "https://ikoncloud.keross.com/bms-express-server/data"
    weather_api_url: str = "https://api.open-meteo.com/v1/forecast"
    
    # Location (Sharjah)
    latitude: float = 25.34
    longitude: float = 55.41
    
    lookback_minutes: int = 10
    lag_timeout_minutes: int = 20  # If last optimization is older, use current as lag
    
    required_datapoints: List[str] = None
    
    model_dir: str = "artifacts"
    
    table_name: str = "mpc_optimization_results_36c27828d0b44f1e8a94d962d342e7c2"
    
    def __post_init__(self):
        if self.required_datapoints is None:
            self.required_datapoints = [
                'SpTREff',  
                'FbVFD',    
                'TempSu',    
                'TempSp1',   
                'FbFAD',     
                'Co2RA'    
            ]


# =============================================================================
# DATA FETCHERS
# =============================================================================

class BMSDataFetcher:
    """Fetches live sensor data from BMS API."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        
    def fetch_last_10_minutes(self, equipment_id: str) -> Dict[str, float]:
        """
        Fetch sensor data from last 10 minutes and resample to single values.
        
        Args:
            equipment_id: Equipment identifier (e.g., 'Ahu13')
            
        Returns:
            Dict with resampled sensor values
        """
        now_utc = datetime.now(timezone.utc)
        from_time = now_utc - timedelta(minutes=self.config.lookback_minutes)
        
        from_str = from_time.strftime("%Y-%m-%dT%H:%M:%S UTC")
        
        cleaned_building_id = self.config.building_id.replace("-", "")
        table_name = f"datapoint_live_monitoring_{cleaned_building_id}"
        
        datapoints_str = ",".join([f"'{dp}'" for dp in self.config.required_datapoints])
        
        query = f"""
            SELECT * FROM {table_name} 
            WHERE equipment_id = '{equipment_id}' 
            AND datapoint IN ({datapoints_str}) 
            AND system_type = 'AHU' 
            AND data_received_on >= '{from_str}' 
            ALLOW FILTERING;
        """
        
        payload = {"query": query.strip()}
        
        log.info(f"[BMSFetcher] Fetching data for {equipment_id} from last {self.config.lookback_minutes} minutes")
        log.debug(f"[BMSFetcher] Query: {query}")
        
        try:
            response = requests.post(
                self.config.bms_api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if not data:
                log.warning(f"[BMSFetcher] No data returned for {equipment_id}")
                return None
            
            return self._parse_and_resample(data)
            
        except requests.exceptions.RequestException as e:
            log.error(f"[BMSFetcher] API request failed: {e}")
            return None
        except Exception as e:
            log.error(f"[BMSFetcher] Error processing data: {e}")
            return None
    
    def _parse_and_resample(self, data_list: List[Dict]) -> Dict[str, float]:
        """Parse raw API data and resample to single values."""
        
        if not data_list:
            return None
        
        records = []
        for item in data_list:
            try:
                timestamp_str = item.get('data_received_on', '')
                timestamp = pd.to_datetime(timestamp_str.replace(' UTC', ''), utc=True)
                
                datapoint = item.get('datapoint', '')
                value = float(item.get('monitoring_data', 0))
                
                records.append({
                    'timestamp': timestamp,
                    'datapoint': datapoint,
                    'value': value
                })
            except Exception as e:
                log.debug(f"[BMSFetcher] Skipping invalid record: {e}")
                continue
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        
        df_pivot = df.pivot_table(
            index='timestamp',
            columns='datapoint',
            values='value',
            aggfunc='mean'
        )
        
        result = {}
        for col in df_pivot.columns:
            result[col] = df_pivot[col].mean()
        
        log.info(f"[BMSFetcher] Resampled {len(records)} records to {len(result)} datapoints")
        log.debug(f"[BMSFetcher] Values: {result}")
        
        return result


class WeatherFetcher:
    """Fetches current weather from Open-Meteo API."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        
    def fetch_current_weather(self) -> Dict[str, float]:
        """
        Fetch current weather (temperature, humidity) from Open-Meteo.
        
        Returns:
            Dict with 'temperature' and 'humidity' keys
        """
        params = {
            "latitude": self.config.latitude,
            "longitude": self.config.longitude,
            "current": "temperature_2m,relative_humidity_2m",
            "timezone": "GMT"
        }
        
        log.info("[WeatherFetcher] Fetching current weather from Open-Meteo")
        
        try:
            response = requests.get(
                self.config.weather_api_url,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            current = data.get('current', {})
            
            result = {
                'temperature': current.get('temperature_2m', 35.0),
                'humidity': current.get('relative_humidity_2m', 50.0)
            }
            
            log.info(f"[WeatherFetcher] Current weather: {result['temperature']}°C, {result['humidity']}% RH")
            return result
            
        except Exception as e:
            log.error(f"[WeatherFetcher] Failed to fetch weather: {e}")
            return None


class OccupancyFetcher:
    """Fetches movie schedule and occupancy status."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._schedule_cache = None
        self._cache_time = None
        self._cache_ttl_minutes = 30  
        
    def fetch_occupancy_status(
        self, 
        screen_id: str, 
        ticket: str,
        ticket_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch current occupancy status for a screen.
        
        Args:
            screen_id: Screen identifier (e.g., 'Screen 13')
            ticket: Ticket for API authentication
            ticket_type: Optional ticket type (e.g., 'jobUser' to set User-Agent header)
            
        Returns:
            Dict with occupancy status, movie name, time remaining, etc.
        """
        log.info(f"[OccupancyFetcher] Fetching occupancy for {screen_id}")
        
        now = datetime.now()
        if (self._schedule_cache is not None and 
            self._cache_time is not None and
            (now - self._cache_time).total_seconds() < self._cache_ttl_minutes * 60):
            schedule_data = self._schedule_cache
            log.debug("[OccupancyFetcher] Using cached schedule data")
        else:
            schedule_data = fetch_movie_schedule(ticket=ticket, ticket_type=ticket_type)
            if schedule_data:
                self._schedule_cache = schedule_data
                self._cache_time = now
                log.info(f"[OccupancyFetcher] Fetched {len(schedule_data)} schedule instances")
        
        if not schedule_data:
            log.error("[OccupancyFetcher] Failed to fetch movie schedule")
            return None
        
        status = get_current_movie_occupancy_status(
            schedule_data=schedule_data,
            screens=[screen_id],
            for_which_time=0, 
            instance_index=0  
        )
        
        if screen_id not in status:
            log.warning(f"[OccupancyFetcher] No status found for {screen_id}")
            return None
        
        screen_status = status[screen_id]
        log.info(f"[OccupancyFetcher] Status for {screen_id}: {screen_status}")
        
        return screen_status


class CassandraDataHandler:
    """Handles Cassandra read/write operations for MPC results."""
    
    def __init__(self, config: InferenceConfig, session: Session):
        self.config = config
        self.session = session
        
    def get_last_optimization(self, equipment_id: str, timeout_minutes: Optional[int] = None, only_successful: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the most recent optimization result from Cassandra.
        
        Args:
            equipment_id: Equipment identifier
            timeout_minutes: Custom timeout in minutes (uses config default if None)
            only_successful: If True, only return records where optimization_status is 'success' or 'bypassed' (default: True)
            
        Returns:
            Dict with last optimization data (includes 'age_minutes' field), or None if not found/too old
        """
        effective_timeout = timeout_minutes if timeout_minutes is not None else self.config.lag_timeout_minutes
        
        success_filter = "AND optimization_status IN ('success', 'bypassed')" if only_successful else ""
        
        query = f"""
            SELECT * FROM {self.config.table_name}
            WHERE equipment_id = '{equipment_id}' {success_filter}
            LIMIT 1;
        """
        
        log.debug(f"[CassandraHandler] Fetching last optimization for {equipment_id} (timeout: {effective_timeout} min)")
        
        try:
            rows = self.session.execute(query)
            row = rows.one()
            
            if not row:
                log.info(f"[CassandraHandler] No previous optimization found for {equipment_id}")
                return None
            
            result = {col: getattr(row, col, None) for col in row._fields}
            
            timestamp_utc = result.get('timestamp_utc')
            if timestamp_utc:
                now_utc = datetime.now(timezone.utc)
                if timestamp_utc.tzinfo is None:
                    timestamp_utc = timestamp_utc.replace(tzinfo=timezone.utc)
                
                age_minutes = (now_utc - timestamp_utc).total_seconds() / 60
                result['age_minutes'] = age_minutes  
                
                if age_minutes > effective_timeout:
                    log.info(f"[CassandraHandler] Last optimization is {age_minutes:.1f} min old (> {effective_timeout} min timeout)")
                    return None
                
                log.info(f"[CassandraHandler] Found optimization from {age_minutes:.1f} min ago")
            
            return result
            
        except Exception as e:
            log.error(f"[CassandraHandler] Failed to fetch last optimization: {e}")
            return None
    
    def save_optimization_result(self, result: Dict[str, Any]) -> bool:
        """
        Save optimization result to Cassandra.
        
        Args:
            result: Dict with all optimization data
            
        Returns:
            True if successful, False otherwise
        """
        columns = [
            'equipment_id', 'timestamp_utc', 'optimized_setpoint', 'actual_sptreff',
            'actual_tempsp1', 'target_temperature', 'setpoint_difference',
            'occupied', 'movie_name', 'mode', 'is_precooling', 'time_until_next_movie',
            'outside_temp', 'outside_humidity', 'fb_vfd', 'fb_fad', 'co2_ra',
            'co2_load_category', 'co2_load_factor', 'optimization_status',
            'objective_value', 'used_features', 'next_tempsp1_lag', 'next_sptreff_lag',
            'previous_setpoint', 'screen_id', 'timestamp_sharjah'
        ]
        
        values = []
        for col in columns:
            val = result.get(col)
            if val is None:
                values.append('NULL')
            elif isinstance(val, str):
                val = val.replace("'", "''")
                values.append(f"'{val}'")
            elif isinstance(val, bool):
                values.append('true' if val else 'false')
            elif isinstance(val, datetime):
                values.append(f"'{val.strftime('%Y-%m-%d %H:%M:%S')}'")
            elif isinstance(val, (int, float)):
                if np.isnan(val) if isinstance(val, float) else False:
                    values.append('NULL')
                else:
                    values.append(str(val))
            else:
                values.append(f"'{str(val)}'")
        
        query = f"""
            INSERT INTO {self.config.table_name}
            ({', '.join(columns)})
            VALUES ({', '.join(values)});
        """
        
        log.debug(f"[CassandraHandler] Saving optimization result")
        
        try:
            self.session.execute(query)
            log.info(f"[CassandraHandler] Saved optimization result for {result.get('equipment_id')}")
            return True
        except Exception as e:
            log.error(f"[CassandraHandler] Failed to save optimization result: {e}")
            return False


# =============================================================================
# MPC INFERENCE PIPELINE
# =============================================================================

class MPCInferencePipeline:
    """
    Complete MPC inference pipeline for real-time optimization.
    
    This class orchestrates:
    1. Fetching live sensor data from BMS
    2. Fetching lag values from Cassandra
    3. Fetching current weather
    4. Fetching occupancy status
    5. Running MPC optimization
    6. Saving results to Cassandra
    """
    
    _model_cache: Dict[str, CinemaAHUMPCSystem] = {}
    
    def __init__(
        self,
        cassandra_session: Session,
        config: Optional[InferenceConfig] = None
    ):
        self.config = config or InferenceConfig()
        self.session = cassandra_session
        
        self.bms_fetcher = BMSDataFetcher(self.config)
        self.weather_fetcher = WeatherFetcher(self.config)
        self.occupancy_fetcher = OccupancyFetcher(self.config)
        self.cassandra_handler = CassandraDataHandler(self.config, self.session)
        
    @classmethod
    def load_model_on_startup(cls, equipment_id: str = "Ahu13") -> bool:
        """
        Load MPC model into class-level cache. Call this at app startup.
        
        Args:
            equipment_id: Equipment identifier
            
        Returns:
            True if model loaded successfully
        """
        log.info(f"[MPCInference] Loading MPC model for {equipment_id} at startup...")
        
        try:
            from src.bms_ai.mpc.mpc_training_pipeline import MPCTrainingPipeline
            
            pipeline = MPCTrainingPipeline()
            mpc_system = pipeline.load_model(equipment_id=equipment_id)
            cls._model_cache[equipment_id] = mpc_system
            log.info(f"[MPCInference] Model for {equipment_id} loaded successfully")
            return True
        except FileNotFoundError as e:
            log.error(f"[MPCInference] Model file not found for {equipment_id}: {e}")
            log.error(f"[MPCInference] Expected path: artifacts/{equipment_id.lower()}/mpc_model.joblib")
            return False
        except Exception as e:
            log.error(f"[MPCInference] Failed to load model for {equipment_id}: {e}")
            import traceback
            log.error(f"[MPCInference] Traceback: {traceback.format_exc()}")
            return False
    
    def get_model(self, equipment_id: str) -> Optional[CinemaAHUMPCSystem]:
        """Get model from cache, loading if necessary."""
        if equipment_id not in self._model_cache:
            if not self.load_model_on_startup(equipment_id):
                return None
        return self._model_cache.get(equipment_id)
    
    def _create_fail_record(
        self,
        equipment_id: str,
        screen_id: str,
        now_utc: datetime,
        now_sharjah: datetime,
        reason: str,
        sensor_data: Optional[Dict] = None,
        weather: Optional[Dict] = None,
        occupancy: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a record for failed optimization to save to Cassandra."""
        return {
            'equipment_id': equipment_id,
            'timestamp_utc': now_utc.replace(tzinfo=None),
            'timestamp_sharjah': now_sharjah.strftime("%Y-%m-%d %H:%M:%S"),
            'optimized_setpoint': None,
            'actual_sptreff': sensor_data.get('SpTREff') if sensor_data else None,
            'actual_tempsp1': sensor_data.get('TempSp1') if sensor_data else None,
            'target_temperature': None,
            'setpoint_difference': None,
            'occupied': occupancy.get('status') if occupancy else None,
            'movie_name': occupancy.get('movie_name') if occupancy else None,
            'mode': None,
            'is_precooling': None,
            'time_until_next_movie': occupancy.get('time_until_next_movie') if occupancy and isinstance(occupancy.get('time_until_next_movie'), int) else None,
            'outside_temp': weather.get('temperature') if weather else None,
            'outside_humidity': weather.get('humidity') if weather else None,
            'fb_vfd': sensor_data.get('FbVFD') if sensor_data else None,
            'fb_fad': sensor_data.get('FbFAD') if sensor_data else None,
            'co2_ra': sensor_data.get('Co2RA') if sensor_data else None,
            'co2_load_category': None,
            'co2_load_factor': None,
            'optimization_status': f'fail: {reason}',
            'objective_value': None,
            'used_features': None,
            'next_tempsp1_lag': sensor_data.get('TempSp1') if sensor_data else None,
            'next_sptreff_lag': None,
            'previous_setpoint': None,
            'screen_id': screen_id
        }
    
    def run_inference(
        self,
        equipment_id: str = "Ahu13",
        screen_id: str = "Screen 13",
        ticket: str = "",
        building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2",
        ticket_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete MPC inference pipeline.
        
        Args:
            equipment_id: AHU identifier
            screen_id: Screen identifier for occupancy lookup
            ticket: Ticket for movie schedule API
            building_id: Building identifier
            ticket_type: Optional ticket type (e.g., 'jobUser' to set User-Agent header)
            
        Returns:
            Dict with optimization result and all saved data
        """
        log.info(f"[MPCInference] Starting inference for {equipment_id} / {screen_id}")
        
        if building_id:
            self.config.building_id = building_id
        
        now_utc = datetime.now(timezone.utc)
        now_sharjah = now_utc.astimezone(SHARJAH_OFFSET)
        
        mpc_system = self.get_model(equipment_id)
        if mpc_system is None:
            fail_record = self._create_fail_record(
                equipment_id=equipment_id,
                screen_id=screen_id,
                now_utc=now_utc,
                now_sharjah=now_sharjah,
                reason='MODEL_NOT_LOADED'
            )
            self.cassandra_handler.save_optimization_result(fail_record)
            return {
                'success': False,
                'error': f'MPC model not loaded for {equipment_id}',
                'error_type': 'MODEL_NOT_LOADED',
                'saved_to_cassandra': True
            }
        
        sensor_data = self.bms_fetcher.fetch_last_10_minutes(equipment_id)
        if sensor_data is None:
            fail_record = self._create_fail_record(
                equipment_id=equipment_id,
                screen_id=screen_id,
                now_utc=now_utc,
                now_sharjah=now_sharjah,
                reason='SENSOR_DATA_FETCH_FAILED'
            )
            self.cassandra_handler.save_optimization_result(fail_record)
            return {
                'success': False,
                'error': f'Failed to fetch sensor data for {equipment_id}',
                'error_type': 'SENSOR_DATA_FETCH_FAILED',
                'saved_to_cassandra': True
            }
        
        log.debug(f"[MPCInference] Sensor data (10-min average) for {equipment_id}: {sensor_data}")
        
        weather = self.weather_fetcher.fetch_current_weather()
        if weather is None:
            log.error("[MPCInference] Weather data unavailable - cannot proceed with optimization")
            fail_record = self._create_fail_record(
                equipment_id=equipment_id,
                screen_id=screen_id,
                now_utc=now_utc,
                now_sharjah=now_sharjah,
                reason='WEATHER_FETCH_FAILED',
                sensor_data=sensor_data
            )
            self.cassandra_handler.save_optimization_result(fail_record)
            return {
                'success': False,
                'error': 'Failed to fetch weather data. Cannot proceed without weather information.',
                'error_type': 'WEATHER_FETCH_FAILED',
                'saved_to_cassandra': True
            }
        
        occupancy = self.occupancy_fetcher.fetch_occupancy_status(screen_id, ticket, ticket_type)
        if occupancy is None or occupancy == {}:
            log.error(f"[MPCInference] Schedule data unavailable for {screen_id} - cannot proceed")
            fail_record = self._create_fail_record(
                equipment_id=equipment_id,
                screen_id=screen_id,
                now_utc=now_utc,
                now_sharjah=now_sharjah,
                reason='SCHEDULE_DATA_UNAVAILABLE',
                sensor_data=sensor_data,
                weather=weather
            )
            self.cassandra_handler.save_optimization_result(fail_record)
            return {
                'success': False,
                'error': f'Failed to fetch schedule/occupancy status for {screen_id}. No valid schedule found for current date.',
                'error_type': 'SCHEDULE_DATA_UNAVAILABLE',
                'saved_to_cassandra': True
            }
        
        last_optimization = self.cassandra_handler.get_last_optimization(equipment_id, timeout_minutes=10)
        cassandra_fallback_used = False
        
        if last_optimization is None:
            log.info("[MPCInference] No Cassandra data within 10 min, trying 30 min fallback...")
            last_optimization = self.cassandra_handler.get_last_optimization(equipment_id, timeout_minutes=30)
            if last_optimization:
                cassandra_fallback_used = True
                log.info(f"[MPCInference] Found Cassandra data within 30 min (age: {last_optimization.get('age_minutes', 'unknown'):.1f} min)")
        
        log.debug(f"[MPCInference] Last optimization from Cassandra: {last_optimization is not None}, fallback_used: {cassandra_fallback_used}")
        
        using_sensor_as_lag = False
        
        if last_optimization:
            tempsp1_lag = last_optimization.get('next_tempsp1_lag', sensor_data.get('TempSp1', 24.0))
            sptreff_lag = last_optimization.get('next_sptreff_lag', sensor_data.get('SpTREff', 24.0))
            previous_setpoint = last_optimization.get('previous_setpoint')
            log.debug(f"[MPCInference] Using lag from Cassandra - TempSp1_lag: {tempsp1_lag}, SpTREff_lag: {sptreff_lag}, previous_setpoint: {previous_setpoint}")
        else:
            current_tempsp1 = sensor_data.get('TempSp1')
            current_sptreff = sensor_data.get('SpTREff')
            
            if current_tempsp1 is not None and current_sptreff is not None:
                tempsp1_lag = current_tempsp1
                sptreff_lag = current_sptreff
                previous_setpoint = current_sptreff
                using_sensor_as_lag = True
                log.info(f"[MPCInference] No Cassandra data - using current sensor data as lag: TempSp1_lag={tempsp1_lag}, SpTREff_lag={sptreff_lag}")
            else:
                is_occupied = occupancy.get('status') == 1
                time_until_next = occupancy.get('time_until_next_movie')
                is_precooling = False
                if not is_occupied and isinstance(time_until_next, (int, float)) and time_until_next < 60:
                    is_precooling = True
                
                default_setpoint = 24.0 if (is_occupied or is_precooling) else 27.0
                mode = 'occupied' if is_occupied else ('pre_cooling' if is_precooling else 'unoccupied')
                
                log.warning(f"[MPCInference] No sensor lag data (TempSp1/SpTREff) - returning default setpoint {default_setpoint}°C (mode: {mode})")
                
                fail_record = {
                    'equipment_id': equipment_id,
                    'timestamp_utc': now_utc.replace(tzinfo=None),
                    'timestamp_sharjah': now_sharjah.strftime("%Y-%m-%d %H:%M:%S"),
                    'optimized_setpoint': default_setpoint,
                    'actual_sptreff': sensor_data.get('SpTREff'),
                    'actual_tempsp1': sensor_data.get('TempSp1'),
                    'target_temperature': 23.5 if (is_occupied or is_precooling) else 27.0,
                    'setpoint_difference': None,
                    'occupied': 1 if (is_occupied or is_precooling) else 0,
                    'movie_name': occupancy.get('movie_name'),
                    'mode': mode,
                    'is_precooling': is_precooling,
                    'time_until_next_movie': time_until_next if isinstance(time_until_next, int) else None,
                    'outside_temp': weather['temperature'],
                    'outside_humidity': weather['humidity'],
                    'fb_vfd': sensor_data.get('FbVFD'),
                    'fb_fad': sensor_data.get('FbFAD'),
                    'co2_ra': sensor_data.get('Co2RA'),
                    'co2_load_category': 'unknown',
                    'co2_load_factor': 0.0,
                    'optimization_status': 'fail: NO_SENSOR_LAG_DATA',
                    'objective_value': None,
                    'used_features': None,
                    'next_tempsp1_lag': None,
                    'next_sptreff_lag': default_setpoint,
                    'previous_setpoint': default_setpoint,
                    'screen_id': screen_id
                }
                self.cassandra_handler.save_optimization_result(fail_record)
                
                return {
                    'success': False,
                    'equipment_id': equipment_id,
                    'timestamp_utc': now_utc.isoformat(),
                    'timestamp_sharjah': now_sharjah.strftime("%Y-%m-%d %H:%M:%S"),
                    'optimized_setpoint': default_setpoint,
                    'actual_sptreff': sensor_data.get('SpTREff'),
                    'actual_tempsp1': sensor_data.get('TempSp1'),
                    'target_temperature': 23.5 if (is_occupied or is_precooling) else 27.0,
                    'setpoint_difference': None,
                    'occupied': 1 if (is_occupied or is_precooling) else 0,
                    'movie_name': occupancy.get('movie_name'),
                    'mode': mode,
                    'is_precooling': is_precooling,
                    'time_until_next_movie': time_until_next if isinstance(time_until_next, int) else None,
                    'outside_temp': weather['temperature'],
                    'outside_humidity': weather['humidity'],
                    'fb_vfd': sensor_data.get('FbVFD'),
                    'fb_fad': sensor_data.get('FbFAD'),
                    'co2_ra': sensor_data.get('Co2RA'),
                    'co2_load_category': 'unknown',
                    'co2_load_factor': 0.0,
                    'optimization_status': 'fail: NO_SENSOR_LAG_DATA',
                    'objective_value': None,
                    'used_features': None,
                    'next_tempsp1_lag': None,
                    'next_sptreff_lag': default_setpoint,
                    'previous_setpoint': default_setpoint,
                    'screen_id': screen_id,
                    'saved_to_cassandra': True,
                    'error': 'No sensor lag data (TempSp1/SpTREff) available',
                    'error_type': 'NO_SENSOR_LAG_DATA'
                }
        
        current_measurements = {
            'TempSp1': sensor_data.get('TempSp1', 24.0),
            'TempSp1_lag_10m': tempsp1_lag,
            'SpTREff': sensor_data.get('SpTREff', 24.0),
            'SpTREff_lag_10m': sptreff_lag,
            'TempSu': sensor_data.get('TempSu', 18.0),
            'FbVFD': sensor_data.get('FbVFD', 50.0),
            'FbFAD': sensor_data.get('FbFAD', 50.0),
            'Co2RA': sensor_data.get('Co2RA', 600.0)
        }
        
        log.debug(f"[MPCInference] Current measurements for MPC: {current_measurements}")
        log.debug(f"[MPCInference] Weather: temp={weather['temperature']}°C, humidity={weather['humidity']}%")
        log.debug(f"[MPCInference] Occupancy: status={occupancy.get('status')}, movie={occupancy.get('movie_name')}, time_until_next={occupancy.get('time_until_next_movie')}")
        
        occupancy_response = {
            'status': occupancy.get('status', 0),
            'movie_name': occupancy.get('movie_name'),
            'time_remaining': f"{occupancy.get('time_remaining')} minutes" if occupancy.get('time_remaining') else None,
            'time_until_next_movie': occupancy.get('time_until_next_movie'),
            'next_movie_name': occupancy.get('next_movie_name')
        }
        
        weather_forecast = [{
            'temperature': weather['temperature'],
            'humidity': weather['humidity']
        }] * mpc_system.config.prediction_horizon
        
        if previous_setpoint is not None:
            mpc_system.mpc.previous_setpoint = previous_setpoint
        
        try:
            mpc_result = mpc_system.get_optimal_setpoint(
                current_measurements=current_measurements,
                occupancy_api_response=occupancy_response,
                weather_forecast=weather_forecast,
                current_time=now_utc.replace(tzinfo=None)
            )
        except Exception as e:
            log.error(f"[MPCInference] MPC optimization failed: {e}")
            
            default_setpoint = 24.0 if (is_occupied or is_precooling) else 27.0
            fail_record = self._create_fail_record(
                equipment_id=equipment_id,
                screen_id=screen_id,
                now_utc=now_utc,
                now_sharjah=now_sharjah,
                reason=f'OPTIMIZATION_FAILED: {str(e)}',
                sensor_data=sensor_data,
                weather=weather,
                occupancy=occupancy
            )
            self.cassandra_handler.save_optimization_result(fail_record)
            
            return {
                'success': False,
                'equipment_id': equipment_id,
                'timestamp_utc': now_utc.isoformat(),
                'timestamp_sharjah': now_sharjah.strftime("%Y-%m-%d %H:%M:%S"),
                'optimized_setpoint': default_setpoint,
                'actual_sptreff': sensor_data.get('SpTREff'),
                'actual_tempsp1': sensor_data.get('TempSp1'),
                'optimization_status': f'fail: OPTIMIZATION_FAILED: {str(e)}',
                'saved_to_cassandra': True,
                'error': f'MPC optimization failed: {str(e)}',
                'error_type': 'OPTIMIZATION_FAILED'
            }
        
        is_precooling = False
        movie_name = occupancy.get('movie_name')
        if occupancy.get('status') == 0:
            time_until_next = occupancy.get('time_until_next_movie')
            if isinstance(time_until_next, (int, float)) and time_until_next < 60:
                is_precooling = True
                movie_name = f"[PRE-COOLING] {occupancy.get('next_movie_name', 'Unknown')}"
        
        used_features = {
            'TempSp1': current_measurements['TempSp1'],
            'TempSp1_lag_10m': current_measurements['TempSp1_lag_10m'],
            'SpTREff': current_measurements['SpTREff'],
            'SpTREff_lag_10m': current_measurements['SpTREff_lag_10m'],
            'outside_temp': weather['temperature'],
            'outside_humidity': weather['humidity'],
            'occupied': 1 if occupancy.get('status') == 1 or is_precooling else 0,
            'hour_sin': np.sin(2 * np.pi * now_sharjah.hour / 24),
            'hour_cos': np.cos(2 * np.pi * now_sharjah.hour / 24),
            'day_sin': np.sin(2 * np.pi * now_sharjah.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * now_sharjah.weekday() / 7),
            'FbVFD': current_measurements['FbVFD'],
            'FbFAD': current_measurements['FbFAD']
        }
        
        occupied = 1 if (occupancy.get('status') == 1 or is_precooling) else 0
        
        storage_record = {
            'equipment_id': equipment_id,
            'timestamp_utc': now_utc.replace(tzinfo=None),
            'optimized_setpoint': mpc_result['optimal_setpoint'],
            'actual_sptreff': sensor_data.get('SpTREff'),
            'actual_tempsp1': sensor_data.get('TempSp1'),
            'target_temperature': mpc_result.get('target_temperature', 23.5),
            'setpoint_difference': mpc_result['optimal_setpoint'] - sensor_data.get('SpTREff', 0),
            'occupied': occupied,
            'movie_name': movie_name,
            'mode': mpc_result['mode'],
            'is_precooling': is_precooling,
            'time_until_next_movie': occupancy.get('time_until_next_movie') if isinstance(occupancy.get('time_until_next_movie'), int) else None,
            'outside_temp': weather['temperature'],
            'outside_humidity': weather['humidity'],
            'fb_vfd': sensor_data.get('FbVFD'),
            'fb_fad': sensor_data.get('FbFAD'),
            'co2_ra': sensor_data.get('Co2RA'),
            'co2_load_category': mpc_result.get('co2_load_category', 'unknown'),
            'co2_load_factor': mpc_result.get('co2_load_factor', 0.0),
            'optimization_status': mpc_result['optimization_status'],
            'objective_value': mpc_result.get('objective_value'),
            'used_features': json.dumps(used_features),
            'next_tempsp1_lag': sensor_data.get('TempSp1'),
            'next_sptreff_lag': mpc_result['optimal_setpoint'],
            'previous_setpoint': mpc_result['optimal_setpoint'],
            'screen_id': screen_id,
            'timestamp_sharjah': now_sharjah.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        save_success = self.cassandra_handler.save_optimization_result(storage_record)
        
        if not save_success:
            log.warning("[MPCInference] Failed to save result to Cassandra, returning result anyway")
        
        response = {
            'success': True,
            'equipment_id': equipment_id,
            'timestamp_utc': now_utc.isoformat(),
            'timestamp_sharjah': storage_record['timestamp_sharjah'],
            'optimized_setpoint': storage_record['optimized_setpoint'],
            'actual_sptreff': storage_record['actual_sptreff'],
            'actual_tempsp1': storage_record['actual_tempsp1'],
            'target_temperature': storage_record['target_temperature'],
            'setpoint_difference': storage_record['setpoint_difference'],
            'occupied': storage_record['occupied'],
            'movie_name': storage_record['movie_name'],
            'mode': storage_record['mode'],
            'is_precooling': storage_record['is_precooling'],
            'time_until_next_movie': storage_record['time_until_next_movie'],
            'outside_temp': storage_record['outside_temp'],
            'outside_humidity': storage_record['outside_humidity'],
            'fb_vfd': storage_record['fb_vfd'],
            'fb_fad': storage_record['fb_fad'],
            'co2_ra': storage_record['co2_ra'],
            'co2_load_category': storage_record['co2_load_category'],
            'co2_load_factor': storage_record['co2_load_factor'],
            'optimization_status': storage_record['optimization_status'],
            'objective_value': storage_record['objective_value'],
            'used_features': used_features,
            'next_tempsp1_lag': storage_record['next_tempsp1_lag'],
            'next_sptreff_lag': storage_record['next_sptreff_lag'],
            'previous_setpoint': storage_record['previous_setpoint'],
            'screen_id': storage_record['screen_id'],
            'saved_to_cassandra': save_success
        }
        
        log.info(f"[MPCInference] Inference complete: optimal_setpoint={response['optimized_setpoint']}, mode={response['mode']}")
        
        return response


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_mpc_optimization(
    cassandra_session: Session,
    equipment_id: str = "Ahu13",
    screen_id: str = "Screen 13",
    ticket: str = "",
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2",
    ticket_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run MPC optimization.
    
    Args:
        cassandra_session: Cassandra session
        equipment_id: AHU identifier
        screen_id: Screen identifier
        ticket: Ticket for movie schedule API
        building_id: Building identifier
        ticket_type: Optional ticket type (e.g., 'jobUser' to set User-Agent header)
        
    Returns:
        Dict with optimization result
    """
    pipeline = MPCInferencePipeline(cassandra_session=cassandra_session)
    return pipeline.run_inference(
        equipment_id=equipment_id,
        screen_id=screen_id,
        ticket=ticket,
        building_id=building_id,
        ticket_type=ticket_type
    )
