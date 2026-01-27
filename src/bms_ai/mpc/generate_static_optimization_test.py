"""
Static Optimization Test Data Generator for Cinema AHU MPC
============================================================

This script generates static test data with real movie schedule occupancy
and weather data, then runs MPC optimization for each timestep.

Features:
- Fetches historical weather data from Open-Meteo
- Uses real movie schedule for occupancy (with caching)
- Fetches actual sensor data from BMS API
- Runs MPC optimization for each row
- Outputs CSV with both UTC and Sharjah timestamps

Usage:
    python -m src.bms_ai.mpc.generate_static_optimization_test

Author: BMS-AI Team
Date: January 2026
"""

import os
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from src.bms_ai.logger_config import setup_logger
from src.bms_ai.mpc.mpc_training_pipeline import MPCTrainingPipeline, PipelineConfig
from src.bms_ai.mpc.cinema_ahu_mpc import OccupancyInfo
from src.bms_ai.utils.setpoint_optimization_utils import (
    fetch_movie_schedule,
    get_occupancy_status_for_timestamp,
    SHARJAH_OFFSET
)

log = setup_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class StaticTestConfig:
    """Configuration for static test data generation."""
    
    # Time range (UTC)
    START_DATE_UTC = "2026-01-15 07:00:00"
    END_DATE_UTC = "2026-01-27 00:00:00"
    
    # Sampling interval
    SAMPLE_INTERVAL_MINUTES = 10
    
    # AHU Configuration
    EQUIPMENT_ID = "Ahu13"
    SCREEN_NAME = "Screen 13"
    
    # API Configuration
    BUILDING_ID = "36c27828-d0b4-4f1e-8a94-d962d342e7c2"
    API_URL = "https://ikoncloud.keross.com/bms-express-server/data"
    
    # Weather API (Sharjah coordinates)
    LATITUDE = 25.33
    LONGITUDE = 55.39
    
    # Required datapoints
    REQUIRED_DATAPOINTS = [
        'SpTREff',   # Effective setpoint
        'FbVFD',     # Fan speed feedback
        'TempSu',    # Supply air temperature
        'TempSp1',   # Space air temperature
        'SpTROcc',   # Occupied setpoint
        'FbFAD',     # Fresh air damper feedback
        'Co2RA'      # CO2 level
    ]
    
    # Output paths
    OUTPUT_DIR = "artifacts/ahu13_mpc"
    OUTPUT_FILENAME = "static_optimization_test_with_schedule.csv"
    SCHEDULE_CACHE_FILENAME = "schedule_cache.json"
    
    # Model path
    MODEL_PATH = "artifacts/ahu13_mpc/mpc_model.joblib"


# =============================================================================
# SCHEDULE DATA CACHING
# =============================================================================

class ScheduleCache:
    """Manages caching of movie schedule data."""
    
    def __init__(self, cache_dir: str, cache_filename: str):
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, cache_filename)
        
    def load_cached_schedule(self) -> Optional[List[Dict[str, Any]]]:
        """Load schedule from cache if available."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    cached_data = json.load(f)
                    
                # Check cache age (max 24 hours)
                cached_at = cached_data.get('cached_at', '')
                if cached_at:
                    cache_time = datetime.fromisoformat(cached_at)
                    if datetime.now() - cache_time < timedelta(hours=24):
                        print(f"✓ Loaded schedule from cache ({self.cache_path})")
                        log.info(f"Loaded schedule from cache: {self.cache_path}")
                        return cached_data.get('schedule_data', [])
                        
            except Exception as e:
                log.warning(f"Failed to load cache: {e}")
        
        return None
    
    def save_schedule_to_cache(self, schedule_data: List[Dict[str, Any]]):
        """Save schedule data to cache."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        cache_data = {
            'cached_at': datetime.now().isoformat(),
            'schedule_data': schedule_data
        }
        
        with open(self.cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"✓ Schedule cached to {self.cache_path}")
        log.info(f"Schedule cached to {self.cache_path}")


# =============================================================================
# DATA FETCHING
# =============================================================================

class StaticTestDataFetcher:
    """Fetches all required data for static optimization test."""
    
    def __init__(self, config: StaticTestConfig):
        self.config = config
        self.schedule_cache = ScheduleCache(
            config.OUTPUT_DIR, 
            config.SCHEDULE_CACHE_FILENAME
        )
        
    def fetch_bms_data(self) -> pd.DataFrame:
        """Fetch actual sensor data from BMS API."""
        print("\n" + "=" * 60)
        print("STEP 1: FETCHING BMS SENSOR DATA")
        print("=" * 60)
        
        # Build query
        cleaned_id = self.config.BUILDING_ID.replace("-", "").lower()
        location_table_name = f"datapoint_live_monitoring_{cleaned_id}"
        
        datapoint_list = ', '.join([f"'{dp}'" for dp in self.config.REQUIRED_DATAPOINTS])
        
        query = f"""
        SELECT * FROM {location_table_name} 
        WHERE equipment_id = '{self.config.EQUIPMENT_ID}'
        AND datapoint IN ({datapoint_list})
        AND system_type = 'AHU'
        AND data_received_on >= '{self.config.START_DATE_UTC}'
        AND data_received_on <= '{self.config.END_DATE_UTC}'
        ALLOW FILTERING;
        """
        
        payload = {"query": query.strip()}
        
        print(f"Fetching data for {self.config.EQUIPMENT_ID}...")
        print(f"Date range: {self.config.START_DATE_UTC} to {self.config.END_DATE_UTC}")
        
        try:
            response = requests.post(self.config.API_URL, json=payload, timeout=120)
            response.raise_for_status()
            raw_response = response.json()
            
            if isinstance(raw_response, list):
                data_list = raw_response
            elif isinstance(raw_response, dict) and 'queryResponse' in raw_response:
                data_list = raw_response.get('queryResponse', [])
            else:
                raise ValueError("Invalid API response format")
            
            print(f"✓ Fetched {len(data_list)} raw records")
            log.info(f"Fetched {len(data_list)} raw records from BMS API")
            
            # Parse and pivot data
            df = self._parse_bms_data(data_list)
            return df
            
        except Exception as e:
            print(f"✗ Error fetching BMS data: {e}")
            log.error(f"Error fetching BMS data: {e}")
            raise
    
    def _parse_bms_data(self, data_list: List[Dict]) -> pd.DataFrame:
        """Parse raw BMS data into a pivoted DataFrame."""
        if not data_list:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        
        # Parse monitoring_data - can be string number or JSON
        if 'monitoring_data' in df.columns:
            def extract_value(monitoring_data):
                try:
                    if monitoring_data is None:
                        return 0.0
                    
                    # If it's already a number
                    if isinstance(monitoring_data, (int, float)):
                        return float(monitoring_data)
                    
                    # If it's a string
                    if isinstance(monitoring_data, str):
                        # Try direct float conversion first (e.g., '27')
                        try:
                            return float(monitoring_data)
                        except ValueError:
                            pass
                        
                        # Try JSON parsing (e.g., '{"value": 27}')
                        try:
                            data = json.loads(monitoring_data)
                            if isinstance(data, dict):
                                return float(data.get('value', 0))
                            return float(data)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    
                    return 0.0
                except Exception as e:
                    log.debug(f"Failed to parse monitoring_data: {monitoring_data}, error: {e}")
                    return 0.0
            
            df['value'] = df['monitoring_data'].apply(extract_value)
        else:
            df['value'] = 0.0
        
        # Force value to numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
        
        print(f"  Value stats: min={df['value'].min():.1f}, max={df['value'].max():.1f}, mean={df['value'].mean():.1f}")
        
        # Parse timestamps
        df['data_received_on'] = pd.to_datetime(df['data_received_on'], utc=True)
        
        # Filter to the requested time range (make comparison timestamps tz-aware)
        start_time = pd.to_datetime(self.config.START_DATE_UTC).tz_localize('UTC')
        end_time = pd.to_datetime(self.config.END_DATE_UTC).tz_localize('UTC')
        df = df[(df['data_received_on'] >= start_time) & (df['data_received_on'] <= end_time)]
        
        if len(df) == 0:
            print(f"  ✗ No data in requested range {start_time} to {end_time}")
            return pd.DataFrame()
        
        print(f"  Filtered to {len(df)} records in date range")
        
        # Sort by time
        df = df.sort_values('data_received_on')
        
        # Pivot to get datapoints as columns
        pivot_df = df.pivot_table(
            index='data_received_on',
            columns='datapoint',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        # Resample to 10-minute intervals
        pivot_df = pivot_df.set_index('data_received_on')
        pivot_df = pivot_df.resample(f'{self.config.SAMPLE_INTERVAL_MINUTES}min').mean()
        pivot_df = pivot_df.reset_index()
        
        # Forward fill missing values
        pivot_df = pivot_df.ffill().bfill()
        
        # Add lagged features
        if 'TempSp1' in pivot_df.columns:
            pivot_df['TempSp1_lag_10m'] = pivot_df['TempSp1'].shift(1).fillna(pivot_df['TempSp1'])
        if 'SpTREff' in pivot_df.columns:
            pivot_df['SpTREff_lag_10m'] = pivot_df['SpTREff'].shift(1).fillna(pivot_df['SpTREff'])
        
        print(f"✓ Parsed into {len(pivot_df)} rows with {len(pivot_df.columns)} columns")
        return pivot_df
    
    def fetch_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch historical weather data from Open-Meteo."""
        print("\n" + "=" * 60)
        print("STEP 2: FETCHING WEATHER DATA")
        print("=" * 60)
        
        df = df.copy()
        
        # Ensure data_received_on is timezone-naive for merge
        if df['data_received_on'].dt.tz is not None:
            df['data_received_on'] = df['data_received_on'].dt.tz_localize(None)
        
        start_date = df['data_received_on'].min().strftime('%Y-%m-%d')
        end_date = df['data_received_on'].max().strftime('%Y-%m-%d')
        
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.config.LATITUDE,
            "longitude": self.config.LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m",
            "timezone": "UTC"  # Use UTC to match BMS data
        }
        
        try:
            print(f"Fetching weather data from {start_date} to {end_date}...")
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            weather_df = pd.DataFrame({
                'ds_weather': pd.to_datetime(data['hourly']['time']),
                'outside_temp': data['hourly']['temperature_2m'],
                'outside_humidity': data['hourly']['relative_humidity_2m']
            })
            
            # Ensure weather timestamps are also timezone-naive
            if weather_df['ds_weather'].dt.tz is not None:
                weather_df['ds_weather'] = weather_df['ds_weather'].dt.tz_localize(None)
            
            # Merge using nearest timestamp
            df = df.sort_values('data_received_on')
            combined_df = pd.merge_asof(
                df,
                weather_df.sort_values('ds_weather'),
                left_on='data_received_on',
                right_on='ds_weather',
                direction='nearest'
            )
            
            print(f"✓ Added weather data ({len(weather_df)} hourly records)")
            print(f"  Temperature range: {weather_df['outside_temp'].min():.1f}°C to {weather_df['outside_temp'].max():.1f}°C")
            log.info("Added weather data successfully")
            return combined_df.drop(columns=['ds_weather'], errors='ignore')
            
        except Exception as e:
            print(f"✗ Weather fetch failed: {e}")
            log.warning(f"Weather fetch failed: {e}, using defaults")
            df['outside_temp'] = 25.0
            df['outside_humidity'] = 50.0
            return df
    
    def fetch_schedule_data(self) -> List[Dict[str, Any]]:
        """Fetch movie schedule data with caching."""
        print("\n" + "=" * 60)
        print("STEP 3: FETCHING MOVIE SCHEDULE DATA")
        print("=" * 60)
        
        # Try to load from cache first
        schedule_data = self.schedule_cache.load_cached_schedule()
        
        if schedule_data is None:
            print("Cache miss - fetching fresh schedule data...")
            schedule_data = fetch_movie_schedule()
            
            if schedule_data:
                self.schedule_cache.save_schedule_to_cache(schedule_data)
            else:
                print("✗ Failed to fetch schedule data")
                return []
        
        # Print schedule summary
        print(f"✓ Schedule data loaded with {len(schedule_data)} instances:")
        for idx, sched in enumerate(schedule_data):
            print(f"  [{idx}] {sched.get('cinema_name', 'Unknown')} - "
                  f"{sched.get('start_date', '')} to {sched.get('end_date', '')}")
        
        return schedule_data
    
    def add_occupancy_from_schedule(
        self, 
        df: pd.DataFrame, 
        schedule_data: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Add occupancy status from real movie schedule."""
        print("\n" + "=" * 60)
        print("STEP 4: ADDING OCCUPANCY FROM SCHEDULE")
        print("=" * 60)
        
        df = df.copy()
        
        # Initialize occupancy columns
        df['occupied'] = 0
        df['movie_name'] = None
        df['time_remaining_min'] = None
        df['time_until_next_min'] = None
        df['next_movie_name'] = None
        df['schedule_instance_used'] = None
        df['is_overnight'] = False
        df['is_precooling'] = False  # True if occupied due to pre-cooling (< 60 min to next movie)
        
        # Pre-cooling threshold: consider occupied if next movie starts within this many minutes
        PRECOOLING_THRESHOLD_MINUTES = 60
        
        if not schedule_data:
            print("✗ No schedule data available - all timestamps marked as unoccupied")
            return df
        
        total_rows = len(df)
        occupied_count = 0
        precooling_count = 0
        
        print(f"Processing {total_rows} timestamps for Screen 13...")
        
        for idx, row in df.iterrows():
            # data_received_on is in UTC (from Cassandra)
            timestamp_utc = row['data_received_on']
            
            # Ensure timezone awareness
            if timestamp_utc.tzinfo is None:
                timestamp_utc = timestamp_utc.replace(tzinfo=timezone.utc)
            
            # Get occupancy status using the schedule function
            # This automatically converts UTC to Sharjah and finds the right instance
            status = get_occupancy_status_for_timestamp(
                schedule_data=schedule_data,
                screen=self.config.SCREEN_NAME,
                timestamp_utc=timestamp_utc
            )
            
            if status.get('status') == 1:
                # Movie is currently showing - definitely occupied
                df.at[idx, 'occupied'] = 1
                df.at[idx, 'movie_name'] = status.get('movie_name')
                
                # time_remaining is now an integer (minutes)
                time_remaining = status.get('time_remaining')
                if time_remaining is not None:
                    df.at[idx, 'time_remaining_min'] = int(time_remaining)
                
                df.at[idx, 'schedule_instance_used'] = status.get('instance_used')
                df.at[idx, 'is_overnight'] = status.get('is_overnight_from_previous_schedule', False)
                occupied_count += 1
            else:
                # No movie currently showing - check if pre-cooling is needed
                # time_until_next_movie is now an integer (minutes)
                time_until_next = status.get('time_until_next_movie')
                next_movie = status.get('next_movie_name')
                
                if time_until_next is not None and isinstance(time_until_next, (int, float)):
                    df.at[idx, 'time_until_next_min'] = int(time_until_next)
                    df.at[idx, 'next_movie_name'] = next_movie
                    
                    # Pre-cooling logic: if next movie < 60 min, consider occupied
                    if time_until_next <= PRECOOLING_THRESHOLD_MINUTES:
                        df.at[idx, 'occupied'] = 1
                        df.at[idx, 'movie_name'] = f"[PRE-COOLING] {next_movie}" if next_movie else "[PRE-COOLING]"
                        df.at[idx, 'is_precooling'] = True
                        occupied_count += 1
                        precooling_count += 1
        
        occupancy_rate = (occupied_count / total_rows) * 100
        print(f"✓ Occupancy detection complete:")
        print(f"  Total timestamps: {total_rows}")
        print(f"  Occupied timestamps: {occupied_count} (includes {precooling_count} pre-cooling periods)")
        print(f"  Occupancy rate: {occupancy_rate:.1f}%")
        print(f"  Pre-cooling threshold: {PRECOOLING_THRESHOLD_MINUTES} minutes before movie start")
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Sharjah time and cyclic time features."""
        df = df.copy()
        
        # Add Sharjah time column
        df['timestamp_utc'] = df['data_received_on']
        df['timestamp_sharjah'] = df['data_received_on'].apply(
            lambda x: x.tz_localize('UTC').astimezone(SHARJAH_OFFSET).replace(tzinfo=None)
            if x.tzinfo is None else x.astimezone(SHARJAH_OFFSET).replace(tzinfo=None)
        )
        
        # Cyclic time encoding (based on Sharjah time for local patterns)
        df['hour_sharjah'] = df['timestamp_sharjah'].dt.hour
        df['day_of_week'] = df['timestamp_sharjah'].dt.dayofweek
        
        # Cyclic encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_sharjah'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_sharjah'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df


# =============================================================================
# MPC OPTIMIZATION RUNNER
# =============================================================================

class MPCOptimizationRunner:
    """Runs MPC optimization for each row in the dataset."""
    
    def __init__(self, config: StaticTestConfig):
        self.config = config
        self.pipeline = None
        self.mpc_system = None
        
    def load_model(self):
        """Load the trained MPC model."""
        print("\n" + "=" * 60)
        print("STEP 5: LOADING MPC MODEL")
        print("=" * 60)
        
        self.pipeline = MPCTrainingPipeline()
        
        try:
            self.mpc_system = self.pipeline.load_model(
                equipment_id=self.config.EQUIPMENT_ID
            )
            print(f"✓ Model loaded from {self.config.MODEL_PATH}")
            return True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            log.error(f"Failed to load MPC model: {e}")
            return False
    
    def run_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run MPC optimization for each row."""
        print("\n" + "=" * 60)
        print("STEP 6: RUNNING MPC OPTIMIZATION")
        print("=" * 60)
        
        if self.mpc_system is None:
            raise RuntimeError("MPC model not loaded")
        
        results = []
        total_rows = len(df)
        success_count = 0
        bypass_count = 0
        fail_count = 0
        
        print(f"Optimizing {total_rows} timestamps...")
        
        for idx, row in df.iterrows():
            try:
                # Extract current measurements
                current_measurements = {
                    'TempSp1': row.get('TempSp1', 24.0),
                    'TempSp1_lag_10m': row.get('TempSp1_lag_10m', 24.0),
                    'SpTREff': row.get('SpTREff', 24.0),
                    'SpTREff_lag_10m': row.get('SpTREff_lag_10m', row.get('SpTREff', 24.0)),
                    'TempSu': row.get('TempSu', 18.0),
                    'FbVFD': row.get('FbVFD', 50.0),
                    'FbFAD': row.get('FbFAD', 50.0),
                    'Co2RA': row.get('Co2RA', 600.0)
                }
                
                # Build occupancy response
                occupancy_status = int(row.get('occupied', 0))
                movie_name = row.get('movie_name')
                time_remaining = row.get('time_remaining_min')
                
                occupancy_response = {
                    'status': occupancy_status,
                    'movie_name': movie_name if occupancy_status == 1 else None,
                    'time_remaining': f"{time_remaining} minutes" if time_remaining else None,
                    'time_until_next_movie': 'No upcoming shows'
                }
                
                # Build weather forecast (constant for simplicity)
                weather_forecast = [{
                    'temperature': row.get('outside_temp', 25.0),
                    'humidity': row.get('outside_humidity', 50.0)
                }] * 6  # Prediction horizon
                
                # Get optimal setpoint
                timestamp = row.get('data_received_on', pd.Timestamp.now())
                result = self.mpc_system.get_optimal_setpoint(
                    current_measurements=current_measurements,
                    occupancy_api_response=occupancy_response,
                    weather_forecast=weather_forecast,
                    current_time=timestamp
                )
                
                # Track status
                if result.get('bypass_optimization'):
                    bypass_count += 1
                elif 'success' in result.get('optimization_status', ''):
                    success_count += 1
                else:
                    fail_count += 1
                
                # Store result
                results.append({
                    'timestamp_utc': row.get('timestamp_utc', row.get('data_received_on')),
                    'timestamp_sharjah': row.get('timestamp_sharjah'),
                    'actual_SpTREff': row.get('SpTREff', np.nan),
                    'optimized_SpTREff': result['optimal_setpoint'],
                    'setpoint_difference': result['optimal_setpoint'] - row.get('SpTREff', 0),
                    'actual_TempSp1': row.get('TempSp1', np.nan),
                    'target_temperature': result.get('target_temperature'),
                    'occupied': occupancy_status,
                    'movie_name': movie_name,
                    'mode': result['mode'],
                    'outside_temp': row.get('outside_temp', np.nan),
                    'outside_humidity': row.get('outside_humidity', np.nan),
                    'FbVFD': row.get('FbVFD', np.nan),
                    'FbFAD': row.get('FbFAD', np.nan),
                    'Co2RA': row.get('Co2RA', np.nan),
                    'co2_load_category': result.get('co2_load_category', 'unknown'),
                    'co2_load_factor': result.get('co2_load_factor', np.nan),
                    'is_overnight_movie': row.get('is_overnight', False),
                    'is_precooling': row.get('is_precooling', False),
                    'schedule_instance_used': row.get('schedule_instance_used'),
                    'optimization_status': result['optimization_status']
                })
                
                # Progress indicator
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{total_rows} rows...")
                    
            except Exception as e:
                log.warning(f"Failed to optimize row {idx}: {e}")
                fail_count += 1
                continue
        
        print(f"\n✓ Optimization complete:")
        print(f"  Success: {success_count}")
        print(f"  Bypassed (unoccupied): {bypass_count}")
        print(f"  Failed: {fail_count}")
        
        return pd.DataFrame(results)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def generate_static_optimization_test():
    """Main function to generate static optimization test data."""
    print("\n" + "=" * 70)
    print("  STATIC OPTIMIZATION TEST DATA GENERATOR")
    print("  Cinema AHU MPC with Real Movie Schedule")
    print("=" * 70)
    
    config = StaticTestConfig()
    
    # Step 1-4: Fetch and prepare data
    fetcher = StaticTestDataFetcher(config)
    
    # Fetch BMS sensor data
    df = fetcher.fetch_bms_data()
    
    if df.empty:
        print("✗ No BMS data available. Exiting.")
        return
    
    # Add weather data
    df = fetcher.fetch_weather_data(df)
    
    # Fetch and cache schedule data
    schedule_data = fetcher.fetch_schedule_data()
    
    # Add occupancy from real schedule
    df = fetcher.add_occupancy_from_schedule(df, schedule_data)
    
    # Add time features
    df = fetcher.add_time_features(df)
    
    # Step 5-6: Run MPC optimization
    optimizer = MPCOptimizationRunner(config)
    
    if not optimizer.load_model():
        print("✗ Cannot proceed without MPC model. Exiting.")
        return
    
    results_df = optimizer.run_optimization(df)
    
    # Save results
    print("\n" + "=" * 60)
    print("STEP 7: SAVING RESULTS")
    print("=" * 60)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILENAME)
    
    results_df.to_csv(output_path, index=False)
    print(f"✓ Results saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("  SUMMARY STATISTICS")
    print("=" * 70)
    
    avg_diff = results_df['setpoint_difference'].mean()
    std_diff = results_df['setpoint_difference'].std()
    
    occupied_results = results_df[results_df['occupied'] == 1]
    unoccupied_results = results_df[results_df['occupied'] == 0]
    
    success_rate = (results_df['optimization_status'].str.contains('success', na=False).sum() / 
                   len(results_df[results_df['occupied'] == 1])) * 100 if len(occupied_results) > 0 else 0
    
    print(f"\nTotal samples: {len(results_df)}")
    print(f"  - Occupied: {len(occupied_results)} ({len(occupied_results)/len(results_df)*100:.1f}%)")
    print(f"  - Unoccupied: {len(unoccupied_results)} ({len(unoccupied_results)/len(results_df)*100:.1f}%)")
    print(f"\nSetpoint Difference (Optimized - Actual):")
    print(f"  - Mean: {avg_diff:.3f}°C")
    print(f"  - Std: {std_diff:.3f}°C")
    print(f"\nOptimization Success Rate (occupied only): {success_rate:.1f}%")
    
    # Movie breakdown
    movies = results_df[results_df['movie_name'].notna()]['movie_name'].value_counts()
    print(f"\nMovies detected:")
    for movie, count in movies.head(10).items():
        print(f"  - {movie}: {count} timestamps")
    
    print("\n" + "=" * 70)
    print("  STATIC OPTIMIZATION TEST COMPLETE")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    generate_static_optimization_test()
