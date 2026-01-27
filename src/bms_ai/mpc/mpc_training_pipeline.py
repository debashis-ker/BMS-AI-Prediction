"""
MPC Training Pipeline for Cinema AHU
=====================================

This module provides a complete end-to-end training pipeline that:
1. Fetches raw sensor data from the BMS API
2. Adds weather data from Open-Meteo
3. Preprocesses and normalizes the data
4. Trains the MPC thermal prediction model
5. Returns a ready-to-use MPC system for real-time optimization

Usage:
------
    from src.bms_ai.mpc.mpc_training_pipeline import MPCTrainingPipeline
    
    # Initialize and run full pipeline
    pipeline = MPCTrainingPipeline()
    mpc_system = pipeline.run_full_pipeline(
        equipment_id="Ahu13",
        from_date="2025-11-29 07:00:00",
        to_date="2026-01-27 07:00:00"
    )
    
    # Use for optimization
    result = mpc_system.get_optimal_setpoint(measurements, occupancy_response)

Author: BMS-AI Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import os
import joblib
from sklearn.model_selection import train_test_split

from .cinema_ahu_mpc import (
    CinemaAHUMPCSystem,
    MPCConfig,
    MPCState,
    OccupancyInfo
)
from ..logger_config import setup_logger

log = setup_logger(__name__)


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the MPC training pipeline."""
    
    # API Configuration
    building_id: str = "36c27828-d0b4-4f1e-8a94-d962d342e7c2"
    api_url: str = "https://ikoncloud.keross.com/bms-express-server/data"
    system_type: str = "AHU"
    
    # Weather API (Sharjah coordinates)
    latitude: float = 25.33
    longitude: float = 55.39
    
    # Required datapoints for MPC
    required_datapoints: List[str] = None
    
    # Preprocessing
    resample_interval: str = "10min"
    date_column: str = "data_received_on"
    
    # Occupancy detection threshold
    # If SpTREff < 26, system is considered occupied
    occupancy_threshold: float = 26.0
    
    # Model save path (folder per AHU)
    model_base_dir: str = "artifacts"
    model_filename: str = "mpc_model.joblib"
    
    # Train/Test Split
    test_size: float = 0.2              # 20% for testing
    validation_enabled: bool = True      # Enable train-test split
    
    def __post_init__(self):
        if self.required_datapoints is None:
            self.required_datapoints = [
                'SpTREff',   # Effective setpoint
                'FbVFD',     # Fan speed feedback
                'TempSu',    # Supply air temperature
                'TempSp1',   # Space air temperature
                'SpTROcc',   # Occupied setpoint
                'FbFAD',     # Fresh air damper feedback
                'Co2RA'      # CO2 level for occupancy load estimation
            ]


# =============================================================================
# SECTION 2: DATA FETCHING
# =============================================================================

class DataFetcher:
    """Handles fetching raw sensor data from BMS API."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    def fetch_data(
        self,
        equipment_id: str,
        from_date: str = "",
        to_date: str = "",
        zone: str = "",
        datapoints: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch raw sensor data from the BMS API.
        
        Args:
            equipment_id: AHU equipment identifier (e.g., "Ahu13")
            from_date: Start date string (e.g., "2025-11-29 07:00:00")
            to_date: End date string
            zone: Optional zone filter
            datapoints: List of datapoints to fetch (defaults to config)
            
        Returns:
            List of raw data records from API
        """
        datapoints = datapoints or self.config.required_datapoints
        
        # Build database table name
        cleaned_id = self.config.building_id.replace("-", "").lower()
        location_table_name = f"datapoint_live_monitoring_{cleaned_id}"
        
        # Build CQL query
        query = f"select * from {location_table_name} where "
        
        if equipment_id:
            query += f"equipment_id = '{equipment_id}' and "
        
        if datapoints:
            datapoint_list = ', '.join([f"'{f}'" for f in datapoints])
            query += f"datapoint IN ({datapoint_list}) and "
        
        if self.config.system_type:
            query += f"system_type = '{self.config.system_type}' and "
        
        if zone:
            query += f"zone = '{zone}' and "
        
        if from_date and to_date:
            query += f"data_received_on >= '{from_date}' and data_received_on <= '{to_date}' "
        else:
            # Default to last 30 days
            previous_30_days = (
                pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30)
            ).strftime('%Y-%m-%d %H:%M:%S.%f%z')
            current_time = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S.%f%z')
            query += f"data_received_on >= '{previous_30_days}' and "
            query += f"data_received_on <= '{current_time}'"
        
        query += " allow filtering;"
        
        # Execute API request
        payload = {"query": query}
        
        log.debug(f"[DataFetcher] Query: {query}")
        
        try:
            print(f"[DataFetcher] Fetching data for {equipment_id}...")
            response = requests.post(
                self.config.api_url, 
                json=payload, 
                timeout=60
            )
            response.raise_for_status()
            raw_response = response.json()
            
            # Parse response
            if isinstance(raw_response, list):
                data_list = raw_response
            elif isinstance(raw_response, dict) and 'queryResponse' in raw_response:
                data_list = raw_response.get('queryResponse', [])
            else:
                raise ValueError("Invalid API response format")
            
            print(f"[DataFetcher] Fetched {len(data_list)} raw records")
            return data_list
            
        except Exception as e:
            print(f"[DataFetcher] Error fetching data: {e}")
            raise


# =============================================================================
# SECTION 3: WEATHER DATA INTEGRATION
# =============================================================================

class WeatherDataIntegrator:
    """Integrates historical weather data from Open-Meteo API."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    def add_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add historical weather data (temperature, humidity) to the dataframe.
        
        Args:
            df: DataFrame with 'data_received_on' column
            
        Returns:
            DataFrame with added 'outside_temp' and 'outside_humidity' columns
        """
        df = df.copy()
        df['data_received_on'] = pd.to_datetime(df['data_received_on'])
        
        start_date = df['data_received_on'].min().strftime('%Y-%m-%d')
        end_date = df['data_received_on'].max().strftime('%Y-%m-%d')
        
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.config.latitude,
            "longitude": self.config.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m",
            "timezone": "auto"
        }
        
        try:
            print(f"[WeatherIntegrator] Fetching weather data from {start_date} to {end_date}...")
            log.info(f"Fetching weather data from {start_date} to {end_date}...")
            response = requests.get(base_url, params=params, timeout=30)
            data = response.json()
            
            weather_df = pd.DataFrame({
                'ds_weather': pd.to_datetime(data['hourly']['time']),
                'outside_temp': data['hourly']['temperature_2m'],
                'outside_humidity': data['hourly']['relative_humidity_2m']
            })
            
            # Merge using nearest timestamp
            df = df.sort_values('data_received_on')
            combined_df = pd.merge_asof(
                df,
                weather_df.sort_values('ds_weather'),
                left_on='data_received_on',
                right_on='ds_weather',
                direction='nearest'
            )
            
            print(f"[WeatherIntegrator] Added weather data successfully")
            log.info("Added weather data successfully")
            return combined_df.drop(columns=['ds_weather'])
            
        except Exception as e:
            print(f"[WeatherIntegrator] Weather fetch failed: {e}")
            print("[WeatherIntegrator] Using default values (0)")
            log.warning(f"Weather fetch failed: {e}")
            log.warning("Using default values (0) for weather data")
            df['outside_temp'] = 0
            df['outside_humidity'] = 0
            return df


# =============================================================================
# SECTION 4: DATA PREPROCESSOR
# =============================================================================

class DataPreprocessor:
    """Preprocesses raw data for MPC training."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.weather_integrator = WeatherDataIntegrator(config)
        
    def preprocess(
        self, 
        records: List[Dict[str, Any]],
        add_weather: bool = True
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline for MPC training data.
        
        Steps:
        1. Convert to DataFrame
        2. Parse monitoring data
        3. Add weather data
        4. Pivot datapoints to columns
        5. Resample to regular intervals
        6. Add time features (cyclic encoding)
        7. Add lagged features
        8. Add target variable
        9. Add occupancy indicator
        
        Args:
            records: Raw data records from API
            add_weather: Whether to fetch and add weather data
            
        Returns:
            Preprocessed DataFrame ready for MPC training
        """
        if not records:
            print("[Preprocessor] No records to process")
            log.warning("No records to process")
            return pd.DataFrame()
        
        print(f"[Preprocessor] Processing {len(records)} records...")
        log.info(f"Processing {len(records)} records...")
        
        # Step 1: Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Step 2: Parse monitoring_data column
        if 'monitoring_data' in df.columns:
            mapping = {'inactive': 0.0, 'active': 1.0}
            df['monitoring_data'] = df['monitoring_data'].replace(mapping, regex=False)
            df['monitoring_data'] = pd.to_numeric(df['monitoring_data'], errors='coerce')
        
        # Step 3: Parse datetime
        date_col = self.config.date_column
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].dt.tz is not None:
            df[date_col] = df[date_col].dt.tz_localize(None)
        
        # Step 4: Add weather data
        if add_weather:
            df = self.weather_integrator.add_weather_data(df)
        else:
            df['outside_temp'] = 0
            df['outside_humidity'] = 0
        
        # Step 5: Pivot datapoints to columns
        aggregated = df.groupby([date_col, 'datapoint'])['monitoring_data'].agg('first')
        result_df = aggregated.unstack(level='datapoint').reset_index()
        
        # Merge weather back
        weather = df[[date_col, 'outside_temp', 'outside_humidity']].drop_duplicates()
        result_df = pd.merge(result_df, weather, on=date_col, how='left')
        
        # Step 6: Resample to regular intervals
        result_df = result_df.sort_values(date_col).set_index(date_col)
        result_df = result_df.resample(self.config.resample_interval).mean().ffill()
        
        # Step 7: Add time features (cyclic encoding)
        result_df['hour'] = result_df.index.hour
        result_df['day_of_week'] = result_df.index.dayofweek
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
        result_df['day_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['day_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        
        # Step 8: Add lagged features
        if 'TempSp1' in result_df.columns:
            result_df['Target_Temp'] = result_df['TempSp1'].shift(-1)
            result_df['TempSp1_lag_10m'] = result_df['TempSp1'].shift(1)
        
        if 'SpTREff' in result_df.columns:
            result_df['SpTREff_lag_10_min'] = result_df['SpTREff'].shift(1)
        
        # Step 9: Add occupancy indicator
        if 'SpTREff' in result_df.columns:
            result_df['occupied'] = np.where(
                result_df['SpTREff'] < self.config.occupancy_threshold, 
                1, 
                0
            )
        
        # Drop rows with NaN in critical columns
        required_cols = ['Target_Temp', 'TempSp1_lag_10m']
        available_required = [c for c in required_cols if c in result_df.columns]
        
        if available_required:
            result_df = result_df.dropna(subset=available_required)
        
        result_df = result_df.reset_index()
        
        print(f"[Preprocessor] Final dataset: {len(result_df)} samples")
        print(f"[Preprocessor] Columns: {list(result_df.columns)}")
        log.info(f"Final dataset: {len(result_df)} samples")
        log.debug(f"Columns: {list(result_df.columns)}")
        
        return result_df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate preprocessed data for MPC training.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Validation report dictionary
        """
        required_features = [
            'TempSp1', 'TempSp1_lag_10m', 'SpTREff', 'SpTREff_lag_10_min',
            'outside_temp', 'outside_humidity', 'occupied',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'FbVFD', 'FbFAD', 'Co2RA', 'Target_Temp'
        ]
        
        missing = [f for f in required_features if f not in df.columns]
        available = [f for f in required_features if f in df.columns]
        
        # Check for NaN
        nan_counts = df[available].isna().sum().to_dict()
        
        # Data statistics
        stats = df[available].describe().to_dict()
        
        report = {
            'total_samples': len(df),
            'required_features': required_features,
            'available_features': available,
            'missing_features': missing,
            'nan_counts': nan_counts,
            'is_valid': len(missing) == 0 and len(df) > 100,
            'date_range': {
                'start': str(df[self.config.date_column].min()),
                'end': str(df[self.config.date_column].max())
            }
        }
        
        return report


# =============================================================================
# SECTION 5: MAIN TRAINING PIPELINE
# =============================================================================

class MPCTrainingPipeline:
    """
    Complete end-to-end training pipeline for Cinema AHU MPC.
    
    This class orchestrates:
    1. Data fetching from BMS API
    2. Weather data integration
    3. Data preprocessing and normalization
    4. MPC model training
    5. Model saving/loading
    
    Example:
    --------
    >>> pipeline = MPCTrainingPipeline()
    >>> mpc_system = pipeline.run_full_pipeline(
    ...     equipment_id="Ahu13",
    ...     from_date="2025-11-29 07:00:00",
    ...     to_date="2026-01-27 07:00:00"
    ... )
    >>> 
    >>> # Use for optimization
    >>> result = mpc_system.get_optimal_setpoint(
    ...     current_measurements={'TempSp1': 24.5, 'SpTREff': 24.0, ...},
    ...     occupancy_api_response={'status': 1, 'movie_name': 'Avatar', ...}
    ... )
    """
    
    def __init__(
        self,
        pipeline_config: Optional[PipelineConfig] = None,
        mpc_config: Optional[MPCConfig] = None
    ):
        """
        Initialize the training pipeline.
        
        Args:
            pipeline_config: Configuration for data fetching/preprocessing
            mpc_config: Configuration for MPC controller
        """
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.mpc_config = mpc_config or MPCConfig()
        
        # Initialize components
        self.data_fetcher = DataFetcher(self.pipeline_config)
        self.preprocessor = DataPreprocessor(self.pipeline_config)
        
        # State
        self.raw_data: Optional[List[Dict]] = None
        self.preprocessed_data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.mpc_system: Optional[CinemaAHUMPCSystem] = None
        self.training_metrics: Optional[Dict] = None
        self.test_metrics: Optional[Dict] = None
        self.validation_report: Optional[Dict] = None
        self.equipment_id: Optional[str] = None  # Track which AHU we're training for
        
    def fetch_data(
        self,
        equipment_id: str,
        from_date: str = "",
        to_date: str = "",
        zone: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Step 1: Fetch raw data from BMS API.
        
        Args:
            equipment_id: AHU identifier (e.g., "Ahu13")
            from_date: Start date
            to_date: End date
            zone: Optional zone filter
            
        Returns:
            List of raw data records
        """
        print("\n" + "=" * 60)
        print("STEP 1: FETCHING DATA FROM BMS API")
        print("=" * 60)
        log.info("=" * 60)
        log.info("STEP 1: FETCHING DATA FROM BMS API")
        log.info("=" * 60)
        
        self.equipment_id = equipment_id  # Store for model naming
        
        self.raw_data = self.data_fetcher.fetch_data(
            equipment_id=equipment_id,
            from_date=from_date,
            to_date=to_date,
            zone=zone
        )
        
        return self.raw_data
    
    def preprocess_data(
        self,
        raw_data: Optional[List[Dict]] = None,
        add_weather: bool = True
    ) -> pd.DataFrame:
        """
        Step 2: Preprocess raw data for MPC training.
        
        Args:
            raw_data: Raw data records (uses stored data if None)
            add_weather: Whether to add weather data
            
        Returns:
            Preprocessed DataFrame
        """
        print("\n" + "=" * 60)
        print("STEP 2: PREPROCESSING DATA")
        print("=" * 60)
        log.info("=" * 60)
        log.info("STEP 2: PREPROCESSING DATA")
        log.info("=" * 60)
        
        data = raw_data or self.raw_data
        if data is None:
            raise ValueError("No data to preprocess. Call fetch_data() first.")
        
        self.preprocessed_data = self.preprocessor.preprocess(
            records=data,
            add_weather=add_weather
        )
        
        # Validate
        self.validation_report = self.preprocessor.validate_data(self.preprocessed_data)
        
        print(f"\n[Validation] Samples: {self.validation_report['total_samples']}")
        print(f"[Validation] Date range: {self.validation_report['date_range']}")
        print(f"[Validation] Missing features: {self.validation_report['missing_features']}")
        print(f"[Validation] Is Valid: {self.validation_report['is_valid']}")
        log.info(f"Validation - Samples: {self.validation_report['total_samples']}")
        log.info(f"Validation - Date range: {self.validation_report['date_range']}")
        log.info(f"Validation - Missing features: {self.validation_report['missing_features']}")
        log.info(f"Validation - Is Valid: {self.validation_report['is_valid']}")
        
        return self.preprocessed_data
    
    def train_mpc(
        self,
        training_data: Optional[pd.DataFrame] = None
    ) -> CinemaAHUMPCSystem:
        """
        Step 3: Train the MPC model with train-test split validation.
        
        Args:
            training_data: Preprocessed training data (uses stored if None)
            
        Returns:
            Trained CinemaAHUMPCSystem ready for optimization
        """
        print("\n" + "=" * 60)
        print("STEP 3: TRAINING MPC MODEL")
        print("=" * 60)
        log.info("=" * 60)
        log.info("STEP 3: TRAINING MPC MODEL")
        log.info("=" * 60)
        
        data = training_data if training_data is not None else self.preprocessed_data
        if data is None:
            raise ValueError("No training data. Call preprocess_data() first.")
        
        # Train-Test Split if validation enabled
        if self.pipeline_config.validation_enabled:
            print(f"Splitting data: {100*(1-self.pipeline_config.test_size):.0f}% train, {100*self.pipeline_config.test_size:.0f}% test")
            log.info(f"Splitting data: {100*(1-self.pipeline_config.test_size):.0f}% train, {100*self.pipeline_config.test_size:.0f}% test")
            
            # Chronological split to respect time series nature
            split_idx = int(len(data) * (1 - self.pipeline_config.test_size))
            self.train_data = data.iloc[:split_idx].copy()
            self.test_data = data.iloc[split_idx:].copy()
            
            print(f"Train samples: {len(self.train_data)}, Test samples: {len(self.test_data)}")
            log.info(f"Train samples: {len(self.train_data)}, Test samples: {len(self.test_data)}")
            train_dataset = self.train_data
        else:
            print("Training on full dataset (validation disabled)")
            log.info("Training on full dataset (validation disabled)")
            train_dataset = data
            self.train_data = data
            self.test_data = None
        
        # Initialize MPC system
        self.mpc_system = CinemaAHUMPCSystem(self.mpc_config)
        
        # Train on training set
        print("Training thermal prediction model...")
        log.info("Training thermal prediction model...")
        self.training_metrics = self.mpc_system.train(train_dataset)
        
        print(f"\n[Training Complete]")
        print(f"  Train RMSE: {self.training_metrics['rmse']:.4f}°C")
        print(f"  Train MAE: {self.training_metrics['mae']:.4f}°C")
        print(f"  Train R²: {self.training_metrics['r2']:.4f}")
        log.info("Training Complete")
        log.info(f"  Train RMSE: {self.training_metrics['rmse']:.4f}°C")
        log.info(f"  Train MAE: {self.training_metrics['mae']:.4f}°C")
        log.info(f"  Train R²: {self.training_metrics['r2']:.4f}")
        
        # Evaluate on test set if available
        if self.test_data is not None:
            print("\nEvaluating on test set...")
            log.info("Evaluating on test set...")
            self.test_metrics = self._evaluate_on_test_set()
            print("\n[Test Set Performance]")
            print(f"  Test RMSE: {self.test_metrics['rmse']:.4f}°C")
            print(f"  Test MAE: {self.test_metrics['mae']:.4f}°C")
            print(f"  Test R²: {self.test_metrics['r2']:.4f}")
            log.info("Test Set Performance:")
            log.info(f"  Test RMSE: {self.test_metrics['rmse']:.4f}°C")
            log.info(f"  Test MAE: {self.test_metrics['mae']:.4f}°C")
            log.info(f"  Test R²: {self.test_metrics['r2']:.4f}")
            
            # Check for overfitting
            rmse_diff = abs(self.test_metrics['rmse'] - self.training_metrics['rmse'])
            if rmse_diff > 0.5:
                print(f"  ⚠ Potential overfitting detected: RMSE difference = {rmse_diff:.4f}°C")
                log.warning(f"Potential overfitting detected: RMSE difference = {rmse_diff:.4f}°C")
            else:
                print(f"  ✓ Model generalization good: RMSE difference = {rmse_diff:.4f}°C")
                log.info(f"Model generalization good: RMSE difference = {rmse_diff:.4f}°C")
        
        return self.mpc_system
    
    def _evaluate_on_test_set(self) -> Dict[str, float]:
        """
        Evaluate trained model on test set and generate optimized setpoints comparison.
        
        Returns:
            Dictionary with test metrics (rmse, mae, r2)
        """
        if self.test_data is None or self.mpc_system is None:
            raise ValueError("Test data or trained model not available")
        
        # Get thermal model
        thermal_model = self.mpc_system.thermal_model
        
        # Prepare test features
        X_test, y_test = thermal_model.prepare_features(self.test_data)
        
        # Scale and predict
        X_test_scaled = thermal_model.scaler_X.transform(X_test)
        y_pred_scaled = thermal_model.model.predict(X_test_scaled)
        y_pred = thermal_model.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # Calculate metrics
        mse = np.mean((y_test.values - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test.values - y_pred))
        r2 = 1 - (np.sum((y_test.values - y_pred) ** 2) / np.sum((y_test.values - y_test.mean()) ** 2))
        
        # Generate optimized setpoints for test set comparison
        self._generate_test_comparison()
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def _generate_test_comparison(self):
        """
        Generate optimized setpoints for test data and save comparison CSV.
        
        Creates a CSV with:
        - Timestamp
        - Actual SpTREff (historical setpoint)
        - Optimized SpTREff (MPC recommendation)
        - Actual TempSp1 (actual temperature)
        - Predicted TempSp1 (model prediction)
        - Occupancy status
        - Weather conditions
        """
        if self.test_data is None or self.mpc_system is None:
            return
        
        print("\nGenerating test set comparison (Actual vs Optimized Setpoints)...")
        log.info("Generating test set comparison (Actual vs Optimized Setpoints)...")
        
        comparison_data = []
        
        # Sample every 10th row to avoid too many MPC optimizations (speeds up processing)
        sample_interval = 10
        test_sample = self.test_data.iloc[::sample_interval].copy()
        
        for idx, row in test_sample.iterrows():
            try:
                # Extract current measurements (including CO2)
                current_measurements = {
                    'TempSp1': row.get('TempSp1', 24.0),
                    'TempSp1_lag_10m': row.get('TempSp1_lag_10m', 24.0),
                    'SpTREff': row.get('SpTREff', 24.0),
                    'SpTREff_lag_10m': row.get('SpTREff_lag_10_min', row.get('SpTREff', 24.0)),
                    'TempSu': row.get('TempSu', 18.0),
                    'FbVFD': row.get('FbVFD', 50.0),
                    'FbFAD': row.get('FbFAD', 50.0),
                    'Co2RA': row.get('Co2RA', 600.0)  # CO2 for occupancy load
                }
                
                # Create occupancy info from binary occupancy
                occupancy_status = int(row.get('occupied', 0))
                occupancy_response = {
                    'status': occupancy_status,
                    'movie_name': 'Test Movie' if occupancy_status == 1 else None,
                    'time_remaining': '90 minutes' if occupancy_status == 1 else None,
                    'time_until_next_movie': 'No upcoming shows'
                }
                
                # Create simple weather forecast (constant)
                weather_forecast = [{
                    'temperature': row.get('outside_temp', 35.0),
                    'humidity': row.get('outside_humidity', 50.0)
                }] * 6
                
                # Get MPC optimal setpoint
                result = self.mpc_system.get_optimal_setpoint(
                    current_measurements=current_measurements,
                    occupancy_api_response=occupancy_response,
                    weather_forecast=weather_forecast,
                    current_time=row.get('data_received_on', pd.Timestamp.now())
                )
                
                # Store comparison (including CO2 info)
                comparison_data.append({
                    'timestamp': row.get('data_received_on'),
                    'actual_SpTREff': row.get('SpTREff', np.nan),
                    'optimized_SpTREff': result['optimal_setpoint'],
                    'setpoint_difference': result['optimal_setpoint'] - row.get('SpTREff', 0),
                    'actual_TempSp1': row.get('TempSp1', np.nan),
                    'target_temperature': result.get('target_temperature', np.nan),
                    'occupied': occupancy_status,
                    'mode': result['mode'],
                    'outside_temp': row.get('outside_temp', np.nan),
                    'outside_humidity': row.get('outside_humidity', np.nan),
                    'FbVFD': row.get('FbVFD', np.nan),
                    'Co2RA': row.get('Co2RA', np.nan),
                    'co2_load_category': result.get('co2_load_category', 'unknown'),
                    'co2_load_factor': result.get('co2_load_factor', np.nan),
                    'optimization_status': result['optimization_status']
                })
                
            except Exception as e:
                log.warning(f"Failed to optimize for row {idx}: {e}")
                continue
        
        # Save comparison to CSV
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Determine save path
            ahu_name = (self.equipment_id or "unknown_ahu").lower()
            ahu_folder = os.path.join(self.pipeline_config.model_base_dir, f"{ahu_name}_mpc")
            os.makedirs(ahu_folder, exist_ok=True)
            
            comparison_path = os.path.join(ahu_folder, "test_comparison_actual_vs_optimized.csv")
            comparison_df.to_csv(comparison_path, index=False)
            
            # Calculate summary statistics
            avg_diff = comparison_df['setpoint_difference'].mean()
            std_diff = comparison_df['setpoint_difference'].std()
            energy_savings_potential = (comparison_df['setpoint_difference'] > 0).sum() / len(comparison_df) * 100
            
            print(f"\nTest Set Comparison Saved: {comparison_path}")
            print(f"  Samples compared: {len(comparison_df)}")
            print(f"  Avg setpoint difference: {avg_diff:.3f}°C (Optimized - Actual)")
            print(f"  Std deviation: {std_diff:.3f}°C")
            print(f"  Energy savings potential: {energy_savings_potential:.1f}% of time")
            print(f"  (Higher optimized setpoint = less cooling = energy savings)\n")
            
            log.info(f"Test comparison saved: {comparison_path}")
            log.info(f"Samples: {len(comparison_df)}, Avg diff: {avg_diff:.3f}°C, Savings potential: {energy_savings_potential:.1f}%")
    
    def save_model(self, filepath: Optional[str] = None, equipment_id: Optional[str] = None) -> str:
        """
        Save the trained MPC model to disk.
        
        Model is saved in AHU-specific folder:
            artifacts/{ahu_name}/mpc_model.joblib
        
        Args:
            filepath: Custom save path (uses default if None)
            equipment_id: AHU identifier for folder naming (uses stored if None)
            
        Returns:
            Path where model was saved
        """
        if self.mpc_system is None:
            raise ValueError("No trained model to save. Call train_mpc() first.")
        
        # Use stored equipment_id if not provided
        ahu_name = equipment_id or self.equipment_id or "unknown_ahu"
        ahu_name = f"{ahu_name.lower()}_mpc"  # Normalize to lowercase (e.g., "ahu13")
        
        if filepath is None:
            # Create AHU-specific folder: artifacts/{ahu_name}/
            ahu_folder = os.path.join(self.pipeline_config.model_base_dir, ahu_name)
            os.makedirs(ahu_folder, exist_ok=True)
            filepath = os.path.join(
                ahu_folder,
                self.pipeline_config.model_filename
            )
        
        self.mpc_system.save_model(filepath)
        
        # Also save pipeline metadata
        metadata = {
            'pipeline_config': self.pipeline_config,
            'mpc_config': self.mpc_config,
            'training_metrics': self.training_metrics,
            'test_metrics': self.test_metrics,
            'validation_report': self.validation_report,
            'train_samples': len(self.train_data) if self.train_data is not None else 0,
            'test_samples': len(self.test_data) if self.test_data is not None else 0,
            'trained_at': datetime.now().isoformat()
        }
        metadata_path = filepath.replace('.joblib', '_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        log.info(f"Metadata saved to {metadata_path}")
        
        return filepath
    
    def load_model(self, filepath: Optional[str] = None, equipment_id: Optional[str] = None) -> CinemaAHUMPCSystem:
        """
        Load a trained MPC model from disk.
        
        Model is loaded from AHU-specific folder:
            artifacts/{ahu_name}/mpc_model.joblib
        
        Args:
            filepath: Model file path (uses default if None)
            equipment_id: AHU identifier for folder path (e.g., "Ahu13")
            
        Returns:
            Loaded CinemaAHUMPCSystem
        """
        if filepath is None:
            if equipment_id is None:
                raise ValueError("Either filepath or equipment_id must be provided")
            
            ahu_name = f"{equipment_id.lower()}_mpc"
            filepath = os.path.join(
                self.pipeline_config.model_base_dir,
                ahu_name,
                self.pipeline_config.model_filename
            )
        
        self.mpc_system = CinemaAHUMPCSystem(self.mpc_config)
        self.mpc_system.load_model(filepath)
        
        # Try to load metadata
        metadata_path = filepath.replace('.joblib', '_metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.training_metrics = metadata.get('training_metrics')
            self.test_metrics = metadata.get('test_metrics')
            self.validation_report = metadata.get('validation_report')
            log.info(f"Loaded model trained at {metadata.get('trained_at')}")
            if self.test_metrics:
                log.info(f"Test RMSE: {self.test_metrics['rmse']:.4f}°C, Test R²: {self.test_metrics['r2']:.4f}")
        
        return self.mpc_system
    
    def run_full_pipeline(
        self,
        equipment_id: str,
        from_date: str = "",
        to_date: str = "",
        zone: str = "",
        add_weather: bool = True,
        save_model: bool = True,
        save_path: Optional[str] = None
    ) -> CinemaAHUMPCSystem:
        """
        Run the complete training pipeline end-to-end.
        
        This is the main entry point that:
        1. Fetches data from BMS API
        2. Adds weather data
        3. Preprocesses and normalizes
        4. Trains the MPC model
        5. Saves the model (optional)
        
        Args:
            equipment_id: AHU identifier (e.g., "Ahu13")
            from_date: Training data start date
            to_date: Training data end date
            zone: Optional zone filter
            add_weather: Whether to fetch weather data
            save_model: Whether to save trained model
            save_path: Custom model save path
            
        Returns:
            Trained CinemaAHUMPCSystem ready for optimization
            
        Example:
        --------
        >>> pipeline = MPCTrainingPipeline()
        >>> mpc = pipeline.run_full_pipeline(
        ...     equipment_id="Ahu13",
        ...     from_date="2025-11-29 07:00:00",
        ...     to_date="2026-01-27 07:00:00"
        ... )
        """
        print("\n" + "=" * 70)
        print("      CINEMA AHU MPC - FULL TRAINING PIPELINE")
        print("=" * 70)
        print(f"\nEquipment: {equipment_id}")
        print(f"Date Range: {from_date} to {to_date}")
        print(f"Weather Data: {'Enabled' if add_weather else 'Disabled'}")
        print(f"Validation: {'Enabled' if self.pipeline_config.validation_enabled else 'Disabled'}")
        log.info("=" * 70)
        log.info("      CINEMA AHU MPC - FULL TRAINING PIPELINE")
        log.info("=" * 70)
        log.info(f"Equipment: {equipment_id}")
        log.info(f"Date Range: {from_date} to {to_date}")
        log.info(f"Weather Data: {'Enabled' if add_weather else 'Disabled'}")
        log.info(f"Validation: {'Enabled' if self.pipeline_config.validation_enabled else 'Disabled'}")
        
        # Step 1: Fetch data
        self.fetch_data(
            equipment_id=equipment_id,
            from_date=from_date,
            to_date=to_date,
            zone=zone
        )
        
        # Step 2: Preprocess
        self.preprocess_data(add_weather=add_weather)
        
        # Step 3: Train
        self.train_mpc()
        
        # Step 4: Save (optional)
        if save_model:
            print("\n" + "=" * 60)
            print("STEP 4: SAVING MODEL")
            print("=" * 60)
            log.info("=" * 60)
            log.info("STEP 4: SAVING MODEL")
            log.info("=" * 60)
            model_path = self.save_model(save_path)
            print(f"Model saved to: {model_path}")
            log.info(f"Model saved to: {model_path}")
        
        print("\n" + "=" * 70)
        print("      PIPELINE COMPLETE - MPC SYSTEM READY FOR OPTIMIZATION")
        print("=" * 70)
        log.info("=" * 70)
        log.info("      PIPELINE COMPLETE - MPC SYSTEM READY FOR OPTIMIZATION")
        log.info("=" * 70)
        
        return self.mpc_system
    
    def get_optimization_ready_system(self) -> CinemaAHUMPCSystem:
        """
        Get the trained MPC system for optimization.
        
        Returns:
            Trained CinemaAHUMPCSystem
            
        Raises:
            ValueError if system not trained
        """
        if self.mpc_system is None:
            raise ValueError(
                "MPC system not ready. Either:\n"
                "1. Call run_full_pipeline() to train a new model\n"
                "2. Call load_model() to load an existing model"
            )
        return self.mpc_system


# =============================================================================
# SECTION 6: CONVENIENCE FUNCTIONS
# =============================================================================

def train_mpc_from_scratch(
    equipment_id: str,
    from_date: str,
    to_date: str,
    mpc_config: Optional[MPCConfig] = None,
    save_model: bool = True
) -> CinemaAHUMPCSystem:
    """
    Convenience function to train MPC from scratch.
    
    Args:
        equipment_id: AHU identifier
        from_date: Training start date
        to_date: Training end date
        mpc_config: Optional MPC configuration
        save_model: Whether to save the model
        
    Returns:
        Trained CinemaAHUMPCSystem
    """
    pipeline = MPCTrainingPipeline(mpc_config=mpc_config)
    return pipeline.run_full_pipeline(
        equipment_id=equipment_id,
        from_date=from_date,
        to_date=to_date,
        save_model=save_model
    )


def load_trained_mpc(
    equipment_id: str = None,
    filepath: str = None
) -> CinemaAHUMPCSystem:
    """
    Convenience function to load a pre-trained MPC model.
    
    Model is loaded from AHU-specific folder:
        artifacts/{ahu_name}/mpc_model.joblib
    
    Args:
        equipment_id: AHU identifier (e.g., "Ahu13") - recommended
        filepath: Direct path to saved model (optional, overrides equipment_id)
        
    Returns:
        Loaded CinemaAHUMPCSystem
        
    Example:
        >>> mpc = load_trained_mpc(equipment_id="Ahu13")
        >>> # or
        >>> mpc = load_trained_mpc(filepath="artifacts/ahu13/mpc_model.joblib")
    """
    pipeline = MPCTrainingPipeline()
    return pipeline.load_model(filepath=filepath, equipment_id=equipment_id)


# =============================================================================
# SECTION 7: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MPC TRAINING PIPELINE - DEMO")
    print("=" * 70)
    
    # Configure MPC
    mpc_config = MPCConfig(
        prediction_horizon=6,
        control_horizon=3,
        sample_time_minutes=10,
        sp_min_occupied=23.0,
        sp_max_occupied=26.0,
        sp_unoccupied=27.0,
        temp_comfort_target=23.5,
        w_comfort=100.0,
        w_energy=1.0,
        w_smoothness=50.0
    )
    
    # Initialize pipeline
    pipeline = MPCTrainingPipeline(mpc_config=mpc_config)
    
    # Run full pipeline
    mpc_system = pipeline.run_full_pipeline(
        equipment_id="Ahu13",
        from_date="2025-11-29 07:00:00",
        to_date="2026-01-27 07:00:00",
        save_model=True
    )
    
    # Test optimization
    print("\n" + "=" * 70)
    print("TESTING OPTIMIZATION")
    print("=" * 70)
    
    test_measurements = {
        'TempSp1': 24.5,
        'TempSp1_lag_10m': 24.3,
        'SpTREff': 24.0,
        'SpTREff_lag_10m': 24.0,
        'TempSu': 18.5,
        'FbVFD': 65.0,
        'FbFAD': 45.0
    }
    
    test_occupancy = {
        "status": 1,
        "movie_name": "Test Movie",
        "time_remaining": "90 minutes"
    }
    
    result = mpc_system.get_optimal_setpoint(
        current_measurements=test_measurements,
        occupancy_api_response=test_occupancy
    )
    
    print(f"\nOptimization Result:")
    print(f"  Mode: {result['mode']}")
    print(f"  Optimal Setpoint: {result['optimal_setpoint']}°C")
    print(f"  Target Temperature: {result.get('target_temperature', 'N/A')}°C")
    print(f"  Status: {result['optimization_status']}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
