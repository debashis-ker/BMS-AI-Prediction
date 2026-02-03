"""
Cinema AHU Supervisory Model Predictive Control (MPC) System
=============================================================

This module implements a Supervisory MPC for cinema Air Handling Units (AHUs).
The MPC optimizes ONLY the Room Temperature Effective Setpoint (SpTREff) while
all physical actuators (CHW valve, fan VFD, dampers) remain under existing PID control.

System Architecture:
-------------------
- Control Variable: SpTREff (Room Temperature Effective Setpoint)
- Role: Setpoint optimizer running on top of existing BMS
- Actuators: Read-only (controlled by existing PID loops)

"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1: SYSTEM DEFINITIONS & DATA CLASSES
# =============================================================================

class OccupancyMode(Enum):
    """Operational modes based on occupancy logic."""
    OCCUPIED = "occupied"           # Movie is playing (status=1)
    PRE_COOLING = "pre_cooling"     # status=0 AND time_until_next < 60 min
    UNOCCUPIED = "unoccupied"       # status=0 AND time_until_next >= 60 min or null


@dataclass
class MPCConfig:
    """Configuration parameters for the MPC controller."""
    
    prediction_horizon: int = 6         
    control_horizon: int = 3           
    sample_time_minutes: int = 10      
    
    # Setpoint Constraints (SpTREff)
    sp_min_occupied: float = 23.0       
    sp_max_occupied: float = 26.0       
    sp_unoccupied: float = 27.0         
    
    # Temperature Safety Bounds (TempSp1 - Space Air Temperature)
    temp_min_hard: float = 22.0         
    temp_max_hard: float = 26.0    
    temp_comfort_target: float = 23.5  
    
    # Rate of Change Constraints
    delta_sp_max: float = 0.5           
    
    # Pre-cooling Threshold
    precool_threshold_minutes: int = 60  
    
    # Objective Function Weights
    w_comfort: float = 100.0           
    w_energy: float = 1.0               
    w_smoothness: float = 50.0         
    w_soft_constraint: float = 1000.0   
    
    # Model Parameters
    thermal_time_constant: float = 30.0  


@dataclass
class MPCState:
    """
    State Vector (x) for the MPC system.
    
    The state vector captures the current thermal state of the zone:
    x = [TempSp1, TempSp1_lag_10m, SpTREff, SpTREff_lag_10m]
    
    Where:
    - TempSp1: Current space air temperature (°C)
    - TempSp1_lag_10m: Space temperature 10 minutes ago (°C)
    - SpTREff: Current effective setpoint (°C)
    - SpTREff_lag_10m: Setpoint 10 minutes ago (°C)
    """
    TempSp1: float               
    TempSp1_lag_10m: float              
    SpTREff: float                      
    SpTREff_lag_10m: float             
    TempSu: float = 20.0                
    FbVFD: float = 50.0                
    FbFAD: float = 50.0                 
    Co2RA: float = 600.0               
    
    def get_co2_load_category(self) -> str:
        """
        Categorize CO2 level into occupancy load.
        
        CO2 Interpretation for Cinema:
        - 500-600 ppm: Empty/few people (low load)
        - 700-800 ppm: Light occupancy (medium load)
        - 900-1000 ppm: Moderate occupancy (high load)
        - 1100+ ppm: Heavy occupancy (very high load)
        """
        co2_range = int(self.Co2RA / 100)
        if co2_range <= 6:
            return 'low'       
        elif co2_range <= 8:
            return 'medium'    
        elif co2_range <= 10:
            return 'high'      
        else:
            return 'very_high' 
    
    def get_co2_load_factor(self) -> float:
        """
        Get numeric load factor from CO2 (0.0 to 1.0+).
        
        Returns:
            0.0 = minimal load (empty), 1.0 = full load, >1.0 = overcrowded
        """
        return max(0.0, (self.Co2RA - 500) / 500)
    
    def to_vector(self) -> np.ndarray:
        """Convert state to numpy vector."""
        return np.array([
            self.TempSp1, 
            self.TempSp1_lag_10m, 
            self.SpTREff, 
            self.SpTREff_lag_10m
        ])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, **kwargs) -> 'MPCState':
        """Create state from numpy vector."""
        return cls(
            TempSp1=vec[0],
            TempSp1_lag_10m=vec[1],
            SpTREff=vec[2],
            SpTREff_lag_10m=vec[3],
            **kwargs
        )


@dataclass
class DisturbanceVector:
    """
    Disturbance Vector (d) for the MPC system.
    
    d = [outside_temp, outside_humidity, occupancy_mode, hour_sin, hour_cos, 
         day_sin, day_cos, time_until_event]
    
    These are exogenous variables that affect the system but cannot be controlled.
    """
    outside_temp: float                 # Outdoor temperature (°C)
    outside_humidity: float             # Outdoor humidity (%)
    occupancy_mode: OccupancyMode       # Current operational mode
    hour_sin: float                     # Cyclic hour encoding (sin)
    hour_cos: float                     # Cyclic hour encoding (cos)
    day_sin: float                      # Cyclic day encoding (sin)
    day_cos: float                      # Cyclic day encoding (cos)
    time_until_event: Optional[int] = None  # Minutes until next movie (if applicable)
    co2_load_factor: float = 0.5        # CO2-based occupancy load (0.0-1.5)
    
    def to_features(self) -> Dict[str, float]:
        """Convert to feature dictionary for model input."""
        return {
            'outside_temp': self.outside_temp,
            'outside_humidity': self.outside_humidity,
            'occupied': 1.0 if self.occupancy_mode != OccupancyMode.UNOCCUPIED else 0.0,
            'hour_sin': self.hour_sin,
            'hour_cos': self.hour_cos,
            'day_sin': self.day_sin,
            'day_cos': self.day_cos,
            'co2_load_factor': self.co2_load_factor
        }


@dataclass 
class OccupancyInfo:
    """Parsed occupancy information from movie schedule API."""
    status: int                         # 0 = no movie, 1 = movie playing
    movie_name: Optional[str] = None
    time_remaining: Optional[int] = None        # Minutes remaining if movie playing
    time_until_next_movie: Optional[int] = None  # Minutes until next movie
    next_movie_name: Optional[str] = None
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'OccupancyInfo':
        """Parse API response to OccupancyInfo."""
        status = response.get('status', 0)
        
        if status == 1:
            time_remaining_val = response.get('time_remaining', 0)
            if isinstance(time_remaining_val, int):
                time_remaining = time_remaining_val
            elif isinstance(time_remaining_val, str) and time_remaining_val:
                time_remaining = int(time_remaining_val.split()[0])
            else:
                time_remaining = None
            return cls(
                status=1,
                movie_name=response.get('movie_name'),
                time_remaining=time_remaining
            )
        else:
            time_until_val = response.get('time_until_next_movie', '')
            if isinstance(time_until_val, int):
                time_until = time_until_val
            elif isinstance(time_until_val, str) and time_until_val and time_until_val != "No upcoming shows":
                time_until = int(time_until_val.split()[0])
            else:
                time_until = None
                
            return cls(
                status=0,
                time_until_next_movie=time_until,
                next_movie_name=response.get('next_movie_name')
            )


# =============================================================================
# SECTION 2: OCCUPANCY LOGIC HANDLER
# =============================================================================

class OccupancyLogicHandler:
    """
    Handles the occupancy logic for bridging training data (binary) 
    with inference data (detailed timing).
    
    Logic Requirements:
    1. OCCUPIED: status == 1 (movie playing)
    2. PRE_COOLING: status == 0 AND time_until_next_movie < 60 minutes
    3. UNOCCUPIED: status == 0 AND (time_until_next_movie >= 60 OR null)
    """
    
    def __init__(self, config: MPCConfig):
        self.config = config
        
    def determine_mode(self, occupancy_info: OccupancyInfo) -> OccupancyMode:
        """
        Determine operational mode based on occupancy information.
        
        Args:
            occupancy_info: Parsed occupancy data from API
            
        Returns:
            OccupancyMode enum value
        """
        if occupancy_info.status == 1:
            return OccupancyMode.OCCUPIED
            
        time_until = occupancy_info.time_until_next_movie
        
        if time_until is None:
            return OccupancyMode.UNOCCUPIED
            
        if time_until < self.config.precool_threshold_minutes:
            return OccupancyMode.PRE_COOLING
        else:
            return OccupancyMode.UNOCCUPIED
    
    def get_mode_constraints(
        self, 
        mode: OccupancyMode,
        outside_temp: float = None,
        current_room_temp: float = None,
        co2_load_factor: float = None
    ) -> Tuple[float, float]:
        """
        Get setpoint constraints based on operational mode, weather, and CO2 load.
        
        WEATHER + CO2 ADAPTIVE STRATEGY:
        1. HOT outside (>25°C): Force setpoint to target (need max cooling)
        2. MILD outside (20-25°C): Normal target ± 0.5°C
        3. COLD outside (<20°C): Allow higher setpoint for energy savings
           - BUT limit by CO2 (high occupancy = more cooling needed)
        4. HIGH CO2 (>1000 ppm): Tighter cooling bounds (full cinema)
        5. LOW CO2 (<600 ppm): Can relax setpoint (empty/few people)
        
        Args:
            mode: Current operational mode
            outside_temp: Current outdoor temperature (optional)
            current_room_temp: Current room temperature (optional)
            co2_load_factor: CO2-based occupancy load 0.0-1.5 (optional)
            
        Returns:
            Tuple of (min_setpoint, max_setpoint)
        """
        if mode == OccupancyMode.UNOCCUPIED:
            return (self.config.sp_unoccupied, self.config.sp_unoccupied)
        
        target = self.config.temp_comfort_target  # 23.5
        sp_min = self.config.sp_min_occupied      # 23.0
        sp_max = target + 1.0                     # 24.5
        
        if co2_load_factor is not None:
            if co2_load_factor >= 1.0:  
                sp_max = target + 0.5            
            elif co2_load_factor <= 0.2:  
                sp_max = target + 1.5             
        
        if outside_temp is not None:
            if outside_temp > 25.0:
                sp_max = target + 0.5         
                
            elif outside_temp < 20.0:
                sp_min = target                
                
                if co2_load_factor is not None and co2_load_factor >= 1.0:
                    sp_max = target + 0.5      
                elif outside_temp < 15.0:
                    sp_max = target + 1.5         
                else:
                    sp_max = target + 1.0        
        
        if current_room_temp is not None and current_room_temp > target + 1.0:
            sp_min = target - 0.5               
            sp_max = target                       
        
        if sp_min > sp_max:
            sp_min = target
            sp_max = target
        
        return (sp_min, sp_max)
    
    def should_bypass_optimization(self, mode: OccupancyMode) -> bool:
        """Check if optimization should be bypassed (unoccupied mode)."""
        return mode == OccupancyMode.UNOCCUPIED
    
    def get_precool_ramp_target(
        self, 
        occupancy_info: OccupancyInfo,
        current_temp: float,
        target_comfort: float
    ) -> float:
        """
        Calculate gradual pre-cooling setpoint to avoid CHW valve saturation.
        
        During pre-cooling, we want to gradually lower the setpoint to reach
        target_comfort by the time the movie starts.
        
        Args:
            occupancy_info: Current occupancy information
            current_temp: Current space temperature
            target_comfort: Target comfort temperature
            
        Returns:
            Intermediate setpoint target for gradual pre-cooling
        """
        if occupancy_info.time_until_next_movie is None:
            return target_comfort
            
        time_until = occupancy_info.time_until_next_movie
        
        temp_gap = current_temp - target_comfort
        
        if temp_gap <= 0:
            return target_comfort
        
        progress = 1.0 - (time_until / self.config.precool_threshold_minutes)
        progress = np.clip(progress, 0.0, 1.0)
        
        intermediate_target = current_temp - (temp_gap * progress)
        
        return np.clip(
            intermediate_target,
            self.config.sp_min_occupied,
            self.config.sp_max_occupied
        )


# =============================================================================
# SECTION 3: THERMAL PREDICTION MODEL
# =============================================================================

class ThermalPredictionModel:
    """
    Thermal Prediction Model for predicting future TempSp1.
    
    Model Structure: ARX (AutoRegressive with eXogenous inputs)
    
    TempSp1(k+1) = a1*TempSp1(k) + a2*TempSp1(k-1) + b1*SpTREff(k) + 
                   b2*SpTREff(k-1) + c1*outside_temp(k) + c2*occupied(k) + 
                   d1*hour_sin + d2*hour_cos + e
    
    This model captures:
    - Thermal inertia (autoregressive terms)
    - Setpoint influence on temperature trajectory
    - External disturbances (weather, occupancy)
    - Diurnal patterns
    """
    
    def __init__(self, config: MPCConfig):
        self.config = config
        self.model: Optional[Ridge] = None
        self.scaler_X: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for ARX model training.
        
        Handles the input mismatch between binary training data 
        and detailed inference timing by:
        1. Using binary occupancy for training
        2. Mapping pre-cooling mode to occupied=1 during inference
        
        Args:
            df: Preprocessed dataframe with AHU data
            
        Returns:
            Tuple of (X features, y target)
        """
        self.feature_names = [
            'TempSp1',              
            'TempSp1_lag_10m',      
            'SpTREff',              
            'SpTREff_lag_10_min',   
            'outside_temp',         
            'outside_humidity',     
            'occupied',             
            'hour_sin',             
            'hour_cos',
            'day_sin',              
            'day_cos',
            'FbVFD',              
            'FbFAD'                
        ]
        
        available_features = [f for f in self.feature_names if f in df.columns]
        
        if 'SpTREff_lag_10_min' in df.columns:
            df = df.rename(columns={'SpTREff_lag_10_min': 'SpTREff_lag_10m'})
            if 'SpTREff_lag_10m' in self.feature_names:
                pass
            else:
                available_features = [f if f != 'SpTREff_lag_10_min' else 'SpTREff_lag_10m' 
                                    for f in available_features]
        
        self.feature_names = available_features
        
        X = df[available_features].copy()
        y = df['Target_Temp'].copy()  
        
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y
    
    def train(self, df: pd.DataFrame, alpha: float = 1.0) -> Dict[str, float]:
        """
        Train the ARX thermal prediction model.
        
        Uses Ridge regression for regularization to prevent overfitting
        and improve generalization.
        
        Args:
            df: Preprocessed training dataframe
            alpha: Ridge regularization parameter
            
        Returns:
            Dictionary with training metrics
        """
        print("Training Thermal Prediction Model (ARX)...")
        
        X, y = self.prepare_features(df)
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        self.model = Ridge(alpha=alpha)
        self.model.fit(X_scaled, y_scaled)
        
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        mse = np.mean((y.values - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y.values - y_pred))
        r2 = 1 - (np.sum((y.values - y_pred) ** 2) / np.sum((y.values - y.mean()) ** 2))
        
        self.is_trained = True
        
        print(f"  Samples: {len(y)}")
        print(f"  RMSE: {rmse:.4f}°C")
        print(f"  MAE: {mae:.4f}°C")
        print(f"  R²: {r2:.4f}")
        
        print("\n  Feature Coefficients:")
        for name, coef in zip(self.feature_names, self.model.coef_):
            print(f"    {name}: {coef:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def predict_single_step(
        self, 
        state: MPCState, 
        control_input: float,  # SpTREff
        disturbance: DisturbanceVector
    ) -> float:
        """
        Predict TempSp1 for the next timestep.
        
        Args:
            state: Current system state
            control_input: SpTREff setpoint command
            disturbance: Exogenous disturbance vector
            
        Returns:
            Predicted TempSp1 for next timestep
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        features = {
            'TempSp1': state.TempSp1,
            'TempSp1_lag_10m': state.TempSp1_lag_10m,
            'SpTREff': control_input,
            'SpTREff_lag_10m': state.SpTREff,  
            'outside_temp': disturbance.outside_temp,
            'outside_humidity': disturbance.outside_humidity,
            'occupied': 1.0 if disturbance.occupancy_mode != OccupancyMode.UNOCCUPIED else 0.0,
            'hour_sin': disturbance.hour_sin,
            'hour_cos': disturbance.hour_cos,
            'day_sin': disturbance.day_sin,
            'day_cos': disturbance.day_cos,
            'FbVFD': state.FbVFD,
            'FbFAD': state.FbFAD
        }
        
        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
        
        return y_pred
    
    def predict_trajectory(
        self,
        initial_state: MPCState,
        control_sequence: np.ndarray, 
        disturbance_forecast: List[DisturbanceVector]
    ) -> np.ndarray:
        """
        Predict temperature trajectory over prediction horizon.
        
        Args:
            initial_state: Current system state
            control_sequence: Array of SpTREff setpoints [N_p]
            disturbance_forecast: List of disturbance vectors for each step
            
        Returns:
            Predicted TempSp1 trajectory [N_p]
        """
        N_p = len(control_sequence)
        trajectory = np.zeros(N_p)
        
        current_state = MPCState(
            TempSp1=initial_state.TempSp1,
            TempSp1_lag_10m=initial_state.TempSp1_lag_10m,
            SpTREff=initial_state.SpTREff,
            SpTREff_lag_10m=initial_state.SpTREff_lag_10m,
            TempSu=initial_state.TempSu,
            FbVFD=initial_state.FbVFD,
            FbFAD=initial_state.FbFAD
        )
        
        for k in range(N_p):
            next_temp = self.predict_single_step(
                state=current_state,
                control_input=control_sequence[k],
                disturbance=disturbance_forecast[k]
            )
            trajectory[k] = next_temp
            
            new_state = MPCState(
                TempSp1=next_temp,
                TempSp1_lag_10m=current_state.TempSp1,
                SpTREff=control_sequence[k],
                SpTREff_lag_10m=current_state.SpTREff,
                TempSu=current_state.TempSu,
                FbVFD=current_state.FbVFD,
                FbFAD=current_state.FbFAD
            )
            current_state = new_state
        
        return trajectory
    
    def save(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'config': self.config
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model from file."""
        
        if 'src.bms_ai.mpc.mpc_training_pipeline' not in sys.modules:
            try:
                import src.bms_ai.mpc.mpc_training_pipeline
                sys.modules['src.bms_ai.mpc.mpc_training_pipeline'] = src.bms_ai.mpc.mpc_training_pipeline
            except:
                pass
        
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler_X = model_data['scaler_X']
            self.scaler_y = model_data['scaler_y']
            self.feature_names = model_data['feature_names']
            self.config = model_data['config']
            self.is_trained = True
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            raise


# =============================================================================
# SECTION 4: MPC OPTIMIZER
# =============================================================================

class CinemaAHUMPC:
    """
    Supervisory Model Predictive Controller for Cinema AHU.
    
    Optimization Formulation:
    -------------------------
    
    CORRECTED Objective Function J:
    
    J = Σₖ [ w_comfort * (SpTREff(k) - T_target)²     # Setpoint tracks target
           + 0.5*w_comfort * (TempSp1(k) - T_target)²  # Predicted temp reaches target
           + w_smoothness * (ΔSpTREff(k))²             # Smooth transitions
           + w_soft * max(0, T_min - TempSp1(k))²      # Soft lower bound
           + w_soft * max(0, TempSp1(k) - T_max)²      # Soft upper bound
           + CONFLICT_PENALTY if (TempSp1 > target AND SpTREff > target)
         ]
    
    KEY INSIGHT: SpTREff IS the setpoint the AHU PID controller tracks.
    To achieve room temp of 23.5°C, we MUST set SpTREff ≈ 23.5°C.
    
    Energy savings come from:
    - Raising target during unoccupied (bypass mode)
    - Allowing small setpoint relaxation when room is already cool
    
    Subject to:
    - SpTREff_min ≤ SpTREff(k) ≤ SpTREff_max  (Hard constraints)
    - |ΔSpTREff(k)| ≤ Δmax                      (Rate constraint)
    - TempSp1(k) predicted by thermal model
    
    Control Strategy:
    -----------------
    - OCCUPIED: Set SpTREff to achieve target (23.5°C)
    - PRE_COOLING: Gradual ramp to target before movie starts
    - UNOCCUPIED: Bypass optimization, force SpTREff = 27°C
    """
    
    def __init__(
        self, 
        thermal_model: ThermalPredictionModel,
        config: Optional[MPCConfig] = None
    ):
        self.thermal_model = thermal_model
        self.config = config or MPCConfig()
        self.occupancy_handler = OccupancyLogicHandler(self.config)
        
        self.previous_setpoint: Optional[float] = None
        
    def _build_objective(
        self,
        initial_state: MPCState,
        disturbance_forecast: List[DisturbanceVector],
        mode: OccupancyMode,
        target_temp: float
    ):
        """
        Build the MPC objective function.
        
        CORRECTED LOGIC (v2):
        ---------------------
        SpTREff IS the setpoint that the AHU PID controller tracks.
        The room temperature will eventually converge to SpTREff.
        
        Therefore: To achieve target_temp=23.5°C, we MUST set SpTREff = 23.5°C
        
        Energy savings during occupied mode are MINIMAL and come from:
        - Not overcooling below target
        - Smooth transitions (no aggressive hunting)
        
        Major energy savings come from UNOCCUPIED mode (bypass to 27°C)
        
        Args:
            initial_state: Current system state
            disturbance_forecast: Forecast disturbances over horizon
            mode: Current operational mode
            target_temp: Target comfort temperature
            
        Returns:
            Objective function callable
        """
        N_p = self.config.prediction_horizon
        N_c = self.config.control_horizon
        
        avg_outside_temp = np.mean([d.outside_temp for d in disturbance_forecast])
        is_hot_day = avg_outside_temp > 25.0
        is_cold_day = avg_outside_temp < 20.0
        room_is_warm = initial_state.TempSp1 > target_temp + 0.5
        room_is_comfortable = initial_state.TempSp1 <= target_temp + 0.5
        
        co2_load = initial_state.get_co2_load_factor()
        is_high_occupancy = co2_load >= 1.0      # CO2 > 1000 ppm (full cinema)
        is_low_occupancy = co2_load <= 0.2       # CO2 < 600 ppm (nearly empty)
        avg_co2_load = np.mean([d.co2_load_factor for d in disturbance_forecast])
        
        def objective(u_sequence: np.ndarray) -> float:
            """
            Evaluate MPC objective for given control sequence.
            
            u_sequence: Control inputs (SpTREff setpoints) for N_c steps
            """
            full_sequence = np.zeros(N_p)
            full_sequence[:N_c] = u_sequence
            full_sequence[N_c:] = u_sequence[-1] 
            
            temp_trajectory = self.thermal_model.predict_trajectory(
                initial_state=initial_state,
                control_sequence=full_sequence,
                disturbance_forecast=disturbance_forecast
            )
            
            # Initialize cost
            J = 0.0
            
            prev_sp = self.previous_setpoint or initial_state.SpTREff
            
            for k in range(N_p):
                temp_k = temp_trajectory[k]
                sp_k = full_sequence[k]
                outside_k = disturbance_forecast[k].outside_temp
                co2_k = disturbance_forecast[k].co2_load_factor
                
                effective_target = target_temp
                
                if is_cold_day and room_is_comfortable and not is_high_occupancy:
                    cold_bonus = min(1.0, (20.0 - avg_outside_temp) / 10.0)  
                    occupancy_scale = max(0.0, 1.0 - co2_load)  
                    effective_target = target_temp + cold_bonus * occupancy_scale * 0.5
                
                if is_high_occupancy:
                    effective_target = min(effective_target, target_temp)
                
                setpoint_error = (sp_k - effective_target) ** 2
                J += self.config.w_comfort * 2.0 * setpoint_error
                
                temp_target = effective_target if (is_cold_day and not is_high_occupancy) else target_temp
                predicted_comfort_error = (temp_k - temp_target) ** 2
                J += self.config.w_comfort * 0.5 * predicted_comfort_error
                
                if sp_k > effective_target:
                    above_target = sp_k - effective_target
                    if is_hot_day:
                        weather_multiplier = 3.0
                    elif is_cold_day:
                        weather_multiplier = 0.5
                    else:
                        weather_multiplier = 1.0
                    occupancy_multiplier = 1.0 + co2_load  
                    J += self.config.w_comfort * 3.0 * weather_multiplier * occupancy_multiplier * (above_target ** 2)
                
                # High CO2 = full cinema = people expect comfort
                if (outside_k > 25.0 or is_high_occupancy) and sp_k > target_temp:
                    heat_penalty = (sp_k - target_temp) * 10.0
                    if outside_k > 25.0:
                        heat_penalty *= (outside_k - 25.0) 
                    if is_high_occupancy:
                        heat_penalty *= (1.0 + co2_load)   
                    J += self.config.w_comfort * heat_penalty
                
                if outside_k < 20.0 and room_is_comfortable and not is_high_occupancy:
                    if sp_k > target_temp and sp_k <= target_temp + 1.0:
                        cold_saving = (20.0 - outside_k) / 10.0  
                        occupancy_bonus = max(0.0, 1.0 - co2_k)  
                        energy_reward = cold_saving * occupancy_bonus * (sp_k - target_temp) * 5.0
                        J -= self.config.w_energy * energy_reward  
                
                if room_is_warm and sp_k > target_temp:
                    room_error = initial_state.TempSp1 - target_temp
                    J += self.config.w_comfort * 5.0 * room_error * (sp_k - target_temp)
                
                if k < N_c:
                    delta_sp = sp_k - prev_sp
                    smoothness_penalty = delta_sp ** 2
                    J += self.config.w_smoothness * smoothness_penalty
                    prev_sp = sp_k
                
                lower_violation = max(0, self.config.temp_min_hard - temp_k)
                upper_violation = max(0, temp_k - self.config.temp_max_hard)
                J += self.config.w_soft_constraint * (lower_violation ** 2)
                J += self.config.w_soft_constraint * (upper_violation ** 2)
            
            return J
        
        return objective
    
    def _build_constraints(
        self, 
        mode: OccupancyMode,
        sp_min: float = None,
        sp_max: float = None
    ) -> List[Dict]:
        """
        Build optimization constraints based on operational mode.
        
        Args:
            mode: Current operational mode
            sp_min: Minimum setpoint bound (for checking constraint compatibility)
            sp_max: Maximum setpoint bound (for checking constraint compatibility)
            
        Returns:
            List of scipy constraint dictionaries
        """
        N_c = self.config.control_horizon
        
        constraints = []
        
        # Add rate constraints as inequality constraints
        if self.previous_setpoint is not None:
            prev_in_bounds = True
            if sp_min is not None and sp_max is not None:
                delta = self.config.delta_sp_max
                if self.previous_setpoint - delta > sp_max or self.previous_setpoint + delta < sp_min:
                    prev_in_bounds = False  
            
            if prev_in_bounds:
                def rate_constraint_first(u):
                    return self.config.delta_sp_max - np.abs(u[0] - self.previous_setpoint)
                constraints.append({'type': 'ineq', 'fun': rate_constraint_first})
        
        for k in range(N_c - 1):
            def rate_constraint(u, k=k):
                return self.config.delta_sp_max - np.abs(u[k+1] - u[k])
            constraints.append({'type': 'ineq', 'fun': rate_constraint})
        
        return constraints
    
    def compute_optimal_setpoint(
        self,
        current_state: MPCState,
        occupancy_info: OccupancyInfo,
        weather_forecast: List[Dict[str, float]],
        current_time: datetime
    ) -> Dict[str, Any]:
        """
        Compute optimal SpTREff setpoint using MPC.
        
        Args:
            current_state: Current system state
            occupancy_info: Occupancy information from API
            weather_forecast: List of weather dicts for each horizon step
            current_time: Current datetime
            
        Returns:
            Dictionary with optimal setpoint and diagnostic info
        """
        N_p = self.config.prediction_horizon
        N_c = self.config.control_horizon
        
        mode = self.occupancy_handler.determine_mode(occupancy_info)
        
        if self.occupancy_handler.should_bypass_optimization(mode):
            optimal_sp = self.config.sp_unoccupied
            self.previous_setpoint = optimal_sp
            
            return {
                'optimal_setpoint': optimal_sp,
                'mode': mode.value,
                'bypass_optimization': True,
                'reason': 'Unoccupied mode - forcing static setpoint',
                'predicted_trajectory': None,
                'optimization_status': 'bypassed'
            }
        
        if mode == OccupancyMode.PRE_COOLING:
            target_temp = self.occupancy_handler.get_precool_ramp_target(
                occupancy_info=occupancy_info,
                current_temp=current_state.TempSp1,
                target_comfort=self.config.temp_comfort_target
            )
        else:
            target_temp = self.config.temp_comfort_target
        
        disturbance_forecast = []
        co2_load_factor = current_state.get_co2_load_factor()  
        
        for k in range(N_p):
            step_time = current_time + timedelta(minutes=k * self.config.sample_time_minutes)
            hour = step_time.hour + step_time.minute / 60.0
            day = step_time.weekday()
            
            weather_k = weather_forecast[min(k, len(weather_forecast) - 1)]
            
            disturbance_forecast.append(DisturbanceVector(
                outside_temp=weather_k.get('temperature', 30.0),
                outside_humidity=weather_k.get('humidity', 50.0),
                occupancy_mode=mode,
                hour_sin=np.sin(2 * np.pi * hour / 24),
                hour_cos=np.cos(2 * np.pi * hour / 24),
                day_sin=np.sin(2 * np.pi * day / 7),
                day_cos=np.cos(2 * np.pi * day / 7),
                time_until_event=occupancy_info.time_until_next_movie,
                co2_load_factor=co2_load_factor  
            ))
        
        objective = self._build_objective(
            initial_state=current_state,
            disturbance_forecast=disturbance_forecast,
            mode=mode,
            target_temp=target_temp
        )
        
        outside_temp = disturbance_forecast[0].outside_temp if disturbance_forecast else None
        sp_min, sp_max = self.occupancy_handler.get_mode_constraints(
            mode=mode,
            outside_temp=outside_temp,
            current_room_temp=current_state.TempSp1,
            co2_load_factor=co2_load_factor 
        )
        bounds = [(sp_min, sp_max)] * N_c
        
        constraints = self._build_constraints(mode, sp_min, sp_max)
        
        u0 = np.full(N_c, target_temp)
        u0 = np.clip(u0, sp_min, sp_max)
        
        result = minimize(
            objective,
            u0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-5} 
        )
        
        if not result.success and 'directional' in result.message:
            result_lbfgsb = minimize(
                objective,
                u0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200}
            )
            if result_lbfgsb.success or result_lbfgsb.fun < result.fun:
                result = result_lbfgsb
        
        if result.success:
            optimal_sp = float(result.x[0])
            optimization_status = 'success'
        else:
            if outside_temp is not None and outside_temp < 20.0:
                optimal_sp = sp_max
            elif current_state.TempSp1 > target_temp + 0.5:
                optimal_sp = max(sp_min, target_temp - 0.3)
            else:
                optimal_sp = target_temp
            
            optimal_sp = np.clip(optimal_sp, sp_min, sp_max)
            optimization_status = f'failed: {result.message}'
        
        if self.previous_setpoint is not None:
            delta = optimal_sp - self.previous_setpoint
            if abs(delta) > self.config.delta_sp_max:
                optimal_sp = self.previous_setpoint + np.sign(delta) * self.config.delta_sp_max
        
        optimal_sp = np.clip(optimal_sp, sp_min, sp_max)
        
        self.previous_setpoint = optimal_sp
        
        full_sequence = np.zeros(N_p)
        full_sequence[:N_c] = result.x if result.success else u0
        full_sequence[N_c:] = full_sequence[N_c - 1]
        
        predicted_trajectory = self.thermal_model.predict_trajectory(
            initial_state=current_state,
            control_sequence=full_sequence,
            disturbance_forecast=disturbance_forecast
        )
        
        return {
            'optimal_setpoint': round(optimal_sp, 2),
            'mode': mode.value,
            'bypass_optimization': False,
            'target_temperature': round(target_temp, 2),
            'predicted_trajectory': predicted_trajectory.tolist(),
            'control_sequence': result.x.tolist() if result.success else u0.tolist(),
            'optimization_status': optimization_status,
            'objective_value': float(result.fun) if result.success else None,
            'time_until_event': occupancy_info.time_until_next_movie,
            'current_space_temp': current_state.TempSp1,
            'co2_ppm': current_state.Co2RA,
            'co2_load_category': current_state.get_co2_load_category(),
            'co2_load_factor': round(co2_load_factor, 2)
        }


# =============================================================================
# SECTION 5: MPC SYSTEM WRAPPER
# =============================================================================

class CinemaAHUMPCSystem:
    """
    Complete MPC System for Cinema AHU with training and inference capabilities.
    
    This class wraps all components and provides a simple interface for:
    1. Training the thermal prediction model
    2. Running the MPC optimizer
    3. Integrating with the movie schedule API
    """
    
    def __init__(self, config: Optional[MPCConfig] = None):
        self.config = config or MPCConfig()
        self.thermal_model = ThermalPredictionModel(self.config)
        self.mpc: Optional[CinemaAHUMPC] = None
        self.is_initialized = False
        
    def train(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the thermal prediction model.
        
        Args:
            training_data: Preprocessed training dataframe
            
        Returns:
            Training metrics dictionary
        """
        metrics = self.thermal_model.train(training_data)
        self.mpc = CinemaAHUMPC(self.thermal_model, self.config)
        self.is_initialized = True
        return metrics
    
    def get_optimal_setpoint(
        self,
        current_measurements: Dict[str, float],
        occupancy_api_response: Dict[str, Any],
        weather_forecast: Optional[List[Dict[str, float]]] = None,
        current_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get optimal setpoint for a single screen/AHU.
        
        Args:
            current_measurements: Dict with current sensor readings
                Required keys: TempSp1, SpTREff, TempSu, FbVFD, FbFAD
            occupancy_api_response: Response from get_current_movie_occupancy_status
                for a single screen
            weather_forecast: List of weather dicts [{temperature, humidity}, ...]
            current_time: Current datetime (defaults to now)
            
        Returns:
            Dictionary with optimal setpoint and diagnostics
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call train() first.")
        
        current_time = current_time or datetime.now()
        
        state = MPCState(
            TempSp1=current_measurements.get('TempSp1', 24.0),
            TempSp1_lag_10m=current_measurements.get('TempSp1_lag_10m', 
                                                      current_measurements.get('TempSp1', 24.0)),
            SpTREff=current_measurements.get('SpTREff', 24.0),
            SpTREff_lag_10m=current_measurements.get('SpTREff_lag_10m',
                                                      current_measurements.get('SpTREff', 24.0)),
            TempSu=current_measurements.get('TempSu', 18.0),
            FbVFD=current_measurements.get('FbVFD', 50.0),
            FbFAD=current_measurements.get('FbFAD', 50.0),
            Co2RA=current_measurements.get('Co2RA', 600.0)  
        )
        
        occupancy_info = OccupancyInfo.from_api_response(occupancy_api_response)
        
        if weather_forecast is None:
            weather_forecast = [{'temperature': 35.0, 'humidity': 50.0}] * self.config.prediction_horizon
        
        result = self.mpc.compute_optimal_setpoint(
            current_state=state,
            occupancy_info=occupancy_info,
            weather_forecast=weather_forecast,
            current_time=current_time
        )
        
        return result
    
    def get_optimal_setpoints_multi_screen(
        self,
        screen_measurements: Dict[str, Dict[str, float]],
        occupancy_responses: Dict[str, Dict[str, Any]],
        weather_forecast: Optional[List[Dict[str, float]]] = None,
        current_time: Optional[datetime] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get optimal setpoints for multiple screens.
        
        Args:
            screen_measurements: Dict of {screen_name: measurements_dict}
            occupancy_responses: Dict of {screen_name: occupancy_api_response}
            weather_forecast: Shared weather forecast
            current_time: Current datetime
            
        Returns:
            Dict of {screen_name: optimal_setpoint_result}
        """
        results = {}
        
        for screen_name in screen_measurements.keys():
            if screen_name not in occupancy_responses:
                continue
                
            try:
                result = self.get_optimal_setpoint(
                    current_measurements=screen_measurements[screen_name],
                    occupancy_api_response=occupancy_responses[screen_name],
                    weather_forecast=weather_forecast,
                    current_time=current_time
                )
                results[screen_name] = result
            except Exception as e:
                results[screen_name] = {
                    'error': str(e),
                    'optimal_setpoint': self.config.sp_max_occupied  
                }
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        self.thermal_model.save(filepath)
        
    def load_model(self, filepath: str):
        """Load trained model from file."""
        self.thermal_model.load(filepath)
        self.mpc = CinemaAHUMPC(self.thermal_model, self.config)
        self.is_initialized = True



def create_disturbance_forecast_from_weather_api(
    base_time: datetime,
    horizon_steps: int,
    sample_time_minutes: int,
    outdoor_temp: float,
    outdoor_humidity: float
) -> List[Dict[str, float]]:
    """
    Create a simple disturbance forecast assuming constant weather.
    
    For production, this should be replaced with actual weather API forecast.
    
    Args:
        base_time: Starting time for forecast
        horizon_steps: Number of forecast steps
        sample_time_minutes: Time between steps
        outdoor_temp: Current outdoor temperature
        outdoor_humidity: Current outdoor humidity
        
    Returns:
        List of weather forecast dictionaries
    """
    forecast = []
    for k in range(horizon_steps):
        forecast.append({
            'temperature': outdoor_temp,
            'humidity': outdoor_humidity
        })
    return forecast


# if __name__ == "__main__":
    # import json
    
    # # AHU identifier for model naming
    # AHU_ID = "ahu13"
    
    # print("=" * 70)
    # print(f"CINEMA AHU SUPERVISORY MPC SYSTEM - {AHU_ID.upper()}")
    # print("=" * 70)
    
    # # Load training data
    # data_path = f"notebooks/{AHU_ID}_preprocessed_data.csv"
    
    # try:
    #     df = pd.read_csv(data_path)
    #     print(f"\nLoaded training data: {len(df)} samples")
    #     print(f"Columns: {list(df.columns)}")
    # except FileNotFoundError:
    #     print(f"Training data not found at {data_path}")
    #     print("Please ensure the preprocessed data file exists.")
    #     exit(1)
    
    # # Initialize MPC system with configuration
    # config = MPCConfig(
    #     prediction_horizon=6,       # 60 minutes lookahead
    #     control_horizon=3,          # 30 minutes control horizon
    #     sample_time_minutes=10,
    #     sp_min_occupied=23.0,
    #     sp_max_occupied=26.0,
    #     sp_unoccupied=28.0,
    #     temp_comfort_target=23.5,
    #     w_comfort=100.0,
    #     w_energy=1.0,
    #     w_smoothness=50.0
    # )
    
    # mpc_system = CinemaAHUMPCSystem(config)
    
    # # Train the model
    # print("\n" + "-" * 70)
    # print(f"TRAINING THERMAL PREDICTION MODEL FOR {AHU_ID.upper()}")
    # print("-" * 70)
    # metrics = mpc_system.train(df)
    
    # # Save model in AHU-specific folder: artifacts/{ahu_id}/mpc_model.joblib
    # model_dir = os.path.join("artifacts", AHU_ID)
    # os.makedirs(model_dir, exist_ok=True)
    # model_path = os.path.join(model_dir, "mpc_model.joblib")
    # mpc_system.save_model(model_path)
    
    # # Demo: Simulate different occupancy scenarios
    # print("\n" + "-" * 70)
    # print("DEMO: MPC OPTIMIZATION FOR DIFFERENT SCENARIOS")
    # print("-" * 70)
    
    # # Current measurements (simulated)
    # current_measurements = {
    #     'TempSp1': 24.5,
    #     'TempSp1_lag_10m': 24.3,
    #     'SpTREff': 24.0,
    #     'SpTREff_lag_10m': 24.0,
    #     'TempSu': 18.5,
    #     'FbVFD': 65.0,
    #     'FbFAD': 45.0
    # }
    
    # # Weather forecast (simplified)
    # weather_forecast = [{'temperature': 35.0, 'humidity': 55.0}] * 6
    
    # # Scenario 1: Movie currently playing
    # print("\n1. SCENARIO: Movie Currently Playing (OCCUPIED)")
    # occupancy_playing = {
    #     "status": 1,
    #     "movie_name": "The Spongebob Movie",
    #     "time_remaining": "108 minutes"
    # }
    # result1 = mpc_system.get_optimal_setpoint(
    #     current_measurements=current_measurements,
    #     occupancy_api_response=occupancy_playing,
    #     weather_forecast=weather_forecast
    # )
    # print(f"   Mode: {result1['mode']}")
    # print(f"   Optimal Setpoint: {result1['optimal_setpoint']}°C")
    # print(f"   Target Temperature: {result1.get('target_temperature', 'N/A')}°C")
    # print(f"   Predicted Trajectory: {[round(t, 2) for t in result1.get('predicted_trajectory', [])[:3]]}...")
    
    # # Scenario 2: Pre-cooling (movie starting in 30 min)
    # print("\n2. SCENARIO: Pre-Cooling (movie in 30 min)")
    # occupancy_precool = {
    #     "status": 0,
    #     "time_until_next_movie": "30 minutes",
    #     "next_movie_name": "Mercy (18TC)"
    # }
    # result2 = mpc_system.get_optimal_setpoint(
    #     current_measurements=current_measurements,
    #     occupancy_api_response=occupancy_precool,
    #     weather_forecast=weather_forecast
    # )
    # print(f"   Mode: {result2['mode']}")
    # print(f"   Optimal Setpoint: {result2['optimal_setpoint']}°C")
    # print(f"   Target Temperature: {result2.get('target_temperature', 'N/A')}°C")
    # print(f"   Time Until Event: {result2.get('time_until_event')} minutes")
    
    # # Scenario 3: Unoccupied (no movie for 2 hours)
    # print("\n3. SCENARIO: Unoccupied (no movie for 2 hours)")
    # occupancy_unoccupied = {
    #     "status": 0,
    #     "time_until_next_movie": "120 minutes",
    #     "next_movie_name": "Avatar 3"
    # }
    # result3 = mpc_system.get_optimal_setpoint(
    #     current_measurements=current_measurements,
    #     occupancy_api_response=occupancy_unoccupied,
    #     weather_forecast=weather_forecast
    # )
    # print(f"   Mode: {result3['mode']}")
    # print(f"   Optimal Setpoint: {result3['optimal_setpoint']}°C")
    # print(f"   Bypass Optimization: {result3['bypass_optimization']}")
    # print(f"   Reason: {result3.get('reason', 'N/A')}")
    
    # # Scenario 4: No upcoming shows
    # print("\n4. SCENARIO: No Upcoming Shows")
    # occupancy_none = {
    #     "status": 0,
    #     "time_until_next_movie": "No upcoming shows"
    # }
    # result4 = mpc_system.get_optimal_setpoint(
    #     current_measurements=current_measurements,
    #     occupancy_api_response=occupancy_none,
    #     weather_forecast=weather_forecast
    # )
    # print(f"   Mode: {result4['mode']}")
    # print(f"   Optimal Setpoint: {result4['optimal_setpoint']}°C")
    # print(f"   Bypass Optimization: {result4['bypass_optimization']}")
    
    # print("\n" + "=" * 70)
    # print("MPC SYSTEM DEMO COMPLETE")
    # print("=" * 70)
