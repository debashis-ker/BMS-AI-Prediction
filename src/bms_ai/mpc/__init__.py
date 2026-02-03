"""
MPC (Model Predictive Control) Module for BMS-AI
================================================

This module contains supervisory MPC controllers for building HVAC systems.

Components:
-----------
- CinemaAHUMPC: MPC controller for cinema AHU setpoint optimization
- CinemaAHUMPCSystem: Complete system wrapper with training and inference
- MPCConfig: Configuration dataclass for MPC parameters
- OccupancyMode: Enum for operational modes (OCCUPIED, PRE_COOLING, UNOCCUPIED)
- MPCTrainingPipeline: End-to-end training pipeline (fetch → preprocess → train)
"""

from .cinema_ahu_mpc import (
    CinemaAHUMPC,
    CinemaAHUMPCSystem,
    MPCConfig,
    MPCState,
    DisturbanceVector,
    OccupancyMode,
    OccupancyInfo,
    OccupancyLogicHandler,
    ThermalPredictionModel,
    create_disturbance_forecast_from_weather_api
)

from .mpc_training_pipeline import (
    MPCTrainingPipeline,
    PipelineConfig,
    DataFetcher,
    WeatherDataIntegrator,
    DataPreprocessor,
    train_mpc_from_scratch,
    load_trained_mpc
)

__all__ = [
    # Core MPC
    'CinemaAHUMPC',
    'CinemaAHUMPCSystem',
    'MPCConfig',
    'MPCState',
    'DisturbanceVector',
    'OccupancyMode',
    'OccupancyInfo',
    'OccupancyLogicHandler',
    'ThermalPredictionModel',
    'create_disturbance_forecast_from_weather_api',
    # Training Pipeline
    'MPCTrainingPipeline',
    'PipelineConfig',
    'DataFetcher',
    'WeatherDataIntegrator',
    'DataPreprocessor',
    'train_mpc_from_scratch',
    'load_trained_mpc'
]
