from src.bms_ai.logger_config import setup_logger
from src.bms_ai.pipelines.training_pipeline import TrainingPipeline
from src.bms_ai.pipelines.prescriptive_pipeline import find_optimal_setpoints
from unittest.mock import MagicMock
import pandas as pd

log = setup_logger(__name__)

def run_prescriptive_analysis():
    """
    Example function to run the prescriptive analysis pipeline with mock data.
    """
    log.info("="*50)
    log.info("Running Prescriptive Analytics Example")
    log.info("="*50)

    mock_model_fan_power = MagicMock()
    mock_model_fan_power.predict.side_effect = lambda df: df['SA Pressure setpoint'] * 10 + df['RA temperature setpoint'] * 0.5

    mock_model_ra_temp = MagicMock()
    mock_model_ra_temp.predict.side_effect = lambda df: df['RA temperature setpoint'] + (df['OA Temp'] - df['RA temperature setpoint']) * 0.1

    mock_model_ra_co2 = MagicMock()
    mock_model_ra_co2.predict.side_effect = lambda df: 400 + df['SA Pressure setpoint'] * 50



    current_state = {
        'RA temperature setpoint': 22.0,
        'SA Pressure setpoint': 1.5,
        'OA Temp': 30.0,
        'OA Humid': 70.0
    }
    comfort_zone = {
        'RA Temp': {'min': 21.0, 'max': 23.0},
        'RA CO2': {'max': 800}
    }
    target_reduction_pct = 10.0

    recommendation = find_optimal_setpoints(
        current_state,
        target_reduction_pct,
        comfort_zone,
        models
    )

    log.info(f"\\nFinal Recommendation:\\n{recommendation}")


def main():
    """
    Main function to train the model using the existing pipeline.
    """
    try:
        log.info("Starting the model training process...")
        pipeline = TrainingPipeline()
        pipeline.main()
        log.info("Model training completed successfully.")
    except Exception as e:
        log.error(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    main()
    log.debug("Initiating prescriptive analysis...")
    run_prescriptive_analysis()
    log.debug("Prescriptive analysis completed.")
