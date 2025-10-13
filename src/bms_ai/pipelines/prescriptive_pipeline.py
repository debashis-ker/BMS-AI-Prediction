import pandas as pd
import numpy as np
from src.bms_ai.logger_config import setup_logger
import pickle
import joblib
from copy import deepcopy
import os
import itertools
import time

log = setup_logger(__name__)

class PrescriptivePipeline:
    """
    A pipeline for finding optimal operational setpoints to minimize fan power consumption.
    """
    def __init__(self, model_path="artifacts/model.pkl", preprocessor_path="artifacts/preprocessor.pkl"):
        """
        Initializes the pipeline by loading the trained model and preprocessor.

        Args:
            model_path (str, optional): Path to the trained model .pkl file. Defaults to None.
            preprocessor_path (str, optional): Path to the preprocessor .pkl file. Defaults to None.
        """
        log.info("Initializing PrescriptivePipeline...")
        self.model = None
        self.preprocessor = None
        self._load_artifacts(model_path, preprocessor_path)

    def _load_artifacts(self, model_path, preprocessor_path):
        """Loads the model and preprocessor from disk."""
        print(f"DEBUG: Starting _load_artifacts with model_path={model_path}, preprocessor_path={preprocessor_path}")
        try:
            if model_path is None or preprocessor_path is None:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                model_path = model_path or os.path.join(project_root, 'artifacts', 'model.pkl')
                preprocessor_path = preprocessor_path or os.path.join(project_root, 'artifacts', 'preprocessor.pkl')
            log.info(f"Loading model from: {model_path}")
            try:
                self.model = joblib.load(model_path)
            except Exception as model_error:
                try:
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                except Exception as pickle_error:
                    raise Exception(f"Cannot load model.pkl with either joblib or pickle. Joblib error: {model_error}, Pickle error: {pickle_error}")
            log.info(f"Loading preprocessor from: {preprocessor_path}")
            try:
                self.preprocessor = joblib.load(preprocessor_path)
            except Exception as prep_error:
                try:
                    with open(preprocessor_path, 'rb') as f:
                        self.preprocessor = pickle.load(f)
                except Exception as pickle_error:
                    raise Exception(f"Cannot load preprocessor.pkl with either joblib or pickle. Joblib error: {prep_error}, Pickle error: {pickle_error}")
            log.info("Model and preprocessor loaded successfully.")
        except FileNotFoundError as e:
            log.error(f"Error loading artifacts: {e}. Make sure 'model.pkl' and 'preprocessor.pkl' exist.")
            raise
        except Exception as e:
            log.error(f"An unexpected error occurred during artifact loading: {e}")
            raise

    def _find_single_feature_optimal(self, current_conditions: dict, setpoint_feature: str, setpoint_range: np.ndarray) -> dict:
        """
        Private method to find the optimal setpoint for a single feature.
        """
        print(f"DEBUG: Starting optimization for feature: '{setpoint_feature}'")
        log.info(f"Starting optimization for feature: '{setpoint_feature}'")
        best_setpoint = None
        min_fan_power = float('inf')
        if not self.model or not self.preprocessor:
            log.error("Model or preprocessor not loaded. Cannot run optimization.")
            return {"message": "Model or preprocessor not loaded."}
        for test_value in setpoint_range:
            temp_conditions = deepcopy(current_conditions)
            temp_conditions[setpoint_feature] = test_value
            input_df = pd.DataFrame([temp_conditions])
            try:
                input_df_reordered = input_df[self.preprocessor.feature_names_in_]
                transformed_data = self.preprocessor.transform(input_df_reordered)
                feature_names_out = self.preprocessor.get_feature_names_out()
                transformed_df = pd.DataFrame(transformed_data, columns=feature_names_out)
            except Exception as e:
                log.error(f"Failed to align DataFrame columns or transform data: {e}")
                continue
            predicted_values = self.model.predict(transformed_df)
            predicted_fan_power = predicted_values[0][3]
            if predicted_fan_power < min_fan_power:
                min_fan_power = predicted_fan_power
                best_setpoint = test_value
                log.info(f"New best found! Setpoint: {best_setpoint:.2f}, Power: {min_fan_power:.4f} KW")
        if best_setpoint is None:
            log.warning(f"Optimization for '{setpoint_feature}' finished, but no best setpoint was found.")
            return {"message": "Could not determine an optimal setpoint for this feature."}
        result = {
            'best_setpoint': best_setpoint,
            'min_fan_power_kw': min_fan_power
        }
        log.info(f"Optimization for '{setpoint_feature}' complete. Best result: {result}")
        return result

    def run_optimization(self, current_conditions: dict, search_space: dict) -> dict:
        """
        Runs the optimization process testing all combinations of setpoint features.

        Args:
            current_conditions (dict): The current state of the system.
            search_space (dict): A dictionary where keys are feature names and values are iterables of test points.

        Returns:
            dict: A dictionary containing the best combination of setpoints and minimum fan power.
        """
        
        log.info("Starting comprehensive grid search optimization...")
        start_time = time.time()
        if not self.model or not self.preprocessor:
            log.error("Model or preprocessor not loaded. Cannot run optimization.")
            return {"message": "Model or preprocessor not loaded."}
        feature_names = list(search_space.keys())
        feature_ranges = list(search_space.values())
        total_combinations = 1
        for range_vals in feature_ranges:
            total_combinations *= len(range_vals)
        log.info(f"Testing {total_combinations} combinations across {len(feature_names)} features:")
        for feature, range_vals in search_space.items():
            log.info(f"  - {feature}: {len(range_vals)} values from {min(range_vals):.1f} to {max(range_vals):.1f}")
        best_combination = None
        min_fan_power = float('inf')
        combination_count = 0
        for combination in itertools.product(*feature_ranges):
            combination_count += 1
            test_conditions = deepcopy(current_conditions)
            current_setpoints = {}
            for i, feature_name in enumerate(feature_names):
                test_conditions[feature_name] = combination[i]
                current_setpoints[feature_name] = combination[i]
            try:
                input_df = pd.DataFrame([test_conditions])
                input_df_reordered = input_df[self.preprocessor.feature_names_in_]
                transformed_data = self.preprocessor.transform(input_df_reordered)
                feature_names_out = self.preprocessor.get_feature_names_out()
                transformed_df = pd.DataFrame(transformed_data, columns=feature_names_out)
                predicted_values = self.model.predict(transformed_df)
                predicted_fan_power = predicted_values[0][3]
                if predicted_fan_power < min_fan_power:
                    min_fan_power = predicted_fan_power
                    best_combination = current_setpoints.copy()
                    log.info(f"NEW BEST! Combination {combination_count}: {best_combination} -> {min_fan_power:.4f} KW")
            except Exception as e:
                if combination_count <= 10:
                    log.error(f"Error testing combination {combination_count}: {e}")
                continue
        if best_combination is None:
            log.warning("No valid combination found that could be processed.")
            return {"message": "Could not find optimal setpoint combination."}
        end_time = time.time()
        elapsed_time = end_time - start_time
        log.info(f"Optimization completed in {elapsed_time:.2f} seconds.")
        print(f"Optimization completed in {elapsed_time:.2f} seconds.")
        result = {
            'best_setpoints': best_combination,
            'min_fan_power_kw': min_fan_power,
            'total_combinations_tested': combination_count,
            'optimization_method': 'comprehensive_grid_search',
            'optimization_time_seconds': elapsed_time
        }
        log.info(f"\n\n{'='*50} OPTIMIZATION COMPLETE {'='*50}")
        log.info(f"Tested {combination_count} combinations")
        log.info(f"Best setpoint combination found:")
        for feature, value in best_combination.items():
            log.info(f"  - {feature}: {value}")
        log.info(f"Minimum predicted fan power: {min_fan_power:.4f} KW")
        log.info("="*100)
        return result

def get_current_conditions():
    return {
        "Bag filter dirty status": 0,
        "OA Flow": 1000.8904418945312,
        "OA Humid": 80.93263626098633,
        "OA Temp": 40.93263626098633,
        "Plant enable": 1,
        "RA  temperature setpoint": 24.5,
        "RA CO2": 500,
        "RA CO2 setpoint": 750.0,
        "RA Damper feedback": 15.0,
        "RA Temp": 23.376264572143555,
        "SA Fan Speed feedback": 100.0,
        "SA Pressure setpoint": 750.0,
        "SA pressure": 297.919921875,
        "SA temp": 18.363384246826172,
        "Sup fan cmd": 1,
        "Trip status": 1.0,
        "airflow Status": 1,
        "auto Status": 1,
        "pre Filter dirty staus": 0,
        "hour": 6,
        "dayofweek": 2,
        "month": 5,
        "dayofyear": 142,
        'xtra_feature1': 0.0,
    }

if __name__ == '__main__':
    import time
    try:
        pipeline = PrescriptivePipeline()
        search_space = {
            'RA  temperature setpoint': np.arange(20.0, 27.5, 1),
            'RA CO2 setpoint': np.arange(500.0, 825.0, 50.0),
            'SA Pressure setpoint': np.arange(500.0, 1250.0, 50.0)
        }
        total_combos = 1
        for feature, values in search_space.items():
            total_combos *= len(values)
        initial_conditions = get_current_conditions()
        main_start_time = time.time()
        res = pipeline.run_optimization(initial_conditions, search_space)
        main_end_time = time.time()
        main_elapsed_time = main_end_time - main_start_time
        log.info(f"\nFinal Recommendation:\n{res}")
        print(f"\nFinal Recommendation:\n{res=}")
        print(f"Total time for optimization (including overhead): {main_elapsed_time:.2f} seconds.")
        log.info(f"Total time for optimization (including overhead): {main_elapsed_time:.2f} seconds.")
    except Exception as e:
        print(f"Exception caught in main: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        log.critical(f"A critical error occurred during the prescriptive pipeline execution: {e}")
