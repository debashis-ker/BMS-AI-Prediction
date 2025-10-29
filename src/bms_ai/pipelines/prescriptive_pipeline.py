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

    def run_optimization(self, current_conditions: dict, search_space: dict, 
                         optimization_method: str = "grid", n_iterations: int = 1000) -> dict:
        """
        Runs the optimization process using the specified method.

        Args:
            current_conditions (dict): The current state of the system.
            search_space (dict): A dictionary where keys are feature names and values are either:
                - List of discrete values (for grid search)
                - [min, max] tuple (for random/hybrid search)
            optimization_method (str): 'grid', 'random', or 'hybrid'
            n_iterations (int): Number of iterations for random/hybrid methods

        Returns:
            dict: A dictionary containing the best combination of setpoints and minimum fan power.
        """
        
        log.info(f"Starting {optimization_method} optimization...")
        start_time = time.time()
        
        if not self.model or not self.preprocessor:
            log.error("Model or preprocessor not loaded. Cannot run optimization.")
            return {"message": "Model or preprocessor not loaded."}
        
        if optimization_method == "grid":
            result = self._grid_search_optimization(current_conditions, search_space, start_time)
        elif optimization_method == "random":
            result = self._random_search_optimization(current_conditions, search_space, n_iterations, start_time)
        elif optimization_method == "hybrid":
            result = self._hybrid_search_optimization(current_conditions, search_space, n_iterations, start_time)
        else:
            log.error(f"Unknown optimization method: {optimization_method}")
            return {"message": f"Unknown optimization method: {optimization_method}"}
        
        return result
    
    def _grid_search_optimization(self, current_conditions: dict, search_space: dict, start_time: float) -> dict:
        """Grid search: exhaustive search over all combinations."""
        feature_names = list(search_space.keys())
        feature_ranges = list(search_space.values())
        
        total_combinations = 1
        for range_vals in feature_ranges:
            total_combinations *= len(range_vals)
        
        log.info(f"Grid Search: Testing {total_combinations} combinations across {len(feature_names)} features")
        
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
                predicted_fan_power = self._predict_fan_power(test_conditions)
                
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
        
        return {
            'best_setpoints': best_combination,
            'min_fan_power_kw': min_fan_power,
            'total_combinations_tested': combination_count,
            'optimization_method': 'grid',
            'optimization_time_seconds': elapsed_time
        }
    
    def _random_search_optimization(self, current_conditions: dict, search_space: dict, 
                                     n_iterations: int, start_time: float) -> dict:
        """Random search: sample random points from continuous ranges."""
        feature_names = list(search_space.keys())
        
        log.info(f"Random Search: Testing {n_iterations} random combinations")
        
        best_combination = None
        min_fan_power = float('inf')
        
        for iteration in range(n_iterations):
            test_conditions = deepcopy(current_conditions)
            current_setpoints = {}
            
            for feature_name in feature_names:
                feature_range = search_space[feature_name]
                
                if len(feature_range) == 2:
                    random_value = np.random.uniform(feature_range[0], feature_range[1])
                else:
                    random_value = np.random.choice(feature_range)
                
                test_conditions[feature_name] = random_value
                current_setpoints[feature_name] = random_value
            
            try:
                predicted_fan_power = self._predict_fan_power(test_conditions)
                
                if predicted_fan_power < min_fan_power:
                    min_fan_power = predicted_fan_power
                    best_combination = current_setpoints.copy()
                    log.info(f"NEW BEST! Iteration {iteration+1}: {best_combination} -> {min_fan_power:.4f} KW")
            except Exception as e:
                if iteration < 10:
                    log.error(f"Error in iteration {iteration+1}: {e}")
                continue
        
        if best_combination is None:
            log.warning("No valid combination found.")
            return {"message": "Could not find optimal setpoint combination."}
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return {
            'best_setpoints': best_combination,
            'min_fan_power_kw': min_fan_power,
            'total_combinations_tested': n_iterations,
            'optimization_method': 'random',
            'optimization_time_seconds': elapsed_time
        }
    
    def _hybrid_search_optimization(self, current_conditions: dict, search_space: dict, 
                                     n_iterations: int, start_time: float) -> dict:
        """Hybrid search: coarse grid search followed by random refinement."""
        feature_names = list(search_space.keys())
        
        log.info(f"Hybrid Search: Phase 1 - Coarse grid search")
        
        coarse_grid = {}
        for feature_name in feature_names:
            feature_range = search_space[feature_name]
            if len(feature_range) == 2:
                coarse_grid[feature_name] = np.linspace(feature_range[0], feature_range[1], 5)
            else:
                step = max(1, len(feature_range) // 5)
                coarse_grid[feature_name] = feature_range[::step][:5]
        
        grid_result = self._grid_search_optimization(current_conditions, coarse_grid, start_time)
        
        if "message" in grid_result:
            return grid_result
        
        best_grid_setpoints = grid_result['best_setpoints']
        best_grid_power = grid_result['min_fan_power_kw']
        grid_iterations = grid_result['total_combinations_tested']
        
        log.info(f"Hybrid Search: Phase 2 - Random refinement around best grid point")
        
        best_combination = best_grid_setpoints.copy()
        min_fan_power = best_grid_power
        
        refinement_iterations = n_iterations - grid_iterations
        
        for iteration in range(refinement_iterations):
            test_conditions = deepcopy(current_conditions)
            current_setpoints = {}
            
            for feature_name in feature_names:
                feature_range = search_space[feature_name]
                grid_best_value = best_grid_setpoints[feature_name]
                
                if len(feature_range) == 2:
                    range_width = feature_range[1] - feature_range[0]
                    window = range_width * 0.2
                    lower_bound = max(feature_range[0], grid_best_value - window)
                    upper_bound = min(feature_range[1], grid_best_value + window)
                    random_value = np.random.uniform(lower_bound, upper_bound)
                else:
                    random_value = np.random.choice(feature_range)
                
                test_conditions[feature_name] = random_value
                current_setpoints[feature_name] = random_value
            
            try:
                predicted_fan_power = self._predict_fan_power(test_conditions)
                
                if predicted_fan_power < min_fan_power:
                    min_fan_power = predicted_fan_power
                    best_combination = current_setpoints.copy()
                    log.info(f"NEW BEST! Refinement {iteration+1}: {best_combination} -> {min_fan_power:.4f} KW")
            except Exception as e:
                if iteration < 10:
                    log.error(f"Error in refinement iteration {iteration+1}: {e}")
                continue
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return {
            'best_setpoints': best_combination,
            'min_fan_power_kw': min_fan_power,
            'total_combinations_tested': grid_iterations + refinement_iterations,
            'optimization_method': 'hybrid',
            'optimization_time_seconds': elapsed_time
        }
    
    def _predict_fan_power(self, conditions: dict) -> float:
        """Helper method to predict fan power for given conditions."""
        input_df = pd.DataFrame([conditions])
        input_df_reordered = input_df[self.preprocessor.feature_names_in_]
        transformed_data = self.preprocessor.transform(input_df_reordered)
        feature_names_out = self.preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names_out)
        predicted_values = self.model.predict(transformed_df)
        return predicted_values[0][3]

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
