"""
Damper Optimization Pipeline
Single-file implementation for training a surrogate model and optimizing damper setpoints.
Target: Minimize FbFAD (Fresh Air Damper Feedback)
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import itertools
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from logger_config import setup_logger
from exception import CustomException

log = setup_logger(__name__)

FEATURE_COLUMNS = [
    'CMDSpdVFD', 'CmdCHW', 'CmdVFD', 'Co2Avg', 'Co2RA', 'Co2RA2', 'FbVFD',
    'HuAvg1', 'HuR', 'HuR1', 'HuR2', 'HuSu', 'PIDTR', 'PIDVFD', 'SpMinVFD',
    'SpTREff', 'SpTROcc', 'StaFlw', 'StaVFDSf', 'TRe', 'TSu', 'TempSp1',
    'TempSp2', 'TrAvg', 'TsOn',
    'month', 'hour', 'week_number', 'is_weekend'
]

SETPOINT_NAMES = ['SpMinVFD', 'SpTREff', 'SpTROcc']
TARGET_VARIABLE = 'FbFAD'

@dataclass
class DataTransformationConfig:
    """Configuration for data transformation artifacts."""
    scaler_path: str = os.path.join('artifacts', 'production_models', 'damper_scaler.pkl')
    label_encoders_path: str = os.path.join('artifacts', 'production_models', 'damper_label_encoders.pkl')
    processed_train_path: str = os.path.join('artifacts', 'production_models', 'damper_train_processed.csv')
    processed_test_path: str = os.path.join('artifacts', 'production_models', 'damper_test_processed.csv')


class DamperDataTransformation:
    """Handles data preprocessing for damper optimization."""
    
    def __init__(self):
        self.config = DataTransformationConfig()
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        
    def identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numeric and categorical columns using select_dtypes (matching notebook approach).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (numeric_columns, categorical_columns)
        """
        # First convert columns to numeric where possible (matching notebook: pd.to_numeric with errors='ignore')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        # Use select_dtypes to identify column types (matching notebook)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
        log.info(f"Identified {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
        return numeric_cols, categorical_cols
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Preprocess data: convert to numeric, apply MinMax scaling, and label encode.
        
        Args:
            df: Input DataFrame with features
            fit: Whether to fit transformers (True for training, False for inference)
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            log.info(f"Starting preprocessing. Fit mode: {fit}")
            df_processed = df.copy()
            
            # Identify column types (this also converts to numeric where possible)
            numeric_cols, categorical_cols = self.identify_column_types(df_processed)
            
            # Label encode categorical columns first
            if categorical_cols:
                for col in categorical_cols:
                    if fit:
                        le = LabelEncoder()
                        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                        self.label_encoders[col] = le
                        log.info(f"Label encoded column: {col}")
                    else:
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            # Handle unseen labels
                            df_processed[col] = df_processed[col].astype(str).apply(
                                lambda x: le.transform([x])[0] if x in le.classes_ else -1
                            )
                        else:
                            log.warning(f"No encoder found for {col}, filling with -1")
                            df_processed[col] = -1
            
            # Apply MinMax scaling to numeric columns
            if numeric_cols:
                if fit:
                    df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
                    log.info(f"Fitted and scaled {len(numeric_cols)} numeric columns with MinMaxScaler")
                else:
                    df_processed[numeric_cols] = self.scaler.transform(df_processed[numeric_cols])
                    log.info(f"Scaled {len(numeric_cols)} numeric columns with MinMaxScaler")
            
            log.info("Preprocessing completed successfully")
            return df_processed
            
        except Exception as e:
            log.error(f"Error in preprocessing: {e}")
            raise CustomException(e, sys)
    
    def save_transformers(self):
        """Save scaler and label encoders to disk."""
        try:
            os.makedirs(os.path.dirname(self.config.scaler_path), exist_ok=True)
            joblib.dump(self.scaler, self.config.scaler_path)
            joblib.dump(self.label_encoders, self.config.label_encoders_path)
            log.info(f"Saved transformers to {os.path.dirname(self.config.scaler_path)}")
        except Exception as e:
            log.error(f"Error saving transformers: {e}")
            raise CustomException(e, sys)
    
    def load_transformers(self):
        """Load scaler and label encoders from disk."""
        try:
            if os.path.exists(self.config.scaler_path):
                self.scaler = joblib.load(self.config.scaler_path)
                log.info(f"Loaded scaler from {self.config.scaler_path}")
            
            if os.path.exists(self.config.label_encoders_path):
                self.label_encoders = joblib.load(self.config.label_encoders_path)
                log.info(f"Loaded label encoders from {self.config.label_encoders_path}")
                
        except Exception as e:
            log.error(f"Error loading transformers: {e}")
            raise CustomException(e, sys)
    
    def transform_dataset(self, data_path: str, equipment_id: str = "Ahu6") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform entire dataset from CSV with raw BMS data format.
        
        Args:
            data_path: Path to CSV file with raw BMS data (columns: system_type, equipment_id, datapoint, monitoring_data, data_received_on, etc.)
            equipment_id: Equipment ID to filter (default: "Ahu6")
            
        Returns:
            Tuple of (X_processed, y)
        """
        try:
            log.info(f"Loading raw BMS data from {data_path}")
            df = pd.read_csv(data_path)
            log.info(f"Raw data shape: {df.shape}")
            
            # Step 1: Filter for AHU system type
            if 'system_type' in df.columns:
                df = df[df['system_type'] == 'AHU'].copy()
                log.info(f"Filtered for AHU system type. Shape: {df.shape}")
            else:
                log.warning("'system_type' column not found, skipping filter")
            
            # Step 2: Filter for specific equipment
            if 'equipment_id' in df.columns:
                df = df[df['equipment_id'] == equipment_id].copy()
                log.info(f"Filtered for equipment '{equipment_id}'. Shape: {df.shape}")
            else:
                log.warning("'equipment_id' column not found, skipping filter")
            
            if df.empty:
                raise ValueError(f"No data found for system_type='AHU' and equipment_id='{equipment_id}'")
            
            # Step 3: Process timestamp
            if 'data_received_on' in df.columns:
                df['data_received_on'] = pd.to_datetime(df['data_received_on'], errors='coerce')
                
                # Handle timezone
                if df['data_received_on'].dt.tz is not None:
                    df['data_received_on_naive'] = df['data_received_on'].dt.tz_localize(None)
                else:
                    df['data_received_on_naive'] = df['data_received_on']
                
                df.sort_values('data_received_on_naive', inplace=True)
                log.info("Processed timestamps")
            else:
                raise ValueError("'data_received_on' column not found in data")
            
            # Step 4: Pivot the data
            log.info("Pivoting data from long to wide format...")
            pivoted_df = df.pivot_table(
                index='data_received_on_naive',
                columns='datapoint',
                values='monitoring_data',
                aggfunc='first'
            )
            
            # Step 4.5: Extract temporal features from timestamp index before reset
            log.info("Creating temporal features from timestamp...")
            pivoted_df['month'] = pivoted_df.index.month
            pivoted_df['hour'] = pivoted_df.index.hour
            pivoted_df['week_number'] = pivoted_df.index.isocalendar().week
            pivoted_df['is_weekend'] = (pivoted_df.index.dayofweek >= 5).astype(int)  # 5=Saturday, 6=Sunday
            log.info("Created temporal features: month, hour, week_number, is_weekend")
            
            pivoted_df.reset_index(drop=True, inplace=True)
            log.info(f"Pivoted data shape: {pivoted_df.shape}")
            log.info(f"Columns after pivot: {list(pivoted_df.columns)}")
            
            # Step 5: Validate required columns exist after pivoting
            required_cols = FEATURE_COLUMNS + [TARGET_VARIABLE]
            missing_cols = [col for col in required_cols if col not in pivoted_df.columns]
            if missing_cols:
                available_cols = list(pivoted_df.columns)
                raise ValueError(
                    f"Missing required columns after pivoting: {missing_cols}\n"
                    f"Available columns: {available_cols}\n"
                    f"Hint: Check if datapoint names in raw data match expected feature names"
                )
            
            # Step 6: Select only required columns
            df_final = pivoted_df[required_cols].copy()
            
            # Step 7: Convert columns to numeric with errors='ignore' (matching notebook approach)
            for col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors='ignore')
            log.info("Converted columns to numeric where possible")
            
            # Step 8: Drop rows with any NaN values (matching notebook: pivoted_table.dropna(how='any'))
            df_final.dropna(how='any', inplace=True)
            log.info(f"After dropping NaN rows: {df_final.shape}")
            
            if df_final.empty:
                raise ValueError("No valid data remaining after preprocessing")
            
            # Step 9: Separate features and target
            X = df_final[FEATURE_COLUMNS].copy()
            y = df_final[TARGET_VARIABLE].copy()
            
            log.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
            
            # Step 10: Preprocess features (convert to numeric, scale, encode)
            X_processed = self.preprocess_data(X, fit=True)
            
            return X_processed, y
            
        except Exception as e:
            log.error(f"Error transforming dataset: {e}")
            raise CustomException(e, sys)


# ==================== MODEL TRAINER ====================
@dataclass
class ModelTrainerConfig:
    """Configuration for model training."""
    model_path: str = os.path.join('artifacts', 'production_models', 'damper_model.pkl')
    metrics_path: str = os.path.join('artifacts', 'production_models', 'damper_metrics.txt')
    best_model_info_path: str = os.path.join('artifacts', 'production_models', 'best_model_info.txt')


class DamperModelTrainer:
    """Trains the damper optimization surrogate model with multiple algorithms and hyperparameter tuning."""
    
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.model = None
        self.best_model_name = None
        self.best_params = None
        
    def get_models_and_params(self) -> Dict[str, Tuple[Any, Dict]]:
        """
        Define models and their hyperparameter grids for tuning.
        
        Returns:
            Dictionary mapping model names to (model, param_grid) tuples
        """
        models_params = {
            'RandomForest': (
                RandomForestRegressor(random_state=42, n_jobs=-1),
                {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            ),
            'GradientBoosting': (
                GradientBoostingRegressor(random_state=42),
                {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'subsample': [0.8, 0.9, 1.0]
                }
            ),
            'XGBoost': (
                XGBRegressor(random_state=42, n_jobs=-1),
                {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7, 9],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            ),
            'ExtraTrees': (
                ExtraTreesRegressor(random_state=42, n_jobs=-1),
                {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            ),
            'Ridge': (
                Ridge(random_state=42),
                {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                }
            ),
            'Lasso': (
                Lasso(random_state=42, max_iter=10000),
                {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'selection': ['cyclic', 'random']
                }
            ),
            'ElasticNet': (
                ElasticNet(random_state=42, max_iter=10000),
                {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.2, 0.5, 0.8],
                    'selection': ['cyclic', 'random']
                }
            ),
            'KNN': (
                KNeighborsRegressor(n_jobs=-1),
                {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            )
        }
        
        return models_params
        
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_test: pd.DataFrame, y_test: pd.Series,
                   search_method: str = 'random',
                   cv_folds: int = 5,
                   n_iter: int = 20) -> Dict[str, Any]:
        """
        Train multiple models with hyperparameter tuning and select the best one.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            search_method: 'random' for RandomizedSearchCV or 'grid' for GridSearchCV
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for RandomizedSearchCV
            
        Returns:
            Dictionary of evaluation metrics and model info
        """
        try:
            log.info("="*60)
            log.info("TRAINING MULTIPLE MODELS WITH HYPERPARAMETER TUNING")
            log.info(f"Search Method: {search_method.upper()}")
            log.info(f"CV Folds: {cv_folds}")
            if search_method == 'random':
                log.info(f"Random Search Iterations: {n_iter}")
            log.info("="*60)
            
            models_params = self.get_models_and_params()
            best_score = -np.inf
            best_model = None
            best_model_name = None
            best_params = None
            all_results = {}
            
            # Train and evaluate each model
            for model_name, (base_model, param_grid) in models_params.items():
                try:
                    log.info(f"\n{'='*60}")
                    log.info(f"Training: {model_name}")
                    log.info(f"{'='*60}")
                    
                    # Perform hyperparameter search
                    if search_method == 'random':
                        search = RandomizedSearchCV(
                            base_model,
                            param_distributions=param_grid,
                            n_iter=n_iter,
                            cv=cv_folds,
                            scoring='r2',
                            n_jobs=-1,
                            random_state=42,
                            verbose=1
                        )
                    else:  # grid search
                        search = GridSearchCV(
                            base_model,
                            param_grid=param_grid,
                            cv=cv_folds,
                            scoring='r2',
                            n_jobs=-1,
                            verbose=1
                        )
                    
                    # Fit the search
                    search.fit(X_train, y_train)
                    
                    # Get best model from search
                    model = search.best_estimator_
                    
                    # Evaluate on train and test sets
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    train_mae = mean_absolute_error(y_train, y_train_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    
                    all_results[model_name] = {
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'cv_best_score': search.best_score_,
                        'best_params': search.best_params_
                    }
                    
                    log.info(f"\n{model_name} Results:")
                    log.info(f"  CV Best R2: {search.best_score_:.4f}")
                    log.info(f"  Train R2: {train_r2:.4f}")
                    log.info(f"  Test R2: {test_r2:.4f}")
                    log.info(f"  Test RMSE: {test_rmse:.4f}")
                    log.info(f"  Test MAE: {test_mae:.4f}")
                    log.info(f"  Best Params: {search.best_params_}")
                    
                    # Track best model based on test R2 score
                    if test_r2 > best_score:
                        best_score = test_r2
                        best_model = model
                        best_model_name = model_name
                        best_params = search.best_params_
                        
                except Exception as e:
                    log.error(f"Error training {model_name}: {e}")
                    continue
            
            if best_model is None:
                raise ValueError("No model was successfully trained")
            
            # Set the best model
            self.model = best_model
            self.best_model_name = best_model_name
            self.best_params = best_params
            
            log.info("\n" + "="*60)
            log.info("BEST MODEL SELECTED")
            log.info("="*60)
            log.info(f"Model: {best_model_name}")
            log.info(f"Test R2 Score: {best_score:.4f}")
            log.info(f"Best Parameters: {best_params}")
            log.info("="*60)
            
            # Return metrics for best model
            best_metrics = all_results[best_model_name].copy()
            best_metrics['best_model_name'] = best_model_name
            best_metrics['all_results'] = all_results
            
            return best_metrics
            
        except Exception as e:
            log.error(f"Error training models: {e}")
            raise CustomException(e, sys)
    
    def save_model(self):
        """Save trained model to disk."""
        try:
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            joblib.dump(self.model, self.config.model_path)
            log.info(f"Model saved to {self.config.model_path}")
        except Exception as e:
            log.error(f"Error saving model: {e}")
            raise CustomException(e, sys)
    
    def load_model(self):
        """Load trained model from disk."""
        try:
            if os.path.exists(self.config.model_path):
                self.model = joblib.load(self.config.model_path)
                log.info(f"Model loaded from {self.config.model_path}")
            else:
                raise FileNotFoundError(f"Model not found at {self.config.model_path}")
        except Exception as e:
            log.error(f"Error loading model: {e}")
            raise CustomException(e, sys)
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics and model information to text file."""
        try:
            os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)
            
            with open(self.config.metrics_path, 'w') as f:
                f.write("Damper Optimization Model Results\n")
                f.write("=" * 60 + "\n\n")
                
                # Best model info
                f.write(f"Best Model: {metrics.get('best_model_name', 'N/A')}\n")
                f.write(f"Best Parameters: {self.best_params}\n\n")
                
                f.write("Best Model Metrics:\n")
                f.write("-" * 60 + "\n")
                for key, value in metrics.items():
                    if key not in ['best_model_name', 'all_results'] and isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.4f}\n")
                
                # All models comparison
                if 'all_results' in metrics:
                    f.write("\n" + "=" * 60 + "\n")
                    f.write("All Models Comparison:\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for model_name, model_metrics in metrics['all_results'].items():
                        f.write(f"\n{model_name}:\n")
                        f.write("-" * 40 + "\n")
                        for key, value in model_metrics.items():
                            if key != 'best_params' and isinstance(value, (int, float)):
                                f.write(f"  {key}: {value:.4f}\n")
                            elif key == 'best_params':
                                f.write(f"  {key}: {value}\n")
            
            log.info(f"Metrics saved to {self.config.metrics_path}")
        except Exception as e:
            log.error(f"Error saving metrics: {e}")


# ==================== PREDICTION PIPELINE ====================
class DamperPredictionPipeline:
    """Handles predictions using trained model."""
    
    def __init__(self):
        self.model = None
        self.transformer = DamperDataTransformation()
        
    def load_artifacts(self):
        """Load model and transformers."""
        try:
            # Load model
            trainer = DamperModelTrainer()
            trainer.load_model()
            self.model = trainer.model
            
            # Load transformers
            self.transformer.load_transformers()
            
            log.info("All artifacts loaded successfully")
        except Exception as e:
            log.error(f"Error loading artifacts: {e}")
            raise CustomException(e, sys)
    
    def predict(self, input_data: Dict[str, Any]) -> float:
        """
        Make prediction for single input.
        
        Args:
            input_data: Dictionary with feature names and values
            
        Returns:
            Predicted FbFAD value
        """
        try:
            # Validate input
            missing_features = [f for f in FEATURE_COLUMNS if f not in input_data]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Create DataFrame
            df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
            
            # Preprocess
            df_processed = self.transformer.preprocess_data(df, fit=False)
            
            # Predict
            prediction = self.model.predict(df_processed)[0]
            
            return float(prediction)
            
        except Exception as e:
            log.error(f"Error in prediction: {e}")
            raise CustomException(e, sys)


# ==================== PRESCRIPTIVE PIPELINE ====================
class DamperPrescriptivePipeline:
    """Optimizes setpoints to minimize FbFAD."""
    
    def __init__(self):
        self.prediction_pipeline = DamperPredictionPipeline()
        
    def load_artifacts(self):
        """Load model and transformers."""
        self.prediction_pipeline.load_artifacts()
        log.info("Prescriptive pipeline ready")
    
    def _predict_fbfad(self, conditions: Dict[str, Any]) -> float:
        """Predict FbFAD for given conditions."""
        return self.prediction_pipeline.predict(conditions)
    
    def run_optimization(self, current_conditions: Dict[str, Any], 
                        search_space: Dict[str, List[float]],
                        optimization_method: str = "grid",
                        n_iterations: int = 1000) -> Dict[str, Any]:
        """
        Optimize setpoints to minimize FbFAD.
        
        Args:
            current_conditions: Current system state
            search_space: Ranges for setpoint exploration
            optimization_method: 'grid', 'random', or 'hybrid'
            n_iterations: Number of iterations for random/hybrid
            
        Returns:
            Dictionary with best setpoints and minimum FbFAD
        """
        try:
            log.info(f"Starting {optimization_method} optimization for damper...")
            start_time = time.time()
            
            if optimization_method == "grid":
                result = self._grid_search_optimization(current_conditions, search_space, start_time)
            elif optimization_method == "random":
                result = self._random_search_optimization(current_conditions, search_space, n_iterations, start_time)
            elif optimization_method == "hybrid":
                result = self._hybrid_search_optimization(current_conditions, search_space, n_iterations, start_time)
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
            
            return result
            
        except Exception as e:
            log.error(f"Optimization failed: {e}")
            raise CustomException(e, sys)
    
    def _grid_search_optimization(self, current_conditions: Dict[str, Any],
                                 search_space: Dict[str, List[float]],
                                 start_time: float) -> Dict[str, Any]:
        """Grid search optimization."""
        feature_names = list(search_space.keys())
        feature_ranges = list(search_space.values())
        
        total_combinations = np.prod([len(r) for r in feature_ranges])
        log.info(f"Grid Search: Testing {total_combinations} combinations")
        
        best_setpoints = None
        min_fbfad = float('inf')
        combination_count = 0
        
        for combination in itertools.product(*feature_ranges):
            test_conditions = current_conditions.copy()
            
            # Update setpoints
            for feature_name, test_value in zip(feature_names, combination):
                test_conditions[feature_name] = test_value
            
            # Predict
            predicted_fbfad = self._predict_fbfad(test_conditions)
            
            if predicted_fbfad < min_fbfad:
                min_fbfad = predicted_fbfad
                best_setpoints = dict(zip(feature_names, combination))
            
            combination_count += 1
            
            if combination_count % 100 == 0:
                log.info(f"Tested {combination_count}/{total_combinations} combinations")
        
        elapsed_time = time.time() - start_time
        
        return {
            'best_setpoints': best_setpoints,
            'min_fbfad': min_fbfad,
            'total_combinations_tested': combination_count,
            'optimization_method': 'grid',
            'optimization_time_seconds': elapsed_time
        }
    
    def _random_search_optimization(self, current_conditions: Dict[str, Any],
                                   search_space: Dict[str, List[float]],
                                   n_iterations: int,
                                   start_time: float) -> Dict[str, Any]:
        """Random search optimization."""
        feature_names = list(search_space.keys())
        
        log.info(f"Random Search: Testing {n_iterations} combinations")
        
        best_setpoints = None
        min_fbfad = float('inf')
        
        for i in range(n_iterations):
            test_conditions = current_conditions.copy()
            
            # Sample random values
            for feature_name in feature_names:
                feature_range = search_space[feature_name]
                if isinstance(feature_range, list) and len(feature_range) == 2:
                    # Continuous range [min, max]
                    test_value = np.random.uniform(feature_range[0], feature_range[1])
                else:
                    # Discrete values
                    test_value = np.random.choice(feature_range)
                
                test_conditions[feature_name] = test_value
            
            # Predict
            predicted_fbfad = self._predict_fbfad(test_conditions)
            
            if predicted_fbfad < min_fbfad:
                min_fbfad = predicted_fbfad
                best_setpoints = {k: test_conditions[k] for k in feature_names}
            
            if (i + 1) % 100 == 0:
                log.info(f"Tested {i + 1}/{n_iterations} combinations")
        
        elapsed_time = time.time() - start_time
        
        return {
            'best_setpoints': best_setpoints,
            'min_fbfad': min_fbfad,
            'total_combinations_tested': n_iterations,
            'optimization_method': 'random',
            'optimization_time_seconds': elapsed_time
        }
    
    def _hybrid_search_optimization(self, current_conditions: Dict[str, Any],
                                   search_space: Dict[str, List[float]],
                                   n_iterations: int,
                                   start_time: float) -> Dict[str, Any]:
        """Hybrid search: coarse grid + random refinement."""
        feature_names = list(search_space.keys())
        
        log.info("Hybrid Phase 1: Coarse grid search")
        coarse_space = {}
        for name, rng in search_space.items():
            if isinstance(rng, list) and len(rng) == 2:
                coarse_space[name] = list(np.linspace(rng[0], rng[1], 5))
            else:
                coarse_space[name] = rng
        
        grid_result = self._grid_search_optimization(current_conditions, coarse_space, start_time)
        best_setpoints = grid_result['best_setpoints']
        min_fbfad = grid_result['min_fbfad']
        tested = grid_result['total_combinations_tested']
        
        log.info("Hybrid Phase 2: Random refinement")
        refinement_iters = n_iterations - tested
        
        for i in range(refinement_iters):
            test_conditions = current_conditions.copy()
            
            for feature_name in feature_names:
                feature_range = search_space[feature_name]
                best_val = best_setpoints[feature_name]
                
                if isinstance(feature_range, list) and len(feature_range) == 2:
                    range_width = feature_range[1] - feature_range[0]
                    explore_width = 0.2 * range_width
                    
                    test_val = np.random.uniform(
                        max(feature_range[0], best_val - explore_width),
                        min(feature_range[1], best_val + explore_width)
                    )
                else:
                    test_val = np.random.choice(feature_range)
                
                test_conditions[feature_name] = test_val
            
            predicted_fbfad = self._predict_fbfad(test_conditions)
            
            if predicted_fbfad < min_fbfad:
                min_fbfad = predicted_fbfad
                best_setpoints = {k: test_conditions[k] for k in feature_names}
            
            if (i + 1) % 100 == 0:
                log.info(f"Refinement: {i + 1}/{refinement_iters} iterations")
        
        elapsed_time = time.time() - start_time
        
        return {
            'best_setpoints': best_setpoints,
            'min_fbfad': min_fbfad,
            'total_combinations_tested': tested + refinement_iters,
            'optimization_method': 'hybrid',
            'optimization_time_seconds': elapsed_time
        }


def train(data_path: str, equipment_id: str = "Ahu1", test_size: float = 0.2,
          search_method: str = 'random', cv_folds: int = 5, n_iter: int = 20) -> Dict[str, Any]:
    """
    Main training function with multiple models and hyperparameter tuning.
    
    Args:
        data_path: Path to CSV with raw BMS data
        equipment_id: Equipment ID to filter (default: "Ahu1")
        test_size: Fraction for test split
        search_method: 'random' for RandomizedSearchCV or 'grid' for GridSearchCV
        cv_folds: Number of cross-validation folds
        n_iter: Number of iterations for RandomizedSearchCV
        
    Returns:
        Dictionary with training results
    """
    try:
        log.info("="*60)
        log.info("STARTING DAMPER OPTIMIZATION MODEL TRAINING")
        log.info(f"Equipment ID: {equipment_id}")
        log.info(f"Search Method: {search_method.upper()}")
        log.info("="*60)
        
        log.info("Step 1: Data Transformation")
        transformer = DamperDataTransformation()
        X, y = transformer.transform_dataset(data_path, equipment_id=equipment_id)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        log.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Save transformers
        transformer.save_transformers()
        
        # Step 2: Model Training with Hyperparameter Tuning
        log.info("Step 2: Multi-Model Training with Hyperparameter Tuning")
        trainer = DamperModelTrainer()
        metrics = trainer.train_model(
            X_train, y_train, X_test, y_test,
            search_method=search_method,
            cv_folds=cv_folds,
            n_iter=n_iter
        )
        
        # Save model and metrics
        trainer.save_model()
        trainer.save_metrics(metrics)
        
        log.info("="*60)
        log.info("TRAINING COMPLETED SUCCESSFULLY")
        log.info("="*60)
        
        return {
            'status': 'success',
            'best_model_name': metrics.get('best_model_name', 'Unknown'),
            'metrics': metrics,
            'model_path': trainer.config.model_path,
            'scaler_path': transformer.config.scaler_path
        }
        
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise CustomException(e, sys)


def optimize(current_conditions: Dict[str, Any],
            search_space: Optional[Dict[str, List[float]]] = None,
            optimization_method: str = "random",
            n_iterations: int = 1000) -> Dict[str, Any]:
    """
    Main optimization function.
    
    Args:
        current_conditions: Current system state
        search_space: Optional setpoint ranges
        optimization_method: 'grid', 'random', or 'hybrid'
        n_iterations: Iterations for random/hybrid
        
    Returns:
        Optimization results
    """
    try:
        log.info("="*60)
        log.info("STARTING DAMPER SETPOINT OPTIMIZATION")
        log.info("="*60)
        
        # Default search space if not provided
        if not search_space:
            if optimization_method == "grid":
                search_space = {
                    'SpMinVFD': list(np.arange(40.0, 80.0, 5.0)),
                    'SpTREff': list(np.arange(20.0, 26.0, 1.0)),
                    'SpTROcc': list(np.arange(22.0, 27.0, 1.0))
                }
            else:
                search_space = {
                    'SpMinVFD': [40.0, 80.0],
                    'SpTREff': [20.0, 26.0],
                    'SpTROcc': [22.0, 27.0]
                }
        
        pipeline = DamperPrescriptivePipeline()
        pipeline.load_artifacts()
        
        result = pipeline.run_optimization(
            current_conditions,
            search_space,
            optimization_method,
            n_iterations
        )
        
        log.info("="*60)
        log.info("OPTIMIZATION COMPLETED SUCCESSFULLY")
        log.info(f"Best Setpoints: {result['best_setpoints']}")
        log.info(f"Minimum FbFAD: {result['min_fbfad']:.4f}")
        log.info("="*60)
        
        return result
        
    except Exception as e:
        log.error(f"Optimization failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":

    example_conditions = {
        'CMDSpdVFD': 50.0,
        'CmdCHW': 30.0,
        'CmdVFD': 1,
        'Co2Avg': 600.0,
        'Co2RA': 580.0,
        'Co2RA2': 620.0,
        'FbVFD': 45.0,
        'HuAvg1': 55.0,
        'HuR': 54.0,
        'HuR1': 56.0,
        'HuR2': 55.0,
        'HuSu': 50.0,
        'PIDTR': 23.0,
        'PIDVFD': 48.0,
        'SpMinVFD': 40.0,
        'SpTREff': 22.0,
        'SpTROcc': 24.0,
        'StaFlw': 1,
        'StaVFDSf': 1,
        'TRe': 23.5,
        'TSu': 18.0,
        'TempSp1': 24.0,
        'TempSp2': 23.5,
        'TrAvg': 23.7,
        'TsOn': 1
    }
    
    opt_result = optimize(
        current_conditions=example_conditions,
        optimization_method="random",
        n_iterations=500
    )
    print(opt_result)
