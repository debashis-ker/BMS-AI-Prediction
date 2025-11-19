"""
Generic Optimization Pipeline
Dynamic pipeline for training surrogate models and optimizing setpoints for any AHU equipment.
Automatically selects best features using correlation and mutual information.
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
import itertools
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from logger_config import setup_logger
from exception import CustomException

log = setup_logger(__name__)

# Fixed setpoints for all optimizations
SETPOINT_NAMES = ['SpMinVFD', 'SpTREff', 'SpTROcc']


# ==================== FEATURE SELECTION ====================
@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection artifacts."""
    correlation_plot_path: str = os.path.join('artifacts', 'generic_models', 'correlation_plot.png')
    mutual_info_plot_path: str = os.path.join('artifacts', 'generic_models', 'mutual_info_plot.png')
    selected_features_path: str = os.path.join('artifacts', 'generic_models', 'selected_features.txt')


class GenericFeatureSelector:
    """Automatically selects best features using correlation and mutual information."""
    
    def __init__(self, equipment_id: str, target_column: str):
        self.config = FeatureSelectionConfig()
        self.equipment_id = equipment_id
        self.target_column = target_column
        self.selected_features = []
        
    def select_features(self, df: pd.DataFrame, n_features: int = 20) -> List[str]:
        """
        Select top features based on correlation and mutual information.
        
        Args:
            df: DataFrame with features and target
            n_features: Total number of features to select (default: 20)
            
        Returns:
            List of selected feature names
        """
        try:
            log.info("="*60)
            log.info("STARTING AUTOMATIC FEATURE SELECTION")
            log.info(f"Target: {self.target_column}")
            log.info("="*60)
            
            # Ensure target exists and is numeric
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in data")
            
            # Check if target is numeric
            try:
                df[self.target_column] = pd.to_numeric(df[self.target_column], errors='raise')
            except (ValueError, TypeError):
                raise ValueError(f"Target column '{self.target_column}' must contain numeric values")
            
            # Separate features and target
            y = df[self.target_column].copy()
            X = df.drop(columns=[self.target_column]).copy()
            
            log.info(f"Initial features: {X.shape[1]}")
            
            # Transform features for analysis (convert to numeric or label encode)
            X_transformed = self._transform_for_analysis(X)
            
            # Drop rows with NaN in target or features
            valid_idx = X_transformed.notna().all(axis=1) & y.notna()
            X_transformed = X_transformed[valid_idx]
            y = y[valid_idx]
            
            log.info(f"Valid samples after dropping NaN: {X_transformed.shape[0]}")
            
            if X_transformed.empty:
                raise ValueError("No valid data remaining after transformation")
            
            # Calculate correlation
            log.info("Calculating correlation with target...")
            df_temp = pd.concat([X_transformed, y], axis=1)
            corr_matrix = df_temp.corr()
            target_corr = corr_matrix[self.target_column].drop(self.target_column)
            
            # Calculate mutual information
            log.info("Calculating mutual information with target...")
            mi_scores = mutual_info_regression(X_transformed, y, random_state=42)
            mi_series = pd.Series(mi_scores, index=X_transformed.columns)
            
            # Save plots
            self._plot_feature_importance(target_corr, mi_series)
            
            # Select top features
            # Top 10 from mutual information
            top_mi_features = mi_series.nlargest(10).index.tolist()
            
            # Top 10 from correlation (excluding those already in top MI)
            corr_abs = target_corr.abs()
            top_corr_features = []
            for feat in corr_abs.nlargest(n_features).index:
                if feat not in top_mi_features and len(top_corr_features) < 10:
                    top_corr_features.append(feat)
            
            # Combine features
            self.selected_features = top_mi_features + top_corr_features
            
            # Add setpoints if not present
            for setpoint in SETPOINT_NAMES:
                if setpoint in X.columns and setpoint not in self.selected_features:
                    self.selected_features.append(setpoint)
                    log.info(f"Added setpoint: {setpoint}")
            
            log.info(f"Selected {len(self.selected_features)} features")
            log.info(f"Top 10 MI features: {top_mi_features}")
            log.info(f"Top 10 Correlation features: {top_corr_features}")
            log.info(f"Setpoints included: {[s for s in SETPOINT_NAMES if s in self.selected_features]}")
            
            # Save selected features
            self._save_selected_features(target_corr, mi_series)
            
            return self.selected_features
            
        except Exception as e:
            log.error(f"Error in feature selection: {e}")
            raise CustomException(e, sys)
    
    def _transform_for_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features to numeric for correlation/MI analysis."""
        df_transformed = df.copy()
        
        for col in df_transformed.columns:
            # Try to convert to numeric
            df_transformed[col] = pd.to_numeric(df_transformed[col], errors='ignore')
        
        # Label encode categorical columns
        for col in df_transformed.columns:
            if df_transformed[col].dtype == 'object':
                le = LabelEncoder()
                df_transformed[col] = le.fit_transform(df_transformed[col].astype(str))
                log.info(f"Label encoded: {col}")
        
        return df_transformed
    
    def _plot_feature_importance(self, correlation: pd.Series, mutual_info: pd.Series):
        """Create and save plots for correlation and mutual information."""
        try:
            os.makedirs(os.path.dirname(self.config.correlation_plot_path), exist_ok=True)
            
            # Create combined plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot A: Correlation with Target
            sorted_corr = correlation.abs().nlargest(20).sort_values(ascending=True)
            sorted_corr.plot(kind='barh', color='skyblue', ax=axes[0])
            axes[0].set_title(f'Top 20 Features by Correlation with {self.target_column}')
            axes[0].set_xlabel('Absolute Correlation Coefficient')
            axes[0].grid(axis='x', linestyle='--', alpha=0.7)
            
            # Plot B: Mutual Information Scores
            sorted_mi = mutual_info.nlargest(20).sort_values(ascending=True)
            sorted_mi.plot(kind='barh', color='teal', ax=axes[1])
            axes[1].set_title(f'Top 20 Features by Mutual Information with {self.target_column}')
            axes[1].set_xlabel('Mutual Information Score')
            axes[1].grid(axis='x', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(self.config.correlation_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            log.info(f"Feature importance plots saved to {os.path.dirname(self.config.correlation_plot_path)}")
            
        except Exception as e:
            log.error(f"Error creating plots: {e}")
    
    def _save_selected_features(self, correlation: pd.Series, mutual_info: pd.Series):
        """Save selected features and their scores to file."""
        try:
            os.makedirs(os.path.dirname(self.config.selected_features_path), exist_ok=True)
            
            with open(self.config.selected_features_path, 'w') as f:
                f.write(f"Selected Features for {self.equipment_id}\n")
                f.write(f"Target: {self.target_column}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Total Features Selected: {len(self.selected_features)}\n\n")
                
                f.write("Feature Scores:\n")
                f.write("-" * 60 + "\n")
                for feat in self.selected_features:
                    corr_val = correlation.get(feat, 0.0)
                    mi_val = mutual_info.get(feat, 0.0)
                    f.write(f"{feat}:\n")
                    f.write(f"  Correlation: {corr_val:.4f}\n")
                    f.write(f"  Mutual Info: {mi_val:.4f}\n\n")
            
            log.info(f"Selected features info saved to {self.config.selected_features_path}")
            
        except Exception as e:
            log.error(f"Error saving selected features: {e}")


# ==================== DATA TRANSFORMATION ====================
@dataclass
class GenericDataTransformationConfig:
    """Configuration for data transformation artifacts."""
    scaler_path: str = os.path.join('artifacts', 'generic_models', 'scaler.pkl')
    label_encoders_path: str = os.path.join('artifacts', 'generic_models', 'label_encoders.pkl')


class GenericDataTransformation:
    """Handles data preprocessing for generic optimization."""
    
    def __init__(self, equipment_id: str, target_column: str):
        self.equipment_id = equipment_id
        self.target_column = target_column
        # Update config with dynamic paths based on equipment_id and target_column
        if equipment_id and target_column:
            scaler_filename = f"{equipment_id}_{target_column}_scaler.pkl"
            encoders_filename = f"{equipment_id}_{target_column}_label_encoders.pkl"
            self.config = GenericDataTransformationConfig(
                scaler_path=os.path.join('artifacts', 'generic_models', scaler_filename),
                label_encoders_path=os.path.join('artifacts', 'generic_models', encoders_filename)
            )
        else:
            self.config = GenericDataTransformationConfig()
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.selected_features = []
        
    def transform_dataset(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Transform entire dataset from CSV with automatic feature selection.
        
        Args:
            data_path: Path to CSV file with raw BMS data
            
        Returns:
            Tuple of (X_processed, y, selected_features)
        """
        try:
            log.info("="*60)
            log.info("STARTING GENERIC DATA TRANSFORMATION")
            log.info(f"Equipment: {self.equipment_id}, Target: {self.target_column}")
            log.info("="*60)
            
            # Step 1: Load data
            log.info(f"Loading raw BMS data from {data_path}")
            df = pd.read_csv(data_path)
            log.info(f"Raw data shape: {df.shape}")
            
            # Step 2: Filter for AHU system type
            if 'system_type' in df.columns:
                df = df[df['system_type'] == 'AHU'].copy()
                log.info(f"Filtered for AHU system type. Shape: {df.shape}")
            
            # Step 3: Filter for specific equipment
            if 'equipment_id' in df.columns:
                df = df[df['equipment_id'] == self.equipment_id].copy()
                log.info(f"Filtered for equipment '{self.equipment_id}'. Shape: {df.shape}")
            else:
                raise ValueError("'equipment_id' column not found in data")
            
            if df.empty:
                raise ValueError(f"No data found for system_type='AHU' and equipment_id='{self.equipment_id}'")
            
            # Step 4: Process timestamp
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
            
            # Step 5: Pivot the data
            log.info("Pivoting data from long to wide format...")
            pivoted_df = df.pivot_table(
                index='data_received_on_naive',
                columns='datapoint',
                values='monitoring_data',
                aggfunc='first'
            )
            
            # Step 6: Extract temporal features from timestamp
            log.info("Creating temporal features from timestamp...")
            pivoted_df['month'] = pivoted_df.index.month
            pivoted_df['hour'] = pivoted_df.index.hour
            pivoted_df['week_number'] = pivoted_df.index.isocalendar().week
            pivoted_df['is_weekend'] = (pivoted_df.index.dayofweek >= 5).astype(int)
            
            pivoted_df.reset_index(drop=True, inplace=True)
            log.info(f"Pivoted data shape: {pivoted_df.shape}")
            
            # Step 7: Remove columns with only single value
            single_value_cols = []
            for col in pivoted_df.columns:
                if pivoted_df[col].nunique() == 1:
                    single_value_cols.append(col)
            
            if single_value_cols:
                pivoted_df.drop(columns=single_value_cols, inplace=True)
                log.info(f"Removed {len(single_value_cols)} columns with single value: {single_value_cols}")
            
            # Step 8: Remove columns with all NaN or blank
            all_nan_cols = pivoted_df.columns[pivoted_df.isna().all()].tolist()
            if all_nan_cols:
                pivoted_df.drop(columns=all_nan_cols, inplace=True)
                log.info(f"Removed {len(all_nan_cols)} columns with all NaN: {all_nan_cols}")
            
            # Step 9: Convert columns to numeric where possible
            for col in pivoted_df.columns:
                pivoted_df[col] = pd.to_numeric(pivoted_df[col], errors='ignore')
            log.info("Converted columns to numeric where possible")
            
            # Step 10: Drop rows with any NaN
            pivoted_df.dropna(how='any', inplace=True)
            log.info(f"After dropping NaN rows: {pivoted_df.shape}")
            
            if pivoted_df.empty:
                raise ValueError("No valid data remaining after preprocessing")
            
            # Step 11: Feature selection using correlation and mutual information
            log.info("Starting automatic feature selection...")
            feature_selector = GenericFeatureSelector(self.equipment_id, self.target_column)
            selected_features = feature_selector.select_features(pivoted_df, n_features=20)
            self.selected_features = selected_features
            
            # Step 12: Create final X and y
            # Ensure all selected features exist in the data
            available_features = [f for f in selected_features if f in pivoted_df.columns]
            missing_features = [f for f in selected_features if f not in pivoted_df.columns]
            
            if missing_features:
                log.warning(f"Some selected features not in data: {missing_features}")
            
            if not available_features:
                raise ValueError("No selected features available in the data")
            
            X = pivoted_df[available_features].copy()
            y = pivoted_df[self.target_column].copy()
            
            log.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
            log.info(f"Selected features: {available_features}")
            
            X_processed = self._preprocess_data(X, fit=True)
            
            return X_processed, y, available_features
            
        except Exception as e:
            log.error(f"Error transforming dataset: {e}")
            raise CustomException(e, sys)
    
    def _preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Preprocess data: convert to numeric, apply MinMax scaling, and label encode."""
        try:
            log.info(f"Starting preprocessing. Fit mode: {fit}")
            df_processed = df.copy()
            
            for col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='ignore')
            
            numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
            
            log.info(f"Identified {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
            
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
                            df_processed[col] = df_processed[col].astype(str).apply(
                                lambda x: le.transform([x])[0] if x in le.classes_ else -1
                            )
            
            if numeric_cols:
                if fit:
                    df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
                    log.info(f"Fitted and scaled {len(numeric_cols)} numeric columns")
                else:
                    df_processed[numeric_cols] = self.scaler.transform(df_processed[numeric_cols])
                    log.info(f"Scaled {len(numeric_cols)} numeric columns")
            
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


@dataclass
class GenericModelTrainerConfig:
    """Configuration for model training."""
    model_path: str = os.path.join('artifacts', 'generic_models', 'model.pkl')
    metrics_path: str = os.path.join('artifacts', 'generic_models', 'metrics.txt')


class GenericModelTrainer:
    """Trains the generic optimization surrogate model."""
    
    def __init__(self, equipment_id: str = None, target_column: str = None):
        self.equipment_id = equipment_id
        self.target_column = target_column
        # Update config with dynamic model path based on equipment_id and target variable
        if equipment_id and target_column:
            model_filename = f"{equipment_id}_{target_column}_model.pkl"
            metrics_filename = f"{equipment_id}_{target_column}_metrics.txt"
            self.config = GenericModelTrainerConfig(
                model_path=os.path.join('artifacts', 'generic_models', model_filename),
                metrics_path=os.path.join('artifacts', 'generic_models', metrics_filename)
            )
        else:
            self.config = GenericModelTrainerConfig()
        self.model = None
        self.best_model_name = None
        self.best_params = None
        
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_test: pd.DataFrame, y_test: pd.Series,
                   search_method: str = 'random',
                   cv_folds: int = 5,
                   n_iter: int = 20) -> Dict[str, Any]:
        """Train multiple models with hyperparameter tuning."""
        try:
            log.info("="*60)
            log.info("TRAINING MODELS WITH HYPERPARAMETER TUNING")
            log.info(f"Search Method: {search_method.upper()}")
            log.info("="*60)
            
            # Simplified model selection for faster training
            models_params = {
                'RandomForest': (
                    RandomForestRegressor(random_state=42, n_jobs=-1),
                    {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5]
                    }
                ),
                'XGBoost': (
                    XGBRegressor(random_state=42, n_jobs=-1),
                    {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5, 7]
                    }
                ),
                'GradientBoosting': (
                    GradientBoostingRegressor(random_state=42),
                    {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5]
                    }
                )
            }
            
            best_score = -np.inf
            best_model = None
            best_model_name = None
            all_results = {}
            
            for model_name, (base_model, param_grid) in models_params.items():
                try:
                    log.info(f"\nTraining: {model_name}")
                    
                    search = RandomizedSearchCV(
                        base_model,
                        param_distributions=param_grid,
                        n_iter=n_iter,
                        cv=cv_folds,
                        scoring='r2',
                        n_jobs=-1,
                        random_state=42,
                        verbose=0
                    )
                    
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    
                    # Evaluate
                    y_test_pred = model.predict(X_test)
                    test_r2 = r2_score(y_test, y_test_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    
                    all_results[model_name] = {
                        'test_r2': test_r2,
                        'test_rmse': test_rmse,
                        'test_mse': test_mse,
                        'test_mae': test_mae,
                        'best_params': search.best_params_
                    }
                    
                    log.info(f"  Test R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
                    
                    if test_r2 > best_score:
                        best_score = test_r2
                        best_model = model
                        best_model_name = model_name
                        self.best_params = search.best_params_
                        
                except Exception as e:
                    log.error(f"Error training {model_name}: {e}")
                    continue
            
            if best_model is None:
                raise ValueError("No model was successfully trained")
            
            self.model = best_model
            self.best_model_name = best_model_name
            
            log.info(f"\nBest Model: {best_model_name} (R2: {best_score:.4f})")
            
            return {
                'best_model_name': best_model_name,
                'test_r2': best_score,
                'all_results': all_results
            }
            
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


# ==================== MAIN TRAINING FUNCTION ====================
def train_generic(data_path: str, equipment_id: str, target_column: str,
                  test_size: float = 0.2, search_method: str = 'random',
                  cv_folds: int = 5, n_iter: int = 20) -> Dict[str, Any]:
    """
    Main generic training function with automatic feature selection.
    
    Args:
        data_path: Path to CSV with raw BMS data
        equipment_id: Equipment ID to filter
        target_column: Target variable to optimize
        test_size: Fraction for test split
        search_method: 'random' or 'grid'
        cv_folds: Number of CV folds
        n_iter: Number of iterations for RandomizedSearchCV
        
    Returns:
        Dictionary with training results
    """
    try:
        log.info("="*60)
        log.info("STARTING GENERIC OPTIMIZATION MODEL TRAINING")
        log.info(f"Equipment: {equipment_id}, Target: {target_column}")
        log.info("="*60)
        
        # Step 1: Data Transformation with auto feature selection
        transformer = GenericDataTransformation(equipment_id, target_column)
        X, y, selected_features = transformer.transform_dataset(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        log.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Save transformers
        transformer.save_transformers()
        
        # Step 2: Model Training
        trainer = GenericModelTrainer(equipment_id=equipment_id, target_column=target_column)
        metrics = trainer.train_model(
            X_train, y_train, X_test, y_test,
            search_method=search_method,
            cv_folds=cv_folds,
            n_iter=n_iter
        )
        
        # Save model
        trainer.save_model()
        
        log.info("="*60)
        log.info("TRAINING COMPLETED SUCCESSFULLY")
        log.info("="*60)
        
        # Plot paths from FeatureSelectionConfig
        feature_config = FeatureSelectionConfig()
        
        return {
            'status': 'success',
            'best_model_name': metrics['best_model_name'],
            'selected_features': selected_features,
            'metrics': metrics,
            'model_path': trainer.config.model_path,
            'scaler_path': transformer.config.scaler_path,
            'correlation_plot': feature_config.correlation_plot_path,
            'mi_plot': feature_config.mutual_info_plot_path
        }
        
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise CustomException(e, sys)


# ==================== OPTIMIZATION FUNCTION ====================
def optimize_generic(current_conditions: Dict[str, Any],
                    equipment_id: str,
                    target_column: str,
                    search_space: Optional[Dict[str, List[float]]] = None,
                    optimization_method: str = "random",
                    n_iterations: int = 500) -> Dict[str, Any]:
    """
    Generic optimization function using fixed setpoints.
    
    Args:
        current_conditions: Current system state with all features
        equipment_id: Equipment ID (must match training)
        target_column: Target variable to minimize (must match training)
        search_space: Optional setpoint ranges (defaults used if not provided)
        optimization_method: Optimization method (currently only 'random' supported)
        n_iterations: Number of random search iterations
        
    Returns:
        Optimization results with best setpoints
    """
    try:
        log.info("="*60)
        log.info("STARTING GENERIC SETPOINT OPTIMIZATION")
        log.info(f"Equipment: {equipment_id}, Target: {target_column}")
        log.info(f"Setpoints to optimize: {SETPOINT_NAMES}")
        log.info("="*60)
        
        # Load model and transformers with equipment-specific and target-specific paths
        trainer = GenericModelTrainer(equipment_id=equipment_id, target_column=target_column)
        trainer.load_model()
        
        transformer = GenericDataTransformation(equipment_id=equipment_id, target_column=target_column)
        transformer.load_transformers()
        
        # Default search space if not provided
        if not search_space:
            search_space = {
                'SpMinVFD': list(np.linspace(0, 100, 21)),
                'SpTREff': list(np.linspace(18, 26, 17)),
                'SpTROcc': list(np.linspace(20, 28, 17))
            }
        
        # Random search optimization
        start_time = time.time()
        best_setpoints = None
        min_target = float('inf')
        
        for i in range(n_iterations):
            # Generate random setpoint values
            test_conditions = current_conditions.copy()
            for setpoint in SETPOINT_NAMES:
                if setpoint in search_space:
                    test_conditions[setpoint] = np.random.choice(search_space[setpoint])
            
            # Create input DataFrame
            input_df = pd.DataFrame([test_conditions])
            
            # Predict
            predicted_value = trainer.model.predict(input_df)[0]
            
            if predicted_value < min_target:
                min_target = predicted_value
                best_setpoints = {k: test_conditions[k] for k in SETPOINT_NAMES}
            
            if (i + 1) % 100 == 0:
                log.info(f"Progress: {i + 1}/{n_iterations} iterations")
        
        elapsed_time = time.time() - start_time
        
        log.info("="*60)
        log.info("OPTIMIZATION COMPLETED")
        log.info(f"Best Setpoints: {best_setpoints}")
        log.info(f"Minimum {target_column}: {min_target:.4f}")
        log.info("="*60)
        
        return {
            'best_setpoints': best_setpoints,
            'min_target_value': min_target,
            'target_variable': target_column,
            'total_combinations_tested': n_iterations,
            'optimization_method': optimization_method,
            'optimization_time_seconds': elapsed_time
        }
        
    except Exception as e:
        log.error(f"Optimization failed: {e}")
        raise CustomException(e, sys)
