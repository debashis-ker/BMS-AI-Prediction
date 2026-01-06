"""
Generic Optimization Pipeline
Dynamic pipeline for training surrogate models and optimizing setpoints for any AHU equipment.
Automatically selects best features using correlation and mutual information.
"""

import datetime
import json
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

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    log.warning("scikit-optimize not installed. Bayesian optimization will not be available. Install with: pip install scikit-optimize")

sys.path.append(str(Path(__file__).parent.parent))
from logger_config import setup_logger
from exception import CustomException

log = setup_logger(__name__)
#SETPOINT_NAMES = ['SpMinVFD', 'SpTREff', 'SpTROcc']


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection artifacts."""
    correlation_plot_path: str = os.path.join('artifacts', 'generic_models', 'correlation_plot.png')
    mutual_info_plot_path: str = os.path.join('artifacts', 'generic_models', 'mutual_info_plot.png')
    selected_features_path: str = os.path.join('artifacts', 'generic_models', 'selected_features.txt')
    selected_features_json: str = os.path.join('artifacts', 'generic_models', 'selected_features.json')


class GenericFeatureSelector:
    """Automatically selects best features using correlation and mutual information."""
    
    def __init__(self, equipment_id: str, target_column: str):
        self.config = FeatureSelectionConfig()
        self.equipment_id = equipment_id
        self.target_column = target_column
        self.selected_features = []
        
    def select_features(self, df: pd.DataFrame, n_features: int = 20, setpoints: List[str] = None) -> List[str]:
        """
        Select top features based on correlation and mutual information.
        
        Args:
            df: DataFrame with features and target
            n_features: Total number of features to select (default: 20)
            setpoints: List of setpoint names to exclude from feature selection
            
        Returns:
            List of selected feature names
        """
        try:
            log.info("="*60)
            log.info("STARTING AUTOMATIC FEATURE SELECTION")
            log.info(f"Target: {self.target_column}")
            log.info("="*60)
            log.info(f"Data shape before feature selection: {df.shape}")
            log.info(f"Columns: {df.columns.tolist()}")
            
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in data")
            
            try:
                df[self.target_column] = pd.to_numeric(df[self.target_column], errors='raise')
            except (ValueError, TypeError):
                raise ValueError(f"Target column '{self.target_column}' must contain numeric values")
            
            y = df[self.target_column].copy()
            X = df.drop(columns=[self.target_column]).copy()
            
            log.info(f"Initial features: {X.shape[1]}")
            
            X_transformed = self._transform_for_analysis(X)
            
            valid_idx = X_transformed.notna().all(axis=1) & y.notna()
            X_transformed = X_transformed[valid_idx]
            y = y[valid_idx]
            
            log.info(f"Valid samples after dropping NaN: {X_transformed.shape[0]}")
            
            if X_transformed.empty:
                raise ValueError("No valid data remaining after transformation")
            
            log.info("Calculating correlation with target...")
            df_temp = pd.concat([X_transformed, y], axis=1)
            corr_matrix = df_temp.corr()
            target_corr = corr_matrix[self.target_column].drop(self.target_column)
            self.correlation = target_corr 
            
            log.info("Calculating mutual information with target...")
            mi_scores = mutual_info_regression(X_transformed, y, random_state=42)
            mi_series = pd.Series(mi_scores, index=X_transformed.columns)
            self.mutual_info = mi_series 
            
            self._plot_feature_importance(target_corr, mi_series)
            
            top_mi_features = mi_series.nlargest(10).index.tolist()
            
            corr_abs = target_corr.abs()
            top_corr_features = []
            for feat in corr_abs.nlargest(n_features).index:
                if feat not in top_mi_features and len(top_corr_features) < 10:
                    top_corr_features.append(feat)
            
            self.selected_features = top_mi_features + top_corr_features
            if not setpoints:
                setpoints = []
                print("Setpoints list is empty, proceeding without setpoints.")
            for setpoint in setpoints:
                if setpoint in X.columns and setpoint not in self.selected_features:
                    self.selected_features.append(setpoint)
                    log.info(f"Added setpoint: {setpoint}")
            
            log.info(f"Selected {len(self.selected_features)} features")
            log.info(f"Top 10 MI features: {top_mi_features}")
            log.info(f"Top 10 Correlation features: {top_corr_features}")
            log.info(f"Setpoints included: {[s for s in setpoints if s in self.selected_features]}")
            
            self._save_selected_features(target_corr, mi_series)
            
            return self.selected_features
            
        except Exception as e:
            log.error(f"Error in feature selection: {e}")
            raise CustomException(e, sys)
    
    def _transform_for_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features to numeric for correlation/MI analysis."""
        df_transformed = df.copy()
        
        for col in df_transformed.columns:
            df_transformed[col] = pd.to_numeric(df_transformed[col], errors='ignore')
        
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
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            sorted_corr = correlation.abs().nlargest(20).sort_values(ascending=True)
            sorted_corr.plot(kind='barh', color='skyblue', ax=axes[0])
            axes[0].set_title(f'Top 20 Features by Correlation with {self.target_column}')
            axes[0].set_xlabel('Absolute Correlation Coefficient')
            axes[0].grid(axis='x', linestyle='--', alpha=0.7)
            
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
    
    def _save_selected_features(self, correlation: pd.Series, mutual_info: pd.Series, 
                                setpoints: Optional[List[str]] = None, 
                                setpoint_ranges: Optional[Dict[str, Any]] = None):
        """Save selected features and their scores to file, including setpoints and ranges."""
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
            
            import json
            metadata = {
                'equipment_id': self.equipment_id,
                'target_column': self.target_column,
                'selected_features': self.selected_features,
                'feature_scores': {
                    feat: {
                        'correlation': float(correlation.get(feat, 0.0)),
                        'mutual_info': float(mutual_info.get(feat, 0.0))
                    }
                    for feat in self.selected_features
                }
            }
            
            if setpoints:
                metadata['setpoints'] = setpoints
            if setpoint_ranges:
                metadata['setpoint_ranges'] = setpoint_ranges
            
            json_path = os.path.join(
                'artifacts', 'generic_models', 
                f"{self.equipment_id}_{self.target_column}_features.json"
            )
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            log.info(f"Selected features metadata saved to {json_path}")
            if setpoints:
                log.info(f"Saved {len(setpoints)} setpoints with ranges")
            
        except Exception as e:
            log.error(f"Error saving selected features: {e}")


@dataclass
class GenericDataTransformationConfig:
    """Configuration for data transformation artifacts."""
    scaler_path: str = os.path.join('artifacts', 'generic_models', 'scaler.pkl')
    label_encoders_path: str = os.path.join('artifacts', 'generic_models', 'label_encoders.pkl')
    transformation_metadata_path: str = os.path.join('artifacts', 'generic_models', 'transformation_metadata.json')


class GenericDataTransformation:
    """Handles data preprocessing for generic optimization."""
    
    def __init__(self, equipment_id: str, target_column: str):
        self.equipment_id = equipment_id
        self.target_column = target_column
        if equipment_id and target_column:
            scaler_filename = f"{equipment_id}_{target_column}_scaler.pkl"
            encoders_filename = f"{equipment_id}_{target_column}_label_encoders.pkl"
            metadata_filename = f"{equipment_id}_{target_column}_transformation_metadata.json"
            self.config = GenericDataTransformationConfig(
                scaler_path=os.path.join('artifacts', 'generic_models', scaler_filename),
                label_encoders_path=os.path.join('artifacts', 'generic_models', encoders_filename),
                transformation_metadata_path=os.path.join('artifacts', 'generic_models', metadata_filename)
            )
        else:
            self.config = GenericDataTransformationConfig()
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.selected_features = []
        self.numeric_cols_at_training = []
        self.categorical_cols_at_training = []
        
    def transform_dataset(self, data: List[dict], setpoints: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare dataset without scaling or feature selection.
        
        IMPORTANT: This method only loads, filters, pivots, and cleans data.
        Feature selection and scaling must happen AFTER train/test split to avoid data leakage.
        
        Args:
            data: List of dictionaries with raw BMS data
            setpoints: Optional list of setpoint columns to include and track
            
        Returns:
            Tuple of (pivoted_df, setpoints_list) - unscaled data ready for splitting
        """
        try:
            log.info("="*60)
            log.info("STARTING DATA LOADING AND PREPARATION")
            log.info(f"Equipment: {self.equipment_id}, Target: {self.target_column}")
            log.info("="*60)
            
            df = pd.DataFrame(data)
            log.info(f"Raw data shape: {df.shape}")
            
            if 'system_type' in df.columns:
                df = df[df['system_type'] == 'AHU'].copy()
                log.info(f"Filtered for AHU system type. Shape: {df.shape}")
            
            if 'equipment_id' in df.columns:
                df = df[df['equipment_id'] == self.equipment_id].copy()
                log.info(f"Filtered for equipment '{self.equipment_id}'. Shape: {df.shape}")
            else:
                raise ValueError("'equipment_id' column not found in data")
            
            if df.empty:
                raise ValueError(f"No data found for system_type='AHU' and equipment_id='{self.equipment_id}'")
            
            if 'data_received_on' in df.columns:
                df['data_received_on'] = pd.to_datetime(df['data_received_on'], errors='coerce')
                
                if df['data_received_on'].dt.tz is not None:
                    df['data_received_on_naive'] = df['data_received_on'].dt.tz_localize(None)
                else:
                    df['data_received_on_naive'] = df['data_received_on']
                
                df.sort_values('data_received_on_naive', inplace=True)
                log.info("Processed timestamps")
            else:
                raise ValueError("'data_received_on' column not found in data")
        
            log.info("Pivoting data from long to wide format...")
            pivoted_df = df.pivot_table(
                index='data_received_on_naive',
                columns='datapoint',
                values='monitoring_data',
                aggfunc='first'
            )
            
            log.info("Creating temporal features from timestamp...")
            pivoted_df['month'] = pivoted_df.index.month
            pivoted_df['hour'] = pivoted_df.index.hour
            pivoted_df['week_number'] = pivoted_df.index.isocalendar().week
            pivoted_df['is_weekend'] = (pivoted_df.index.dayofweek >= 5).astype(int)
            
            pivoted_df.reset_index(drop=True, inplace=True)
            log.info(f"Pivoted data shape: {pivoted_df.shape}")
            
            single_value_cols = []
            for col in pivoted_df.columns:
                if pivoted_df[col].nunique() == 1:
                    single_value_cols.append(col)
            
            if single_value_cols:
                pivoted_df.drop(columns=single_value_cols, inplace=True)
                log.info(f"Removed {len(single_value_cols)} columns with single value: {single_value_cols}")
            
            all_nan_cols = pivoted_df.columns[pivoted_df.isna().all()].tolist()
            if all_nan_cols:
                pivoted_df.drop(columns=all_nan_cols, inplace=True)
                log.info(f"Removed {len(all_nan_cols)} columns with all NaN: {all_nan_cols}")
            
            for col in pivoted_df.columns:
                pivoted_df[col] = pd.to_numeric(pivoted_df[col], errors='ignore')
            log.info("Converted columns to numeric where possible")
            
            pivoted_df.dropna(how='any', inplace=True)
            log.info(f"After dropping NaN rows: {pivoted_df.shape}")
            
            if pivoted_df.empty:
                raise ValueError("No valid data remaining after preprocessing")
            
            setpoints = setpoints or SETPOINT_NAMES
            present_setpoints = [s for s in setpoints if s in pivoted_df.columns]
            
            log.info("Data preparation complete. Feature selection and scaling will occur after train/test split.")
            log.info(f"Prepared data shape: {pivoted_df.shape}")
            log.info(f"Present setpoints: {present_setpoints}")
            
            return pivoted_df, present_setpoints
            
        except Exception as e:
            log.error(f"Error loading dataset: {e}")
            raise CustomException(e, sys)
    
    def fit_transform_features(self, df: pd.DataFrame, setpoints: List[str], n_features: int = 20) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Fit feature selector and scaler on training data, then transform.
        
        This method should ONLY be called on training data to prevent data leakage.
        Use transform_features() for test data.
        
        Args:
            df: Pivoted dataframe (training data only)
            setpoints: List of setpoint columns to include
            n_features: Number of features to select
            
        Returns:
            Tuple of (X_processed, y, selected_features)
        """
        try:
            log.info("="*60)
            log.info("FITTING FEATURE SELECTOR AND SCALER ON TRAINING DATA")
            log.info("="*60)
            
            # Step 1: Feature selection on training data only
            log.info("Starting automatic feature selection on training data...")
            feature_selector = GenericFeatureSelector(self.equipment_id, self.target_column)
            selected_features = feature_selector.select_features(df, n_features=n_features,setpoints=setpoints)
            self.selected_features = selected_features
            
            # Step 2: Ensure setpoints are included
            present_setpoints = [s for s in setpoints if s in df.columns]
            for sp in present_setpoints:
                if sp not in self.selected_features:
                    self.selected_features.append(sp)
                    log.info(f"Added setpoint to selected features: {sp}")
            
            # Step 3: Calculate and save setpoint ranges
            setpoint_ranges = {}
            for sp in present_setpoints:
                try:
                    col_vals = pd.to_numeric(df[sp].dropna(), errors='coerce')
                    if col_vals.empty:
                        continue
                    min_val = float(col_vals.min())
                    max_val = float(col_vals.max())
                    if abs(max_val - min_val) < 1e-6:
                        min_val = max(0, min_val - 0.1)
                        max_val = max_val + 0.1
                    setpoint_ranges[sp] = {
                        "min": min_val,
                        "max": max_val,
                        "lookup": list(np.linspace(min_val, max_val, num=21))
                    }
                    log.info(f"Setpoint {sp} range: [{min_val:.2f}, {max_val:.2f}]")
                except Exception as e:
                    log.warning(f"Could not compute range for setpoint {sp}: {e}")
            
            feature_selector._save_selected_features(
                correlation=getattr(feature_selector, 'correlation', pd.Series()),
                mutual_info=getattr(feature_selector, 'mutual_info', pd.Series()),
                setpoints=present_setpoints,
                setpoint_ranges=setpoint_ranges
            )
            
            # Step 4: Extract X and y
            available_features = [f for f in self.selected_features if f in df.columns]
            missing_features = [f for f in self.selected_features if f not in df.columns]
            
            if missing_features:
                log.warning(f"Some selected features not in data: {missing_features}")
            
            if not available_features:
                raise ValueError("No selected features available in the data")
            
            X = df[available_features].copy()
            y = df[self.target_column].copy()
            
            log.info(f"Training data shape: X={X.shape}, y={y.shape}")
            log.info(f"Selected {len(available_features)} features")
            
            # Step 5: Fit and transform scaler on training data
            X_processed = self._preprocess_data(X, fit=True)
            
            return X_processed, y, available_features
            
        except Exception as e:
            log.error(f"Error in fit_transform_features: {e}")
            raise CustomException(e, sys)
    
    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data using already-fitted scaler and feature selector.
        
        Args:
            df: Pivoted dataframe (test data)
            
        Returns:
            X_processed: Scaled and transformed features
        """
        try:
            log.info("Transforming test data using fitted transformers...")
            
            if not self.selected_features:
                raise ValueError("No features selected. Call fit_transform_features first.")
            
            # Use only the features selected during training
            available_features = [f for f in self.selected_features if f in df.columns]
            X = df[available_features].copy()
            
            # Transform using fitted scaler (fit=False)
            X_processed = self._preprocess_data(X, fit=False)
            
            log.info(f"Test data transformed. Shape: {X_processed.shape}")
            return X_processed
            
        except Exception as e:
            log.error(f"Error in transform_features: {e}")
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
            
            
            if categorical_cols:
                for col in categorical_cols:
                    if fit:
                        le = LabelEncoder()
                        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                        self.label_encoders[col] = le
                    else:
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            df_processed[col] = df_processed[col].astype(str).apply(
                                lambda x: le.transform([x])[0] if x in le.classes_ else -1
                            )
            
            if numeric_cols:
                if fit:
                    self.numeric_cols_at_training = numeric_cols.copy()
                    self.categorical_cols_at_training = categorical_cols.copy()
                    df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
                  
                else:
                    df_processed[numeric_cols] = self.scaler.transform(df_processed[numeric_cols])
                   
            
            return df_processed
            
        except Exception as e:
            log.error(f"Error in preprocessing: {e}")
            raise CustomException(e, sys)
    
    def save_transformers(self):
        """Save scaler, label encoders, and transformation metadata to disk."""
        try:
            os.makedirs(os.path.dirname(self.config.scaler_path), exist_ok=True)
            
            joblib.dump(self.scaler, self.config.scaler_path)
            joblib.dump(self.label_encoders, self.config.label_encoders_path)
            
            import json
            metadata = {
                'numeric_cols_at_training': self.numeric_cols_at_training,
                'categorical_cols_at_training': self.categorical_cols_at_training,
                'selected_features': self.selected_features,
                'label_encoder_columns': list(self.label_encoders.keys())
            }
            
            with open(self.config.transformation_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            log.info(f"Saved transformers and metadata to {os.path.dirname(self.config.scaler_path)}")
            log.info(f"Transformation metadata: {metadata}")
        except Exception as e:
            log.error(f"Error saving transformers: {e}")
            raise CustomException(e, sys)
    
    def load_transformers(self):
        """Load scaler, label encoders, and transformation metadata from disk."""
        try:
            if os.path.exists(self.config.scaler_path):
                self.scaler = joblib.load(self.config.scaler_path)
                log.info(f"Loaded scaler from {self.config.scaler_path}")
            
            if os.path.exists(self.config.label_encoders_path):
                self.label_encoders = joblib.load(self.config.label_encoders_path)
                log.info(f"Loaded label encoders from {self.config.label_encoders_path}")
            
            if os.path.exists(self.config.transformation_metadata_path):
                import json
                with open(self.config.transformation_metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.numeric_cols_at_training = metadata.get('numeric_cols_at_training', [])
                self.categorical_cols_at_training = metadata.get('categorical_cols_at_training', [])
                self.selected_features = metadata.get('selected_features', [])
                
                log.info(f"Loaded transformation metadata from {self.config.transformation_metadata_path}")
                log.info(f"Numeric columns: {len(self.numeric_cols_at_training)}, Categorical: {len(self.categorical_cols_at_training)}")
            else:
                log.warning(f"Transformation metadata not found at {self.config.transformation_metadata_path}")
                
        except Exception as e:
            log.error(f"Error loading transformers: {e}")
            raise CustomException(e, sys)
    
    def transform_input(self, input_df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """
        Transform input data for prediction using saved transformation metadata.
        Replicates the exact transformation pipeline used during training.
        
        Args:
            input_df: Raw input DataFrame with feature values
            selected_features: List of features expected by the model
            
        Returns:
            Transformed DataFrame ready for prediction
        """
        try:
            df_processed = input_df.copy()
            
            temporal_features = ['month', 'hour', 'week_number', 'is_weekend']
            missing_temporal = [f for f in temporal_features if f in selected_features and f not in df_processed.columns]
            
            if missing_temporal:
                if 'timestamp' in df_processed.columns:
                    log.info("Generating temporal features from provided timestamp")
                    timestamp_val = df_processed['timestamp'].iloc[0]
                    
                    if isinstance(timestamp_val, str):
                        timestamp_dt = pd.to_datetime(timestamp_val)
                    elif isinstance(timestamp_val, pd.Timestamp):
                        timestamp_dt = timestamp_val
                    else:
                        from datetime import datetime
                        timestamp_dt = pd.to_datetime(timestamp_val)
                    
                    if timestamp_dt.tz is not None:
                        timestamp_dt = timestamp_dt.tz_localize(None)
                    
                    for temp_feat in missing_temporal:
                        if temp_feat == 'month':
                            df_processed[temp_feat] = timestamp_dt.month
                        elif temp_feat == 'hour':
                            df_processed[temp_feat] = timestamp_dt.hour
                        elif temp_feat == 'week_number':
                            df_processed[temp_feat] = timestamp_dt.isocalendar()[1]
                        elif temp_feat == 'is_weekend':
                            df_processed[temp_feat] = 1 if timestamp_dt.weekday() >= 5 else 0
                    
                    df_processed.drop(columns=['timestamp'], inplace=True)
                    log.info(f"Generated temporal features from timestamp: {missing_temporal}")
                else:
                    raise ValueError(
                        f"Temporal features {missing_temporal} are required by the model but not provided. "
                        f"Please provide either these temporal features directly or a 'timestamp' field to generate them."
                    )
            for col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='ignore')
            
            for col in self.categorical_cols_at_training:
                if col in df_processed.columns:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        df_processed[col] = df_processed[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        log.warning(f"Column '{col}' was categorical during training but encoder not found")
            
            if self.numeric_cols_at_training:
                missing_numeric = [col for col in self.numeric_cols_at_training if col not in df_processed.columns]
                if missing_numeric:
                    log.warning(f"Missing numeric columns from training: {missing_numeric}")
                    for col in missing_numeric:
                        df_processed[col] = 0
                
                df_for_scaling = df_processed[self.numeric_cols_at_training]
                
                df_scaled = pd.DataFrame(
                    self.scaler.transform(df_for_scaling),
                    columns=self.numeric_cols_at_training,
                    index=df_for_scaling.index
                )
                
                df_processed[self.numeric_cols_at_training] = df_scaled
            
            return df_processed
            
        except Exception as e:
            log.error(f"Error transforming input: {e}")
            log.error(f"Numeric cols from training: {self.numeric_cols_at_training}")
            log.error(f"Categorical cols from training: {self.categorical_cols_at_training}")
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
                    
                    y_test_pred = model.predict(X_test)
                    test_r2 = r2_score(y_test, y_test_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    
                    # Calculate Adjusted R^2
                    n = len(y_test)
                    p = X_test.shape[1] 
                    test_r2_adj = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
                    
                    all_results[model_name] = {
                        'test_r2': test_r2,
                        'test_r2_adjusted': test_r2_adj,
                        'test_rmse': test_rmse,
                        'test_mse': test_mse,
                        'test_mae': test_mae,
                        'best_params': search.best_params_,
                        'n_features': p
                    }
                    
                    log.info(f"  Test R²: {test_r2:.4f}, Adjusted R²: {test_r2_adj:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
                    log.info(f"  Features used: {p}")
                    
                    if test_r2_adj > best_score:
                        best_score = test_r2_adj
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
            
            log.info(f"\nBest Model: {best_model_name} (Adjusted R²: {best_score:.4f})")
            log.info(f"Standard R²: {all_results[best_model_name]['test_r2']:.4f}")
            
            json_path = os.path.join(
                'artifacts',f'best_model_metrics.json'
            )
            os.makedirs(os.path.dirname(json_path), exist_ok=True)

            if not os.path.exists(json_path):
                with open(json_path, 'w') as file:
                    json.dump({}, file)

            with open(json_path, 'r') as file:
                try:
                    file_data = json.load(file)
                    if not isinstance(file_data, dict):
                        file_data = {}
                except Exception:
                    file_data = {}

            new_entry = {
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "accuracy_score": float(all_results[best_model_name]['test_r2'])
            }

            existing = file_data.get(self.equipment_id)
            if existing is None:
                file_data[self.equipment_id] = [new_entry]
            elif isinstance(existing, list):
                existing.append(new_entry)
                file_data[self.equipment_id] = existing
            else:
                file_data[self.equipment_id] = [existing, new_entry]

            with open(json_path, 'w') as f:
                json.dump(file_data, f, indent=2)
            log.info(f"Best Model : {best_model_name} metrics saved.")

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


def train_generic(data: List[dict], equipment_id: str, target_column: str,
                  test_size: float = 0.2, search_method: str = 'random',
                  cv_folds: int = 5, n_iter: int = 20, setpoints: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Main generic training function with automatic feature selection.
    
    CRITICAL: Data is split BEFORE feature selection and scaling to prevent data leakage.
    
    Args:
        data: List of dictionaries with raw BMS data
        equipment_id: Equipment ID to filter
        target_column: Target variable to optimize
        test_size: Fraction for test split
        search_method: 'random' or 'grid'
        cv_folds: Number of CV folds
        n_iter: Number of iterations for RandomizedSearchCV
        setpoints: Optional list of setpoint columns to include and track
        
    Returns:
        Dictionary with training results
    """
    try:
        log.info("="*60)
        log.info("STARTING GENERIC OPTIMIZATION MODEL TRAINING")
        log.info(f"Equipment: {equipment_id}, Target: {target_column}")
        if setpoints:
            log.info(f"Tracking setpoints: {setpoints}")
        log.info("="*60)
        
        transformer = GenericDataTransformation(equipment_id, target_column)
        pivoted_df, present_setpoints = transformer.transform_dataset(data, setpoints=setpoints)
        
        if target_column not in pivoted_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in pivoted data")
        
        y = pivoted_df[target_column]
        df_features = pivoted_df.drop(columns=[target_column])
        
        df_train, df_test, y_train, y_test = train_test_split(
            df_features, y, test_size=test_size, random_state=42, shuffle=True
        )
        log.info(f"Data split complete. Train: {df_train.shape}, Test: {df_test.shape}")
        log.info("CRITICAL: Feature selection and scaling will now be fit on training data only.")
        
        X_train, y_train, selected_features = transformer.fit_transform_features(
            pd.concat([df_train, y_train], axis=1), 
            present_setpoints,
            n_features=20
        )
        
        X_test = transformer.transform_features(df_test)
        
        log.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        log.info(f"No data leakage: Scaler and feature selector fitted on training data only.")
        
        transformer.save_transformers()
        
        trainer = GenericModelTrainer(equipment_id=equipment_id, target_column=target_column)
        metrics = trainer.train_model(
            X_train, y_train, X_test, y_test,
            search_method=search_method,
            cv_folds=cv_folds,
            n_iter=n_iter
        )
        
        trainer.save_model()
        
        log.info("="*60)
        log.info("TRAINING COMPLETED SUCCESSFULLY")
        log.info("="*60)

        feature_config = FeatureSelectionConfig()
        
        setpoints_used = setpoints or SETPOINT_NAMES
        trained_setpoints = [s for s in setpoints_used if s in selected_features]
        
        return {
            'status': 'success',
            'best_model_name': metrics['best_model_name'],
            'selected_features': selected_features,
            'setpoints': trained_setpoints,
            'metrics': metrics,
            'model_path': trainer.config.model_path,
            'scaler_path': transformer.config.scaler_path,
            'correlation_plot': feature_config.correlation_plot_path,
            'mi_plot': feature_config.mutual_info_plot_path,
            'best_model_metrics':metrics.get('best_model_metrics', {})
        }
        
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise CustomException(e, sys)
    
def train_genericV2(data: dict, equipment_id: str, target_column: str,
                  test_size: float = 0.2, search_method: str = 'random',
                  cv_folds: int = 5, n_iter: int = 20, setpoints: Optional[List[str]] = None) -> Dict[str, Any]:
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
        setpoints: Optional list of setpoint columns to include and track
        
    Returns:
        Dictionary with training results
    """
    try:
        log.info("="*60)
        log.info("STARTING GENERIC OPTIMIZATION MODEL TRAINING")
        log.info(f"Equipment: {equipment_id}, Target: {target_column}")
        if setpoints:
            log.info(f"Tracking setpoints: {setpoints}")
        log.info("="*60)
        
        transformer = GenericDataTransformation(equipment_id, target_column)
        X, y, selected_features = transformer.transform_dataset(data_path, setpoints=setpoints)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        log.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        transformer.save_transformers()
        
        trainer = GenericModelTrainer(equipment_id=equipment_id, target_column=target_column)
        metrics = trainer.train_model(
            X_train, y_train, X_test, y_test,
            search_method=search_method,
            cv_folds=cv_folds,
            n_iter=n_iter
        )
        
        trainer.save_model()
        
        log.info("="*60)
        log.info("TRAINING COMPLETED SUCCESSFULLY")
        log.info("="*60)
        
        feature_config = FeatureSelectionConfig()
        
        setpoints_used = setpoints or SETPOINT_NAMES
        trained_setpoints = [s for s in setpoints_used if s in selected_features]
        
        return {
            'status': 'success',
            'best_model_name': metrics['best_model_name'],
            'selected_features': selected_features,
            'setpoints': trained_setpoints,
            'metrics': metrics,
            'model_path': trainer.config.model_path,
            'scaler_path': transformer.config.scaler_path,
            'correlation_plot': feature_config.correlation_plot_path,
            'mi_plot': feature_config.mutual_info_plot_path
        }
        
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise CustomException(e, sys)       
        


def optimize_generic(current_conditions: Dict[str, Any],
                    equipment_id: str,
                    target_column: str,
                    search_space: Optional[Dict[str, List[float]]] = None,
                    optimization_method: str = "random",
                    n_iterations: int = 500,
                    direction: str = "minimize") -> Dict[str, Any]:
    """
    Generic optimization function using fixed setpoints.
    
    Supports three optimization methods:
    - 'grid': Exhaustive search over all combinations (slow for many setpoints)
    - 'random': Random sampling of setpoint combinations
    - 'bayes': Bayesian optimization using Gaussian Process (RECOMMENDED - much faster)
    
    Bayesian optimization intelligently selects the next setpoints to test based on previous
    results, requiring far fewer evaluations than grid search (typically 50-200 vs 8000+).
    
    Args:
        current_conditions: Current system state with all features
        equipment_id: Equipment ID (must match training)
        target_column: Target variable to optimize (must match training)
        search_space: Optional setpoint ranges (defaults used if not provided)
        optimization_method: Optimization method ('grid', 'random', or 'bayes')
        n_iterations: Number of iterations for random/bayes search (ignored for grid search)
        direction: 'minimize' or 'maximize' the target variable
        
    Returns:
        Optimization results with best setpoints
    """
    try:
        log.info("="*60)
        log.info("STARTING GENERIC SETPOINT OPTIMIZATION")
        log.info(f"Equipment: {equipment_id}, Target: {target_column}, Direction: {direction}")
        log.info("="*60)
        
        import time
        start_time = time.time()
        
        import json
        features_json_path = os.path.join(
            'artifacts', 'generic_models',
            f"{equipment_id}_{target_column}_features.json"
        )
        
        with open(features_json_path, 'r') as f:
            features_metadata = json.load(f)
        
        selected_features = features_metadata.get('selected_features', [])
        saved_setpoints = features_metadata.get('setpoints', [])
        setpoint_ranges_meta = features_metadata.get('setpoint_ranges', {})
        
        for sp in saved_setpoints:
            if sp not in selected_features:
                selected_features.append(sp)
        
        if not search_space:
            search_space = {}
            for sp, rng in setpoint_ranges_meta.items():
                if isinstance(rng, dict) and 'lookup' in rng:
                    search_space[sp] = rng['lookup']
                elif isinstance(rng, dict) and 'min' in rng and 'max' in rng:
                    search_space[sp] = list(np.linspace(rng['min'], rng['max'], 21))
            
            if search_space:
                log.info(f"Using saved setpoint ranges from metadata for: {list(search_space.keys())}")
            else:
                log.warning("No setpoint ranges found in metadata, using defaults")
                search_space = {
                    'SpMinVFD': list(np.linspace(0, 100, 21)),
                    'SpTREff': list(np.linspace(18, 26, 17)),
                    'SpTROcc': list(np.linspace(20, 28, 17))
                }
        else:
            log.info("Using user-provided search_space")
        log.info(f"Equipment: {equipment_id}, Target: {target_column}")
        log.info(f"Setpoints to optimize: {SETPOINT_NAMES}")
        log.info("="*60)
        
        import json
        features_json_path = os.path.join(
            'artifacts', 'generic_models',
            f"{equipment_id}_{target_column}_features.json"
        )
        
        if not os.path.exists(features_json_path):
            raise FileNotFoundError(
                f"Selected features metadata not found at {features_json_path}. "
                f"Please train the model first for equipment '{equipment_id}' and target '{target_column}'."
            )
        
        with open(features_json_path, 'r') as f:
            features_metadata = json.load(f)
        
        selected_features = features_metadata['selected_features']
        setpoints = features_metadata['setpoints']
        setpoints_used = setpoints or SETPOINT_NAMES
        log.info(f"Loaded {len(selected_features)} selected features from training")
        log.info(f"Features: {selected_features}")
        
        trainer = GenericModelTrainer(equipment_id=equipment_id, target_column=target_column)
        trainer.load_model()
        
        transformer = GenericDataTransformation(equipment_id=equipment_id, target_column=target_column)
        transformer.load_transformers()
        
        if not search_space:
            search_space = {
                'SpMinVFD': list(np.linspace(0, 100, 21)),
                'SpTREff': list(np.linspace(18, 26, 17)),
                'SpTROcc': list(np.linspace(20, 28, 17))
            }
        
        log.info(f"Optimization method: {optimization_method}")
        log.info(f"Optimization direction: {direction}")
        log.info(f"Search space: {search_space}")
        
        if optimization_method.lower() == 'bayes':
            if not BAYESIAN_AVAILABLE:
                raise ImportError(
                    "Bayesian optimization requires scikit-optimize. "
                    "Install with: pip install scikit-optimize"
                )
            
            log.info("Using Bayesian Optimization with Gaussian Process")
            log.info(f"Maximum iterations: {n_iterations}")
            
            space = []
            setpoint_order = []
            for sp in SETPOINT_NAMES:
                if sp in search_space:
                    sp_values = search_space[sp]
                    space.append(Real(min(sp_values), max(sp_values), name=sp))
                    setpoint_order.append(sp)
            
            if not space:
                raise ValueError("No valid setpoints found in search_space for Bayesian optimization")
            
            log.info(f"Optimizing setpoints: {setpoint_order}")
            
            @use_named_args(space)
            def objective(**params):
                test_conditions = current_conditions.copy()
                for sp in setpoint_order:
                    test_conditions[sp] = params[sp]
                
                input_df = pd.DataFrame([test_conditions])
                
                temporal_features = ['month', 'hour', 'week_number', 'is_weekend']
                for feat in selected_features:
                    if feat not in input_df.columns and feat not in temporal_features:
                        input_df[feat] = 0
                
                try:
                    input_df = transformer.transform_input(input_df, selected_features)
                except ValueError as e:
                    if 'Temporal features' in str(e) and 'timestamp' in str(e):
                        raise ValueError(
                            f"Temporal features are required for optimization. "
                            f"Please provide either {temporal_features} in 'current_conditions' "
                            f"or include a 'timestamp' field."
                        ) from e
                    raise
                
                input_df = input_df[selected_features]
                predicted_value = trainer.model.predict(input_df)[0]
                
                if direction.lower() == 'minimize':
                    return predicted_value
                else:
                    return -predicted_value
            
            log.info("Starting Bayesian optimization...")
            # result = gp_minimize(
            #     objective,
            #     space,
            #     n_calls=n_iterations,
            #     random_state=42,
            #     verbose=False,
            #     n_initial_points=min(10, n_iterations // 2)  # Initial random exploration
            # )
            
            result = gp_minimize(
                objective,
                space,
                n_calls=n_iterations,
                n_initial_points=10, 
                acq_func="EI",       
                random_state=42
            )
            
            best_setpoints = {setpoint_order[i]: result.x[i] for i in range(len(setpoint_order))}
            best_target = result.fun if direction.lower() == 'minimize' else -result.fun
            total_tested = len(result.func_vals)
            
            log.info(f"Bayesian optimization completed with {total_tested} evaluations")
            
        elif optimization_method.lower() == 'grid':
            import itertools
            setpoint_combinations = list(itertools.product(
                search_space.get('SpMinVFD', [80]),
                search_space.get('SpTREff', [21]),
                search_space.get('SpTROcc', [21])
            ))
            log.info(f"Grid search: Testing {len(setpoint_combinations)} combinations")
        else:
            setpoint_combinations = None
        
        start_time = time.time()
        best_setpoints = None
        
        if direction.lower() == 'minimize':
            best_target = float('inf')
            is_better = lambda new, best: new < best
        else: 
            best_target = float('-inf')
            is_better = lambda new, best: new > best
        
        if optimization_method.lower() == 'grid':
            for i, (sp_min, sp_eff, sp_occ) in enumerate(setpoint_combinations):
                test_conditions = current_conditions.copy()
                test_conditions['SpMinVFD'] = sp_min
                test_conditions['SpTREff'] = sp_eff
                test_conditions['SpTROcc'] = sp_occ
                
                input_df = pd.DataFrame([test_conditions])
                
                temporal_features = ['month', 'hour', 'week_number', 'is_weekend']
                for feat in selected_features:
                    if feat not in input_df.columns and feat not in temporal_features:
                        input_df[feat] = 0
                
                try:
                    input_df = transformer.transform_input(input_df, selected_features)
                except ValueError as e:
                    if 'Temporal features' in str(e) and 'timestamp' in str(e):
                        raise ValueError(
                            f"Temporal features are required for optimization. "
                            f"Please provide either {temporal_features} in 'current_conditions' "
                            f"or include a 'timestamp' field."
                        ) from e
                    raise
                input_df = input_df[selected_features]
                predicted_value = trainer.model.predict(input_df)[0]
                
                if is_better(predicted_value, best_target):
                    best_target = predicted_value
                    best_setpoints = {
                        'SpMinVFD': sp_min,
                        'SpTREff': sp_eff,
                        'SpTROcc': sp_occ
                    }
                
                if (i + 1) % 100 == 0 or (i + 1) == len(setpoint_combinations):
                    log.info(f"Progress: {i + 1}/{len(setpoint_combinations)} combinations tested")
            
            total_tested = len(setpoint_combinations)
        else:
            total_tested = n_iterations
            for i in range(n_iterations):
                test_conditions = current_conditions.copy()
                for setpoint in setpoints_used:
                    if setpoint in search_space:
                        test_conditions[setpoint] = np.random.choice(search_space[setpoint])
                
                input_df = pd.DataFrame([test_conditions])
                
                temporal_features = ['month', 'hour', 'week_number', 'is_weekend']
                for feat in selected_features:
                    if feat not in input_df.columns and feat not in temporal_features:
                        input_df[feat] = 0
                
                try:
                    input_df = transformer.transform_input(input_df, selected_features)
                except ValueError as e:
                    if 'Temporal features' in str(e) and 'timestamp' in str(e):
                        raise ValueError(
                            f"Temporal features are required for optimization. "
                            f"Please provide either {temporal_features} in 'current_conditions' "
                            f"or include a 'timestamp' field."
                        ) from e
                    raise
                input_df = input_df[selected_features]
                predicted_value = trainer.model.predict(input_df)[0]
                
                if is_better(predicted_value, best_target):
                    best_target = predicted_value
                    #best_setpoints = {k: test_conditions[k] for k in SETPOINT_NAMES}
                    best_setpoints = {k: test_conditions[k] for k in setpoints_used}
                
                if (i + 1) % 100 == 0:
                    log.info(f"Progress: {i + 1}/{n_iterations} iterations")
        
        elapsed_time = time.time() - start_time
        
        log.info("="*60)
        log.info("OPTIMIZATION COMPLETED")
        log.info(f"Optimization method: {optimization_method.upper()}")
        log.info(f"Best Setpoints: {best_setpoints}")
        log.info(f"Best {target_column} value ({direction}): {best_target:.4f}")
        log.info(f"Total evaluations: {total_tested}")
        log.info(f"Time elapsed: {elapsed_time:.2f} seconds")
        if optimization_method.lower() == 'bayes':
            log.info(f"Efficiency: Bayesian optimization required {total_tested} evaluations vs {len(list(itertools.product(*[search_space[sp] for sp in SETPOINT_NAMES if sp in search_space])))} for grid search")
        log.info("="*60)
        
        return {
            'best_setpoints': best_setpoints,
            'best_target_value': best_target,
            'target_variable': target_column,
            'optimization_direction': direction,
            'selected_features_used': selected_features,
            'total_combinations_tested': total_tested,
            'optimization_method': optimization_method,
            'optimization_time_seconds': elapsed_time,
            "timestamp" : time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }
        
    except Exception as e:
        log.error(f"Optimization failed: {e}")
        raise CustomException(e, sys)
