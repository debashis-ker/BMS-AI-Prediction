import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from dataclasses import dataclass
from src.bms_ai.logger_config import setup_logger
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import (
    Ridge, 
    Lasso, 
    ElasticNet,
    LinearRegression
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import uniform, randint

import xgboost as xgb
from xgboost import XGBRegressor

# Import MLflow and other dependencies
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from src.bms_ai.utils.mlflow_config import MLflowConfig

from src.bms_ai.exception import CustomException
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.main_utils import save_object

warnings.filterwarnings('ignore')

# Setup logger BEFORE optional imports
log = setup_logger(__name__)

# Optional library imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    log.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    log.warning("CatBoost not available. Install with: pip install catboost")

@dataclass
class ModelTrainerSingleConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model_single.pkl")
    best_model_name_path: str = os.path.join("artifacts", "best_model_info_single.txt")

class ModelTrainerSingle:
    def __init__(self, use_mlflow: bool = True, search_type: str = 'random'):
        """
        Initialize ModelTrainerSingle for single output (Fan Power) prediction.
        
        Args:
            use_mlflow: Whether to use MLflow tracking
            search_type: Type of hyperparameter search ('random', 'grid')
        """
        self.model_trainer_config = ModelTrainerSingleConfig()
        self.use_mlflow = use_mlflow
        self.search_type = search_type
        
        if self.use_mlflow:
            MLflowConfig.setup_mlflow()
            self.experiment_id = MLflowConfig.get_or_create_experiment(
                "BMS_AI_Single_Output_Training"
            )
            log.info(f"MLflow tracking enabled. Experiment ID: {self.experiment_id}")

    def get_model_configurations(self):
        """
        Define all models with optimized hyperparameter grids for faster training.
        Reduced parameter space for efficiency while maintaining model quality.
        """
        models = {
            'RandomForest': {
                'estimator': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt']
                }
            },
            'ExtraTrees': {
                'estimator': ExtraTreesRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt']
                }
            },
            'GradientBoosting': {
                'estimator': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'min_samples_split': [2, 5],
                    'subsample': [0.8, 1.0]
                }
            },
            'XGBoost': {
                'estimator': XGBRegressor(
                    objective='reg:squarederror',
                    random_state=42,
                    n_jobs=-1,
                    tree_method='hist'
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [5, 7],
                    'min_child_weight': [1, 3],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'reg_alpha': [0, 0.1],
                    'reg_lambda': [0.1, 1]
                }
            },
            'Ridge': {
                'estimator': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1, 10],
                    'solver': ['auto']
                }
            },
            'Lasso': {
                'estimator': Lasso(random_state=42, max_iter=10000),
                'params': {
                    'alpha': [0.01, 0.1, 1]
                }
            },
            'ElasticNet': {
                'estimator': ElasticNet(random_state=42, max_iter=10000),
                'params': {
                    'alpha': [0.01, 0.1],
                    'l1_ratio': [0.3, 0.5, 0.7]
                }
            }
        }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = {
                'estimator': lgb.LGBMRegressor(
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [5, 7],
                    'num_leaves': [31, 50],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            }
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = {
                'estimator': cb.CatBoostRegressor(
                    random_state=42,
                    verbose=0,
                    thread_count=-1
                ),
                'params': {
                    'iterations': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'depth': [6, 8],
                    'l2_leaf_reg': [3, 5]
                }
            }
        
        return models

    def tune_models(self, X_train, y_train, X_test, y_test):
        """
        Tune all models using RandomizedSearchCV or GridSearchCV.
        """
        try:
            log.info(f"Starting model tuning using {self.search_type.upper()} search...")
            
            if self.use_mlflow and MLflowConfig.AUTOLOG_SKLEARN:
                mlflow.sklearn.autolog(log_models=False)
            
            models_config = self.get_model_configurations()
            tuned_models = {}
            results = {}
            
            for name, config in models_config.items():
                log.info(f"\n{'='*60}")
                log.info(f"Tuning {name}...")
                log.info(f"{'='*60}")
                
                try:
                    if self.use_mlflow:
                        with mlflow.start_run(
                            experiment_id=self.experiment_id,
                            run_name=f"{name}_tuning",
                            nested=True
                        ):
                            mlflow.set_tags({
                                "model_type": name,
                                "stage": "hyperparameter_tuning",
                                "search_type": self.search_type
                            })
                            
                            # Perform hyperparameter search
                            if self.search_type == 'grid':
                                search = GridSearchCV(
                                    estimator=config['estimator'],
                                    param_grid=config['params'],
                                    cv=3,  # Reduced from 5 for faster training
                                    scoring='neg_mean_squared_error',
                                    verbose=1,
                                    n_jobs=-1
                                )
                            else:  # random search
                                search = RandomizedSearchCV(
                                    estimator=config['estimator'],
                                    param_distributions=config['params'],
                                    n_iter=10,  # Reduced from 20 for faster training
                                    cv=3,  # Reduced from 5 for faster training
                                    scoring='neg_mean_squared_error',
                                    verbose=1,
                                    random_state=42,
                                    n_jobs=-1
                                )
                            
                            # Fit the search
                            search.fit(X_train, y_train)
                            best_model = search.best_estimator_
                            
                            # Make predictions
                            y_train_pred = best_model.predict(X_train)
                            y_test_pred = best_model.predict(X_test)
                            
                            # Calculate metrics
                            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                            train_mae = mean_absolute_error(y_train, y_train_pred)
                            test_mae = mean_absolute_error(y_test, y_test_pred)
                            train_r2 = r2_score(y_train, y_train_pred)
                            test_r2 = r2_score(y_test, y_test_pred)
                            
                            # Log parameters and metrics
                            mlflow.log_params(search.best_params_)
                            mlflow.log_metrics({
                                "train_rmse": train_rmse,
                                "test_rmse": test_rmse,
                                "train_mae": train_mae,
                                "test_mae": test_mae,
                                "train_r2": train_r2,
                                "test_r2": test_r2,
                                "best_cv_score": -search.best_score_
                            })
                            
                            # Log model
                            if name in ['XGBoost']:
                                mlflow.xgboost.log_model(best_model, f"{name}_model")
                            elif name in ['LightGBM'] and LIGHTGBM_AVAILABLE:
                                mlflow.lightgbm.log_model(best_model, f"{name}_model")
                            elif name in ['CatBoost'] and CATBOOST_AVAILABLE:
                                mlflow.catboost.log_model(best_model, f"{name}_model")
                            else:
                                mlflow.sklearn.log_model(best_model, f"{name}_model")
                            
                            log.info(f"{name} - Test RMSE: {test_rmse:.4f}, Test R^2: {test_r2:.4f}")
                    
                    else:
                        # Without MLflow
                        if self.search_type == 'grid':
                            search = GridSearchCV(
                                estimator=config['estimator'],
                                param_grid=config['params'],
                                cv=3,  # Reduced from 5 for faster training
                                scoring='neg_mean_squared_error',
                                verbose=1,
                                n_jobs=-1
                            )
                        else:
                            search = RandomizedSearchCV(
                                estimator=config['estimator'],
                                param_distributions=config['params'],
                                n_iter=10,  # Reduced from 20 for faster training
                                cv=3,  # Reduced from 5 for faster training
                                scoring='neg_mean_squared_error',
                                verbose=1,
                                random_state=42,
                                n_jobs=-1
                            )
                        
                        search.fit(X_train, y_train)
                        best_model = search.best_estimator_
                        
                        y_test_pred = best_model.predict(X_test)
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                        test_r2 = r2_score(y_test, y_test_pred)
                        
                        log.info(f"{name} - Test RMSE: {test_rmse:.4f}, Test R^2: {test_r2:.4f}")
                    
                    tuned_models[name] = best_model
                    results[name] = {
                        'rmse': test_rmse,
                        'r2': test_r2,
                        'best_params': search.best_params_
                    }
                    
                except Exception as e:
                    log.error(f"Error tuning {name}: {e}")
                    continue
            
            log.info("\n" + "="*60)
            log.info("Model tuning complete!")
            log.info("="*60)
            
            return tuned_models, results
            
        except Exception as e:
            log.error(f"Exception occurred during model tuning: {e}")
            raise CustomException(e, sys)

    def select_best_model(self, tuned_models, results, X_test, y_test):
        """
        Select the best model based on test RMSE.
        """
        try:
            log.info("Selecting best model based on test RMSE...")
            
            best_name = min(results, key=lambda k: results[k]['rmse'])
            best_model = tuned_models[best_name]
            best_rmse = results[best_name]['rmse']
            best_r2 = results[best_name]['r2']
            
            log.info(f"\nBest Model: {best_name}")
            log.info(f"Test RMSE: {best_rmse:.4f}")
            log.info(f"Test R^2: {best_r2:.4f}")
            log.info(f"Best Parameters: {results[best_name]['best_params']}")
            
            return best_model, best_name, best_rmse, best_r2
            
        except Exception as e:
            log.error(f"Exception occurred in select_best_model: {e}")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_data_path):
        """
        Main method to train and select the best model for single output prediction.
        """
        try:
            log.info("="*60)
            log.info("Starting Single Output Model Training Pipeline")
            log.info("="*60)
            
            df = pd.read_csv(train_data_path)
            log.info(f"Loaded training data: {df.shape}")
            
            X = df.drop(columns=['Fan Power meter (KW)'])
            y = df['Fan Power meter (KW)']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            log.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            tuned_models, results = self.tune_models(X_train, y_train, X_test, y_test)
            
            best_model, best_name, best_rmse, best_r2 = self.select_best_model(
                tuned_models, results, X_test, y_test
            )
            
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            log.info(f"Saved best model to: {self.model_trainer_config.trained_model_file_path}")
            
            with open(self.model_trainer_config.best_model_name_path, 'w') as f:
                f.write(f"Best Model: {best_name}\n")
                f.write(f"Test RMSE: {best_rmse:.4f}\n")
                f.write(f"Test R^2: {best_r2:.4f}\n")
                f.write(f"Best Parameters: {results[best_name]['best_params']}\n")
            
            log.info("="*60)
            log.info("Training Pipeline Complete!")
            log.info("="*60)
            
            return self.model_trainer_config.trained_model_file_path
            
        except Exception as e:
            log.error(f"Exception occurred in initiate_model_trainer: {e}")
            raise CustomException(e, sys)
