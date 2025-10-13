import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from dataclasses import dataclass
from src.bms_ai.logger_config import setup_logger
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import keras_tuner as kt

from src.bms_ai.exception import CustomException
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.utils.main_utils import save_object

warnings.filterwarnings('ignore')

log = setup_logger(__name__)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def tune_sklearn_models(self, X_train, y_train):
        """
        Defines and tunes scikit-learn models using RandomizedSearchCV.
        """
        try:
            log.info("Starting Training for Scikit-learn Models")
            models_to_tune = {
                'RandomForest': {
                    'estimator': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30, None],
                        'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]
                    }
                },
                'GradientBoosting': {
                    'estimator': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]
                    }
                },
                'XGBoost': {
                    'estimator': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7], 'colsample_bytree': [0.7, 0.8, 1.0],
                    }
                }
            }

            best_sklearn_models = {}
            for name, config in models_to_tune.items():
                log.info(f"Tuning {name}...")
                random_search = RandomizedSearchCV(
                    estimator=config['estimator'], param_distributions=config['params'],
                    n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1
                )
                search_wrapper = MultiOutputRegressor(random_search)
                search_wrapper.fit(X_train, y_train)
                best_sklearn_models[name] = search_wrapper
                log.info(f"Finished tuning {name}.")
            
            log.info("Scikit-learn model tuning complete.")
            return best_sklearn_models
        except Exception as e:
            log.error(f"Exception occurred during Scikit-learn model tuning: {e}")
            raise CustomException(e, sys)

    # def create_keras_model_builder(self, input_shape, output_shape):
    #     """
    #     Factory function to create the Keras model builder with specific input/output shapes.
    #     """
    #     def build_model(hp):
    #         inputs = keras.Input(shape=(input_shape,))
    #         x = inputs
    #         for i in range(hp.Int('num_layers', 1, 3)):
    #             x = layers.Dense(
    #                 units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
    #                 activation=hp.Choice('activation', ['relu', 'tanh'])
    #             )(x)
    #             x = layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1))(x)
    #         outputs = layers.Dense(output_shape)(x)
    #         model = keras.Model(inputs=inputs, outputs=outputs)
    #         learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    #         model.compile(
    #             optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    #             loss="mean_squared_error",
    #             metrics=["mean_absolute_error","mean_squared_error"],
    #         )
    #         return model
    #     return build_model

    # def tune_keras_model(self, X_train, y_train):
    #     """
    #     Tunes and trains a deep learning model using Keras Tuner.
    #     """
    #     try:
    #         log.info("Starting Training for Deep Learning Model")
    #         model_builder = self.create_keras_model_builder(X_train.shape[1], y_train.shape[1])
    #         tuner = kt.RandomSearch(
    #             model_builder, objective='val_loss', max_trials=10, executions_per_trial=2,
    #             directory='keras_tuner_dir', project_name='multi_output_regression'
    #         )
    #         tuner.search_space_summary()
    #         early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    #         log.info("Running Keras Tuner search...")
    #         log.info("Starting Keras Tuner search for hyperparameter optimization")
    #         tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    #         
    #         best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    #         keras_model = tuner.get_best_models(num_models=1)[0]
    #         log.info(f"Optimal Keras hyperparameters found: Learning Rate={best_hps.get('lr'):.4f}")
    #         log.info("Keras model tuning complete.")
    #         return keras_model
    #     except Exception as e:
    #         log.error(f"Exception occurred during Keras model tuning: {e}")
    #         raise CustomException(e, sys)

    def evaluate_models(self, models: dict, X_test, y_test):
        """
        Evaluates a dictionary of trained models on the test set and returns a results DataFrame.
        """
        try:
            log.info("Evaluating All Models on Test Set")
            evaluation_results = {}
            for name, model in models.items():
                predictions = model.predict(X_test)
                log.info(f"Evaluating model: {name}")
                log.info(f"Predictions shape: {predictions.shape}, y_test shape: {y_test.shape}")
                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                evaluation_results[name] = {'MAE': mae, 'MSE': mse, 'R2 Score': r2}
                log.info(f"{name} Metrics: MAE={mae:.4f}, MSE={mse:.4f}, R2 Score={r2:.4f}")
                
            results_df = pd.DataFrame(evaluation_results).T
            log.info("Model Comparison:")
            log.info(results_df)
            log.info("Model evaluation complete.")
            return results_df
        except Exception as e:
            log.error(f"Exception occurred during model evaluation: {e}")
            raise CustomException(e, sys)

    def initiate_model_training(self, train_data_path):
        try:
            log.info(f"Initiating model training with data from: {train_data_path}")
            train_df = pd.read_csv(train_data_path)
            log.info("Loaded training data.")
            
            target_columns = ["RA damper control", "RA Temp control( Valve Feedback)", "SA Fan Speed control", "Fan Power meter (KW)"]
            
            y = train_df[target_columns]
            X = train_df.drop(columns=target_columns, axis=1)
            log.info("Separated features and target variables.")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            log.info("Split data into training and testing sets.")

            log.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            log.debug(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


            best_sklearn_models = self.tune_sklearn_models(X_train, y_train)
            # keras_model = self.tune_keras_model(X_train, y_train)
            # all_models = {**best_sklearn_models, 'Keras_Functional_API': keras_model}
            all_models = best_sklearn_models
            results_df = self.evaluate_models(all_models, X_test, y_test)

            best_model_name = results_df['R2 Score'].idxmax()
            best_model = all_models[best_model_name]

            log.info(f"Best performing model is: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            log.info(f"Saved best model to: {self.model_trainer_config.trained_model_file_path}")

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            log.error(f"Exception occurred during model training initiation: {e}")
            raise CustomException(e, sys)
