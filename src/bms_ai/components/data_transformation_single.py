import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from src.bms_ai.exception import CustomException
from src.bms_ai.logger_config import setup_logger
import os
from src.bms_ai.utils.main_utils import save_object

log = setup_logger(__name__)

@dataclass
class DataTransformationSingleConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor_single.pkl")
    scaler_obj_file_path: str = os.path.join('artifacts', "scaler_single.pkl")
    train_data_path: str = os.path.join('artifacts', "train_single.csv")
    test_data_path: str = os.path.join('artifacts', "test_single.csv")

class DataTransformationSingle:
    def __init__(self, scaler_type: str = 'minmax'):
        """
        Initialize DataTransformationSingle.
        
        Args:
            scaler_type: Type of scaler to use ('minmax', 'standard', 'robust')
        """
        self.data_transformation_config = DataTransformationSingleConfig()
        self.scaler_type = scaler_type

    def transform_dataframe(self, df: pd.DataFrame):
        """
        Helper function to apply the core data transformation logic for single output (Fan Power).
        """
        try:
            log.info("Starting data transformation for single output (Fan Power)...")
            df['data_received_on'] = pd.to_datetime(df['data_received_on'])
            df['data_received_on_naive'] = df['data_received_on'].dt.tz_localize(None)
            df.sort_values('data_received_on_naive', inplace=True)

            converted_df = df.pivot_table(
                index=['data_received_on_naive'],
                columns='datapoint',
                values='monitoring_data',
                aggfunc='first'
            )
            converted_df.reset_index(inplace=True)

            numeric_cols = [
                "RA Damper feedback", "SA Pressure setpoint", "OA Humid", "RA Temp",
                "RA CO2", "RA CO2 setpoint", "SA Fan Speed feedback", "SA Fan Speed control",
                "RA Temp control( Valve Feedback)", "SA pressure", "Fan Power meter (KW)",
                "RA damper control", "OA Temp", "OA Flow", "SA temp", "RA  temperature setpoint"
            ]
            present_numeric_cols = [col for col in numeric_cols if col in converted_df.columns]
            converted_df[present_numeric_cols] = converted_df[present_numeric_cols].apply(pd.to_numeric, errors='coerce')

            if "Sup fan cmd" in converted_df.columns:
                mappings = {'active': 1, 'inactive': 0}
                converted_df["Sup fan cmd"] = converted_df["Sup fan cmd"].replace(mappings).fillna(0)

            log.info("Engineering time-based features...")
            converted_df['hour'] = converted_df['data_received_on_naive'].dt.hour
            converted_df['dayofweek'] = converted_df['data_received_on_naive'].dt.dayofweek
            converted_df['month'] = converted_df['data_received_on_naive'].dt.month
            converted_df['dayofyear'] = converted_df['data_received_on_naive'].dt.dayofyear
            
            converted_df['hour_sin'] = np.sin(2 * np.pi * converted_df['hour'] / 24)
            converted_df['hour_cos'] = np.cos(2 * np.pi * converted_df['hour'] / 24)
            converted_df['month_sin'] = np.sin(2 * np.pi * converted_df['month'] / 12)
            converted_df['month_cos'] = np.cos(2 * np.pi * converted_df['month'] / 12)
            converted_df['dayofweek_sin'] = np.sin(2 * np.pi * converted_df['dayofweek'] / 7)
            converted_df['dayofweek_cos'] = np.cos(2 * np.pi * converted_df['dayofweek'] / 7)

            target_column = "Fan Power meter (KW)"
            
            if target_column not in converted_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset!")

            converted_df.dropna(subset=[target_column], inplace=True)
            
            converted_df.fillna(method='ffill', inplace=True)
            converted_df.fillna(method='bfill', inplace=True)
            converted_df.fillna(0, inplace=True)

            y = converted_df[target_column]
            X = converted_df.drop(columns=[target_column, 'data_received_on_naive'], errors='ignore')
            
            other_targets = ["RA damper control", "RA Temp control( Valve Feedback)", "SA Fan Speed control"]
            X = X.drop(columns=[col for col in other_targets if col in X.columns], errors='ignore')
            
            log.info(f"Data transformation complete. Features: {X.shape[1]}, Samples: {X.shape[0]}")
            return X, y

        except Exception as e:
            log.error(f"Exception occurred in transform_dataframe: {e}")
            raise CustomException(e, sys)

    def get_data_transformer_object(self, X: pd.DataFrame):
        """
        This function creates the data transformation object with configurable scaler.
        """
        try:
            log.info(f"Creating data transformer object with {self.scaler_type} scaler.")
            
            encoded_categorical_features = []
            if 'Sup fan cmd' in X.columns:
                encoded_categorical_features.append('Sup fan cmd')

            numeric_features = [
                col for col in X.columns 
                if pd.api.types.is_numeric_dtype(X[col]) and col not in encoded_categorical_features
            ]
            
            if self.scaler_type == 'standard':
                scaler = StandardScaler()
            elif self.scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                scaler = MinMaxScaler()
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', scaler, numeric_features),
                    ('cat', 'passthrough', encoded_categorical_features)
                ],
                remainder='drop'
            )
            
            log.info(f"Data transformer object created successfully with {len(numeric_features)} numeric features.")
            return preprocessor

        except Exception as e:
            log.error(f"Exception occurred in get_data_transformer_object: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_data_path):
        """
        Initiates data transformation for single output prediction.
        """
        try:
            log.info(f"Initiating data transformation with raw data from: {raw_data_path}")
            df = pd.read_csv(raw_data_path)
            log.info(f"Read raw data: {df.shape[0]} rows, {df.shape[1]} columns")
            
            X, y = self.transform_dataframe(df)
            
            preprocessor = self.get_data_transformer_object(X)
            
            log.info("Applying preprocessor to the data.")
            X_transformed = preprocessor.fit_transform(X)
            
            transformed_cols = preprocessor.get_feature_names_out()
            X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_cols, index=X.index)

            train_df = pd.concat([X_transformed_df, y.rename('Fan Power meter (KW)')], axis=1)
            train_df.dropna(inplace=True)

            os.makedirs(os.path.dirname(self.data_transformation_config.train_data_path), exist_ok=True)
            
            log.info(f"Saving training data to: {self.data_transformation_config.train_data_path}")
            train_df.to_csv(self.data_transformation_config.train_data_path, index=False)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            log.info("Saved preprocessing object.")
            log.info(f"Data transformation complete. Final dataset shape: {train_df.shape}")

            return (
                self.data_transformation_config.train_data_path,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            log.error(f"Exception occurred in initiate_data_transformation: {e}")
            raise CustomException(e, sys)
