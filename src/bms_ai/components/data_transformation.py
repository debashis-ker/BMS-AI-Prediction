import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.bms_ai.exception import CustomException
from src.bms_ai.logger_config import setup_logger
import os
from src.bms_ai.utils.main_utils import save_object

log = setup_logger(__name__)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def transform_dataframe(self, df: pd.DataFrame):
        """
        Helper function to apply the core data transformation logic.
        """
        try:
            log.info("Starting data transformation...")
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

            target_columns = ["RA damper control", "RA Temp control( Valve Feedback)", "SA Fan Speed control", "Fan Power meter (KW)"]
            present_target_cols = [col for col in target_columns if col in converted_df.columns]

            converted_df.dropna(subset=present_target_cols, inplace=True)
            converted_df.fillna(method='ffill', inplace=True)
            converted_df.fillna(method='bfill', inplace=True)

            y = converted_df[present_target_cols]
            X = converted_df.drop(columns=present_target_cols + ['data_received_on_naive'], errors='ignore')
            log.info("Data transformation logic applied successfully.")
            return X, y

        except Exception as e:
            log.error(f"Exception occurred in transform_dataframe: {e}")
            raise CustomException(e, sys)

    def get_data_transformer_object(self, X: pd.DataFrame):
        '''
        This function is responsible for creating the data transformation object.
        '''
        try:
            log.info("Creating data transformer object.")
            encoded_categorical_features = []
            if 'Sup fan cmd' in X.columns:
                encoded_categorical_features.append('Sup fan cmd')

            numeric_features = [
                col for col in X.columns 
                if pd.api.types.is_numeric_dtype(X[col]) and col not in encoded_categorical_features
            ]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', MinMaxScaler(), numeric_features),
                    ('cat', 'passthrough', encoded_categorical_features)
                ],
                remainder='drop'
            )
            log.info("Data transformer object created successfully.")
            return preprocessor

        except Exception as e:
            log.error(f"Exception occurred in get_data_transformer_object: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_data_path):
        try:
            log.info(f"Initiating data transformation with raw data from: {raw_data_path}")
            df = pd.read_csv(raw_data_path)
            log.info("Read raw data for transformation.")
            
            X, y = self.transform_dataframe(df)
            
            
            preprocessor = self.get_data_transformer_object(X)
            
            log.info("Applying preprocessor to the data.")
            X_transformed = preprocessor.fit_transform(X)
            
            transformed_cols = preprocessor.get_feature_names_out()
            X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_cols, index=X.index)

            train_df = pd.concat([X_transformed_df, y], axis=1)
            train_df.dropna(inplace=True)

            os.makedirs(os.path.dirname(self.data_transformation_config.train_data_path), exist_ok=True)
            log.info(f"Saving training data to: {self.data_transformation_config.train_data_path}")
            train_df.to_csv(self.data_transformation_config.train_data_path, index=False)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            log.info("Saved preprocessing object.")
            log.info("Data transformation complete.")

            return (
                self.data_transformation_config.train_data_path,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            log.error(f"Exception occurred in initiate_data_transformation: {e}")
            raise CustomException(e, sys)
