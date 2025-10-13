import os
import sys
from src.bms_ai.exception import CustomException
from src.bms_ai.logger_config import setup_logger
import pandas as pd
from dataclasses import dataclass

log = setup_logger(__name__)

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, file_path: str = 'C:/Users/debas/OneDrive/Desktop/output.csv'):
        log.info("Entered the data ingestion method or component")
        try:
            log.info(f"Reading data from {file_path}")
            df = pd.read_csv(file_path)
            log.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            log.info("Filtering the dataframe for Ground Floor AHU")
            df = df[(df['site'] == "Ground Floor") & (df['system_type'] == "AHU")]
            log.info("Filtered the dataframe")

            log.info(f"Saving raw data to {self.ingestion_config.raw_data_path}")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            log.info("Ingestion of the data is completed")

            return self.ingestion_config.raw_data_path

        except Exception as e:
            log.error(f"Exception occurred in data ingestion: {e}")
            raise CustomException(e, sys)
