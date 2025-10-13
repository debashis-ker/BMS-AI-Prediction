from src.bms_ai.logger_config import setup_logger
from src.bms_ai.components.data_ingestion import DataIngestion
from src.bms_ai.components.data_transformation import DataTransformation
from src.bms_ai.components.model_trainer import ModelTrainer

log = setup_logger(__name__)

class TrainingPipeline:
    def main(self):
        log.info("Starting the training pipeline.")
        
        
        # Data Ingestion
        log.info("Initiating data ingestion.")
        data_ingestion = DataIngestion()
        raw_data_path = data_ingestion.initiate_data_ingestion()
        log.info(f"Data ingestion complete. Raw data at: {raw_data_path}")

        
        # Data Transformation
        log.info("Initiating data transformation.")
        data_transformation = DataTransformation()
        train_data_path, _ = data_transformation.initiate_data_transformation(raw_data_path)
        log.info(f"Data transformation complete. Training data at: {train_data_path}")


        # Model Training
        log.info("Initiating model training.")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_data_path)
        log.info("Model training complete.")
        log.info("Training pipeline finished successfully.")

if __name__ == '__main__':
    pipeline = TrainingPipeline()
    pipeline.main()
