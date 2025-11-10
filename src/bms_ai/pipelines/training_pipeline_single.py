import sys
import os
from src.bms_ai.exception import CustomException
from src.bms_ai.logger_config import setup_logger
from src.bms_ai.components.data_ingestion import DataIngestion
from src.bms_ai.components.data_transformation_single import DataTransformationSingle
from src.bms_ai.components.model_trainer_single import ModelTrainerSingle

log = setup_logger(__name__)

class TrainingPipelineSingle:
    """
    Training pipeline for single output (Fan Power) prediction.
    """
    
    def __init__(self, scaler_type: str = 'minmax', search_type: str = 'random', use_mlflow: bool = True):
        """
        Initialize the training pipeline.
        
        Args:
            scaler_type: Type of scaler ('minmax', 'standard', 'robust')
            search_type: Type of hyperparameter search ('random', 'grid')
            use_mlflow: Whether to use MLflow tracking
        """
        self.scaler_type = scaler_type
        self.search_type = search_type
        self.use_mlflow = use_mlflow
        
    def run_pipeline(self, raw_data_path: str = None):
        """
        Run the complete training pipeline.
        
        Args:
            raw_data_path: Path to raw data CSV. If None, uses DataIngestion.
        """
        try:
            log.info("="*80)
            log.info("STARTING SINGLE OUTPUT TRAINING PIPELINE (FAN POWER PREDICTION)")
            log.info("="*80)
            
            if raw_data_path is None:
                log.info("\nStep 1: Data Ingestion")
                log.info("-" * 80)
                data_ingestion = DataIngestion()
                raw_data_path = data_ingestion.initiate_data_ingestion()
                log.info(f"Data ingestion complete. Raw data path: {raw_data_path}")
            else:
                log.info(f"\nUsing provided raw data path: {raw_data_path}")
            
            log.info("\nStep 2: Data Transformation (Single Output)")
            log.info("-" * 80)
            log.info(f"Scaler Type: {self.scaler_type}")
            
            data_transformation = DataTransformationSingle(scaler_type=self.scaler_type)
            train_data_path, preprocessor_path = data_transformation.initiate_data_transformation(raw_data_path)
            
            log.info(f"Data transformation complete.")
            log.info(f"Train data path: {train_data_path}")
            log.info(f"Preprocessor path: {preprocessor_path}")
            
            log.info("\nStep 3: Model Training (Single Output)")
            log.info("-" * 80)
            log.info(f"Search Type: {self.search_type}")
            log.info(f"MLflow Tracking: {self.use_mlflow}")
            
            model_trainer = ModelTrainerSingle(
                use_mlflow=self.use_mlflow,
                search_type=self.search_type
            )
            model_path = model_trainer.initiate_model_trainer(train_data_path)
            
            log.info(f"\nModel training complete.")
            log.info(f"Best model saved at: {model_path}")
            
            log.info("\n" + "="*80)
            log.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            log.info("="*80)
            
            return {
                'model_path': model_path,
                'preprocessor_path': preprocessor_path,
                'train_data_path': train_data_path
            }
            
        except Exception as e:
            log.error(f"Exception occurred in training pipeline: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    """
    Example usage:
    
    # Basic usage with defaults (minmax scaler, random search, MLflow enabled)
    pipeline = TrainingPipelineSingle()
    results = pipeline.run_pipeline()
    
    # Custom configuration
    pipeline = TrainingPipelineSingle(
        scaler_type='standard',
        search_type='grid',
        use_mlflow=True
    )
    results = pipeline.run_pipeline(raw_data_path='path/to/data.csv')
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train single output model for Fan Power prediction')
    parser.add_argument('--scaler', type=str, default='minmax', 
                       choices=['minmax', 'standard', 'robust'],
                       help='Type of scaler to use')
    parser.add_argument('--search', type=str, default='grid',
                       choices=['random', 'grid'],
                       help='Type of hyperparameter search')
    parser.add_argument('--mlflow', type=bool, default=True,
                       help='Enable MLflow tracking')
    parser.add_argument('--data', type=str, default="C:\\Users\\debas\\OneDrive\\Desktop\\output.csv",
                       help='Path to raw data CSV (optional)')
    
    args = parser.parse_args()
    
    pipeline = TrainingPipelineSingle(
        scaler_type=args.scaler,
        search_type=args.search,
        use_mlflow=args.mlflow
    )
    
    results = pipeline.run_pipeline(raw_data_path=args.data)
    
    print("\n" + "="*80)
    print("TRAINING RESULTS:")
    print("="*80)
    print(f"Model Path: {results['model_path']}")
    print(f"Preprocessor Path: {results['preprocessor_path']}")
    print(f"Training Data Path: {results['train_data_path']}")
    print("="*80)
