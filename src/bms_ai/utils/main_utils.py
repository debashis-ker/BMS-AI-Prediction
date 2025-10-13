import os
import sys
import joblib
from src.bms_ai.exception import CustomException
from src.bms_ai.logger_config import setup_logger



log = setup_logger(__name__)

def save_object(file_path, obj):
    try:
        log.info(f"Saving object to {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)
        log.info("Object saved successfully.")

    except Exception as e:
        log.error(f"Exception occurred in save_object: {e}")
        raise CustomException(e, sys)
