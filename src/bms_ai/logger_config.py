import logging
import os
from logging.handlers import RotatingFileHandler

LOG_FOLDER = os.path.join(os.getcwd(), "log_info")

print("AT logger_config.py")
print(f"__name__ from logger_config.py = {__name__}")
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

class LevelFilter(logging.Filter):
    def __init__(self, level):
        self.__level = level

    def filter(self, record):
        return record.levelno == self.__level

def setup_logger(name=__name__):
    """
    Set up a logger where each file only contains logs of its specific level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler_configs = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
    }

    for level_name, level in handler_configs.items():
        handler = RotatingFileHandler(
            os.path.join(LOG_FOLDER, f'{level_name}.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        
        handler.addFilter(LevelFilter(level))
        
        logger.addHandler(handler)
        
    return logger

if __name__ == "__main__":
    log = setup_logger(__name__)
    print(f"Inside main of logger_config.py")
    log.debug("This will only go to debug.log")
    log.info("This will only go to info.log")
    log.warning("This will only go to warning.log")
    log.error("This will only go to error.log")
    #print(f"Log files have been created in: {os.path.abspath(LOG_FOLDER)}")