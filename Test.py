from src.bms_ai.logger_config import setup_logger

log = setup_logger(__name__)

if __name__ == '__main__':
    #log.info("This is a test log message from Test.py")
    print(f"__name__ from Test.py = {__name__}")
    print("Test.py executed successfully.")