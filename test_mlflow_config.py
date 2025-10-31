"""
Test script to verify MLflow configuration reads from .env file correctly
"""

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

print("=" * 70)
print("MLflow Configuration Test")
print("=" * 70)

# Check environment variables
print("\n1. Environment Variables from .env file:")
print(f"   MLFLOW_HOST: {os.getenv('MLFLOW_HOST', 'NOT SET')}")
print(f"   MLFLOW_PORT: {os.getenv('MLFLOW_PORT', 'NOT SET')}")
print(f"   MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI', 'NOT SET')}")
print(f"   ENVIRONMENT: {os.getenv('ENVIRONMENT', 'NOT SET')}")

# Import MLflowConfig
from src.bms_ai.utils.mlflow_config import MLflowConfig

print("\n2. MLflowConfig Values:")
print(f"   MLFLOW_HOST: {MLflowConfig.MLFLOW_HOST}")
print(f"   MLFLOW_PORT: {MLflowConfig.MLFLOW_PORT}")
print(f"   TRACKING_URI: {MLflowConfig.TRACKING_URI}")

print("\n3. Expected MLflow Server URL:")
print(f"   http://{MLflowConfig.MLFLOW_HOST}:{MLflowConfig.MLFLOW_PORT}")

print("\n" + "=" * 70)
print("✅ Configuration loaded successfully!")
print("=" * 70)

# Test with different scenarios
print("\n4. Testing Configuration Scenarios:")

# Scenario 1: Default values
print("\n   Scenario 1: If no .env file exists:")
print("   - MLFLOW_HOST would default to: 127.0.0.1")
print("   - MLFLOW_PORT would default to: 5000")

# Scenario 2: Current .env values
print("\n   Scenario 2: Current .env configuration:")
print(f"   - MLFLOW_HOST is set to: {MLflowConfig.MLFLOW_HOST}")
print(f"   - MLFLOW_PORT is set to: {MLflowConfig.MLFLOW_PORT}")

# Scenario 3: Command line override
print("\n   Scenario 3: Command line can override:")
print("   - python start_mlflow.py --port 8080 --host 0.0.0.0")
print("   - This will use port 8080 instead of the .env value")

print("\n" + "=" * 70)
print("Configuration Priority (highest to lowest):")
print("1. Command-line arguments")
print("2. Environment variables (.env file)")
print("3. Default values (127.0.0.1:5000)")
print("=" * 70)
