"""
MLflow Server Startup Script
This script starts the MLflow tracking server with the configured settings.
"""

import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.bms_ai.utils.mlflow_config import MLflowConfig

def start_mlflow_server(host=None, port=None):
    """
    Start MLflow tracking server.
    
    Args:
        host: Host address (default: from env MLFLOW_HOST or 127.0.0.1)
        port: Port number (default: from env MLFLOW_PORT or 5000)
    """
    # Use values from MLflowConfig if not provided
    host = host or MLflowConfig.MLFLOW_HOST
    port = port or MLflowConfig.MLFLOW_PORT
    
    # Ensure mlruns directory exists
    MLflowConfig.MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Starting MLflow Tracking Server")
    print("=" * 70)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Tracking URI: {MLflowConfig.TRACKING_URI}")
    print(f"MLflow Directory: {MLflowConfig.MLFLOW_DIR}")
    print("=" * 70)
    print()
    print("MLflow UI will be available at:")
    print(f"  http://{host}:{port}")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    
    # Start MLflow server
    try:
        # Convert Windows paths to file:// URI format for MLflow
        backend_uri = MLflowConfig.MLFLOW_DIR.as_posix()
        if not backend_uri.startswith('file:'):
            backend_uri = f"file:///{backend_uri}"
        
        artifact_root = MLflowConfig.ARTIFACTS_DIR.as_posix()
        if not artifact_root.startswith('file:'):
            artifact_root = f"file:///{artifact_root}"
        
        subprocess.run([
            "mlflow", "server",
            "--host", host,
            "--port", str(port),
            "--backend-store-uri", backend_uri,
            "--default-artifact-root", artifact_root
        ])
    except KeyboardInterrupt:
        print("\n\nMLflow server stopped.")
    except Exception as e:
        print(f"\nError starting MLflow server: {e}")
        print("\nMake sure MLflow is installed:")
        print("  pip install mlflow")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start MLflow Tracking Server")
    parser.add_argument("--host", default=None, help=f"Host address (default: from env MLFLOW_HOST or {MLflowConfig.MLFLOW_HOST})")
    parser.add_argument("--port", type=int, default=None, help=f"Port number (default: from env MLFLOW_PORT or {MLflowConfig.MLFLOW_PORT})")
    
    args = parser.parse_args()
    
    start_mlflow_server(host=args.host, port=args.port)
