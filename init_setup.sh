# #!/bin/bash

# echo "[$(date)]: START"

# echo "[$(date)]: creating env with python 3.10 version"
# conda create --prefix ./env python=3.10 -y

# echo "[$(date)]: activating the environment"

# source activate ./env
# 
# # run this before pip install uv
# echo "[$(date)]: installing the dev requirements"
# uv pip install -r requirements.txt

# echo "[$(date)]: END"

#!/bin/bash

# Make sure uv is installed before running this script
# You can install it with: curl -LsSf https://astral.sh/uv/install.sh | sh
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex" 
echo "[$(date)]: START"

echo "[$(date)]: Creating virtual environment with uv and Python 3.10"

uv venv -p 3.10 .venv

echo "[$(date)]: Activating the virtual environment"

source .venv/bin/activate

echo "[$(date)]: Installing requirements with uv"

uv pip install -r requirements.txt

echo "[$(date)]: END"