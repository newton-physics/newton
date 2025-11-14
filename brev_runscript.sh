#!/bin/bash

echo "Activating Jupyter virtual environment"
source /home/ubuntu/.venv/bin/activate

echo "Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /home/ubuntu/newton
VIRTUAL_ENV=/home/ubuntu/.venv /home/ubuntu/.local/bin/uv sync --extra notebook --extra torch-cu12 --active

echo "Running robot_policy example"
/home/ubuntu/.local/bin/uv run --extra torch-cu12 -m newton.examples robot_policy --viewer null

# fix messed up Jupyter Lab installation
VIRTUAL_ENV=/home/ubuntu/.venv /home/ubuntu/.local/bin/uv pip install --upgrade --force-reinstall --no-cache-dir jupyter

# force restart of Jupyter Lab (somehow Brev will restart Jupyter Lab automatically)
pkill jupyter
