#!/bin/bash

cd newton

# activate the Jupyter virtual environment
source /home/ubuntu/.venv/bin/activate

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies into the currently active venv
/home/ubuntu/.local/bin/uv sync --extra notebook --active
