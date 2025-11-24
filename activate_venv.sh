#!/bin/bash

#path to the virtual environment
VENV_PATH=".venv"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Run your desired command
"$@"

# Deactivate the virtual environment (optional)
deactivate
