#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the model download script
echo "Downloading and extracting models..."
python download_models.py
