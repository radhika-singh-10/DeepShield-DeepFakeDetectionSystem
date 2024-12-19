#!/bin/bash

# Ensure the script exits on any error
set -e

# Define the Python files
PREPROCESSING_SCRIPT="frame-extraction.py"
MODEL_SCRIPT="state-of-the-art-model.py"
VISUALIZATION_SCRIPT="visualization-generation.py"

# Check if preprocessing script exists and run it
if [ -f "$PREPROCESSING_SCRIPT" ]; then
    echo "Running preprocessing script: $PREPROCESSING_SCRIPT"
    python "$PREPROCESSING_SCRIPT"
else
    echo "Error: Preprocessing script $PREPROCESSING_SCRIPT not found."
    exit 1
fi

# Check if model training script exists and run it
if [ -f "$MODEL_SCRIPT" ]; then
    echo "Running model training script: $MODEL_SCRIPT"
    python "$MODEL_SCRIPT"
else
    echo "Error: Model training script $MODEL_SCRIPT not found."
    exit 1
fi

# Check if visualization script exists and run it
if [ -f "$VISUALIZATION_SCRIPT" ]; then
    echo "Running visualization script: $VISUALIZATION_SCRIPT"
    python "$VISUALIZATION_SCRIPT"
else
    echo "Error: Visualization script $VISUALIZATION_SCRIPT not found."
    exit 1
fi

# Completion message
echo "All scripts executed successfully."
