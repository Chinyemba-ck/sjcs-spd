#!/bin/bash
# Script to run SPD decomposition on Gemma-3-270M-IT
# This runs the full SPD training to decompose weight matrices into U,V components with gate MLPs

set -e  # Exit on error

echo "================================"
echo "Running SPD on Gemma-3-270M-IT"
echo "================================"

# Activate virtual environment
source .venv/bin/activate

# Run without WandB for now - save locally
export WANDB_MODE=offline

# Run SPD decomposition
echo "Starting SPD decomposition..."
python -m spd.experiments.lm.lm_decomposition spd/experiments/lm/gemma_270m_multilingual_config.yaml

echo "SPD decomposition complete!"
echo "Outputs saved in wandb/latest-run/"