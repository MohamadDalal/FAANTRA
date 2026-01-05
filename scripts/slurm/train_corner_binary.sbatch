#!/bin/bash
#SBATCH --job-name=corners_binary
#SBATCH --partition=acltr
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/corners_binary_%j.out
#SBATCH --error=logs/corners_binary_%j.err

# FAANTRA Corner Kick Binary Shot Prediction
# Predicts SHOT vs NO_SHOT from corner kick videos

set -e

cd /home/mseo/CornerTactics/FAANTRA

# Create directories
mkdir -p logs checkpoints/corners-binary wandb_logs

# Activate environment
source venv/bin/activate

# Disable wandb login requirement (use offline mode)
export WANDB_MODE=offline

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

# Print job info
echo "========================================"
echo "FAANTRA Corner Binary Training"
echo "Task: SHOT vs NO_SHOT prediction"
echo "========================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi
echo "========================================"

# Configuration
CONFIG=${CONFIG:-"config/SoccerNetBall/Corner-Binary.json"}
MODEL=${MODEL:-"corner_binary"}
CHECKPOINT=${CHECKPOINT:-""}

echo "Config: $CONFIG"
echo "Model: $MODEL"
echo "Checkpoint: $CHECKPOINT"

# Run training (with optional checkpoint resume)
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint..."
    python -u main.py "$CONFIG" "$MODEL" --checkpoint-path "$CHECKPOINT"
else
    python -u main.py "$CONFIG" "$MODEL"
fi

echo "========================================"
echo "Training completed: $(date)"
echo "========================================"
