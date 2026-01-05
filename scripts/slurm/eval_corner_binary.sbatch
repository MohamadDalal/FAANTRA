#!/bin/bash
#SBATCH --job-name=corners_binary_eval
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/corners_binary_eval_%j.out
#SBATCH --error=logs/corners_binary_eval_%j.err

# FAANTRA Corner Kick Binary Evaluation
# Evaluates SHOT vs NO_SHOT prediction on test set

set -e

cd /home/mseo/CornerTactics/FAANTRA

source venv/bin/activate

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

echo "=== FAANTRA Corner Binary Evaluation ==="
echo "Job: $SLURM_JOB_ID, Node: $(hostname)"
echo "$(date)"
nvidia-smi

# Configuration
CONFIG=${CONFIG:-"config/SoccerNetBall/Corner-Binary.json"}
CHECKPOINT=${CHECKPOINT:-"checkpoints/corners-binary/corner_binary/transformer/checkpoint/checkpoint.ckpt"}

echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"

# Run evaluation
python -u test.py "$CONFIG" "$CHECKPOINT" corner_binary --split test

echo "=== Evaluation Complete ==="
echo "$(date)"
