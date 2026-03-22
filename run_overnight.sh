#!/bin/bash
# Overnight run: finish QA training, then train agent levels
set -e

echo "=== Overnight Run Started: $(date) ==="

# Wait for current RMT QA training to finish
echo "Waiting for RMT QA training (v3.30) to complete..."
while pgrep -f "exp_v3_30_rmt_qa" > /dev/null 2>&1; do
    sleep 60
done
echo "RMT QA training complete: $(date)"

# Find best checkpoint from QA training
CKPT=$(ls -t results/rmt_qa/checkpoint_epoch*.pt 2>/dev/null | head -1)
echo "Best checkpoint: $CKPT"

# Run agent levels training (Level 1 + 2 training, all 4 eval)
echo ""
echo "=== Agent Levels Training: $(date) ==="
PYTHONUNBUFFERED=1 .venv/bin/python -m rpvt.experiments.exp_v3_31_agent_levels \
    --checkpoint "$CKPT" \
    --epochs 10 \
    --n-train 1500 \
    --lr 5e-5 \
    --output-dir results/agent_levels

echo ""
echo "=== Overnight Run Complete: $(date) ==="
