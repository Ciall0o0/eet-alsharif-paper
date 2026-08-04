#!/usr/bin/env bash
# Reproduce paper training: 3 seeds x config.
# Usage: ./scripts/train_all.sh configs/config_gru_shared_event_d.yaml 3
set -euo pipefail
CONFIG="${1:?config path required}"
N_SEEDS="${2:-3}"
SEEDS=(42 360 712)
mkdir -p checkpoints logs
for s in "${SEEDS[@]:0:$N_SEEDS}"; do
  name="$(basename "$CONFIG" .yaml)_s${s}"
  echo "=== training $name ==="
  python src/train.py --config "$CONFIG" --seed "$s" --epochs 60 \
    --no-swanlab \
    --checkpoint-dir "checkpoints/$name" > "logs/$name.log" 2>&1
done
echo "Done. Checkpoints in checkpoints/ (ppo_elevator_best.pt per seed)."
