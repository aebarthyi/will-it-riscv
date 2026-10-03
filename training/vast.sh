#!/bin/bash
# Move a run to and from a rented box (vast.ai: ssh -p PORT root@HOST).
#   training/vast.sh push HOST PORT configs/qwen3-4b-h200.yaml   # training/ + the config's data
#   training/vast.sh start HOST PORT configs/qwen3-4b-h200.yaml  # cloud.sh, detached; HOURS, TORCH as there
#   training/vast.sh pull HOST PORT configs/qwen3-4b-h200.yaml   # the run's adapters and logs
# The box's copy lives in /workspace/will-it-riscv; runs in /scratch/runs, as at home.
set -eu
cd "$(dirname "$0")/.."
ACTION=$1; HOST=$2; PORT=$3; CONFIG=training/${4#training/}; NAME=$(basename "$CONFIG" .yaml)
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
REMOTE=/workspace/will-it-riscv
DATA=$(sed -n 's/^data: *//p' "$CONFIG")
case $ACTION in
  push)
    # Some images come without rsync.
    $SSH root@$HOST "mkdir -p $REMOTE/$DATA /scratch/runs; command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)"
    rsync -az -e "$SSH" --exclude __pycache__ training/ root@$HOST:$REMOTE/training/
    rsync -az -e "$SSH" "$DATA/" root@$HOST:$REMOTE/$DATA/
    # A run that starts from another's adapter finds it at the same path there.
    INIT=$(sed -n 's/^init_adapter: *//p' "$CONFIG")
    if [ -n "$INIT" ]; then
      $SSH root@$HOST "mkdir -p $INIT"
      rsync -az -e "$SSH" "$INIT/adapter_config.json" "$INIT/adapter_model.safetensors" root@$HOST:$INIT/
    fi ;;
  start)
    $SSH root@$HOST "cd $REMOTE && HOURS=${HOURS:-22} TORCH=${TORCH:-} setsid nohup training/cloud.sh $CONFIG \
      > /scratch/runs/$NAME.out 2>&1 < /dev/null & echo started" ;;
  pull)
    # Adapters and logs; the optimizer states (state.pt, dcp/) stay behind.
    mkdir -p /scratch/runs/$NAME
    rsync -az -e "$SSH" root@$HOST:/scratch/runs/$NAME.out /scratch/runs/
    rsync -az -e "$SSH" --exclude dcp --exclude state.pt root@$HOST:/scratch/runs/$NAME/ /scratch/runs/$NAME/ ;;
  *) echo "push | start | pull" >&2; exit 2 ;;
esac
