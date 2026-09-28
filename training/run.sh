#!/bin/bash
# Train under the watchdog; after a crash or a kill, wait and resume from the last checkpoint.
#   training/run.sh training/configs/qwen3-4b-12k.yaml
# Output goes to /scratch/runs/<config name>.out, temperatures to <name>-temps.log.
cd "$(dirname "$0")/.."
CONFIG=$1; NAME=$(basename "$CONFIG" .yaml); TRIES=${TRIES:-6}
export HF_HOME=${HF_HOME:-/scratch/hf} TEMPS=/scratch/runs/$NAME-temps.log
export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}   # what the memory probe measured under
CAP_MAX=${CAP_MAX:-190}
# A GPU reset puts the card's power cap and fan back to their defaults (190 W, auto fan).
ready() {
  for h in /sys/bus/pci/devices/0000:{67,b5}:00.0/hwmon/hwmon*; do
    cap=$(( $(cat $h/power1_cap)/1000000 )); fan=$(cat $h/pwm1_enable)
    if [ $cap -gt $CAP_MAX ] || [ $fan != 1 ]; then
      echo "=== not ready: $h cap ${cap}W fan mode $fan (want <= ${CAP_MAX}W, manual fan)"; return 1
    fi
  done
}
for try in $(seq 1 $TRIES); do
  ready || exit 3
  echo "=== attempt $try $(date '+%F %T')"
  training/watchdog.sh .venv/bin/torchrun --nproc_per_node=2 training/train.py "$CONFIG" < /dev/null
  code=$?
  echo "=== exit $code $(date '+%F %T')"
  [ $code = 0 ] && exit 0
  sleep 180     # cool down; let a reset card settle
done
exit 1
