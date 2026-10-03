#!/bin/bash
# On a rented CUDA card: install what train.py needs, then train until done or
# out of hours; after a crash, resume from last/. Run from the repo copy:
#   HOURS=22 training/cloud.sh training/configs/qwen3-4b-h200.yaml
# Output goes to /scratch/runs/<config name>/ and <config name>.out, as at home.
cd "$(dirname "$0")/.."
CONFIG=$1; NAME=$(basename "$CONFIG" .yaml); HOURS=${HOURS:-22}; TRIES=${TRIES:-6}
export HF_HOME=${HF_HOME:-/scratch/hf}
mkdir -p /scratch/runs "$HF_HOME"
# vast's base image keeps its Python in /venv/main, without torch.
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
PIP="pip"; command -v uv >/dev/null && PIP="uv pip"
if [ ! -f /scratch/.cloud-ready ]; then
  # TORCH=2.9.1 replaces an image's own torch (h200-318 trained on 2.9.1, the v5 runs on
  # an image's 2.11); the image's torchvision/torchaudio, built for its torch, would
  # break the transformers import, and train.py needs neither.
  if [ -n "${TORCH:-}" ]; then
    $PIP install -q "torch==$TORCH" --index-url ${TORCH_INDEX:-https://download.pytorch.org/whl/cu126} || exit 3
    $PIP uninstall $([ "$PIP" = pip ] && echo -y) torchvision torchaudio torchcodec 2>/dev/null || true
  fi
  # An image's own CUDA torch stays; otherwise the one v2-153 trained with, for CUDA 12
  # (a 12.x wheel runs on any 12.x driver). The rest is pinned to v2-153's too.
  python -c "import torch" 2>/dev/null \
    || $PIP install -q "torch==2.9.1" --index-url ${TORCH_INDEX:-https://download.pytorch.org/whl/cu126} || exit 3
  $PIP install -q "transformers==5.17.0" "peft==0.21.0" "accelerate==1.15.0" pyyaml safetensors \
    && python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())" \
    && touch /scratch/.cloud-ready || exit 3
fi
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
for try in $(seq 1 $TRIES); do
  left=$(( DEADLINE - $(date +%s) ))
  [ $left -le 0 ] && { echo "=== out of time"; exit 124; }
  echo "=== attempt $try $(date '+%F %T'), $(( left / 60 )) min left"
  timeout --signal=TERM --kill-after=60 $left torchrun --nproc_per_node=1 training/train.py "$CONFIG" < /dev/null
  code=$?
  echo "=== exit $code $(date '+%F %T')"
  [ $code = 0 ] && exit 0
  [ $code = 124 ] && { echo "=== out of time"; exit 124; }
  sleep 30
done
exit 1
