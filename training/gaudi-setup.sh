#!/bin/bash
# On the Gaudi box, once: find the Python with Intel Gaudi's PyTorch, make a venv
# beside it (system site packages, so torch stays Gaudi's), add what train.py
# needs with torch pinned, and fetch the base models. gaudi.sh setup runs this.
#   SYS_PY=python3 VENV=~/venv-wir HF_HOME=~/hf training/gaudi-setup.sh
set -u
cd "$(dirname "$0")/.."
SYS_PY=${SYS_PY:-python3}; VENV=${VENV:-$HOME/venv-wir}
export HF_HOME=${HF_HOME:-$HOME/hf}
hl-smi | head -24 || { echo "=== no hl-smi: is the Gaudi driver installed?"; exit 3; }
$SYS_PY -c "import habana_frameworks.torch, torch; print('torch', torch.__version__)" || {
  echo "=== $SYS_PY has no habana_frameworks: run in Intel's PyTorch container, or set SYS_PY"; exit 3; }
[ -x "$VENV/bin/python" ] || $SYS_PY -m venv --system-site-packages "$VENV" || exit 3
# The pin makes pip fail rather than swap Gaudi's torch for a CUDA one.
"$VENV/bin/python" -c "import torch; print('torch==' + torch.__version__)" > "$VENV/constraints.txt"
"$VENV/bin/pip" install -q -c "$VENV/constraints.txt" "transformers==5.17.0" "peft==0.21.0" \
  pyyaml safetensors || exit 3
"$VENV/bin/python" -c "
import habana_frameworks.torch, peft, torch, transformers
print('transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__)" || exit 3
for m in Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B; do
  "$VENV/bin/hf" download "$m" > /dev/null && echo "fetched $m" || exit 3
done
touch "$VENV/.ready"
echo "=== ready"
