#!/bin/bash
# On the Gaudi box, once: find the Python with Intel Gaudi's PyTorch, make a venv
# beside it (system site packages, so torch stays Gaudi's), add what train.py and
# export.py need with torch pinned, fetch llama.cpp's converter and the base
# models. Everything goes under WIR_ROOT (the big drive). gaudi.sh setup runs this.
#   WIR_ROOT=/mnt/nvme SYS_PY=python3 training/gaudi-setup.sh
set -u
cd "$(dirname "$0")/.."
ROOT=${WIR_ROOT:-$HOME}; SYS_PY=${SYS_PY:-python3}; VENV=$ROOT/venv-wir; LLAMA=$ROOT/llama.cpp
export HF_HOME=$ROOT/hf
# The converter Lagutenko-4B's GGUF was written with (b11177), so exports compare.
LLAMA_COMMIT=1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865
hl-smi | head -24 || { echo "=== no hl-smi: is the Gaudi driver installed?"; exit 3; }
df -h "$ROOT" | tail -1; grep -E 'MemTotal|MemAvailable' /proc/meminfo
$SYS_PY -c "import habana_frameworks.torch, torch; print('torch', torch.__version__)" || {
  echo "=== $SYS_PY has no habana_frameworks: run in Intel's PyTorch container, or set SYS_PY"; exit 3; }
[ -x "$VENV/bin/python" ] || $SYS_PY -m venv --system-site-packages "$VENV" || exit 3
# The pin makes pip fail rather than swap Gaudi's torch for a CUDA one.
"$VENV/bin/python" -c "import torch; print('torch==' + torch.__version__)" > "$VENV/constraints.txt"
"$VENV/bin/pip" install -q -c "$VENV/constraints.txt" "transformers==5.17.0" "peft==0.21.0" \
  pyyaml safetensors sentencepiece || exit 3
"$VENV/bin/python" -c "
import habana_frameworks.torch, peft, torch, transformers
print('transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__)" || exit 3
if [ ! -f "$LLAMA/convert_hf_to_gguf.py" ]; then
  git init -q "$LLAMA" && git -C "$LLAMA" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp \
    "$LLAMA_COMMIT" && git -C "$LLAMA" checkout -q FETCH_HEAD || exit 3
fi
echo "llama.cpp converter at $(git -C "$LLAMA" rev-parse --short HEAD)"
for m in Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-8B; do
  "$VENV/bin/hf" download "$m" > /dev/null && echo "fetched $m" || exit 3
done
touch "$VENV/.ready"
echo "=== ready"
