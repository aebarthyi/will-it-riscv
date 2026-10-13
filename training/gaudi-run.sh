#!/bin/bash
# On the Gaudi box: train one config until it is done, resuming from last/ after
# a crash. One card by default (two runs side by side); CARDS=2 runs DDP on both.
#   CARD=0 training/gaudi-run.sh training/configs/qwen3-4b-gaudi-s0.yaml
#   CARD=1 training/gaudi-run.sh training/configs/qwen3-4b-gaudi-s1.yaml
# Runs go to $WIR_ROOT/runs/<config name>/ (WIR_ROOT: the big drive, default ~).
cd "$(dirname "$0")/.."
CONFIG=$1; TRIES=${TRIES:-4}; ROOT=${WIR_ROOT:-$HOME}; VENV=$ROOT/venv-wir
export WILL_IT_RISCV_RUNS=$ROOT/runs HF_HOME=$ROOT/hf
export PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-1}
if [ -n "${CARDS:-}" ]; then N=$CARDS; TAG=all; else
  export HABANA_VISIBLE_MODULES=${CARD:-0}; N=1; TAG=card${CARD:-0}; fi
# Compiled graphs kept on disk, one cache a card: a resumed run, or the next seed
# on that card, compiles nothing again.
mkdir -p "$WILL_IT_RISCV_RUNS/recipes-$TAG"
export PT_HPU_RECIPE_CACHE_CONFIG=${PT_HPU_RECIPE_CACHE_CONFIG:-$WILL_IT_RISCV_RUNS/recipes-$TAG,false,32768}
for try in $(seq 1 "$TRIES"); do
  echo "=== attempt $try $(date '+%F %T'): $N card(s), $TAG, lazy mode $PT_HPU_LAZY_MODE"
  # --standalone picks a free port, so two runs side by side don't collide.
  "$VENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node="$N" \
    training/train.py "$CONFIG" < /dev/null
  code=$?
  echo "=== exit $code $(date '+%F %T')"
  [ $code = 0 ] && exit 0
  sleep 60
done
exit 1
