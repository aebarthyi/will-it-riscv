#!/bin/bash
# Drive runs on the Gaudi box over SSH: GAUDI=user@host (GAUDI_PORT, default 22).
# GAUDI_ROOT is where everything lives on the box (its big drive; default the
# home directory): the code, the venv, the HF cache, llama.cpp's converter, the
# runs. Runs stay there; this machine takes logs, and a checkpoint's adapter and
# GGUF when it is exported to be scored, into /scratch/runs as before.
#   training/gaudi.sh push                                   # training/ and the configs' data
#   training/gaudi.sh setup                                  # gaudi-setup.sh: venv, converter, base models
#   training/gaudi.sh check                                  # gaudi_check.py on both cards
#   CARD=0 training/gaudi.sh start configs/gaudi-smoke-4b.yaml   # gaudi-run.sh, detached
#   training/gaudi.sh status                                 # each run's latest line, and the cards
#   training/gaudi.sh pull configs/qwen3-4b-gaudi-s0.yaml    # the run's logs
#   training/gaudi.sh export qwen3-4b-gaudi-s0/checkpoint-318    # GGUF on the box, then both here
#   training/gaudi.sh stop configs/qwen3-4b-gaudi-s0.yaml
set -eu
cd "$(dirname "$0")/.."
: "${GAUDI:?set GAUDI=user@host}"
SSH="ssh -p ${GAUDI_PORT:-22} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60"
# The root as an absolute path on the box ("." is the home directory ssh starts in).
ROOT=$($SSH "$GAUDI" "mkdir -p ${GAUDI_ROOT:-.} && cd ${GAUDI_ROOT:-.} && pwd")
CODE=$ROOT/will-it-riscv; RUNS=$ROOT/runs; ENV="WIR_ROOT=$ROOT"
config() { CONFIG=training/${1#training/}; NAME=$(basename "$CONFIG" .yaml); }
case ${1:-} in
  push)
    $SSH "$GAUDI" "mkdir -p $CODE/training $RUNS"
    rsync -az -e "$SSH" --exclude __pycache__ training/ "$GAUDI:$CODE/training/"
    for data in $(sed -n 's/^data: *//p' training/configs/*gaudi*.yaml | sort -u); do
      $SSH "$GAUDI" "mkdir -p $CODE/$data"
      rsync -az -e "$SSH" "$data/" "$GAUDI:$CODE/$data/"
    done ;;
  setup)
    $SSH "$GAUDI" "$ENV $CODE/training/gaudi-setup.sh" ;;
  check)
    $SSH "$GAUDI" "cd $CODE/training && HF_HOME=$ROOT/hf PT_HPU_LAZY_MODE=1 $ROOT/venv-wir/bin/python \
      -m torch.distributed.run --standalone --nproc_per_node=2 gaudi_check.py" ;;
  start)
    config "$2"
    $SSH "$GAUDI" "cd $CODE && $ENV CARD=${CARD:-0} ${CARDS:+CARDS=$CARDS} setsid nohup \
      training/gaudi-run.sh $CONFIG > $RUNS/$NAME.out 2>&1 < /dev/null & echo started $NAME" ;;
  status)
    $SSH "$GAUDI" "for f in $RUNS/*.out; do echo \"== \$f\"; grep -E '^\{|===|Error' \"\$f\" | tail -2; done;
      df -h $ROOT | tail -1; hl-smi | head -20" ;;
  pull)
    config "$2"
    mkdir -p "/scratch/runs/$NAME"
    rsync -az -e "$SSH" "$GAUDI:$RUNS/$NAME.out" /scratch/runs/
    rsync -az -e "$SSH" --include '*.json' --include '*.jsonl' --exclude '*' \
      "$GAUDI:$RUNS/$NAME/" "/scratch/runs/$NAME/" ;;
  export)
    # RUN/checkpoint-N: export.py writes the Q8_0 GGUF beside the adapter on the
    # box (once), then the adapter and the GGUF come here; no optimizer state.
    ck=${2%/}
    $SSH "$GAUDI" "ls $RUNS/$ck/*.gguf > /dev/null 2>&1 || (cd $CODE && HF_HOME=$ROOT/hf \
      LLAMA_CPP=$ROOT/llama.cpp $ROOT/venv-wir/bin/python training/export.py $RUNS/$ck)"
    mkdir -p "/scratch/runs/$ck"
    rsync -az --info=progress2 -e "$SSH" --exclude state.pt "$GAUDI:$RUNS/$ck/" "/scratch/runs/$ck/" ;;
  stop)
    config "$2"
    # gaudi-run.sh first, or it relaunches; then the config's trainers. The [x]
    # keeps pkill's pattern from matching the remote shell running pkill.
    p=${CONFIG#training/}
    $SSH "$GAUDI" "pkill -f '[g]audi-run.sh .*$p'; sleep 2; pkill -f '[t]rain.py .*$p'; echo stopped $NAME" ;;
  *)
    echo "push | setup | check | start CONFIG | status | pull CONFIG | export RUN/checkpoint-N | stop CONFIG" >&2
    exit 2 ;;
esac
