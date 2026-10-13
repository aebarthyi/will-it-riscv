#!/bin/bash
# Drive runs on the Gaudi box over SSH: GAUDI=user@host (GAUDI_PORT, default 22).
# The box keeps its copy in ~/will-it-riscv and its runs in ~/runs; pulled runs
# land in /scratch/runs here, as every other run does.
#   training/gaudi.sh push                                   # training/ and the configs' data
#   training/gaudi.sh setup                                  # gaudi-setup.sh: the venv, the base models
#   training/gaudi.sh check                                  # gaudi_check.py on both cards
#   CARD=0 training/gaudi.sh start configs/gaudi-smoke-4b.yaml   # gaudi-run.sh, detached
#   training/gaudi.sh status                                 # each run's latest line, and the cards
#   training/gaudi.sh pull configs/qwen3-4b-gaudi-s0.yaml    # adapters and logs, no optimizer state
#   training/gaudi.sh stop configs/qwen3-4b-gaudi-s0.yaml
set -eu
cd "$(dirname "$0")/.."
: "${GAUDI:?set GAUDI=user@host}"
SSH="ssh -p ${GAUDI_PORT:-22} -o StrictHostKeyChecking=accept-new"
REMOTE=will-it-riscv; RUNS=runs; VENV='~/venv-wir'
config() { CONFIG=training/${1#training/}; NAME=$(basename "$CONFIG" .yaml); }
case ${1:-} in
  push)
    $SSH "$GAUDI" "mkdir -p $REMOTE/training $RUNS"
    rsync -az -e "$SSH" --exclude __pycache__ training/ "$GAUDI:$REMOTE/training/"
    for data in $(sed -n 's/^data: *//p' training/configs/*gaudi*.yaml | sort -u); do
      $SSH "$GAUDI" "mkdir -p $REMOTE/$data"
      rsync -az -e "$SSH" "$data/" "$GAUDI:$REMOTE/$data/"
    done ;;
  setup)
    $SSH "$GAUDI" "$REMOTE/training/gaudi-setup.sh" ;;
  check)
    $SSH "$GAUDI" "cd $REMOTE/training && PT_HPU_LAZY_MODE=1 $VENV/bin/python \
      -m torch.distributed.run --standalone --nproc_per_node=2 gaudi_check.py" ;;
  start)
    config "$2"
    $SSH "$GAUDI" "cd $REMOTE && CARD=${CARD:-0} ${CARDS:+CARDS=$CARDS} setsid nohup \
      training/gaudi-run.sh $CONFIG > ~/$RUNS/$NAME.out 2>&1 < /dev/null & echo started $NAME" ;;
  status)
    $SSH "$GAUDI" "for f in $RUNS/*.out; do echo \"== \$f\"; grep -E '^\{|===|Error' \"\$f\" | tail -2; done;
      hl-smi | head -20" ;;
  pull)
    config "$2"
    mkdir -p "/scratch/runs/$NAME"
    rsync -az -e "$SSH" "$GAUDI:$RUNS/$NAME.out" /scratch/runs/
    rsync -az -e "$SSH" --exclude state.pt --exclude dcp "$GAUDI:$RUNS/$NAME/" "/scratch/runs/$NAME/" ;;
  stop)
    config "$2"
    # gaudi-run.sh first, or it relaunches; then the config's trainers. The [x]
    # keeps pkill's pattern from matching the remote shell running pkill.
    p=${CONFIG#training/}
    $SSH "$GAUDI" "pkill -f '[g]audi-run.sh .*$p'; sleep 2; pkill -f '[t]rain.py .*$p'; echo stopped $NAME" ;;
  *)
    echo "push | setup | check | start CONFIG | status | pull CONFIG | stop CONFIG" >&2; exit 2 ;;
esac
