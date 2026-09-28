#!/bin/bash
# Run a command; kill it if either card goes over the limits. Temps go to $TEMPS.
#   training/watchdog.sh torchrun --nproc_per_node=2 training/train.py <config>
EDGE_MAX=${EDGE_MAX:-85}; HOT_MAX=${HOT_MAX:-95}; TEMPS=${TEMPS:-/scratch/runs/temps.log}
SENSORS=(/sys/bus/pci/devices/0000:67:00.0/hwmon/hwmon* /sys/bus/pci/devices/0000:b5:00.0/hwmon/hwmon*)
setsid "$@" &
P=$!
trap 'kill -- -$P 2>/dev/null' INT TERM
while kill -0 $P 2>/dev/null; do
  line="$(date +%T)"; over=0
  for i in 0 1; do
    e=$(( $(cat ${SENSORS[$i]}/temp1_input)/1000 )); h=$(( $(cat ${SENSORS[$i]}/temp2_input)/1000 ))
    p=$(( $(cat ${SENSORS[$i]}/power1_input)/1000000 )); f=$(cat ${SENSORS[$i]}/fan1_input)
    line="$line gpu$i edge=$e hot=$h ${p}W ${f}rpm"
    if [ $e -gt $EDGE_MAX ] || [ $h -gt $HOT_MAX ]; then over=1; fi
  done
  echo "$line" >> $TEMPS
  if [ $over = 1 ]; then echo "WATCHDOG KILL $line" | tee -a $TEMPS; kill -- -$P; sleep 10; kill -9 -- -$P 2>/dev/null; exit 2; fi
  sleep 5
done
wait $P
