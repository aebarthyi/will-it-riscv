#!/bin/bash
# Put both GPUs into training mode, or give them back to the desktop. Run as root.
#   sudo training/gpu-mode.sh           # stop GDM, fans manual at 70%
#   sudo training/gpu-mode.sh desktop   # fans back on auto, start GDM
# GDM's gnome-shell renders on card 0 and hung it mid-run; the auto fan curve
# holds a 93 °C hotspot. A GPU reset puts that card's fan back on auto, so
# run this again after one (run.sh refuses to start while a fan is on auto).
set -e
MODE=${1:-train}; PERCENT=${PERCENT:-70}
[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0 $*"; exit 1; }
# Whichever of the two cards are on the bus: b5 went missing on 2026-10-08.
shopt -s nullglob
CARDS=(/sys/bus/pci/devices/0000:{67,b5}:00.0/hwmon/hwmon*)
[ ${#CARDS[@]} -gt 0 ] || { echo "no GPU found"; exit 1; }
case $MODE in
  train)
    systemctl stop gdm
    pwm=$(( 255 * PERCENT / 100 ))
    for h in "${CARDS[@]}"; do echo 1 > $h/pwm1_enable; echo $pwm > $h/pwm1; done ;;
  desktop)
    for h in "${CARDS[@]}"; do echo 2 > $h/pwm1_enable; done
    systemctl start gdm ;;
  *) echo "usage: $0 [train|desktop]"; exit 2 ;;
esac
echo "gdm: $(systemctl is-active gdm)"
for h in "${CARDS[@]}"; do
  echo "$h fan mode $(cat $h/pwm1_enable) pwm $(cat $h/pwm1) ($(cat $h/fan1_input) RPM)" \
       "edge $(( $(cat $h/temp1_input)/1000 ))°C hotspot $(( $(cat $h/temp2_input)/1000 ))°C"
done
