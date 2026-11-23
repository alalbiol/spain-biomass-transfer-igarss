#!/usr/bin/env bash
# Overnight follow-up to run_baselines.sh (2026-11-19). A fixed queue, no branching,
# so the result covers every outcome of the random-init baseline:
#   1-2  random init on 25% / 10% of the training plots (low-data claim; same subsets,
#        folds and seeds as the existing BioMassters frac runs; baseline recipe: 30 ep, lr 1e-3)
#   3    ImageNet init with enc_lr_mult 1.0 (the first ImageNet run used 0.1 - not comparable
#        to random init, which used 1.0)
#   4-5  ImageNet (enc 1.0) on 25% / 10%
#   6    BioMassters (Finland) init with the SAME recipe as the baselines (30 ep, lr 1e-3,
#        enc 1.0), so one comparison differs only in the starting weights
#   7    paired province-bootstrap comparison of everything -> generated/ifn_overnight.{json,md}
# Each run is independent: a failure is logged and the queue moves on.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/home/alalbiol/miniconda3/envs/sim_case1/lib
PY="/media/HD/alalbiol/envs/uc1_infer/bin/python -W ignore"
LOG=/media/HD/Data/agb_spain_ifn/logs
SSD_CACHE=/home/alalbiol/Data/agb_spain_ifn/cache_112
[ -f "$SSD_CACHE/ids.csv" ] && export IFN_CACHE="$SSD_CACHE"
STATUS=$LOG/overnight_status.txt
note() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

# Wait for run_baselines.sh (random-init full-data run + its comparison) to finish.
WAIT_PID=${WAIT_PID:-}
if [ -n "$WAIT_PID" ]; then
  note "waiting for PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
note "queue start; chip cache ${IFN_CACHE:-HDD}"

BASE="--model efficientnet_early --regime full --epochs 30 --lr 1e-3 --no_save_weights"
run() {  # run <logname> <args...>
  local name=$1; shift
  note "START $name: $*"
  if $PY finetune_ifn.py $BASE "$@" > "$LOG/$name.log" 2>&1; then
    note "DONE  $name: $(tail -1 "$LOG/$name.log")"
  else
    note "FAIL  $name (exit $?): $(tail -3 "$LOG/$name.log" | tr '\n' ' ')"
  fi
}

run scratch_random_frac0.25   --init random   --enc_lr_mult 1.0 --train_frac 0.25
run scratch_random_frac0.1    --init random   --enc_lr_mult 1.0 --train_frac 0.1
run scratch_imagenet_enc1     --init imagenet --enc_lr_mult 1.0 --tag _enc1
run scratch_imagenet_enc1_frac0.25 --init imagenet --enc_lr_mult 1.0 --train_frac 0.25 --tag _enc1
run scratch_imagenet_enc1_frac0.1  --init imagenet --enc_lr_mult 1.0 --train_frac 0.1  --tag _enc1
run ft_effnet_early_full_samerecipe --init finland --enc_lr_mult 1.0 --tag _samerecipe

note "comparison"
$PY /home/alalbiol/Programacion/AI4DIS/papers/paper1-case1-IGARSS/analysis/compare_ifn_overnight.py \
  > "$LOG/compare_overnight.log" 2>&1 && note "comparison DONE" || note "comparison FAIL"
note "ALL DONE"
