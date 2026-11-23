#!/usr/bin/env bash
# Weekend queue for the IGARSS IFN pretraining ablation (2026-11-20), ~37 h on the RTX 3060.
# Addresses two reviewer points on the BioMassters-vs-random-vs-ImageNet comparison:
#   A  (issue 1) BioMassters with the SAME 30-epoch recipe as the controls on the 25% / 10%
#      subsets, so every point of Fig. 2a differs only in the starting weights.       ~2 h
#   B  (issue 5) seeds 1-2 of the full-data 30-epoch comparison (BioMassters, random,
#      ImageNet enc 1.0). Seed 0 = the existing runs.                                  ~23 h
#   C  seeds 1-2 of the three inits on the 25% / 10% subsets.                          ~10 h
#   D  seeds 1-2 of the paper's 8-epoch fine-tuning (Table IV main row).               ~2 h
# --seed changes init/augmentation/batch order only; folds, inner-val plots and train_frac
# subsets are fixed, so seeds measure training variance on identical data.
# Each run is independent: a failure is logged and the queue moves on.
# Optional START_AT="18:30" delays the start (date -d syntax).
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/home/alalbiol/miniconda3/envs/sim_case1/lib
PY="/media/HD/alalbiol/envs/uc1_infer/bin/python -W ignore"
LOG=/media/HD/Data/agb_spain_ifn/logs
SSD_CACHE=/home/alalbiol/Data/agb_spain_ifn/cache_112
[ -f "$SSD_CACHE/ids.csv" ] && export IFN_CACHE="$SSD_CACHE"
STATUS=$LOG/weekend_status.txt
note() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

if [ -n "${START_AT:-}" ]; then
  target=$(date -d "$START_AT" +%s)
  note "waiting until $(date -d @"$target" '+%F %T')"
  while [ "$(date +%s)" -lt "$target" ]; do sleep 60; done
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
BM="--init finland --enc_lr_mult 1.0 --tag _samerecipe"
RND="--init random --enc_lr_mult 1.0"
IMN="--init imagenet --enc_lr_mult 1.0 --tag _enc1"
PAPER="--init finland --epochs 8 --lr 1e-4 --enc_lr_mult 0.1"   # overrides BASE

# A: matched BioMassters points on the subsets
run ft_samerecipe_frac0.25 $BM --train_frac 0.25
run ft_samerecipe_frac0.1  $BM --train_frac 0.1

# B: full-data seeds, grouped by seed so a stopped queue still leaves complete seed sets
for s in 1 2; do
  run ft_samerecipe_seed$s  $BM  --seed $s
  run scratch_random_seed$s $RND --seed $s
  run scratch_imagenet_enc1_seed$s $IMN --seed $s
done

# C: subset seeds
for s in 1 2; do
  for f in 0.25 0.1; do
    run ft_samerecipe_frac${f}_seed$s  $BM  --train_frac $f --seed $s
    run scratch_random_frac${f}_seed$s $RND --train_frac $f --seed $s
    run scratch_imagenet_enc1_frac${f}_seed$s $IMN --train_frac $f --seed $s
  done
done

# D: the paper's 8-epoch fine-tuning
for s in 1 2; do
  run ft_effnet_early_full_seed$s $PAPER --seed $s
done

note "ALL DONE"
