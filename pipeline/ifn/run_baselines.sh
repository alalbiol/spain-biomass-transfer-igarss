#!/usr/bin/env bash
# M4 no-pretraining baselines for the IGARSS IFN experiment (needs a working GPU).
# Same folds, splits, crops, loss and epoch selection as the paper's fine-tuning;
# only the starting weights differ. Hyperparameters fixed before running:
#   fresh decoder/attention/head need a larger lr (1e-3, the BioMassters training lr)
#   and more epochs (30) than the 8-epoch fine-tuning of an already-trained model.
# ~90 s/epoch on the previous GPU -> ~4 h per run.
set -euo pipefail
cd "$(dirname "$0")"
# The venv sits on the sim_case1 conda env; its libstdc++ is needed by scipy (via timm/lightning).
export LD_LIBRARY_PATH=/home/alalbiol/miniconda3/envs/sim_case1/lib
PY="/media/HD/alalbiol/envs/uc1_infer/bin/python -W ignore"
LOG=/media/HD/Data/agb_spain_ifn/logs
# Read the chip cache from the SSD copy when present (the HDD copy starves the GPU).
SSD_CACHE=/home/alalbiol/Data/agb_spain_ifn/cache_112
[ -f "$SSD_CACHE/ids.csv" ] && export IFN_CACHE="$SSD_CACHE"
echo "chip cache: ${IFN_CACHE:-/media/HD/Data/agb_spain_ifn/cache_112}"
mkdir -p "$LOG"

# CPU, minutes: tree ensembles on plot-level S1/S2 statistics.
[ -f /media/HD/Data/agb_spain_ifn/finetune/tabular_hgb/oof.csv ] || \
  $PY baseline_tabular_ifn.py 2>&1 | tee "$LOG/baseline_tabular.log"

# Main deep baseline: ImageNet encoder (generic pretraining), no BioMassters weights.
$PY finetune_ifn.py --model efficientnet_early --regime full --init imagenet \
  --epochs 30 --lr 1e-3 --enc_lr_mult 0.1 --no_save_weights 2>&1 | tee "$LOG/scratch_imagenet.log"

# Optional: fully random initialisation.
if [ "${WITH_RANDOM:-0}" = 1 ]; then
  $PY finetune_ifn.py --model efficientnet_early --regime full --init random \
    --epochs 30 --lr 1e-3 --enc_lr_mult 1.0 --no_save_weights 2>&1 | tee "$LOG/scratch_random.log"
fi

# Paper-side comparison with paired province-bootstrap intervals.
$PY /home/alalbiol/Programacion/AI4DIS/papers/paper1-case1-IGARSS/analysis/compare_ifn_baselines.py
