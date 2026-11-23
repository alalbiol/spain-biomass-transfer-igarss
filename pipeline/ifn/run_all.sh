#!/usr/bin/env bash
# Full IFN fine-tuning experiment, after build_ifn_plot_labels.py, export_ifn_chips.py and fetch_cci_at_plots.py.
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/media/HD/alalbiol/envs/uc1_infer/lib
PY="/media/HD/alalbiol/envs/uc1_infer/bin/python -W ignore"
LOG=/media/HD/Data/agb_spain_ifn/logs
mkdir -p "$LOG"

[ -f /media/HD/Data/agb_spain_ifn/cache_112/ids.csv ] || $PY build_chip_cache.py 2>&1 | tee "$LOG/build_cache.log"
$PY zero_shot_ifn.py 2>&1 | tee "$LOG/zero_shot.log"
$PY finetune_ifn.py --model efficientnet_early --regime full 2>&1 | tee "$LOG/ft_effnet_early_full.log"
$PY finetune_ifn.py --model efficientnet_early --regime head 2>&1 | tee "$LOG/ft_effnet_early_head.log"
for frac in 0.1 0.25; do
  $PY finetune_ifn.py --model efficientnet_early --regime full --train_frac $frac --no_save_weights 2>&1 | tee "$LOG/ft_effnet_early_full_frac$frac.log"
done
$PY evaluate_ifn.py 2>&1 | tee "$LOG/evaluate.log"
