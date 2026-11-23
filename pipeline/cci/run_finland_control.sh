#!/usr/bin/env bash
# Finland control for the Spain zero-shot gap (IGARSS paper, label-offset question).
#
# BioMassters chips carry no georeferencing, so their ALS-derived labels cannot be
# overlaid on ESA CCI directly. Instead: sample Finnish forest chips with the SAME
# exporter used for Spain (CORINE forest mask, CCI 2021 >= 5 Mg/ha, stratified),
# run the same 10 checkpoints, and score against CCI with the same block-level
# evaluation. The model is in-domain here, so any residual vs CCI (in particular
# the high-biomass underestimation) that also appears in Finland is a model/label
# vs CCI offset, not Spain domain shift.
#
#   nohup setsid bash run_finland_control.sh > context/logs/finland_control_$(date +%Y%m%d_%H%M).log 2>&1 &
#
# Safe to re-run: every stage skips work already done.
REPO=/home/alalbiol/Programacion/AI4DIS/use_case_1_spain
DATA=/media/HD/Data/agb_finland_control
PY=/media/HD/alalbiol/envs/uc1_infer/bin/python
export LD_LIBRARY_PATH=/home/alalbiol/miniconda3/envs/sim_case1/lib:${LD_LIBRARY_PATH:-}
export MPLBACKEND=Agg
source /home/alalbiol/miniconda3/etc/profile.d/conda.sh
cd "$REPO" || exit 1
N_TARGET=${N_TARGET:-200}; WORKERS=${WORKERS:-4}; MAX_ATTEMPTS=${MAX_ATTEMPTS:-6}; WAIT_S=${WAIT_S:-1200}

echo "=== $(date) Finland control start (n_target=$N_TARGET) ==="
conda activate sim_case1
if [ ! -f "$DATA/validation/metadata.csv" ]; then
  # oversample hard: high-AGB (CCI >= 120) chips are rare in Finland
  python3 export_validation_set.py --country Finland --n_target "$N_TARGET" --oversample_factor 12 \
    --out_dir "$DATA/validation" --workers "$WORKERS"
fi
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  python3 export_bm_style_chips.py --all_chips --data_dir "$DATA/validation" --out_dir "$DATA/validation_bm" --workers "$WORKERS"
  rc=$?
  echo "=== $(date) bm export attempt $attempt/$MAX_ATTEMPTS exit $rc ==="
  [ "$rc" -eq 0 ] && break
  [ "$attempt" -lt "$MAX_ATTEMPTS" ] && sleep "$WAIT_S"
done
python3 fetch_cci_100m.py --data_dir "$DATA/validation" --out_dir "$DATA/cci_100m" --workers "$WORKERS"
conda deactivate

# Chips usable by both recipes: all 12 S1 months present, >= 6 S2 months in each
# recipe (high-latitude winter months can legitimately have no scene), CCI fetched.
$PY - <<PYEOF
import pandas as pd
from pathlib import Path
D = Path("$DATA")
meta = pd.read_csv(D / "validation/metadata.csv").drop_duplicates("chip_id")
def ok(c):
    s1 = all((D / "validation" / f"{c}_S1_{m:02d}.tif").exists() for m in range(12))
    s2 = [sum((D / v / f"{c}_S2_{m:02d}.tif").exists() for m in range(12)) for v in ("validation", "validation_bm")]
    return s1 and min(s2) >= 6 and (D / "cci_100m" / f"{c}.tif").exists()
good = meta[meta.chip_id.map(ok)]
good.to_csv(D / "metadata_complete.csv", index=False)
print(f"{len(good)} of {len(meta)} chips usable by both recipes")
PYEOF
META=$DATA/metadata_complete.csv

for v in validation:zero_shot validation_bm:zero_shot_bm; do
  src=${v%%:*}; dst=${v##*:}
  echo "=== $(date) inference: $src ==="
  $PY run_zero_shot.py --data_dir "$DATA/$src" --metafile "$META" --out_dir "$DATA/$dst" --workers 6 2>&1 \
    | grep -E "^\[.*\] [0-9]+/[0-9]+ chips.*~0\.0 min|checkpoint missing"
  echo "=== $(date) RESULT: block-level evaluation vs CCI, Finland, $src ==="
  $PY evaluate_zero_shot_blocks.py --metafile "$META" --pred_dir "$DATA/$dst" --cci_dir "$DATA/cci_100m" 2>&1 | grep -v Warning
done
echo "=== $(date) Finland control done ==="
