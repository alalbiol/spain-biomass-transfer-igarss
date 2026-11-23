#!/usr/bin/env bash
# Unattended pipeline: BioMassters-style re-export of ALL Spain chips (resumable) ->
# zero-shot inference -> block-level evaluation and paired comparison vs the old
# median composites. Safe to re-run: every stage skips work that is already done.
#
#   nohup setsid bash run_bm_full_pipeline.sh > context/logs/bm_full_pipeline_$(date +%Y%m%d_%H%M).log 2>&1 &
#
# If Earth Engine quota/rate limits stop the export (exit code 3 after a streak of
# failed chips) it waits WAIT_S seconds and resumes, up to MAX_ATTEMPTS times, then
# evaluates whatever chips are complete. Env overrides: WORKERS, MAX_ATTEMPTS, WAIT_S.
REPO=/home/alalbiol/Programacion/AI4DIS/use_case_1_spain
DATA=/media/HD/Data/agb_spain
PY=/media/HD/alalbiol/envs/uc1_infer/bin/python
export LD_LIBRARY_PATH=/home/alalbiol/miniconda3/envs/sim_case1/lib:${LD_LIBRARY_PATH:-}
export MPLBACKEND=Agg
source /home/alalbiol/miniconda3/etc/profile.d/conda.sh
cd "$REPO" || exit 1
WORKERS=${WORKERS:-3}; MAX_ATTEMPTS=${MAX_ATTEMPTS:-8}; WAIT_S=${WAIT_S:-1800}

echo "=== $(date) pipeline start (workers=$WORKERS) ==="
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  conda activate sim_case1
  python3 export_bm_style_chips.py --all_chips --workers "$WORKERS"
  rc=$?
  conda deactivate
  echo "=== $(date) export attempt $attempt/$MAX_ATTEMPTS finished with exit code $rc ==="
  [ "$rc" -eq 0 ] && break
  [ "$attempt" -lt "$MAX_ATTEMPTS" ] && { echo "waiting ${WAIT_S}s before resuming"; sleep "$WAIT_S"; }
done

$PY - <<'PYEOF'
import pandas as pd
from pathlib import Path
d = Path("/media/HD/Data/agb_spain/validation_bm")
meta = pd.read_csv("/media/HD/Data/agb_spain/validation/metadata.csv")
ok = [c for c in meta.chip_id if all((d / f"{c}_S2_{m:02d}.tif").exists() for m in range(12))]
meta[meta.chip_id.isin(ok)].to_csv(d / "metadata_complete.csv", index=False)
print(f"{len(ok)} of {len(meta)} chips have all 12 re-exported S2 months")
PYEOF
META=$DATA/validation_bm/metadata_complete.csv
N=$(( $(wc -l < "$META") - 1 ))
if [ "$N" -lt 20 ]; then echo "only $N complete chips; skipping evaluation"; exit 1; fi

conda activate sim_case1
python3 export_bm_style_chips.py --variant_out "$DATA/validation_bm_c20" --max_cloud 20
conda deactivate

echo "=== $(date) inference: bm recipe ==="
$PY run_zero_shot.py --data_dir "$DATA/validation_bm" --metafile "$META" --out_dir "$DATA/zero_shot_bm" --workers 6 2>&1 | grep -E "^\[.*\] [0-9]+/[0-9]+ chips.*~0\.0 min|checkpoint missing"
echo "=== $(date) inference: bm recipe, residual-cloud months dropped (c20) ==="
$PY run_zero_shot.py --data_dir "$DATA/validation_bm_c20" --metafile "$META" --out_dir "$DATA/zero_shot_bm_c20" --workers 6 2>&1 | grep -E "^\[.*\] [0-9]+/[0-9]+ chips.*~0\.0 min|checkpoint missing"

echo "=== $(date) RESULT 1: block-level evaluation, bm recipe, $N chips ==="
$PY evaluate_zero_shot_blocks.py --metafile "$META" --pred_dir "$DATA/zero_shot_bm" 2>&1 | grep -v Warning
echo "=== $(date) RESULT 2: paired comparison old vs bm vs bm_c20 on the same $N chips ==="
$PY compare_composites.py old="$DATA/zero_shot/preds" bm="$DATA/zero_shot_bm/preds" bm_c20="$DATA/zero_shot_bm_c20/preds" --subset "$META" 2>&1 | grep -v Warning
echo "=== $(date) pipeline done ==="
