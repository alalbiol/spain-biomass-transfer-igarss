#!/usr/bin/env bash
# M1: run all 5 backbones x {early,late} plus the 5-model ensemble x {early,late}
# on the real BioMassters test set (2773 chips), using the project's own
# test.py / test_ensemble.py (unmodified) so numbers match established
# convention. Neither script saves output anywhere -- this captures each run's
# full stdout to its own log and parses the final metrics into one CSV.
#
#   nohup setsid bash run_finland_benchmark.sh > context_log 2>&1 &
#
# At ~0.8s/chip wall-clock (measured on a 30-chip timing run) x 2773 chips,
# each single-backbone run is ~35-40 min; ensemble runs cost ~5x that (5
# models evaluated per chip) per mode. Total estimated ~12-13h for all 12 runs.
set -uo pipefail
REPO=/home/alalbiol/Programacion/AI4DIS/use_case_1
PY=/media/HD/alalbiol/envs/uc1_infer/bin/python
export LD_LIBRARY_PATH=/home/alalbiol/miniconda3/envs/sim_case1/lib:${LD_LIBRARY_PATH:-}
export TEST_LABEL_FOLDER=/media/HD/Data/BioMassters/test_agbm
export TEST_FEATURES_FOLDER=/media/HD/Data/BioMassters/test_features
export TEST_METAFILE_PATH=/media/HD/Data/BioMassters/test_meta_chip_ids.csv
export MODEL_DIR=/media/HD/checkpoints/AI4DIS/case1
export OMP_NUM_THREADS=8  # 8 CPUs available; the DataModule default of 31 workers over-subscribes and warns
cd "$REPO" || exit 1

OUT=context_logs/finland_benchmark
mkdir -p "$OUT"
RESULTS_CSV="$OUT/finland_benchmark_results.csv"
echo "model,mode,rmse,mae,r2,sci,frac_over,frac_under,uncertainty_ratio,n_chips" > "$RESULTS_CSV"

BACKBONES="convnextv2 efficientnet efficientnetv2 mobilenetv4 resnetrs101"

for mode in early late; do
  for bb in $BACKBONES; do
    log="$OUT/test_${bb}_${mode}.log"
    if [ -f "$log" ] && grep -q "TEST SUMMARY" "$log"; then
      echo "=== $(date) skip $bb/$mode (already done) ==="
    else
      echo "=== $(date) running $bb/$mode ==="
      $PY test.py --mode "$mode" --backbone "$bb" > "$log" 2>&1
    fi
    line=$(grep "TEST SUMMARY" "$log")
    rmse=$(echo "$line" | grep -oP 'RMSE=\K[0-9.]+')
    mae=$(echo "$line" | grep -oP 'MAE=\K[0-9.]+')
    r2=$(echo "$line" | grep -oP 'R2=\K[0-9.]+')
    over=$(echo "$line" | grep -oP 'sobreestimado\(>2x\)=\K[0-9.]+')
    under=$(echo "$line" | grep -oP 'subestimado\(<0.500x\)=\K[0-9.]+')
    npix=$(echo "$line" | grep -oP 'n_pixels=\K[0-9]+')
    sci=$(grep "Spatial Consistency" "$log" | grep -oP '= \K[0-9.]+')
    echo "$bb,$mode,$rmse,$mae,$r2,$sci,$over,$under,,$((npix/65536))" >> "$RESULTS_CSV"
  done

  log="$OUT/test_ensemble_${mode}.log"
  if [ -f "$log" ] && grep -q "ENSEMBLE SUMMARY" "$log"; then
    echo "=== $(date) skip ensemble/$mode (already done) ==="
  else
    echo "=== $(date) running ensemble/$mode ==="
    $PY test_ensemble.py --mode "$mode" > "$log" 2>&1
  fi
  line=$(grep "RMSE(mean ensemble)" "$log")
  rmse=$(echo "$line" | grep -oP 'RMSE\(mean ensemble\)=\K[0-9.]+')
  mae=$(echo "$line" | grep -oP 'MAE=\K[0-9.]+')
  r2=$(echo "$line" | grep -oP 'R2=\K[0-9.]+')
  unc=$(grep "Prediction uncertainty" "$log" | grep -oP '[0-9.]+(?=% \(esperado)')
  npix_line=$(grep "ENSEMBLE SUMMARY" "$log")
  npix=$(echo "$npix_line" | grep -oP 'n_pixels=\K[0-9]+')
  echo "ensemble,$mode,$rmse,$mae,$r2,,,,${unc:-},$((npix/65536))" >> "$RESULTS_CSV"
done

echo "=== $(date) ALL DONE. Results: $RESULTS_CSV ==="
column -s, -t "$RESULTS_CSV"
