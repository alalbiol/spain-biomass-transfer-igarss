# Reproducing the reported results

`python scripts/reproduce.py` runs the six analyses below and compares every value with `expected/` (relative tolerance 1e-9). It needs only `data/` from the EOTDL snapshot and the packages in `requirements.txt` (Python 3.11; tested with the versions listed there). It takes about a minute on a CPU. Outputs go to `outputs/` (override with `OUTPUT_DIR`; set `DATA_DIR` if `data/` is elsewhere).

| Step | Script | Output |
|---|---|---|
| 1 | `analysis/ifn_adaptation.py` | `ifn_adaptation.json`, and `ifn_plot_predictions.csv` used by steps 2–3 |
| 2 | `analysis/ifn_pretraining.py` | `ifn_pretraining.json`, `ifn_pretraining.md` (readable summary) |
| 3 | `analysis/ifn_baselines.py` | `ifn_baselines.json` |
| 4 | `analysis/cci_blocks.py` | `cci_blocks_<spain|finland_control>_<recipe>.csv` |
| 5 | `analysis/cci_recipes.py` | `cci_recipes.json` |
| 6 | `analysis/finland_benchmark.py` | `finland_benchmark_100m.csv` |

## Where each reported number comes from

"ALL rows" means the rows with `stratum == ALL`. `ens_early`/`ens_late` average the five encoders of one fusion mode; `ens_all` averages all ten models.

### Finnish benchmark and CCI comparison

| Manuscript item | Source |
|---|---|
| Table I (BioMassters test set, native 10 m) | `expected/finland_benchmark_10m.csv`: **not recomputed**, see below |
| Table II (chips per forest type and CCI bin) | `cci_recipes.json` → `table2_strata` |
| Table III (bias, RMSE, R², r for recipes M and B) | `cci_blocks_spain_median.csv`, `cci_blocks_spain_best.csv`: ALL rows of `ens_early`/`ens_late`, columns `bias`, `rmse`, `r2`, `pearson_r`; `n_chips` 499, `n_blocks` 289,600 |
| Within-mode change M → B (late R² +0.092, r +0.055; early −0.033, −0.046) and its intervals | `cci_recipes.json` → `within_mode`, rows with `recipe` `best`: `change_r2`, `change_r` |
| Change in the late-minus-early gap (0.125, [0.099, 0.151]) | `cci_recipes.json` → `gap_change_best_minus_median.r2` |
| Late minus early under B (R² 0.013, r −0.034, with intervals) | `cci_recipes.json` → `late_minus_early.best` |
| Fig. 1 (per-model change, B minus M) | `r2` and `pearson_r` of each model's ALL row: `cci_blocks_spain_best.csv` minus `cci_blocks_spain_median.csv` |
| 234 cloudy chip-months; early/late R² 0.330/0.329 without them | `cci_recipes.json` → `cloudy_chip_months`; `cci_blocks_spain_best_c20.csv` ALL rows of `ens_early`/`ens_late` |
| Coniferous vs broadleaved under B (R² 0.39/0.27, bias −5.2/−23.7) | `cci_blocks_spain_best.csv`: `ens_all`, strata `type=coniferous`, `type=broadleaved` |
| Bias in blocks ≥ 120 Mg/ha (−78.9 early, −74.3 late) | `cci_recipes.json` → `residual_by_cci_level`, recipe `best`, key `>=120` |
| BioMassters pooled to 100 m vs lidar (R² 0.944/0.873, RMSE 13.50/20.35) | `finland_benchmark_100m.csv`, `ensemble` rows |
| Finnish CCI control: bias −13.7 (M) / −8.3 (B), R² 0.56/0.55, r 0.82/0.79 | `cci_blocks_finland_control_<median|best>.csv`, ALL row of `ens_all` (188 chips) |
| Spanish ten-model bias −16.6 (M) / −11.9 (B), r 0.62–0.65 | `cci_blocks_spain_<median|best>.csv`, ALL row of `ens_all` |

### Inventory plots (IFN4)

| Manuscript item | Source |
|---|---|
| Table IV, first block (original, linear, decoder, full eight epochs) | `ifn_adaptation.json` → `cohorts.all` |
| Reduction 4.28 Mg/ha (9.4%), interval 2.03–5.88; 13 of 14 provinces; Fig. 3 | `ifn_adaptation.json` → `cohorts.all`, `paired_rmse_change_ci.full`, `improved_provinces`, `provinces` |
| Three-seed reduction 4.10 (1.95–5.57); 12–13 provinces per seed; one province worse in every seed | `ifn_pretraining.json` → `seeds.orig_vs_bm8` |
| Cáceres (code 10, n = 363, +5.3 Mg/ha) | `ifn_adaptation.json` → `provinces` |
| Year-matched plots (5773): 45.51 and 40.64 | `ifn_adaptation.json` → `cohorts.exact_year` |
| Table IV, 30-epoch block (three-seed means of RMSE, bias, R², r) | `ifn_pretraining.json` → `seeds.summary["1.0"]`: `rmse_mean`, `bias_mean`, `r2_mean`, `r_mean` |
| Table IV, tree ensembles; fine-tuning vs gradient boosting 6.7 (4.1–9.8), 13 of 14 | `ifn_baselines.json` → `overall`, `full_minus_method_rmse.tabular_hgb`, `provinces_full_better` |
| BioMassters vs random 4.18 (2.46–6.02), vs ImageNet 4.02 (1.11–6.02), random minus ImageNet 0.16, eight vs 30 epochs 1.41 (0.27–2.58), and the 10% comparisons | `ifn_pretraining.json` → `seeds.pairs` |
| 25% and 10% mean RMSE; eight-epoch mean 41.55 (41.38–41.79); Fig. 2a; seed ranges (4.8, 1.5) | `ifn_pretraining.json` → `seeds.summary` (`rmse_mean`, `rmse_min`, `rmse_max`) |
| ImageNet encoder at a tenfold lower rate (47.33) | `ifn_pretraining.json` → `fractions["1.0"].efficientnet_early_scratch_imagenet` |
| Error by biomass class (573 plots > 150 Mg/ha: bias −91.9 → −70.8, RMSE 108.9 → 90.6; 1535 plots ≤ 25: 18.4 → 20.8); Fig. 2b | `ifn_baselines.json` → `by_biomass_class`, methods `original` and `full` |
| Leave-one-province-out and context-separated checks (69 cross-fold plot pairs within 1.58 km) | `ifn_adaptation.json` → `leave_one_province_out_gain`, `cross_fold_nearby_pairs_context_envelope`, `context_separated_sensitivity` |

## Notes on the analyses

- **Bootstrap.** CCI intervals resample chips (2000 draws), keeping each chip's blocks together. Inventory intervals resample the 14 provinces (2000 draws) with the saved out-of-fold predictions fixed. Seeds are fixed in each script. The original within-mode CCI intervals used 200 draws; `python analysis/cci_recipes.py --legacy-200` reproduces that run exactly. Both settings give the same conclusion: all four within-mode changes exclude zero.
- **Exactness.** The analyses are ports of the original scripts (in `pipeline/`), reading the packed data instead of the original folders. Their outputs equal the original results files exactly. The only exception is the bootstrap draws in `cci_recipes.py`, which use per-chip sums and differ from the original concatenation by less than 1e-13.

## The 10-m prediction maps

The analyses read the 100-m blocks. `python scripts/check_prediction_maps.py` (a few minutes) pools every 10-m map in `data/cci/*/predictions_10m/` and confirms that it equals the stored blocks exactly, so the maps are the ones that were scored. To recompute a map from imagery, download the chip ([GEE.md](GEE.md)) and run the model with sentinel-biomass-inference; on the original inputs in `data/cci/spain/example_inputs/` this agrees with the stored maps within 0.07 Mg/ha (CPU against the original GPU run).

## Table I: native 10-m benchmark (needs GPU inference)

Table I scores the ten models on the 2773 BioMassters test chips at 10 m. Recomputing it needs:

1. the BioMassters test features and labels (2773 chips) from the NeurIPS 2023 [BioMassters release](https://huggingface.co/datasets/nascetti-a/BioMassters), which includes test labels. The original run read the features converted to one safetensors file per chip (about 122 GB);
2. the ten checkpoints from the [sentinel-biomass-inference EOTDL model](https://www.eotdl.com/models/sentinel-biomass-inference);
3. a GPU (about 12 hours for all models on one GPU).

The original run used `use_case_1/test.py` and `test_ensemble.py` of the AI4DIS project, driven by `pipeline/cci/run_finland_benchmark.sh`. These scripts pool RMSE, MAE and R² over all test pixels (2773 × 256 × 256), and the ensemble averages the five encoders of one fusion mode. Their printed summaries are stored in `expected/finland_benchmark_10m.csv`. The 100-m version of the same benchmark (`finland_benchmark.py`) is recomputed from saved pooled predictions.

## Going further back

`pipeline/` holds the code that produced `data/`, from Earth Engine exports through inference and fine-tuning. It is provided as used, not re-run for this release; see [pipeline/README.md](pipeline/README.md).
