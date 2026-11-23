# Data

All files below are in the EOTDL snapshot under `data/`, not on GitHub. Biomass is **Mg of aboveground dry biomass per hectare** throughout. `scripts/build_ifn_data.py` and `scripts/build_cci_data.py` record how each file was derived from the original experiment folders, and `data/*/build_record.json` lists the source hashes.

## `data/cci/`: comparison with ESA CCI Biomass and the Finnish benchmark

Every CCI result is computed on 100-m blocks. Each 256 × 256 prediction at 10 m was averaged into a 25 × 25 grid of 100-m blocks (the last 6 pixels in each direction are dropped). The `.npz` files store these blocks, not the 10-m maps.

| File | Contents |
|---|---|
| `spain/chips.csv` | The 500 Spanish chips (2021): `chip_id`, centre `lon`/`lat`, CCI chip mean `agb_mean`, `forest_fraction`, CORINE 2018 `forest_type`, CCI biomass bin `agb_bin` (low < 50, medium 50–120, high ≥ 120), `agb_year`, `grid_size_m` (2560) |
| `spain/cci_100m.npz` | `chip_id`, `agb` and `sd` (500 × 25 × 25, float64; NaN where CCI has no value): ESA CCI Biomass 2021 and its standard deviation, sampled at each block centre |
| `spain/blocks_median.npz` | Recipe M (pixelwise monthly median, Sen2Cor cloud probability 0–100) |
| `spain/blocks_best.npz` | Recipe B (least-cloudy scene per month, s2cloudless probability 0–255, as in BioMassters) |
| `spain/blocks_best_c20.npz` | Recipe B with the 234 chip-months whose chosen scene exceeds 20% cloud probability marked missing |
| `spain/best_scene_months.csv` | Recipe B choice per chip and month: `n_scenes` available, `chosen_cld` (mean cloud probability of the chosen scene, %), `kept_in_c20` |
| `finland_control/chips.csv`, `cci_100m.npz`, `blocks_median.npz`, `blocks_best.npz` | The same for the 188 Finnish control chips (2021) |
| `finland_benchmark/pooled_100m.npz` | BioMassters test set (2773 chips): `pred` (10 models × 2773 × 25 × 25) and the pooled lidar-derived `ground_truth`, both float32 |

Each `blocks_*.npz` holds `models` (10 names, `<encoder>_<early|late>`), `chip_id`, and `pred` (models × chips × 25 × 25, float64).

## `data/ifn/`: Spanish National Forest Inventory plots

| File | Contents |
|---|---|
| `plots.csv` | 6117 IFN4 plots where five pine species give at least 80% of the basal area. Labels and plot attributes: `agb_mg_ha` (label), `agb_large`/`agb_small` (trees ≥ 7.5 cm and the small-stem approximation), `ba_total`, `ba_pine`, `pine_ba_share`, `n_trees_measured`, `trees_per_ha`, `h_mean`, `main_pine`, survey `date`, `plot_class`, source `database`, `province_code`, `survey_year`, `image_year`, `chip_id`, and `cross_fold_neighbour` (see below) |
| `zero_shot/plot_preds.csv` | Unadapted predictions of the ten models, averaged over a 25-m-radius disk at each plot centre |
| `cci_at_plots.csv` | CCI biomass at the plot centre, in the CCI year closest to the image year (5885 plots have a value) |
| `finetune/<run>/oof.csv` | Out-of-fold predictions: `chip_id`, `plot_id`, `province_code`, `fold` (provinces grouped into five folds), `main_pine`, `agb_mg_ha`, `pred` |
| `finetune/<run>/args.json`, `history.json` | Run arguments and, for neural runs, inner-validation RMSE per epoch and fold |
| `finetune/efficientnet_early_full/fold{0-4}.safetensors` | Full fine-tuning weights (eight epochs, seed 0), one model per outer fold |
| `finetune/efficientnet_early_head/fold{0-4}.safetensors` | Decoder fine-tuning weights, one model per outer fold |

**Run names.** `efficientnet_early_full` is the eight-epoch full fine-tuning and `efficientnet_early_head` the decoder fine-tuning. `_samerecipe` marks BioMassters initialization with the 30-epoch recipe, and `scratch_random`/`scratch_imagenet` the other two initializations (`_enc1`: encoder at the full learning rate; without it, the lower-rate ImageNet run in Table IV). `_frac0.25`/`_frac0.1` are the fitting subsets and `_seed1`/`_seed2` the extra seeds. `tabular_hgb` and `tabular_rf` are the gradient-boosting and random-forest baselines. The weights are state dicts of the `UnetVFLOW` network in [sentinel-biomass-inference](https://github.com/alalbiol/sentinel-biomass-inference) (`efficientnet`, early fusion), converted to safetensors without changing any value. Other runs did not keep weights.

**Plot locations.** Coordinates are not distributed. `plot_id` and `province_code` identify each plot in the public IFN4 databases, which give its location. The only analysis that used coordinates flags plots whose 1.12-km input context may reach a plot in another fold (`cross_fold_neighbour`, 103 plots in 69 pairs within 1.58 km). It was computed before the coordinates were removed.

**Imagery.** The Sentinel-1/2 inputs are not distributed. They can be exported again with `pipeline/ifn/export_ifn_chips.py` and `pipeline/cci/export_*.py`. Exports re-query Earth Engine and may not be bit-identical.

## Sources and attribution

- **Copernicus Sentinel-1 and Sentinel-2** data (2017–2023), accessed through Google Earth Engine; contains modified Copernicus Sentinel data.
- **ESA CCI Biomass** (Santoro et al., Climate Change Initiative Biomass), via the sat-io community catalogue on Earth Engine.
- **BioMassters** (Nascetti et al., NeurIPS 2023 Datasets and Benchmarks), CC BY 4.0: pooled test labels and the training data of the evaluated models.
- **Spanish National Forest Inventory (IFN4)**, Ministerio para la Transición Ecológica y el Reto Demográfico (MITECO): plot biomass is calculated from the field tree lists with species-specific allometric equations (Ruiz-Peinado et al.).
- **CORINE Land Cover 2018**, Copernicus Land Monitoring Service: forest type used for stratification.

The derived files are released under CC BY 4.0 (LICENSE-DATA). The original sources keep their own terms.
