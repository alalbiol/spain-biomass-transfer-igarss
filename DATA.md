# Data

All files below are in the EOTDL snapshot under `data/` (about 5 GB), not on GitHub. Biomass is **Mg of aboveground dry biomass per hectare** throughout. `scripts/build_ifn_data.py`, `build_cci_data.py`, `build_prediction_maps.py` and `build_example_inputs.py` record how each file was derived from the original experiment folders, and `data/*/build_record.json` lists the source hashes.

## `data/cci/`: comparison with ESA CCI Biomass and the Finnish benchmark

Every CCI result is computed on 100-m blocks: each 256 × 256 prediction at 10 m is averaged into a 25 × 25 grid of 100-m blocks, dropping the last 6 pixels in each direction. Both the 10-m maps and the blocks are provided; `scripts/check_prediction_maps.py` confirms that the first pools exactly to the second.

| File | Contents |
|---|---|
| `spain/chips.csv` | The 500 Spanish chips (2021): `chip_id`, centre `lon`/`lat` (WGS84), CCI chip mean `agb_mean`, `forest_fraction`, CORINE 2018 `forest_type`, CCI biomass bin `agb_bin` (low < 50, medium 50–120, high ≥ 120), `agb_year`, `grid_size_m` (2560), and `left_3035`/`top_3035`: upper-left corner of the 10-m grid in EPSG:3035 |
| `spain/predictions_10m/<recipe>/<chip_id>.tif` | 10-m predictions of the ten models: GeoTIFF, EPSG:3035, 256 × 256, 10 float32 bands named `<encoder>_<early|late>` in the band descriptions. Recipes `median` (M), `best` (B) and `best_c20` (B without the 234 chip-months above 20% cloud) |
| `spain/blocks_<recipe>.npz` | The same predictions pooled to 100 m: `models` (10 names), `chip_id`, `pred` (models × chips × 25 × 25, float64) |
| `spain/cci_100m.npz` | `chip_id`, `agb` and `sd` (500 × 25 × 25, float64; NaN where CCI has no value): ESA CCI Biomass 2021 and its standard deviation, sampled at each block centre |
| `spain/best_scene_months.csv` | Recipe B per chip and month (index 0–11): `n_scenes` covering the whole chip, `chosen_cld` (chip-mean cloud probability of the chosen scene, %), `scene` (Earth Engine `system:index` in `COPERNICUS/S2_SR_HARMONIZED`), `kept_in_c20` |
| `spain/example_inputs/<recipe>/` | Original Sentinel-1/2 inputs of chips `ES_2021_40p7800_W4p0495` (coniferous) and `ES_2021_42p7594_W2p5491` (broadleaved), recipes `median` and `best`, as exported in the study ([GEE.md](GEE.md) describes the bands) |
| `finland_control/` | The same for the 188 Finnish control chips (2021), recipes `median` and `best`; months without a full-coverage scene have an empty `scene` |
| `finland_benchmark/pooled_100m.npz` | BioMassters test set (2773 chips): `pred` (10 models × 2773 × 25 × 25) and the pooled lidar-derived `ground_truth`, both float32 |

## `data/ifn/`: Spanish National Forest Inventory plots

| File | Contents |
|---|---|
| `plots.csv` | 6117 IFN4 plots where five pine species give at least 80% of the basal area. Label `agb_mg_ha` = `agb_large` (trees ≥ 7.5 cm) + `agb_small` (small-stem approximation); `ba_total`, `ba_pine`, `pine_ba_share`, `n_trees_measured`, `trees_per_ha`, `h_mean`, `main_pine`, survey `date`, `plot_class`, source `database`, `province_code`, `survey_year`, `image_year`, `chip_id`. Location: IFN4 `x`/`y` in ETRS89 / UTM zone `huso` (29–31), and `lon`/`lat` (WGS84) |
| `zero_shot/plot_preds.csv` | Unadapted predictions of the ten models, averaged over a 25-m-radius disk at each plot centre |
| `cci_at_plots.csv` | CCI biomass at the plot centre, in the CCI year closest to the image year (5885 plots have a value) |
| `finetune/<run>/oof.csv` | Out-of-fold predictions: `chip_id`, `plot_id`, `province_code`, `fold` (provinces grouped into five folds), `main_pine`, `agb_mg_ha`, `pred` |
| `finetune/<run>/args.json`, `history.json` | Run arguments and, for neural runs, inner-validation RMSE per epoch and fold |
| `finetune/efficientnet_early_full/fold{0-4}.safetensors` | Full fine-tuning weights (eight epochs, seed 0), one model per outer fold |
| `finetune/efficientnet_early_head/fold{0-4}.safetensors` | Decoder fine-tuning weights, one model per outer fold |

**Run names.** `efficientnet_early_full` is the eight-epoch full fine-tuning and `efficientnet_early_head` the decoder fine-tuning. `_samerecipe` marks BioMassters initialization with the 30-epoch recipe, and `scratch_random`/`scratch_imagenet` the other two initializations (`_enc1`: encoder at the full learning rate; without it, the lower-rate ImageNet run in the text). `_frac0.25`/`_frac0.1` are the fitting subsets and `_seed1`/`_seed2` the extra seeds. `tabular_hgb` and `tabular_rf` are the gradient-boosting and random-forest baselines. The weights are state dicts of the `UnetVFLOW` network in [sentinel-biomass-inference](https://github.com/alalbiol/sentinel-biomass-inference) (`efficientnet`, early fusion), converted to safetensors without changing any value. Other runs did not keep weights.

**Imagery.** The Sentinel-1/2 inputs are not distributed, except for the two example chips. `scripts/download_gee.py` downloads them from Earth Engine ([GEE.md](GEE.md)).

## Sources and attribution

- **Copernicus Sentinel-1 and Sentinel-2** data (2017–2023), accessed through Google Earth Engine; contains modified Copernicus Sentinel data.
- **ESA CCI Biomass** (Santoro et al., Climate Change Initiative Biomass), via the sat-io community catalogue on Earth Engine.
- **BioMassters** (Nascetti et al., NeurIPS 2023 Datasets and Benchmarks), CC BY 4.0: pooled test labels and the training data of the evaluated models.
- **Spanish National Forest Inventory (IFN4)**, Ministerio para la Transición Ecológica y el Reto Demográfico (MITECO): plot locations and the tree lists from which plot biomass is calculated with species-specific allometric equations (Ruiz-Peinado et al.).
- **CORINE Land Cover 2018**, Copernicus Land Monitoring Service: forest type used for stratification.
- **Natural Earth** (public domain): basemaps of the README figures.

The derived files are released under CC BY 4.0 (LICENSE-DATA). The original sources keep their own terms.
