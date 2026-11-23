# Original pipeline (as used)

These scripts produced the data in `data/`. They are copied from the AI4DIS project as they were run, and they were **not re-run for this release**. They need resources that the release does not provide: Google Earth Engine access, the BioMassters checkpoints and training code, the IFN4 Access databases, about 60 GB of exported imagery, and a GPU. You do not need them to reproduce the reported numbers; `scripts/reproduce.py` starts from their saved outputs.

The only changes from the original files are these: Earth Engine is initialized with the project in the `EE_PROJECT` environment variable instead of a hard-coded project, and `import os` is added where needed. Paths still point to the authors' machine (`/media/HD/...`, `/home/...`) and must be edited for another setup.

## Dependencies outside this repository

- `models` (`UnetVFLOW`), `utils` (`CustomModel`, the Lightning wrapper) and `inference` (`Config`) come from the AI4DIS `use_case_1` training code. The released equivalent of the network is the `biomass/` package of [sentinel-biomass-inference](https://github.com/alalbiol/sentinel-biomass-inference), and its EOTDL snapshot contains the ten checkpoints as safetensors (the original runs read Lightning `.ckpt` files).
- `pipeline/ifn/build_ifn_plot_labels.py` imports the IFN readers and Spanish biomass equations from the AI4DIS growth-model code. The same equations are released in [iberian-pines-ifn-validation](https://github.com/alalbiol/iberian-pines-ifn-validation) (`models/`).
- Python environments: the inference and training scripts ran with Python 3.11, PyTorch 2.10, timm, segmentation-models-pytorch 0.5 and Lightning 2.6. The Earth Engine exports ran in a separate environment with `earthengine-api`, `rasterio` and `requests`.

## `cci/`: Spanish and Finnish chips scored against CCI

| Script | Role |
|---|---|
| `export_validation_set.py` | Draws the stratified chips (CORINE forest type × CCI biomass bin) and exports recipe M imagery (monthly median, Sen2Cor cloud probability); `--country Finland` for the control |
| `export_bm_style_chips.py` | Re-exports Sentinel-2 with recipe B (least-cloudy scene per month, s2cloudless probability); `--max_cloud 20 --variant_out ...` builds the cloud-filtered variant |
| `fetch_cci_100m.py` | ESA CCI Biomass 2021 (AGB and SD) on the 25 × 25 block grid of each chip |
| `run_zero_shot.py` | 10-m predictions of the ten models for every chip |
| `evaluate_zero_shot_blocks.py` | Block-level metrics (ported as `analysis/cci_blocks.py`) |
| `compare_composites.py` | Paired recipe comparison (ported into `analysis/cci_recipes.py`) |
| `run_finland_pooled_benchmark.py` | BioMassters test set pooled to 100 m (inference plus the metrics ported as `analysis/finland_benchmark.py`) |
| `run_bm_full_pipeline.sh`, `run_finland_control.sh`, `run_finland_benchmark.sh` | Drivers. Three Spanish chips with invalid values were re-exported and re-scored by hand on 22 September 2026 after the first run of `run_bm_full_pipeline.sh`; the released data come from the corrected run. |

The late-minus-early gap intervals and Fig. 1 were computed by two short one-off scripts that are not preserved as files. Their logic is reproduced in `analysis/cci_recipes.py` (same chips, draws and random seeds; the results match the manuscript), and Fig. 1 plots columns of `analysis/cci_blocks.py` output.

## `ifn/`: inventory plots

| Script | Role |
|---|---|
| `build_ifn_plot_labels.py` | Plot biomass from the IFN4 tree lists (pine-dominated plots, allometric equations, subplot expansion, small-stem approximation) |
| `export_ifn_chips.py`, `build_chip_cache.py` | 128-px Sentinel-1/2 chips per plot (recipe B) and the 112-px training cache |
| `fetch_cci_at_plots.py` | CCI biomass at the plot centres |
| `ifn_common.py` | Shared loading, disk weights, province folds, model initialization |
| `zero_shot_ifn.py` | Unadapted predictions of the ten models |
| `finetune_ifn.py` | Decoder or full fine-tuning, random and ImageNet initializations, fitting subsets and seeds; writes `oof.csv`, `history.json`, `args.json` and fold weights |
| `baseline_tabular_ifn.py` | Gradient boosting and random forest on monthly channel means |
| `run_all.sh`, `run_baselines.sh`, `run_overnight.sh`, `run_weekend.sh` | Drivers for each batch of runs |

The first runs (`efficientnet_early_full`, `_head`, and the eight-epoch subsets) were made before the `--seed`, `--init`, `--enc_lr_mult` and `--train_bn` options existed. Their defaults reproduce that behaviour, so those runs' `args.json` files simply lack these keys.
