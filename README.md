---
name: spain-biomass-deployment
authors:
  - Alberto Albiol
  - Miquel Obrador-Reina
  - Juan B. Pedro
  - Albert Fernández
license: CC-BY-4.0
source: https://github.com/alalbiol/spain-biomass-deployment
---

# Deploying Sentinel-1/2 biomass models in Spain

**Code and saved predictions for testing Finland-trained biomass models in Spain: Sentinel-2 preprocessing sensitivity against ESA CCI Biomass, and adaptation with Spanish National Forest Inventory plots.**

This repository accompanies the manuscript *Deploying Sentinel-1/2 Biomass Models in Spain: Preprocessing Sensitivity and Inventory-Based Adaptation*, by Alberto Albiol, Miquel Obrador-Reina, Juan B. Pedro, and Albert Fernández.

**Publication status:** the manuscript is under review. It will be linked here, or its arXiv version, after acceptance.

**Data:** GitHub holds the code and documentation only. The predictions, reference grids, plot tables and adapted weights are in the **[EOTDL dataset `spain-biomass-deployment`](https://www.eotdl.com/datasets/spain-biomass-deployment)**, which contains this repository plus `data/`.

**Models:** the ten BioMassters-trained models evaluated here, with inference code and example patches, are released separately as **[sentinel-biomass-inference](https://github.com/alalbiol/sentinel-biomass-inference)** ([EOTDL model](https://www.eotdl.com/models/sentinel-biomass-inference)).

## What the study covers

Ten Sentinel-1/2 models (five encoders, early and late temporal fusion) trained on the Finnish BioMassters benchmark are applied in Spain.

| Experiment | Reference | Sample | Result files |
|---|---|---|---|
| Sentinel-2 recipe: monthly median (M) vs least-cloudy scene (B) | ESA CCI Biomass, 100-m blocks | 499 chips, 289,600 blocks | `cci_blocks_spain_*.csv`, `cci_recipes.json` |
| Finnish control | CCI 2021 (188 chips); pooled lidar labels (2773 BioMassters test chips) | as stated | `cci_blocks_finland_control_*.csv`, `finland_benchmark_100m.csv` |
| Inventory adaptation and pretraining | IFN4 plot biomass, 6117 pine plots, 14 provinces held out by province | province-grouped 5-fold CV, 3 seeds | `ifn_adaptation.json`, `ifn_pretraining.json`, `ifn_baselines.json` |

On the inventory plots, fine-tuning the early-fusion EfficientNet from its BioMassters weights gives a three-seed mean RMSE of 40.15 Mg/ha, against 45.65 without adaptation and 44.32 and 44.17 when trained from random or ImageNet weights.

## Quick start: reproduce the reported numbers

Download the snapshot with the [EOTDL client](https://www.eotdl.com/docs/getting-started/install) (Python 3.12+, kept in its own environment). The snapshot already contains this code:

```bash
eotdl auth login
eotdl datasets get spain-biomass-deployment --path ./downloads
```

Then, with Python 3.11, inside the downloaded folder (the one containing `release-manifest.json`):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

To use a Git checkout instead, set `DATA_DIR` to the snapshot's `data/` folder, or copy that folder to the checkout root.

The script runs the six analyses in `analysis/` (about a minute on a CPU), writes `outputs/`, and checks every value against `expected/`. [REPRODUCING.md](REPRODUCING.md) maps each table and number in the manuscript to its output field. No GPU, Earth Engine account or inventory download is needed.

## Contents

| Path | Contents | Where |
|---|---|---|
| `analysis/` | The six analyses: CCI block scoring, recipe comparison and bootstraps, Finnish 100-m benchmark, IFN adaptation, pretraining and seeds, no-pretraining baselines | GitHub + EOTDL |
| `expected/` | Reference outputs, including the Table I results that need GPU inference | GitHub + EOTDL |
| `scripts/reproduce.py` | Runs all analyses and checks them against `expected/` | GitHub + EOTDL |
| `scripts/build_*_data.py`, `scripts/prepare_eotdl.py` | How `data/` was built from the original experiment folders, and the snapshot builder (maintainers only) | GitHub + EOTDL |
| `pipeline/` | Original acquisition, inference and training code, as used; not re-run for this release ([pipeline/README.md](pipeline/README.md)) | GitHub + EOTDL |
| `data/cci/` | 100-m pooled predictions of the ten models for each recipe, CCI grids, chip tables, BioMassters pooled predictions and labels | **EOTDL only** |
| `data/ifn/` | Plot table, unadapted predictions, out-of-fold predictions and training logs of all 36 runs, adapted fold weights | **EOTDL only** |

[DATA.md](DATA.md) describes every data file and its sources.

## Licences

- Code and software documentation: MIT ([LICENSE](LICENSE)).
- Data in `data/` and the reference results in `expected/`: CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)), with the source attributions in [DATA.md](DATA.md).

## Funding

This work was carried out within the AI4DIS project, funded by the European Space Agency under Contract No. 4000147555/25/I-DT. The views expressed are those of the authors and do not necessarily reflect the official position of the European Space Agency.
