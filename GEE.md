# Getting the imagery from Google Earth Engine

The Sentinel-1/2 inputs of the study are not redistributed (Spain alone is about 23 GB, and the inventory chips another 28 GB). They are all public in Earth Engine, and `scripts/download_gee.py` downloads them again with the study's recipes, on the same grids, for any subset of chips or plots.

## 1. Set up Earth Engine

1. Use a Google account with [Earth Engine access](https://earthengine.google.com/) and a Google Cloud project registered for Earth Engine (noncommercial registration is free).
2. Install the client in its own environment and authenticate once:

```bash
python3.11 -m venv .venv-gee && source .venv-gee/bin/activate
python -m pip install -r requirements-gee.txt
earthengine authenticate
export EE_PROJECT=<your-cloud-project-id>
```

The downloader reads chip and plot locations from `data/` (EOTDL snapshot). Set `DATA_DIR` if `data/` is elsewhere.

## 2. Download

```bash
# Two Spanish chips with each Sentinel-2 recipe (2.56 km, 12 monthly GeoTIFFs per sensor)
python scripts/download_gee.py chips --dataset spain --recipe median --limit 2 --out-dir downloads/spain_median
python scripts/download_gee.py chips --dataset spain --recipe best   --limit 2 --out-dir downloads/spain_best

# Specific chips, or the Finnish control chips
python scripts/download_gee.py chips --dataset spain --recipe best --chips ES_2021_40p7800_W4p0495 --out-dir downloads/spain_best
python scripts/download_gee.py chips --dataset finland_control --recipe best --limit 5 --out-dir downloads/finland_best

# Inventory plots (1.28 km centred on the plot, in the plot's image year)
python scripts/download_gee.py ifn --limit 10 --out-dir downloads/ifn
python scripts/download_gee.py ifn --chips IFN4_02_0002_A1 IFN4_02_0003_A1 --out-dir downloads/ifn
```

Drop `--limit` to download everything. Existing files are skipped, so an interrupted run resumes where it stopped. Each 2.56-km chip takes one to two minutes and 24 requests. `--workers` (default 4) downloads several chips at once; Earth Engine's rate limits (HTTP 429) are retried with back-off. The 500 Spanish chips take several hours per recipe.

## 3. What each recipe does

| | Sentinel-1 (every recipe) | Sentinel-2 | Cloud band (11th S2 band) |
|---|---|---|---|
| `chips --recipe median` (M) | Per-orbit monthly median of IW GRD VV and VH (dB); bands VV_asc, VH_asc, VV_desc, VH_desc; -9999 where an orbit has no pass | Pixelwise median of all L2A scenes (`COPERNICUS/S2_SR_HARMONIZED`) in the month | Sen2Cor `MSK_CLDPRB`, 0–100 |
| `chips --recipe best` (B) | as above | The scene that covers the whole chip with the lowest chip-mean s2cloudless probability (`COPERNICUS/S2_CLOUD_PROBABILITY`). A month without such a scene has no file | s2cloudless probability × 2.55, 0–255 (the BioMassters scale) |
| `ifn` | as above | as B, for the plot's image year | as B |

Recipe B uses the scenes chosen in the study (`data/cci/<dataset>/best_scene_months.csv`). Add `--reselect` to rank the scenes again instead. The paper compares M and B on the same chips (Section III-B of the manuscript).

**Grids.** The CCI chips use the exact grid of the original exports: 256 × 256 pixels of 10 m in EPSG:3035 (ETRS89-LAEA), with the upper-left corner in `left_3035`/`top_3035` of `chips.csv`. Inventory chips are 128 × 128 pixels centred on the plot (`lon`/`lat` in `plots.csv`); the plot centre is the corner shared by pixels 63 and 64.

**Outputs.**

- `chips`: `<chip_id>_S1_<mm>.tif` (4 bands, float32, dB) and `<chip_id>_S2_<mm>.tif` (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 as L2A digital numbers, then the cloud band), for months `<mm>` = 00–11 of 2021. This is the layout read by [sentinel-biomass-inference](https://github.com/alalbiol/sentinel-biomass-inference).
- `ifn`: `<chip_id>.npz` with `s2` (uint16, 12 × 11 × 128 × 128), `s1` (float32, 12 × 4 × 128 × 128, dB, -9999 = no pass), `missing` (months without an S2 scene) and `cld` (chosen scene's cloud probability, %). This is the layout read by `pipeline/ifn/build_chip_cache.py`.

## 4. Run the models on a download

The ten models and their inference code are in [sentinel-biomass-inference](https://github.com/alalbiol/sentinel-biomass-inference) (checkpoints on its [EOTDL model page](https://www.eotdl.com/models/sentinel-biomass-inference)):

```bash
cd sentinel-biomass-inference
python inference.py --model efficientnet_late --input-dir ../spain-biomass-transfer-igarss/downloads/spain_best \
  --chip-id ES_2021_40p7800_W4p0495 --output outputs/ES_2021_40p7800_W4p0495.tif --device cpu
```

Run on the original inputs of this chip (`data/cci/spain/example_inputs/best/`), the CPU result is within 0.07 Mg/ha of the stored map (`data/cci/spain/predictions_10m/best/`, band `efficientnet_late`). The stored maps were computed on a GPU.

## 5. How close a new download is to the original

The downloader requests the same images on the same grids, and for recipe B the same scenes. Earth Engine assets do change over time, however. We downloaded chip `ES_2021_40p7800_W4p0495` (both recipes) and plot `IFN4_02_0002_A1` again on 23 November 2026, two to three months after the study's exports:

- inventory chip: identical to the original, array for array;
- Sentinel-1: identical up to float32 rounding (< 1e-6 dB; the originals were stored as float64);
- Sentinel-2: 9 of 12 median months and 10 of 12 best-scene months identical. The other months differ although the same scenes are selected: Earth Engine has updated these scene assets since the original export.

The original inputs of two chips are included for comparison: `data/cci/spain/example_inputs/{median,best}/`. For new locations, use `scripts/download_gee.py` of sentinel-biomass-inference, which takes any longitude and latitude.
