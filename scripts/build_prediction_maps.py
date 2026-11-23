"""Maintainer tool: write the 10-m prediction maps of the CCI comparison as GeoTIFFs in data/cci/.

One file per chip and recipe, data/cci/<dataset>/predictions_10m/<recipe>/<chip_id>.tif:
10 float32 bands (one per model, named in the band descriptions), 256 x 256 pixels at 10 m in
EPSG:3035, on the chip's original grid. Values are copied unchanged from the saved predictions
(Mg/ha); DEFLATE compression is lossless. scripts/check_prediction_maps.py confirms that pooling
these maps reproduces the 100-m blocks used by the analyses.

Run with an environment whose rasterio resolves EPSG codes (needs a PROJ database).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'spain': (Path('/media/HD/Data/agb_spain'), {'median': 'zero_shot', 'best': 'zero_shot_bm', 'best_c20': 'zero_shot_bm_c20'}),
    'finland_control': (Path('/media/HD/Data/agb_finland_control'), {'median': 'zero_shot', 'best': 'zero_shot_bm'}),
}
BACKBONES = ['convnextv2', 'efficientnet', 'efficientnetv2', 'mobilenetv4', 'resnetrs101']
MODELS = [f'{b}_{m}' for b in BACKBONES for m in ('early', 'late')]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    crs = CRS.from_epsg(3035)
    for dataset, (src, recipes) in SOURCES.items():
        chips = pd.read_csv(args.data / 'cci' / dataset / 'chips.csv')
        for recipe, folder in recipes.items():
            out = args.data / 'cci' / dataset / 'predictions_10m' / recipe
            out.mkdir(parents=True, exist_ok=False)
            for c in chips.itertuples():
                arr = np.stack([np.load(src / folder / 'preds' / m / f'{c.chip_id}.npy') for m in MODELS])
                assert arr.shape == (10, 256, 256) and arr.dtype == np.float32
                profile = dict(driver='GTiff', width=256, height=256, count=10, dtype='float32', crs=crs,
                               transform=from_origin(c.left_3035, c.top_3035, 10, 10),
                               compress='deflate', predictor=3, tiled=True, blockxsize=256, blockysize=256)
                with rasterio.open(out / f'{c.chip_id}.tif', 'w', **profile) as dst:
                    dst.write(arr)
                    dst.descriptions = tuple(MODELS)
                    dst.update_tags(units='Mg/ha aboveground dry biomass', recipe=recipe, chip_id=c.chip_id)
            print(f'{dataset}/{recipe}: {len(chips)} maps', flush=True)


if __name__ == '__main__':
    main()
