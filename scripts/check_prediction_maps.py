"""Check that the 10-m prediction maps reproduce the 100-m blocks used by the analyses.

    python scripts/check_prediction_maps.py

Each map in data/cci/<dataset>/predictions_10m/<recipe>/ is averaged into 25 x 25 blocks of
100 m (last 6 pixels dropped, float64) and compared with data/cci/<dataset>/blocks_<recipe>.npz.
The comparison is exact (no tolerance). Takes a few minutes; needs rasterio.
"""
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
CCI = Path(os.environ.get('DATA_DIR', ROOT / 'data')) / 'cci'


def main():
    failures, n = [], 0
    for blocks_file in sorted(CCI.glob('*/blocks_*.npz')):
        dataset, recipe = blocks_file.parent.name, blocks_file.stem.removeprefix('blocks_')
        z = np.load(blocks_file)
        models = list(z['models'])
        for j, chip in enumerate(z['chip_id']):
            with rasterio.open(CCI / dataset / 'predictions_10m' / recipe / f'{chip}.tif') as src:
                assert list(src.descriptions) == models
                maps = src.read().astype('float64')
            pooled = maps[:, :250, :250].reshape(len(models), 25, 10, 25, 10).mean(axis=(2, 4))
            if not np.array_equal(pooled, z['pred'][:, j]):
                failures.append(f'{dataset}/{recipe}/{chip}')
            n += 1
        print(f'{dataset}/{recipe}: {len(z["chip_id"])} maps checked', flush=True)
    if failures:
        sys.exit(f'{len(failures)} maps do not pool to the stored blocks, e.g. {failures[:3]}')
    print(f'All {n} maps reproduce their 100-m blocks exactly.')


if __name__ == '__main__':
    main()
