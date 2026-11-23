"""Maintainer tool: pack the saved CCI-comparison predictions into data/cci/ for the EOTDL snapshot.

Readers do not need this script; they download data/ from EOTDL. Every CCI number in
the paper is computed on 100-m blocks: a 256 x 256 prediction at 10 m is averaged into
25 x 25 blocks (the last 6 pixels dropped). This tool applies that pooling once, in
float64 exactly as the original evaluation did, and stores the blocks, so the analysis
reproduces the original metrics without the 10-m prediction maps (about 5 GB).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from skimage import io

ROOT = Path(__file__).resolve().parents[1]
SPAIN = Path('/media/HD/Data/agb_spain')
FINLAND = Path('/media/HD/Data/agb_finland_control')
BIOMASSTERS = Path('/media/HD/Data/BioMassters/pooled_preds')
BACKBONES = ['convnextv2', 'efficientnet', 'efficientnetv2', 'mobilenetv4', 'resnetrs101']
MODELS = [f'{b}_{m}' for b in BACKBONES for m in ('early', 'late')]
NODATA = -9999


def pool(pred):
    return pred[:250, :250].reshape(25, 10, 25, 10).mean(axis=(1, 3))


def add_origin(meta, image_dir):
    """Upper-left corner (EPSG:3035, metres) of each chip's 256 x 256 grid, read from its S1 GeoTIFF.
    With these, the Earth Engine download requests exactly the original grid."""
    left, top = [], []
    for c in meta.chip_id:
        with rasterio.open(image_dir / f'{c}_S1_00.tif') as s:
            assert abs(s.transform.a - 10) < 1e-6 and abs(s.transform.e + 10) < 1e-6 and s.shape == (256, 256)
            left.append(s.transform.c)
            top.append(s.transform.f)
    return meta.assign(left_3035=left, top_3035=top)


def pack_predictions(pred_root, chips, out):
    pred = np.empty((len(MODELS), len(chips), 25, 25))
    for i, m in enumerate(MODELS):
        for j, c in enumerate(chips):
            pred[i, j] = pool(np.load(pred_root / m / f'{c}.npy').astype('float64'))
    np.savez_compressed(out, models=np.array(MODELS), chip_id=np.array(chips), pred=pred)


def pack_cci(cci_dir, chips, out):
    agb, sd = np.empty((len(chips), 25, 25)), np.empty((len(chips), 25, 25))
    for j, c in enumerate(chips):
        a = io.imread(cci_dir / f'{c}.tif')
        assert a.shape == (25, 25, 2), a.shape
        agb[j] = np.where(a[..., 0] == NODATA, np.nan, a[..., 0].astype('float64'))
        sd[j] = np.where(a[..., 1] == NODATA, np.nan, a[..., 1].astype('float64'))
    np.savez_compressed(out, chip_id=np.array(chips), agb=agb, sd=sd)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--output', type=Path, default=ROOT / 'data/cci')
    args = parser.parse_args()
    out = args.output
    (out / 'spain').mkdir(parents=True, exist_ok=False)
    (out / 'finland_control').mkdir()
    (out / 'finland_benchmark').mkdir()

    # Spain: the 500 stratified chips; all have the 12 re-exported best-scene months.
    meta = pd.read_csv(SPAIN / 'validation_bm/metadata_complete.csv')
    assert meta.chip_id.equals(pd.read_csv(SPAIN / 'validation/metadata.csv').chip_id) and len(meta) == 500
    chips = meta.chip_id.tolist()
    add_origin(meta, SPAIN / 'validation').to_csv(out / 'spain/chips.csv', index=False)
    pack_cci(SPAIN / 'cci_100m', chips, out / 'spain/cci_100m.npz')
    for name, src in [('median', 'zero_shot'), ('best', 'zero_shot_bm'), ('best_c20', 'zero_shot_bm_c20')]:
        pack_predictions(SPAIN / src / 'preds', chips, out / f'spain/blocks_{name}.npz')

    # Best-scene choice per chip-month: the last logged attempt is the exported scene. The
    # c20 variant dropped months whose chosen scene exceeded 20% cloud probability; the
    # flag is read from the c20 folder itself rather than recomputed.
    log = pd.read_csv(SPAIN / 'validation_bm/best_scene_log.csv')
    months = log.groupby(['chip_id', 'month'], sort=False).tail(1)[['chip_id', 'month', 'n_scenes', 'chosen_cld', 'scene']]
    months = months[months.chip_id.isin(chips)].sort_values(['chip_id', 'month']).reset_index(drop=True)
    assert len(months) == 12 * len(chips)
    c20 = SPAIN / 'validation_bm_c20'
    months['kept_in_c20'] = [(c20 / f'{c}_S2_{m:02d}.tif').exists() for c, m in zip(months.chip_id, months.month)]
    months.to_csv(out / 'spain/best_scene_months.csv', index=False)

    # Finnish CCI control: 188 chips after the 30-valid-block rule is applied in analysis.
    fmeta = pd.read_csv(FINLAND / 'metadata_complete.csv')
    fchips = fmeta.chip_id.tolist()
    add_origin(fmeta, FINLAND / 'validation').to_csv(out / 'finland_control/chips.csv', index=False)
    pack_cci(FINLAND / 'cci_100m', fchips, out / 'finland_control/cci_100m.npz')
    for name, src in [('median', 'zero_shot'), ('best', 'zero_shot_bm')]:
        pack_predictions(FINLAND / src / 'preds', fchips, out / f'finland_control/blocks_{name}.npz')
    flog = pd.read_csv(FINLAND / 'validation_bm/best_scene_log.csv')
    fmonths = flog.groupby(['chip_id', 'month'], sort=False).tail(1)[['chip_id', 'month', 'n_scenes', 'chosen_cld', 'scene']]
    fmonths = fmonths[fmonths.chip_id.isin(fchips)].sort_values(['chip_id', 'month']).reset_index(drop=True)
    assert len(fmonths) == 12 * len(fchips) and (fmonths.n_scenes >= 0).all()
    fmonths.to_csv(out / 'finland_control/best_scene_months.csv', index=False)

    # BioMassters test set pooled to 100 m against the pooled lidar labels (saved as float32).
    bchips = sorted(p.stem for p in (BIOMASSTERS / '_ground_truth_pooled').glob('*.npy'))
    assert len(bchips) == 2773
    pred = np.stack([np.stack([np.load(BIOMASSTERS / m / f'{c}.npy') for c in bchips]) for m in MODELS])
    gt = np.stack([np.load(BIOMASSTERS / '_ground_truth_pooled' / f'{c}.npy') for c in bchips])
    assert pred.dtype == gt.dtype == np.float32
    np.savez_compressed(out / 'finland_benchmark/pooled_100m.npz', models=np.array(MODELS),
                        chip_id=np.array(bchips), pred=pred, ground_truth=gt)

    record = {'sources': {'spain': str(SPAIN), 'finland_control': str(FINLAND), 'biomassters_pooled': str(BIOMASSTERS)},
              'files': {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(out.rglob('*')) if p.is_file()},
              'c20_dropped_months': int((~months.kept_in_c20).sum())}
    (out / 'build_record.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
