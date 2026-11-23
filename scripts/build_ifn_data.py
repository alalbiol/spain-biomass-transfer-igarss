"""Maintainer tool: copy the saved IFN experiment outputs into data/ifn/ for the EOTDL snapshot.

Readers do not need this script; they download data/ from EOTDL. It records how the
released tables were derived from the original AI4DIS experiment folder, so it reads
the authors' local paths. Plot coordinates are not copied. The only use the analysis
made of them, the cross-fold neighbour sensitivity, is precomputed here as a flag.
Weights are converted from PyTorch state dicts to safetensors without changing values.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/media/HD/Data/agb_spain_ifn')

# Every run read by analysis/ (seed suffixes added below). The smoke-test run is excluded.
RUNS = [
    'efficientnet_early_full', 'efficientnet_early_head',
    'efficientnet_early_full_frac0.25', 'efficientnet_early_full_frac0.1',
    'efficientnet_early_full_samerecipe', 'efficientnet_early_full_frac0.25_samerecipe',
    'efficientnet_early_full_frac0.1_samerecipe',
    'efficientnet_early_scratch_random', 'efficientnet_early_scratch_random_frac0.25',
    'efficientnet_early_scratch_random_frac0.1',
    'efficientnet_early_scratch_imagenet', 'efficientnet_early_scratch_imagenet_enc1',
    'efficientnet_early_scratch_imagenet_frac0.25_enc1', 'efficientnet_early_scratch_imagenet_frac0.1_enc1',
    'tabular_hgb', 'tabular_rf',
]
SEEDED = [
    'efficientnet_early_full', 'efficientnet_early_full_samerecipe',
    'efficientnet_early_full_frac0.25_samerecipe', 'efficientnet_early_full_frac0.1_samerecipe',
    'efficientnet_early_scratch_random', 'efficientnet_early_scratch_random_frac0.25',
    'efficientnet_early_scratch_random_frac0.1', 'efficientnet_early_scratch_imagenet_enc1',
    'efficientnet_early_scratch_imagenet_frac0.25_enc1', 'efficientnet_early_scratch_imagenet_frac0.1_enc1',
]
WEIGHT_RUNS = ['efficientnet_early_full', 'efficientnet_early_head']
COORDINATES = ['x', 'y', 'huso', 'lon', 'lat']
CONTEXT_RADIUS_M = 1120 * np.sqrt(2)  # half-diagonal envelope of a 112-px (1.12 km) input


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--output', type=Path, default=ROOT / 'data/ifn')
    parser.add_argument('--skip-weights', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}

    plots = pd.read_csv(SOURCE / 'plots.csv', dtype={'province_code': str})
    assert plots.chip_id.is_unique and len(plots) == 6117
    folds = pd.read_csv(SOURCE / 'finetune/efficientnet_early_full/oof.csv').set_index('chip_id').fold
    plots['fold'] = plots.chip_id.map(folds)
    assert plots.fold.notna().all()
    # Same computation as the original audit (audit_ifn_adaptation.py), done before coordinates are dropped.
    xy = np.c_[np.deg2rad(plots.lon) * 6371000 * np.cos(np.deg2rad(plots.lat.mean())), np.deg2rad(plots.lat) * 6371000]
    pairs = cKDTree(xy).query_pairs(CONTEXT_RADIUS_M, output_type='ndarray')
    fold = plots.fold.to_numpy()
    cross = pairs[fold[pairs[:, 0]] != fold[pairs[:, 1]]]
    plots['cross_fold_neighbour'] = False
    plots.loc[plots.index[np.unique(cross.ravel())], 'cross_fold_neighbour'] = True
    context = {'radius_m': float(CONTEXT_RADIUS_M), 'cross_fold_pairs': int(len(cross)),
               'flagged_plots': int(plots.cross_fold_neighbour.sum()),
               'duplicate_coordinates': int(plots.duplicated(['lon', 'lat']).sum())}
    plots.drop(columns=COORDINATES + ['fold']).to_csv(args.output / 'plots.csv', index=False)
    sources['plots.csv'] = sha256(SOURCE / 'plots.csv')

    (args.output / 'zero_shot').mkdir()
    shutil.copyfile(SOURCE / 'zero_shot/plot_preds.csv', args.output / 'zero_shot/plot_preds.csv')
    shutil.copyfile(SOURCE / 'cci_at_plots.csv', args.output / 'cci_at_plots.csv')
    sources['zero_shot/plot_preds.csv'] = sha256(SOURCE / 'zero_shot/plot_preds.csv')
    sources['cci_at_plots.csv'] = sha256(SOURCE / 'cci_at_plots.csv')

    names = RUNS + [f'{r}_seed{s}' for r in SEEDED for s in (1, 2)]
    for run in names:
        src, dst = SOURCE / 'finetune' / run, args.output / 'finetune' / run
        dst.mkdir(parents=True)
        for name in ('oof.csv', 'args.json', 'history.json'):
            if (src / name).exists():
                shutil.copyfile(src / name, dst / name)
                sources[f'finetune/{run}/{name}'] = sha256(src / name)
        assert (dst / 'oof.csv').exists(), run

    if not args.skip_weights:
        import torch
        from safetensors.torch import save_file
        for run in WEIGHT_RUNS:
            for k in range(5):
                src = SOURCE / 'finetune' / run / f'fold{k}.pt'
                state = torch.load(src, map_location='cpu', weights_only=True)
                dst = args.output / 'finetune' / run / f'fold{k}.safetensors'
                save_file({n: t.contiguous() for n, t in state.items()}, str(dst),
                          metadata={'source': f'{run}/fold{k}.pt', 'source_sha256': sha256(src)})
                sources[f'finetune/{run}/fold{k}.pt'] = sha256(src)

    record = {'source_dir': str(SOURCE), 'source_sha256': sources, 'context_sensitivity': context,
              'removed_columns': COORDINATES, 'runs': names, 'weights': [] if args.skip_weights else WEIGHT_RUNS}
    (args.output / 'build_record.json').write_text(json.dumps(record, indent=1) + '\n')
    print(f'Wrote {args.output}: {len(names)} runs, context flag on {context["flagged_plots"]} plots')


if __name__ == '__main__':
    main()
