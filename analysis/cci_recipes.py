"""Paired Spanish recipe comparison (Section IV-A), Table II counts and the cloud filter.

- Within-mode change M -> B (best scene minus median) for the early, late and all-model
  ensembles, with chip-bootstrap 95% intervals (port of the original compare_composites.py).
- Late-minus-early gap under each recipe and its change between recipes (interaction),
  with 2000-draw chip-bootstrap intervals on R2 and Pearson r.
- Mean residual (prediction minus CCI) by block-level CCI level, including >= 120 Mg/ha.
- Table II: chips per CORINE forest type and CCI biomass bin (before the one-chip exclusion).
- Best-scene chip-months above 20% cloud probability (dropped in the best_c20 variant).

Writes outputs/cci_recipes.json. --legacy-200 reproduces the original run of the within-mode
intervals, which used the first 200 of 500 draws; the default uses 2000 draws as stated in
the paper. Both give the same conclusion (all four within-mode changes exclude zero).
"""
import argparse
import json

import numpy as np
import pandas as pd

from cci_blocks import load_blocks, load_cci
from common import CCI, OUT

RECIPES = ['median', 'best', 'best_c20']
KEYS = ['bias', 'rmse', 'r2', 'r', 'within_r']
LEVELS = {'CCI<50': (0, 50), '50-120': (50, 120), '>=120': (120, 1e9)}


def metrics(P, R, within):
    err = P - R
    return {'bias': err.mean(), 'rmse': np.sqrt((err ** 2).mean()),
            'r2': 1 - (err ** 2).sum() / ((R - R.mean()) ** 2).sum(), 'r': np.corrcoef(P, R)[0, 1],
            'within_r': np.nanmean(within), 'under_frac': np.mean((R >= 10) & (P < 0.5 * R))}


def sums_metric(a):
    """R2 and r from per-chip sums [n, sy, syy, sx, sxx, sxy, sse] (as in the original pair check)."""
    n, sy, syy, sx, sxx, sxy, sse = np.sum(a, axis=0)
    return np.array([1 - sse / (syy - sy * sy / n), (sxy - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--legacy-200', action='store_true', help='original within-mode bootstrap (200 of 500 draws)')
    args = parser.parse_args()

    meta = pd.read_csv(CCI / 'spain/chips.csv')
    chip_ids, agb, _ = load_cci('spain')
    valid_chips = [j for j in range(len(chip_ids)) if np.isfinite(agb[j]).sum() >= 30]
    blocks = {}
    for recipe in RECIPES:
        names, b, ids = load_blocks('spain', recipe)
        assert ids == chip_ids
        # The original comparison averaged all ten models in sorted-name order.
        b['ens_all'] = np.mean([b[n] for n in names], axis=0)
        blocks[recipe] = {'early': b['ens_early'], 'late': b['ens_late'], 'all': b['ens_all']}
    mask = {j: np.isfinite(agb[j]) for j in valid_chips}

    def compute(recipe, g, idx):
        chips = [valid_chips[i] for i in idx]
        P = np.concatenate([blocks[recipe][g][j][mask[j]] for j in chips])
        R = np.concatenate([agb[j][mask[j]] for j in chips])
        within = []
        for j in chips:
            a, b = blocks[recipe][g][j][mask[j]], agb[j][mask[j]]
            within.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
        return metrics(P, R, within)

    # Bootstrap draws use per-chip sums, which give the same metrics as concatenating the
    # resampled chips' blocks (up to floating-point rounding) in a fraction of the time.
    chip_sums, chip_within = {}, {}
    for recipe in RECIPES:
        for g in ['early', 'late', 'all']:
            rows, within = [], []
            for j in valid_chips:
                p, r = blocks[recipe][g][j][mask[j]], agb[j][mask[j]]
                rows.append([len(r), p.sum(), r.sum(), (p * p).sum(), (r * r).sum(), (p * r).sum(), ((p - r) ** 2).sum()])
                within.append(np.corrcoef(p, r)[0, 1] if p.std() > 0 and r.std() > 0 else np.nan)
            chip_sums[recipe, g], chip_within[recipe, g] = np.array(rows), np.array(within)

    def compute_draw(recipe, g, idx):
        n_, sp, sr, spp, srr, spr, see = chip_sums[recipe, g][idx].sum(axis=0)
        return {'bias': (sp - sr) / n_, 'rmse': np.sqrt(see / n_), 'r2': 1 - see / (srr - sr * sr / n_),
                'r': (spr - sp * sr / n_) / np.sqrt((spp - sp * sp / n_) * (srr - sr * sr / n_)),
                'within_r': np.nanmean(chip_within[recipe, g][idx])}

    rng = np.random.default_rng(0)
    n = len(valid_chips)
    if args.legacy_200:
        boots = [rng.integers(n, size=n) for _ in range(500)][:200]
    else:
        boots = [rng.integers(n, size=n) for _ in range(2000)]
    result = {'n_chips': n, 'n_blocks': int(sum(m.sum() for m in mask.values())),
              'within_mode_draws': len(boots), 'within_mode': [], 'residual_by_cci_level': []}
    all_idx = np.arange(n)
    for g in ['early', 'late', 'all']:
        base = compute('median', g, all_idx)
        for recipe in RECIPES:
            m = compute(recipe, g, all_idx)
            row = {'ensemble': f'ens_{g}', 'recipe': recipe, **{k: float(v) for k, v in m.items()}}
            if recipe != 'median':
                bd = pd.DataFrame([{k: v - compute_draw('median', g, b)[k] for k, v in compute_draw(recipe, g, b).items()}
                                   for b in boots])
                for k in KEYS:
                    lo, hi = np.percentile(bd[k], [2.5, 97.5])
                    row[f'change_{k}'] = {'point': float(m[k] - base[k]), 'ci95': [float(lo), float(hi)],
                                          'excludes_zero': bool(lo > 0 or hi < 0)}
            result['within_mode'].append(row)
            P = np.concatenate([blocks[recipe][g][j][mask[j]] for j in valid_chips])
            R = np.concatenate([agb[j][mask[j]] for j in valid_chips])
            result['residual_by_cci_level'].append(
                {'ensemble': f'ens_{g}', 'recipe': recipe,
                 **{k: float((P[(R >= a) & (R < b)] - R[(R >= a) & (R < b)]).mean()) for k, (a, b) in LEVELS.items()}})

    # Late-minus-early gap per recipe and its change between recipes: 2000 chip-bootstrap draws.
    stats = {}
    for recipe in ['median', 'best']:
        for g in ['early', 'late']:
            rows = []
            for j in valid_chips:
                r, p = agb[j][mask[j]], blocks[recipe][g][j][mask[j]]
                rows.append([len(r), r.sum(), (r * r).sum(), p.sum(), (p * p).sum(), (p * r).sum(), ((p - r) ** 2).sum()])
            stats[recipe, g] = np.array(rows)
    rng = np.random.default_rng(42)
    result['late_minus_early'] = {}
    for recipe in ['median', 'best']:
        e, l = stats[recipe, 'early'], stats[recipe, 'late']
        idx = rng.integers(0, n, size=(2000, n))
        diff = np.array([sums_metric(l[b]) - sums_metric(e[b]) for b in idx])
        point = sums_metric(l) - sums_metric(e)
        q = np.quantile(diff, [.025, .975], axis=0)
        result['late_minus_early'][recipe] = {k: {'point': float(point[i]), 'ci95': q[:, i].tolist()}
                                              for i, k in enumerate(['r2', 'r'])}
    rng = np.random.default_rng(43)
    idx = rng.integers(0, n, size=(2000, n))

    def interaction(b):
        return ((sums_metric(stats['best', 'late'][b]) - sums_metric(stats['best', 'early'][b]))
                - (sums_metric(stats['median', 'late'][b]) - sums_metric(stats['median', 'early'][b])))
    point = interaction(np.arange(n))
    q = np.quantile(np.array([interaction(b) for b in idx]), [.025, .975], axis=0)
    result['gap_change_best_minus_median'] = {k: {'point': float(point[i]), 'ci95': q[:, i].tolist()}
                                              for i, k in enumerate(['r2', 'r'])}

    counts = meta.groupby(['forest_type', 'agb_bin']).size().unstack()[['low', 'medium', 'high']]
    result['table2_strata'] = {t: {b: int(v) for b, v in row.items()} for t, row in counts.iterrows()}
    months = pd.read_csv(CCI / 'spain/best_scene_months.csv')
    assert ((months.chosen_cld > 20) == ~months.kept_in_c20).all()
    result['cloudy_chip_months'] = {'threshold_pct': 20, 'dropped': int((~months.kept_in_c20).sum()),
                                    'total': len(months)}

    OUT.mkdir(exist_ok=True)
    name = 'cci_recipes_legacy200.json' if args.legacy_200 else 'cci_recipes.json'
    (OUT / name).write_text(json.dumps(result, indent=1) + '\n')
    for row in result['within_mode']:
        if row['recipe'] == 'best' and row['ensemble'] != 'ens_all':
            print(row['ensemble'], 'M->B', {k: (round(row[f'change_{k}']['point'], 3), [round(v, 3) for v in row[f'change_{k}']['ci95']])
                                            for k in ['r2', 'r']})
    print('gap change (R2, r):', result['gap_change_best_minus_median'])
    print('late-minus-early under B:', result['late_minus_early']['best'])
    print('Table II:', result['table2_strata'], '| cloudy months:', result['cloudy_chip_months'])


if __name__ == '__main__':
    main()
