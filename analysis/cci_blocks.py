"""Block-level (100 m) agreement with ESA CCI Biomass for every model, ensemble and stratum.

Port of the original evaluate_zero_shot_blocks.py, reading pooled blocks from data/cci/.
Ensembles average the 25 x 25 blocks of their members (five encoders per fusion mode, ten
for ens_all). Only blocks with a CCI value are scored, and chips with fewer than 30 such
blocks are skipped (one Spanish chip, ten Finnish ones). Metric constants follow the
training code: eps 10 Mg/ha, over/under ratios 2 and 0.5, SSIM window 5, data range 500.

Writes outputs/cci_blocks_<dataset>_<recipe>.csv for Spain (median, best, best_c20) and the
Finnish control (median, best). Table III and Fig. 1 use the ALL rows of the Spanish files;
the per-type stratum rows of ens_all and the Finnish ens_all rows are quoted in Section IV.
"""
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

from common import CCI, OUT

EPS, OVER, UNDER = 10.0, 2.0, 0.5
RUNS = [('spain', 'median'), ('spain', 'best'), ('spain', 'best_c20'),
        ('finland_control', 'median'), ('finland_control', 'best')]


def load_cci(dataset):
    z = np.load(CCI / dataset / 'cci_100m.npz')
    return list(z['chip_id']), z['agb'], z['sd']


def load_blocks(dataset, recipe):
    """{model: array (chips, 25, 25)} plus the early/late/all ensembles, and the chip order."""
    z = np.load(CCI / dataset / f'blocks_{recipe}.npz')
    names = sorted(z['models'])  # the original read model folders in sorted order
    order = list(z['models'])
    blocks = {n: z['pred'][order.index(n)] for n in names}
    early = [n for n in names if n.endswith('_early')]
    late = [n for n in names if n.endswith('_late')]
    for g, members in {'ens_early': early, 'ens_late': late, 'ens_all': early + late}.items():
        blocks[g] = np.mean([blocks[m] for m in members], axis=0)
    return names, blocks, list(z['chip_id'])


def within_chip_r(pred, ref):
    m = np.isfinite(ref)
    if m.sum() < 30 or np.std(pred[m]) == 0 or np.std(ref[m]) == 0:
        return np.nan
    return np.corrcoef(pred[m], ref[m])[0, 1]


def summarize(P, R, SD, ssim_scores, within_r, within_r_flip):
    err = P - R
    sst = np.sum((R - R.mean()) ** 2)
    sig = R >= EPS
    return {
        'n_chips': len(ssim_scores), 'n_blocks': len(R), 'ref_mean': R.mean(), 'bias': err.mean(),
        'rel_bias_pct': 100 * err.mean() / R.mean(), 'mae': np.abs(err).mean(), 'rmse': np.sqrt((err ** 2).mean()),
        'r2': 1 - np.sum(err ** 2) / sst, 'pearson_r': np.corrcoef(P, R)[0, 1],
        'frac_over': np.mean(P > OVER * np.maximum(R, EPS)), 'frac_under': np.mean(sig & (P < UNDER * R)),
        'sci_100m': np.nanmean(ssim_scores), 'within_chip_r': np.nanmean(within_r),
        'within_chip_r_flipped_ctrl': np.nanmean(within_r_flip),
        'cci_sd_rms': np.sqrt(np.nanmean(SD ** 2)) if np.isfinite(SD).any() else np.nan,
    }


def evaluate(dataset, recipe):
    meta = pd.read_csv(CCI / dataset / 'chips.csv')
    chip_ids, agb, sd = load_cci(dataset)
    names, blocks, pred_ids = load_blocks(dataset, recipe)
    assert chip_ids == pred_ids == meta.chip_id.tolist()
    pos = {c: i for i, c in enumerate(chip_ids)}
    mi = meta.set_index('chip_id')
    strata = {'ALL': lambda c: True}
    for t in sorted(meta.forest_type.unique()):
        strata[f'type={t}'] = lambda c, t=t: mi.forest_type[c] == t
    for b in ['low', 'medium', 'high']:
        strata[f'bin={b}'] = lambda c, b=b: mi.agb_bin[c] == b
    rows = []
    for m in names + ['ens_early', 'ens_late', 'ens_all']:
        for s, sel in strata.items():
            chips = [pos[c] for c in meta.chip_id if sel(c) and np.isfinite(agb[pos[c]]).sum() >= 30]
            P = np.concatenate([blocks[m][j][np.isfinite(agb[j])] for j in chips])
            R = np.concatenate([agb[j][np.isfinite(agb[j])] for j in chips])
            SD = np.concatenate([sd[j][np.isfinite(agb[j])] for j in chips])
            ssim, wr, wrf = [], [], []
            for j in chips:
                r = agb[j]
                ssim.append(structural_similarity(np.where(np.isfinite(r), r, 0.0), blocks[m][j],
                                                  data_range=500.0, win_size=5))
                wr.append(within_chip_r(blocks[m][j], r))
                wrf.append(within_chip_r(blocks[m][j], r[::-1]))
            rows.append({'model': m, 'stratum': s, **summarize(P, R, SD, ssim, wr, wrf)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    for dataset, recipe in RUNS:
        res = evaluate(dataset, recipe)
        res.to_csv(OUT / f'cci_blocks_{dataset}_{recipe}.csv', index=False)
        e = res[(res.stratum == 'ALL') & res.model.str.startswith('ens_')].set_index('model')
        print(f'{dataset:15s} {recipe:9s} chips {e.n_chips.iloc[0]} blocks {e.n_blocks.iloc[0]}  ' + '  '.join(
            f"{m}: bias {e.bias[m]:+.1f} RMSE {e.rmse[m]:.1f} R2 {e.r2[m]:.3f} r {e.pearson_r[m]:.3f}" for m in e.index))


if __name__ == '__main__':
    main()
