"""Score every IFN run (BioMassters fine-tuning and no-pretraining baselines) at each
training-data fraction, from the out-of-fold predictions in <DATA>/finetune/<run>/oof.csv.

For each fraction, every run is compared with a reference on the plots both predict,
with 2000-draw paired province-bootstrap 95% intervals of RMSE(ref) - RMSE(run)
(negative = the reference is better), the same resampling as compare_ifn_baselines.py.
Also reports, per run, the selected epoch per fold (from history.json): a selected
epoch equal to the last one means the run may still have been improving.
Missing runs are skipped.

Seeds section: seed 0 is the original run, seeds 1-2 add `_seed1`/`_seed2` (init, augmentation
and batch order change; folds, inner validation and subsets do not). Per method and fraction it
reports RMSE per seed, mean and range, and the RMSE of the seed-averaged prediction. Paired
comparisons average RMSE(ref) - RMSE(run) over matched seeds inside each province-bootstrap draw,
so the interval covers province resampling around the three-seed mean; the per-seed point
deltas show the seed spread. Writes outputs/ifn_pretraining.{json,md}.
Run analysis/ifn_adaptation.py first: it writes outputs/ifn_plot_predictions.csv.
"""
import json

import numpy as np
import pandas as pd

from common import IFN, OUT

FT = IFN / 'finetune'
B = 2000

# (label, run dir) per fraction; the first entry of each block is the default reference.
FRACS = {
    '1.0': [('BioMassters, paper recipe (8 ep, 1e-4, enc x0.1)', 'efficientnet_early_full'),
            ('BioMassters, baseline recipe (30 ep, 1e-3, enc x1)', 'efficientnet_early_full_samerecipe'),
            ('Random init (30 ep, 1e-3, enc x1)', 'efficientnet_early_scratch_random'),
            ('ImageNet init (30 ep, 1e-3, enc x1)', 'efficientnet_early_scratch_imagenet_enc1'),
            ('ImageNet init (30 ep, 1e-3, enc x0.1)', 'efficientnet_early_scratch_imagenet'),
            ('Tabular HGB', 'tabular_hgb'),
            ('Tabular RF', 'tabular_rf')],
    '0.25': [('BioMassters, paper recipe', 'efficientnet_early_full_frac0.25'),
             ('BioMassters, baseline recipe', 'efficientnet_early_full_frac0.25_samerecipe'),
             ('Random init', 'efficientnet_early_scratch_random_frac0.25'),
             ('ImageNet init (enc x1)', 'efficientnet_early_scratch_imagenet_frac0.25_enc1')],
    '0.1': [('BioMassters, paper recipe', 'efficientnet_early_full_frac0.1'),
            ('BioMassters, baseline recipe', 'efficientnet_early_full_frac0.1_samerecipe'),
            ('Random init', 'efficientnet_early_scratch_random_frac0.1'),
            ('ImageNet init (enc x1)', 'efficientnet_early_scratch_imagenet_frac0.1_enc1')],
}
# Extra pairs worth an interval of their own: (reference run, other run).
EXTRA = [('efficientnet_early_full_samerecipe', 'efficientnet_early_scratch_random'),
         ('efficientnet_early_full_samerecipe', 'efficientnet_early_scratch_imagenet_enc1'),
         ('efficientnet_early_scratch_imagenet_enc1', 'efficientnet_early_scratch_random')]


def metrics(y, p):
    e = p - y
    return dict(n=int(len(y)), bias=float(e.mean()), rmse=float(np.sqrt(np.mean(e * e))),
                r2=float(1 - np.sum(e * e) / np.sum((y - y.mean()) ** 2)), r=float(np.corrcoef(y, p)[0, 1]))


def load(run):
    f = FT / run / 'oof.csv'
    if not f.exists():
        return None
    o = pd.read_csv(f, dtype={'province_code': str})
    return o if len(o) else None


def best_epochs(run):
    f = FT / run / 'history.json'
    if not f.exists():
        return None
    h = json.load(open(f))
    out = {}
    for fold, hist in h.items():
        tr = [e for e in hist if e['epoch'] > 0]
        if tr:
            b = min(tr, key=lambda e: e['val_rmse'])
            out[fold] = f"{b['epoch']}/{tr[-1]['epoch']}"
    return out


def paired(a, b, ra, rb, seed):
    """RMSE(a) - RMSE(b) on common plots, province-bootstrap CI."""
    d = a[['chip_id', 'province_code', 'agb_mg_ha', 'pred']].merge(
        b[['chip_id', 'pred']], on='chip_id', suffixes=('_a', '_b'), validate='one_to_one')
    rm = lambda t, c: np.sqrt(np.mean((t[c] - t.agb_mg_ha) ** 2))
    groups = [g for _, g in d.groupby('province_code')]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        t = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(rm(t, 'pred_a') - rm(t, 'pred_b'))
    better = sum(rm(g, 'pred_a') < rm(g, 'pred_b') for g in groups)
    return dict(ref=ra, other=rb, n=len(d), n_prov=len(groups), point=float(rm(d, 'pred_a') - rm(d, 'pred_b')),
                ci95=np.quantile(draws, [.025, .975]).tolist(), provinces_ref_better=int(better))


res, md = {'fractions': {}, 'pairs': []}, ['# IFN overnight comparison', '',
      'Out-of-fold, province-grouped 5-fold CV on the IFN4 pine plots. Delta = RMSE(reference) - RMSE(run), '
      'Mg/ha; negative = reference better; 95% paired province-bootstrap interval (2000 draws). '
      'Sel. epoch = epoch chosen on inner validation / epochs trained, per fold.', '']
runs = {}
for frac, entries in FRACS.items():
    present = [(lab, r, load(r)) for lab, r in entries]
    present = [(lab, r, o) for lab, r, o in present if o is not None]
    if not present:
        continue
    for _, r, o in present:
        runs[r] = o
    ref_lab, ref_run, ref = present[0]
    md += [f'## Training fraction {frac} (reference: {ref_lab})', '',
           '| Run | n | RMSE | Bias | R² | r | Delta vs ref [95% CI] | Prov. ref better | Sel. epoch |',
           '|---|---|---|---|---|---|---|---|---|']
    block = {}
    for i, (lab, r, o) in enumerate(present):
        m = metrics(o.agb_mg_ha.to_numpy(), o.pred.to_numpy())
        ep = best_epochs(r)
        row = {'label': lab, 'run': r, **m, 'selected_epoch': ep}
        cell, prov = '—', '—'
        if i > 0:
            p = paired(ref, o, ref_run, r, seed=20261119)
            row['vs_ref'] = p
            cell = f"{p['point']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]"
            prov = f"{p['provinces_ref_better']}/{p['n_prov']}"
        block[r] = row
        eps = ', '.join(ep.values()) if ep else '—'
        md.append(f"| {lab} | {m['n']} | {m['rmse']:.2f} | {m['bias']:+.2f} | {m['r2']:.3f} | {m['r']:.3f} | {cell} | {prov} | {eps} |")
    res['fractions'][frac] = block
    md.append('')

pairs = [(a, b) for a, b in EXTRA if a in runs and b in runs]
if pairs:
    md += ['## Extra paired comparisons (full data)', '', '| Reference | Run | Delta [95% CI] | Prov. ref better |', '|---|---|---|---|']
    for a, b in pairs:
        p = paired(runs[a], runs[b], a, b, seed=20261119)
        res['pairs'].append(p)
        md.append(f"| {a} | {b} | {p['point']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] | {p['provinces_ref_better']}/{p['n_prov']} |")
    md.append('')

# ---------------------------------------------------------------- seeds
SEEDS = [0, 1, 2]
METHODS = {  # key: (label, {fraction: seed-0 run})
    'bm30': ('BioMassters (30 ep)', {'1.0': 'efficientnet_early_full_samerecipe',
                                      '0.25': 'efficientnet_early_full_frac0.25_samerecipe',
                                      '0.1': 'efficientnet_early_full_frac0.1_samerecipe'}),
    'random': ('Random (30 ep)', {'1.0': 'efficientnet_early_scratch_random',
                                  '0.25': 'efficientnet_early_scratch_random_frac0.25',
                                  '0.1': 'efficientnet_early_scratch_random_frac0.1'}),
    'imagenet': ('ImageNet (30 ep)', {'1.0': 'efficientnet_early_scratch_imagenet_enc1',
                                      '0.25': 'efficientnet_early_scratch_imagenet_frac0.25_enc1',
                                      '0.1': 'efficientnet_early_scratch_imagenet_frac0.1_enc1'}),
    'bm8': ('BioMassters (8 ep, paper)', {'1.0': 'efficientnet_early_full'}),
}
SEED_PAIRS = [('bm30', 'random'), ('bm30', 'imagenet'), ('imagenet', 'random'), ('bm8', 'bm30')]


def seed_run(base, s):
    return base if s == 0 else f'{base}_seed{s}'


def load_seeds(base):
    """Prediction matrix (plots x seeds) aligned on chip_id, or None if any seed is missing."""
    frames = [load(seed_run(base, s)) for s in SEEDS]
    if any(f is None for f in frames):
        return None
    ref = frames[0].sort_values('chip_id').reset_index(drop=True)
    preds = []
    for f in frames:
        f = f.sort_values('chip_id').reset_index(drop=True)
        assert f.chip_id.equals(ref.chip_id) and f.fold.equals(ref.fold) and np.allclose(f.agb_mg_ha, ref.agb_mg_ha)
        preds.append(f.pred.to_numpy())
    return ref[['chip_id', 'province_code', 'fold', 'agb_mg_ha']], np.column_stack(preds)


def prov_sse(meta, P):
    """Per-province squared-error sums (provinces x seeds) and plot counts."""
    e2 = (P - meta.agb_mg_ha.to_numpy()[:, None]) ** 2
    g = pd.DataFrame(e2).groupby(meta.province_code.to_numpy())
    return g.sum().to_numpy(), g.size().to_numpy()


rng_seeds = np.random.default_rng(20261122)
md += ['## Seeds (0-2): RMSE per seed, mean and range', '',
       'Seed-ens. = RMSE of the prediction averaged over the three seeds.', '',
       '| Fraction | Method | Seed 0 | Seed 1 | Seed 2 | Mean | Range | Seed-ens. | Mean bias |',
       '|---|---|---|---|---|---|---|---|---|']
res['seeds'] = {'summary': {}, 'pairs': []}
loaded = {}
for frac in ['1.0', '0.25', '0.1']:
    for key, (lab, runs_by_frac) in METHODS.items():
        if frac not in runs_by_frac:
            continue
        got = load_seeds(runs_by_frac[frac])
        if got is None:
            continue
        meta, P = got
        loaded[(key, frac)] = (meta, P)
        y = meta.agb_mg_ha.to_numpy()
        r = [float(np.sqrt(np.mean((P[:, i] - y) ** 2))) for i in range(len(SEEDS))]
        ens = float(np.sqrt(np.mean((P.mean(1) - y) ** 2)))
        bias = float((P - y[:, None]).mean())
        # Table IV reports three-seed means of R2 and r as well as RMSE and bias.
        per_seed = [metrics(y, P[:, i]) for i in range(len(SEEDS))]
        res['seeds']['summary'].setdefault(frac, {})[key] = dict(
            label=lab, run=runs_by_frac[frac], rmse_per_seed=r, rmse_mean=float(np.mean(r)),
            rmse_min=min(r), rmse_max=max(r), rmse_seed_ensemble=ens, bias_mean=bias,
            r2_mean=float(np.mean([m['r2'] for m in per_seed])), r_mean=float(np.mean([m['r'] for m in per_seed])))
        md.append(f"| {frac} | {lab} | {r[0]:.2f} | {r[1]:.2f} | {r[2]:.2f} | {np.mean(r):.2f} | "
                  f"{min(r):.2f}–{max(r):.2f} | {ens:.2f} | {bias:+.2f} |")
md += ['', '## Seeds: paired comparisons (matched seeds, averaged)', '',
       'Delta = mean over seeds of RMSE(ref) - RMSE(run); negative = reference better. 95% interval: '
       'province bootstrap (2000 draws) of the seed-averaged delta. Prov. = provinces where the '
       'reference has lower seed-mean RMSE.', '',
       '| Fraction | Reference | Run | Delta [95% CI] | Per-seed deltas | Prov. ref better |',
       '|---|---|---|---|---|---|']
for frac in ['1.0', '0.25', '0.1']:
    for a, b in SEED_PAIRS:
        if (a, frac) not in loaded or (b, frac) not in loaded:
            continue
        (ma, Pa), (mb, Pb) = loaded[(a, frac)], loaded[(b, frac)]
        assert ma.chip_id.equals(mb.chip_id)
        sa, n = prov_sse(ma, Pa)
        sb, _ = prov_sse(mb, Pb)
        rm = lambda S, N: np.sqrt(S.sum(0) / N.sum())  # per-seed pooled RMSE
        per_seed = rm(sa, n) - rm(sb, n)
        draws = []
        for _ in range(B):
            i = rng_seeds.integers(0, len(n), len(n))
            draws.append(float(np.mean(rm(sa[i], n[i]) - rm(sb[i], n[i]))))
        prov_better = int(np.sum(np.sqrt(sa / n[:, None]).mean(1) < np.sqrt(sb / n[:, None]).mean(1)))
        p = dict(fraction=frac, ref=a, other=b, n=int(n.sum()), n_prov=len(n), point=float(per_seed.mean()),
                 per_seed=per_seed.tolist(), ci95=np.quantile(draws, [.025, .975]).tolist(),
                 provinces_ref_better=prov_better)
        res['seeds']['pairs'].append(p)
        md.append(f"| {frac} | {METHODS[a][0]} | {METHODS[b][0]} | {p['point']:+.2f} "
                  f"[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] | "
                  f"{', '.join(f'{v:+.2f}' for v in per_seed)} | {prov_better}/{len(n)} |")
md.append('')

# Original (unadapted) model vs eight-epoch full fine-tuning, over the three fine-tuning seeds.
audit = OUT / 'ifn_plot_predictions.csv'
if ('bm8', '1.0') in loaded and audit.exists():
    mf, Pf = loaded[('bm8', '1.0')]
    orig = pd.read_csv(audit).set_index('chip_id').loc[mf.chip_id, 'efficientnet_early'].to_numpy()
    so, n = prov_sse(mf, orig[:, None])
    sf, _ = prov_sse(mf, Pf)
    rm = lambda S, N: np.sqrt(S.sum(0) / N.sum())
    per_seed = rm(so, n) - rm(sf, n)
    draws = []
    for _ in range(B):
        i = rng_seeds.integers(0, len(n), len(n))
        draws.append(float(np.mean(rm(so[i], n[i]) - rm(sf[i], n[i]))))
    gain = np.sqrt(so / n[:, None]) - np.sqrt(sf / n[:, None])  # provinces x seeds
    res['seeds']['orig_vs_bm8'] = dict(
        per_seed_reduction=per_seed.tolist(), mean_reduction=float(per_seed.mean()),
        mean_reduction_pct=float(100 * per_seed.mean() / rm(so, n)[0]),
        ci95=np.quantile(draws, [.025, .975]).tolist(),
        provinces_improved_per_seed=(gain > 0).sum(0).tolist(),
        provinces_worse_every_seed=int((gain < 0).all(1).sum()))
    md += ['## Seeds: original vs eight-epoch full fine-tuning', '',
           f"RMSE reduction per seed: {', '.join(f'{v:.2f}' for v in per_seed)}; mean {per_seed.mean():.2f} "
           f"({res['seeds']['orig_vs_bm8']['mean_reduction_pct']:.1f}%), 95% interval "
           f"[{np.quantile(draws, .025):.2f}, {np.quantile(draws, .975):.2f}]. Provinces improved per seed: "
           f"{(gain > 0).sum(0).tolist()} of {len(n)}; worse in every seed: {int((gain < 0).all(1).sum())}.", '']

OUT.mkdir(exist_ok=True)
json.dump(res, open(OUT / 'ifn_pretraining.json', 'w'), indent=1)
(OUT / 'ifn_pretraining.md').write_text('\n'.join(md) + '\n')
print('\n'.join(md))
