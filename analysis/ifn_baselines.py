"""Table IV, no-pretraining baselines: compare full fine-tuning with the tabular models and the
networks trained from random or ImageNet weights.

Reads the audited predictions (original, linear, decoder, full) and every
baseline run present in data/ifn/finetune/<run>/oof.csv for BASELINES, scores
them on their common plots, and gives 2000-draw paired province-bootstrap
intervals for RMSE(full fine-tuning) - RMSE(baseline), the same resampling as
ifn_adaptation.py, which must run first. Writes outputs/ifn_baselines.json.
"""
import json

import numpy as np
import pandas as pd

from common import IFN, OUT
BASELINES = ['tabular_hgb', 'tabular_rf', 'efficientnet_early_scratch_imagenet', 'efficientnet_early_scratch_random']
BINS = [0, 25, 50, 100, 150, np.inf]


def metrics(y, p):
    e = p - y
    return dict(n=len(y), bias=float(e.mean()), rmse=float(np.sqrt(np.mean(e * e))),
                r2=float(1 - np.sum(e * e) / np.sum((y - y.mean()) ** 2)), r=float(np.corrcoef(y, p)[0, 1]))


d = pd.read_csv(OUT / 'ifn_plot_predictions.csv', dtype={'province_code': str})
d = d.rename(columns={'efficientnet_early': 'original'})
methods = ['original', 'linear', 'decoder', 'full']
for run in BASELINES:
    f = IFN / 'finetune' / run / 'oof.csv'
    if f.exists():
        o = pd.read_csv(f)[['chip_id', 'pred']].rename(columns={'pred': run})
        d = d.merge(o, on='chip_id', how='left', validate='one_to_one')
        methods.append(run)
d = d.dropna(subset=methods)
y = d.agb_mg_ha.to_numpy()
res = {'n_plots': len(d), 'n_provinces': int(d.province_code.nunique()),
       'overall': {m: metrics(y, d[m].to_numpy()) for m in methods}}
cls = pd.cut(d.agb_mg_ha, BINS, right=True)
res['by_biomass_class'] = {str(c): {'n': len(t), **{m: {'rmse': metrics(t.agb_mg_ha.to_numpy(), t[m].to_numpy())['rmse'],
                                                         'bias': metrics(t.agb_mg_ha.to_numpy(), t[m].to_numpy())['bias']}
                                                     for m in methods}}
                           for c, t in d.groupby(cls, observed=True)}
res['provinces_full_better'] = {m: int(sum(metrics(t.agb_mg_ha.to_numpy(), t.full.to_numpy())['rmse'] <
                                           metrics(t.agb_mg_ha.to_numpy(), t[m].to_numpy())['rmse']
                                           for _, t in d.groupby('province_code')))
                                for m in methods if m != 'full'}
rng = np.random.default_rng(20261118)
groups = [g for _, g in d.groupby('province_code')]
others = [m for m in methods if m != 'full']
draws = []
for _ in range(2000):
    t = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    yt = t.agb_mg_ha.to_numpy()
    full = metrics(yt, t.full.to_numpy())['rmse']
    draws.append([full - metrics(yt, t[m].to_numpy())['rmse'] for m in others])
draws = np.array(draws)
res['full_minus_method_rmse'] = {m: {'point': res['overall']['full']['rmse'] - res['overall'][m]['rmse'],
                                     'ci95': np.quantile(draws[:, i], [.025, .975]).tolist()}
                                 for i, m in enumerate(others)}
OUT.mkdir(exist_ok=True)
json.dump(res, open(OUT / 'ifn_baselines.json', 'w'), indent=1)
for m in methods:
    s = res['overall'][m]
    extra = res['full_minus_method_rmse'].get(m)
    ci = f"  full-minus {extra['point']:+.2f} [{extra['ci95'][0]:+.2f}, {extra['ci95'][1]:+.2f}]" if extra else ''
    print(f"{m:38s} RMSE {s['rmse']:6.2f} bias {s['bias']:+6.2f} R2 {s['r2']:.3f} r {s['r']:.3f}{ci}")
print('provinces where full beats method:', res['provinces_full_better'])
