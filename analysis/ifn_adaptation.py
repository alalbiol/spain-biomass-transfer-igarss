"""Table IV, upper block: unadapted model, linear recalibration, decoder and full fine-tuning
on the 6117 IFN4 pine plots (province-grouped 5-fold CV), with the sensitivity checks quoted
in Section IV-C. Reads saved out-of-fold predictions only; writes outputs/ifn_adaptation.json.
"""
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import IFN, OUT, metrics, read


def main():
    plots = read(IFN / 'plots.csv')
    z = read(IFN / 'zero_shot/plot_preds.csv')
    for t in (plots, z):
        assert t.chip_id.is_unique
    d = plots.merge(z, on='chip_id', validate='one_to_one')
    for name, col in [('efficientnet_early_full', 'full'), ('efficientnet_early_head', 'decoder')]:
        o = read(IFN / 'finetune' / name / 'oof.csv')
        assert o.chip_id.is_unique and o.fold.nunique() == 5
        assert o.groupby('province_code').fold.nunique().max() == 1
        assert np.allclose(o.agb_mg_ha, o.chip_id.map(plots.set_index('chip_id').agb_mg_ha))
        if 'fold' in d:
            assert (o.set_index('chip_id').fold.reindex(d.chip_id).to_numpy() == d.fold.to_numpy()).all()
        keep = ['chip_id', 'pred'] + ([] if 'fold' in d else ['fold'])
        d = d.merge(o[keep].rename(columns={'pred': col}), on='chip_id', validate='one_to_one')
    assert np.isfinite(d[['efficientnet_early', 'full', 'decoder', 'agb_mg_ha']]).all().all()

    # Linear recalibration fitted on the outer-training provinces of each fold.
    d['linear'] = np.nan
    for f in sorted(d.fold.unique()):
        tr = d.fold != f
        b, a = np.polyfit(d.loc[tr, 'efficientnet_early'], d.loc[tr, 'agb_mg_ha'], 1)
        d.loc[~tr, 'linear'] = a + b * d.loc[~tr, 'efficientnet_early']

    methods = ['efficientnet_early', 'linear', 'decoder', 'full']
    score = lambda t: {m: metrics(t.agb_mg_ha.to_numpy(), t[m].to_numpy()) for m in methods}
    result = {'n_plots': len(d),
              'cohorts': {'all': score(d), 'exact_year': score(d[d.survey_year == d.image_year])}}
    prov = []
    for p, t in d.groupby('province_code'):
        r = {'province': p, 'n': len(t), 'fold': int(t.fold.iloc[0])}
        r.update({m: metrics(t.agb_mg_ha.to_numpy(), t[m].to_numpy())['rmse'] for m in methods})
        prov.append(r)
    result['provinces'] = prov
    result['improved_provinces'] = sum(r['full'] < r['efficientnet_early'] for r in prov)

    rng = np.random.default_rng(20261117)
    groups = [g for _, g in d.groupby('province_code')]
    draws = []
    for _ in range(2000):
        t = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        y = t.agb_mg_ha.to_numpy()
        base = metrics(y, t.efficientnet_early.to_numpy())['rmse']
        draws.append([metrics(y, t[m].to_numpy())['rmse'] - base for m in methods[1:]])
    result['paired_rmse_change_ci'] = {m: np.quantile(np.array(draws)[:, i], [.025, .975]).tolist()
                                       for i, m in enumerate(methods[1:])}
    result['leave_one_province_out_gain'] = [
        float(metrics(t.agb_mg_ha.to_numpy(), t.efficientnet_early.to_numpy())['rmse']
              - metrics(t.agb_mg_ha.to_numpy(), t.full.to_numpy())['rmse'])
        for p in d.province_code.unique() for t in [d[d.province_code != p]]]
    # Cross-province context overlap: drop plots within the half-diagonal of a 112-px (1.12 km)
    # input of a plot in another fold (a conservative circular envelope around the square input).
    xy = np.c_[np.deg2rad(d.lon) * 6371000 * np.cos(np.deg2rad(d.lat.mean())), np.deg2rad(d.lat) * 6371000]
    pairs = cKDTree(xy).query_pairs(1120 * np.sqrt(2), output_type='ndarray')
    fold = d.fold.to_numpy()
    cross = pairs[fold[pairs[:, 0]] != fold[pairs[:, 1]]]
    result['cross_fold_nearby_pairs_context_envelope'] = int(len(cross))
    result['context_separated_sensitivity'] = score(d.drop(index=d.index[np.unique(cross.ravel())]))
    cci = read(IFN / 'cci_at_plots.csv')
    historical = d.merge(cci[['chip_id', 'cci_agb']], on='chip_id', validate='one_to_one')
    result['cci_available_cohort'] = score(historical[historical.cci_agb.notna()])
    result['duplicate_coordinates'] = int(d.duplicated(['lon', 'lat']).sum())

    OUT.mkdir(exist_ok=True)
    (OUT / 'ifn_adaptation.json').write_text(json.dumps(result, indent=2) + '\n')
    d.to_csv(OUT / 'ifn_plot_predictions.csv', index=False)  # read by ifn_pretraining.py
    for m, label in zip(methods, ['Original', 'Linear recalibration', 'Decoder fine-tuning', 'Full fine-tuning']):
        r = result['cohorts']['all'][m]
        print(f"{label:22s} RMSE {r['rmse']:.2f}  bias {r['bias']:.2f}  R2 {r['r2']:.3f}")
    print('provinces improved by full fine-tuning:', result['improved_provinces'], 'of', len(prov))


if __name__ == '__main__':
    main()
