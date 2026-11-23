"""
No-pretraining baselines for the IFN4 pine experiment: tree ensembles on
plot-level Sentinel-1/2 statistics, trained only on Spanish inventory plots.

They answer the practitioner's question the fine-tuning result leaves open:
does starting from the Finland checkpoint beat a standard model fitted to the
same local labels?

Protocol is identical to finetune_ifn.py: the same province-grouped outer
folds, and inside each fold the same shuffled 90% training / 10% inner
validation split (same random_state). Hyperparameters are fixed up front; the
only selection is the boosting iteration, chosen on the inner validation plots
(the analogue of epoch selection). The random forest uses no selection.

Features, per month (12) and channel (4 S1 + 10 S2 + cloud probability):
  * mean over the 25 m plot disk (the footprint the neural models are scored on);
  * mean over a 50 m radius neighbourhood (some spatial context);
plus the per-channel median of the valid months for both. Months flagged
missing are NaN (both models handle NaN natively).

Output: <DATA>/finetune/tabular_{hgb,rf}/oof.csv in the finetune_ifn.py format.
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from ifn_common import CACHE, DATA, N_FOLDS, PX, disk_weights, metrics, plots

HALF = 8  # 16 x 16 px window around the chip centre is enough for a 50 m radius


def window_weights(radius_m):
    w = disk_weights(PX, radius_m=radius_m)
    c = PX // 2
    w = w[c - HALF:c + HALF, c - HALF:c + HALF]
    assert np.isclose(w.sum(), 1.0), "disk does not fit the window"
    return w


def features(chip_ids):
    s1 = np.load(CACHE / "s1.npy", mmap_mode="r")
    s2 = np.load(CACHE / "s2.npy", mmap_mode="r")
    missing = np.load(CACHE / "missing.npy")
    index = dict(zip(pd.read_csv(CACHE / "ids.csv").chip_id, range(s1.shape[0])))
    c = PX // 2
    ws = {"d25": window_weights(25.0), "d50": window_weights(50.0)}
    rows = []
    for cid in chip_ids:
        i = index[cid]
        a1 = np.array(s1[i, :, :, c - HALF:c + HALF, c - HALF:c + HALF], dtype=np.float32)
        a1[a1 < -9000] = np.nan  # S1 nodata
        a2 = np.array(s2[i, :, :, c - HALF:c + HALF, c - HALF:c + HALF], dtype=np.float32)
        x = np.concatenate([a1, a2], axis=1)  # [12, 15, 16, 16]
        x[missing[i]] = np.nan
        feats = []
        for w in ws.values():
            m = np.nansum(x * w, axis=(-2, -1)) / np.nansum(np.isfinite(x) * w, axis=(-2, -1))  # [12, 15]
            feats += [m.ravel(), np.nanmedian(m, axis=0)]
        rows.append(np.concatenate(feats))
    return np.vstack(rows)


def main():
    df = plots()
    print(f"{len(df)} plots; extracting features", flush=True)
    X = features(df.chip_id)
    print(f"feature matrix {X.shape}, NaN share {np.isnan(X).mean():.3f}", flush=True)
    models = {
        "tabular_hgb": lambda: HistGradientBoostingRegressor(
            max_iter=1500, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=False, random_state=0),
        "tabular_rf": lambda: RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, max_features=0.33, n_jobs=-1, random_state=0),
    }
    summary = {}
    for run, make in models.items():
        oof, info = [], {}
        for fold in range(N_FOLDS):
            test = df[df.fold == fold]
            rest = df[df.fold != fold].sample(frac=1.0, random_state=fold)  # as finetune_ifn.py
            n_va = int(0.1 * len(rest))
            va, tr = rest.iloc[:n_va], rest.iloc[n_va:]
            m = make().fit(X[tr.index], tr.agb_mg_ha)
            if run == "tabular_hgb":
                va_rmse = [np.sqrt(np.mean((p - va.agb_mg_ha) ** 2)) for p in m.staged_predict(X[va.index])]
                best = int(np.argmin(va_rmse))
                pred = next(p for k, p in enumerate(m.staged_predict(X[test.index])) if k == best)
                info[fold] = {"best_iter": best + 1, "inner_val_rmse": float(va_rmse[best])}
            else:
                pred = m.predict(X[test.index])
            part = test[["chip_id", "plot_id", "province_code", "fold", "main_pine", "agb_mg_ha"]].assign(pred=pred)
            oof.append(part)
            s = metrics(pred, test.agb_mg_ha)
            print(f"  {run} fold {fold}: train {len(tr)} test {len(test)} RMSE {s['rmse']:.2f} "
                  f"bias {s['bias']:.2f} R2 {s['r2']:.3f} {info.get(fold, '')}", flush=True)
        allp = pd.concat(oof)
        out = DATA / "finetune" / run
        out.mkdir(parents=True, exist_ok=True)
        allp.to_csv(out / "oof.csv", index=False)
        json.dump({"features": int(X.shape[1]), "params": make().get_params(), "folds": info},
                  open(out / "args.json", "w"), indent=1, default=str)
        summary[run] = {k: round(float(v), 3) for k, v in metrics(allp.pred, allp.agb_mg_ha).items()}
        print(run, json.dumps(summary[run]), flush=True)


if __name__ == "__main__":
    main()
