"""
Paired comparison of zero-shot block-level performance across S2 composite
recipes on the SAME subset of chips (see export_bm_style_chips.py).

    python compare_composites.py old=/.../zero_shot/preds bm=/.../zero_shot_bm/preds [bm_c25=...]

The first label is the baseline. Metrics are computed on 100 m blocks where CCI
has a value (same pooling/reference as evaluate_zero_shot_blocks.py). Reported
for early-fusion members, late-fusion members and the two ensembles; 95% CIs
are chip-level bootstrap on the paired difference (variant - baseline).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_zero_shot_blocks import load_cci, pool

RNG = np.random.default_rng(0)


def metrics(P, R, within):
    err = P - R
    return {
        "bias": err.mean(),
        "rmse": np.sqrt((err**2).mean()),
        "r2": 1 - (err**2).sum() / ((R - R.mean()) ** 2).sum(),
        "r": np.corrcoef(P, R)[0, 1],
        "within_r": np.nanmean(within),
        "under_frac": np.mean((R >= 10) & (P < 0.5 * R)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("variants", nargs="+", help="label=preds_dir (first is the baseline)")
    p.add_argument("--subset", default="/media/HD/Data/agb_spain/validation_bm/metadata_subset.csv")
    p.add_argument("--cci_dir", default="/media/HD/Data/agb_spain/cci_100m")
    p.add_argument("--n_boot", type=int, default=500)
    args = p.parse_args()
    labels = [v.split("=")[0] for v in args.variants]
    roots = {v.split("=")[0]: Path(v.split("=")[1]) for v in args.variants}
    sub = pd.read_csv(args.subset)

    def members(root, mode=None):
        names = sorted(d.name for d in root.iterdir() if d.is_dir())
        return [n for n in names if mode is None or n.endswith(f"_{mode}")]

    groups = {"early": "early", "late": "late", "all": None}
    ref, blocks = {}, {l: {g: {} for g in groups} for l in labels}
    for cid in sub.chip_id:
        r = load_cci(Path(args.cci_dir) / f"{cid}.tif")[0]
        if np.isfinite(r).sum() < 30:
            continue
        ref[cid] = r
        for l in labels:
            for g, mode in groups.items():
                blocks[l][g][cid] = np.mean(
                    [pool(np.load(roots[l] / n / f"{cid}.npy").astype("float64")) for n in members(roots[l], mode)], axis=0
                )
    chips = list(ref)
    print(f"chips: {len(chips)}; strata: {sub.groupby(['forest_type', 'agb_bin']).size().to_dict()}\n")

    def compute(label, g, idx):
        P = np.concatenate([blocks[label][g][chips[i]][np.isfinite(ref[chips[i]])] for i in idx])
        R = np.concatenate([ref[chips[i]][np.isfinite(ref[chips[i]])] for i in idx])
        within = []
        for i in idx:
            m = np.isfinite(ref[chips[i]])
            a, b = blocks[label][g][chips[i]][m], ref[chips[i]][m]
            within.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
        return metrics(P, R, within)

    all_idx = np.arange(len(chips))
    boots = [RNG.integers(len(chips), size=len(chips)) for _ in range(args.n_boot)]
    rows = []
    for g in groups:
        base = compute(labels[0], g, all_idx)
        for l in labels:
            m = compute(l, g, all_idx)
            row = {"ensemble": f"ens_{g}", "variant": l, **{k: round(v, 3) for k, v in m.items()}}
            if l != labels[0]:
                bd = []
                for b in boots[: min(args.n_boot, 200)]:
                    mv, mb = compute(l, g, b), compute(labels[0], g, b)
                    bd.append({k: mv[k] - mb[k] for k in ["bias", "rmse", "r2", "r", "within_r"]})
                bd = pd.DataFrame(bd)
                for k in ["bias", "rmse", "r2", "r", "within_r"]:
                    d = m[k] - base[k]
                    lo, hi = np.percentile(bd[k], [2.5, 97.5])
                    row[f"d_{k}"] = f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]" + ("*" if lo > 0 or hi < 0 else "")
            rows.append(row)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(pd.DataFrame(rows).to_string(index=False))
    levels = {"CCI<50": (0, 50), "50-120": (50, 120), ">=120": (120, 1e9)}
    rows2 = []
    for g in groups:
        for l in labels:
            P = np.concatenate([blocks[l][g][c][np.isfinite(ref[c])] for c in chips])
            R = np.concatenate([ref[c][np.isfinite(ref[c])] for c in chips])
            row = {"ensemble": f"ens_{g}", "variant": l}
            for name, (a, b) in levels.items():
                m = (R >= a) & (R < b)
                row[name] = round((P[m] - R[m]).mean(), 1)
            rows2.append(row)
    print("\nmean residual (pred - CCI, Mg/ha) by block-level CCI level (dilution-prone view; same blocks in every variant):")
    print(pd.DataFrame(rows2).to_string(index=False))
    print("\n(d_* = variant - baseline, chip-bootstrap 95% CI, * = CI excludes 0; bias in Mg/ha: closer to 0 is better, "
          "so a POSITIVE d_bias is an improvement when the baseline bias is negative)")


if __name__ == "__main__":
    main()
