"""
Block-level (100 m) evaluation of the zero-shot Spain predictions.

Each 10 m prediction (256x256) is average-pooled 10x10 into 25x25 blocks of
100 m (the last 6 px are dropped) and compared with the ESA CCI AGB 100 m grid
fetched by fetch_cci_100m.py. Only blocks where CCI has a value are used --
CCI masks non-forest/water, and comparing the model over masked areas against
nothing would bias the result (the earlier chip-mean comparison in
evaluate_zero_shot.py averaged the prediction over ALL pixels but the CCI
scalar only over valid ones, which inflates the apparent underestimation).

Metric definitions follow use_case_1/utils.py (pooled RMSE/MAE/R2, over/under
fractions with eps=10 Mg/ha and ratios 2 and 0.5, SSIM-based SCI with a 5x5
window and data_range=500). NOTE: SCI here is on 25x25 blocks vs a 100 m
reference, so it is NOT comparable to the 10 m SCI on the Finland test set.

Also reported:
  - within-chip Pearson r (pattern captured inside a chip), with a control
    that flips the CCI grid vertically -- if georeferencing is right the
    aligned value must clearly beat the control;
  - CCI's own SD as an indicative reference-noise floor.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io
from skimage.metrics import structural_similarity

NODATA = -9999
EPS, OVER, UNDER = 10.0, 2.0, 0.5


def pool(pred):
    return pred[:250, :250].reshape(25, 10, 25, 10).mean(axis=(1, 3))


def load_cci(path):
    a = io.imread(path)
    assert a.shape == (25, 25, 2), a.shape
    agb, sd = a[..., 0].astype("float64"), a[..., 1].astype("float64")
    return np.where(agb == NODATA, np.nan, agb), np.where(sd == NODATA, np.nan, sd)


def within_chip_r(pred, ref):
    m = np.isfinite(ref)
    if m.sum() < 30 or np.std(pred[m]) == 0 or np.std(ref[m]) == 0:
        return np.nan
    return np.corrcoef(pred[m], ref[m])[0, 1]


def summarize(P, R, SD, ssim_scores, within_r, within_r_flip):
    """P, R: 1-D arrays of valid block preds/refs pooled over chips."""
    err = P - R
    sst = np.sum((R - R.mean()) ** 2)
    sig = R >= EPS
    return {
        "n_chips": len(ssim_scores),
        "n_blocks": len(R),
        "ref_mean": R.mean(),
        "bias": err.mean(),
        "rel_bias_pct": 100 * err.mean() / R.mean(),
        "mae": np.abs(err).mean(),
        "rmse": np.sqrt((err**2).mean()),
        "r2": 1 - np.sum(err**2) / sst,
        "pearson_r": np.corrcoef(P, R)[0, 1],
        "frac_over": np.mean(P > OVER * np.maximum(R, EPS)),
        "frac_under": np.mean(sig & (P < UNDER * R)),
        "sci_100m": np.nanmean(ssim_scores),
        "within_chip_r": np.nanmean(within_r),
        "within_chip_r_flipped_ctrl": np.nanmean(within_r_flip),
        "cci_sd_rms": np.sqrt(np.nanmean(SD**2)) if np.isfinite(SD).any() else np.nan,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metafile", default="/media/HD/Data/agb_spain/validation/metadata.csv")
    p.add_argument("--pred_dir", default="/media/HD/Data/agb_spain/zero_shot")
    p.add_argument("--cci_dir", default="/media/HD/Data/agb_spain/cci_100m")
    args = p.parse_args()

    meta = pd.read_csv(args.metafile)
    pred_root = Path(args.pred_dir) / "preds"
    names = sorted(d.name for d in pred_root.iterdir() if d.is_dir())
    early = [n for n in names if n.endswith("_early")]
    late = [n for n in names if n.endswith("_late")]
    groups = {"ens_early": early, "ens_late": late, "ens_all": early + late}

    # blocks[model][chip] -> 25x25 pooled prediction
    ref, sd = {}, {}
    blocks = {n: {} for n in names}
    for cid in meta.chip_id:
        ref[cid], sd[cid] = load_cci(Path(args.cci_dir) / f"{cid}.tif")
        for n in names:
            blocks[n][cid] = pool(np.load(pred_root / n / f"{cid}.npy").astype("float64"))
    for g, members in groups.items():
        blocks[g] = {cid: np.mean([blocks[m][cid] for m in members], axis=0) for cid in meta.chip_id}
        blocks[g + "_disagreement"] = {cid: np.std([blocks[m][cid] for m in members], axis=0) for cid in meta.chip_id}

    mi = meta.set_index("chip_id")
    strata = {"ALL": lambda c: True}
    for t in sorted(meta.forest_type.unique()):
        strata[f"type={t}"] = lambda c, t=t: mi.forest_type[c] == t
    for b in ["low", "medium", "high"]:
        strata[f"bin={b}"] = lambda c, b=b: mi.agb_bin[c] == b

    models = names + list(groups)
    rows = []
    for m in models:
        for s, sel in strata.items():
            chips = [c for c in meta.chip_id if sel(c) and np.isfinite(ref[c]).sum() >= 30]
            P = np.concatenate([blocks[m][c][np.isfinite(ref[c])] for c in chips])
            R = np.concatenate([ref[c][np.isfinite(ref[c])] for c in chips])
            SD = np.concatenate([sd[c][np.isfinite(ref[c])] for c in chips])
            ssim, wr, wrf = [], [], []
            for c in chips:
                r = ref[c]
                filled = np.where(np.isfinite(r), r, 0.0)
                ssim.append(structural_similarity(filled, blocks[m][c], data_range=500.0, win_size=5))
                wr.append(within_chip_r(blocks[m][c], r))
                wrf.append(within_chip_r(blocks[m][c], r[::-1]))
            rows.append({"model": m, "stratum": s, **summarize(P, R, SD, ssim, wr, wrf)})
    res = pd.DataFrame(rows)
    res.to_csv(Path(args.pred_dir) / "zero_shot_metrics_block_level.csv", index=False)

    allr = res[res.stratum == "ALL"]
    n_valid = sum(np.isfinite(r).sum() for r in ref.values())
    print(f"blocks with CCI value: {n_valid}/{len(meta)*625} ({100*n_valid/(len(meta)*625):.1f}%) over {len(meta)} chips")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format)
    cols = ["n_blocks", "ref_mean", "bias", "rel_bias_pct", "mae", "rmse", "r2", "pearson_r", "frac_over", "frac_under",
            "sci_100m", "within_chip_r", "within_chip_r_flipped_ctrl"]
    print(allr.set_index("model")[cols].to_string())
    print("\nens_all by stratum:")
    print(res[res.model == "ens_all"].set_index("stratum")[["n_chips"] + cols].to_string())
    print(f"\nCCI SD (rms over valid blocks, indicative reference noise floor): {allr.cci_sd_rms.iloc[0]:.1f} Mg/ha")


if __name__ == "__main__":
    main()
