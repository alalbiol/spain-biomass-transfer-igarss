"""
M5 prep: recompute the Finland benchmark at the SAME 100 m block resolution
used for the Spain zero-shot evaluation, so the two sides of the comparison
table are on comparable footing.

Why this is needed: `use_case_1/test.py`/`test_ensemble.py` compute RMSE/MAE/
R2/SCI/over-under at native 10 m (256x256 px, pixel-accumulated). Spain has no
native-resolution ground truth -- only ESA CCI AGB at ~100 m -- so its
evaluation (`evaluate_zero_shot_blocks.py`) pools predictions to 25x25 100 m
blocks before comparing. Comparing Finland's native-10m RMSE/R2 directly
against Spain's pooled-100m RMSE/R2 is statistically misleading: pooling
reduces RMSE and inflates R2 by averaging out pixel noise, independent of any
real accuracy difference. This script pools BOTH the Finland prediction and
its real AGBM ground truth the same 10x10->1 way (last 6 px dropped, matching
`evaluate_zero_shot_blocks.py:pool()` exactly) and recomputes the identical
metric set with the identical constants (`use_case_1/utils.py:CustomModel`:
EPS=10 Mg/ha, OVER_RATIO=2, UNDER_RATIO=0.5, SSIM win=5, data_range=500).

What this does NOT fix, and must stay a stated caveat in the paper: even at
matched spatial support, Finland's reference (pooled BioMassters/ALS-derived
AGBM) and Spain's reference (ESA CCI AGB, itself model-derived from SAR+
optical+ML) are produced by fundamentally different processes. Resolution-
matching removes the noise-averaging confound, not the reference-source one.

Per-model 25x25 pooled predictions are saved (tiny: ~2.5 KB/chip) so the
ensemble can be computed post-hoc from disk without reloading 5 checkpoints
at once, and so figures can reuse them later without rerunning inference.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.numpy import load_file
from skimage import io
from skimage.metrics import structural_similarity

UC1 = Path(__file__).resolve().parents[1] / "use_case_1"
sys.path.insert(0, str(UC1))

from inference import Config  # noqa: E402
from models import UnetVFLOW  # noqa: E402
from utils import CustomModel  # noqa: E402

BACKBONES = ["convnextv2", "efficientnet", "efficientnetv2", "mobilenetv4", "resnetrs101"]
MODES = ["early", "late"]
EPS, OVER, UNDER = CustomModel.OVER_UNDER_EPS, CustomModel.OVER_RATIO, CustomModel.UNDER_RATIO
SSIM_WIN, SSIM_RANGE = CustomModel.SSIM_WIN_SIZE, CustomModel.SSIM_DATA_RANGE


def pool(a):
    return a[:250, :250].reshape(25, 10, 25, 10).mean(axis=(1, 3))


def configure(backbone):
    Config.backbone = backbone
    if "efficientnet" in backbone or "resnet" in backbone:
        Config.out_indices, Config.dec_channels = [0, 1, 2, 3, 4], [256, 160, 128, 64, 32]
    elif "convnext" in backbone or "swin" in backbone:
        Config.out_indices, Config.dec_channels = [0, 1, 2, 3], [256, 128, 64, 32]
    else:
        Config.out_indices, Config.dec_channels = [0, 1, 2, 3, 4], [256, 128, 64, 32, 16]
    return Config()


class FinlandDS(torch.utils.data.Dataset):
    def __init__(self, chip_ids, features_dir, labels_dir):
        self.chip_ids = chip_ids
        self.features_dir = Path(features_dir)
        self.labels_dir = Path(labels_dir)

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, i):
        cid = self.chip_ids[i]
        d = load_file(str(self.features_dir / f"{cid}.safetensors"))
        imgs = torch.from_numpy(d["features"]).float()
        mask = torch.zeros(12, dtype=torch.bool)
        mask[d["missing"]] = True
        target = torch.from_numpy(io.imread(self.labels_dir / f"{cid}_agbm.tif").astype("float32"))
        return imgs, mask, target, i


def metrics(P, R, ssim_scores):
    err = P - R
    sst = np.sum((R - R.mean()) ** 2)
    sig = R >= EPS
    return {
        "n_blocks": len(R),
        "rmse": np.sqrt((err**2).mean()),
        "mae": np.abs(err).mean(),
        "r2": 1 - np.sum(err**2) / sst if sst > 0 else np.nan,
        "pearson_r": np.corrcoef(P, R)[0, 1],
        "frac_over": np.mean(P > OVER * np.maximum(R, EPS)),
        "frac_under": np.mean(sig & (P < UNDER * R)),
        "sci_100m": np.nanmean(ssim_scores),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metafile", default="/media/HD/Data/BioMassters/test_meta_chip_ids.csv")
    p.add_argument("--features_dir", default="/media/HD/Data/BioMassters/test_features")
    p.add_argument("--labels_dir", default="/media/HD/Data/BioMassters/test_agbm")
    p.add_argument("--model_dir", default="/media/HD/checkpoints/AI4DIS/case1")
    p.add_argument("--out_dir", default="/media/HD/Data/BioMassters/pooled_preds")
    p.add_argument(
        "--results_csv",
        default="/home/alalbiol/Programacion/AI4DIS/use_case_1/context_logs/finland_benchmark/finland_benchmark_results_100m.csv",
    )
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")
    chip_ids = pd.read_csv(args.metafile).chip_id.tolist()
    if args.limit:
        chip_ids = chip_ids[: args.limit]

    ds = FinlandDS(chip_ids, args.features_dir, args.labels_dir)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers, prefetch_factor=4)

    out_root = Path(args.out_dir)
    gt_dir = out_root / "_ground_truth_pooled"
    gt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for backbone in BACKBONES:
        for mode in MODES:
            name = f"{backbone}_{mode}"
            ckpt = Path(args.model_dir) / f"{name}.ckpt"
            out = out_root / name
            out.mkdir(parents=True, exist_ok=True)
            todo = [c for c in chip_ids if not (out / f"{c}.npy").exists()]

            if todo:
                model = UnetVFLOW(configure(backbone), mode=mode)
                lm = CustomModel.load_from_checkpoint(ckpt, model=model, batch_size=1).eval().to(device)
                todo_set = set(todo)
                t0, done = time.time(), 0
                for imgs, mask, target, idx in loader:
                    cid = chip_ids[int(idx)]
                    if cid not in todo_set:
                        continue
                    with torch.no_grad():
                        pred = lm(imgs.to(device), mask.to(device)).squeeze().cpu().numpy().astype("float64")
                    pooled_pred = pool(pred)
                    np.save(out / f"{cid}.npy", pooled_pred.astype("float32"))
                    gt_path = gt_dir / f"{cid}.npy"
                    if not gt_path.exists():
                        np.save(gt_path, pool(target.squeeze().numpy().astype("float64")).astype("float32"))
                    done += 1
                    if done % 200 == 0 or done == len(todo):
                        el = time.time() - t0
                        print(f"[{name}] {done}/{len(todo)} ({el/done:.2f}s/chip, ~{el/done*(len(todo)-done)/60:.1f} min left)", flush=True)
                del lm, model
                torch.cuda.empty_cache()

            P = np.stack([np.load(out / f"{c}.npy").astype("float64") for c in chip_ids])
            R = np.stack([np.load(gt_dir / f"{c}.npy").astype("float64") for c in chip_ids])
            ssim = [structural_similarity(R[i], P[i], data_range=SSIM_RANGE, win_size=SSIM_WIN) for i in range(len(chip_ids))]
            m = metrics(P.reshape(-1), R.reshape(-1), ssim)
            rows.append({"model": backbone, "mode": mode, **m})
            print(f"[{name}] RMSE={m['rmse']:.2f} R2={m['r2']:.3f} r={m['pearson_r']:.3f} SCI={m['sci_100m']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(args.results_csv, index=False)

    for mode in MODES:
        members = [f"{b}_{mode}" for b in BACKBONES]
        P = np.stack([np.mean([np.load(out_root / m / f"{c}.npy").astype("float64") for m in members], axis=0) for c in chip_ids])
        R = np.stack([np.load(gt_dir / f"{c}.npy").astype("float64") for c in chip_ids])
        ssim = [structural_similarity(R[i], P[i], data_range=SSIM_RANGE, win_size=SSIM_WIN) for i in range(len(chip_ids))]
        m = metrics(P.reshape(-1), R.reshape(-1), ssim)
        std = np.std(
            np.stack([[np.load(out_root / m_ / f"{c}.npy").astype("float64") for c in chip_ids] for m_ in members]), axis=0
        )
        m["uncertainty_ratio"] = float(std.mean() / P.mean())
        rows.append({"model": "ensemble", "mode": mode, **m})
        print(f"[ensemble_{mode}] RMSE={m['rmse']:.2f} R2={m['r2']:.3f} r={m['pearson_r']:.3f} SCI={m['sci_100m']:.3f} unc={m['uncertainty_ratio']:.2%}", flush=True)

    pd.DataFrame(rows).to_csv(args.results_csv, index=False)
    print(f"DONE. Results: {args.results_csv}", flush=True)


if __name__ == "__main__":
    main()
