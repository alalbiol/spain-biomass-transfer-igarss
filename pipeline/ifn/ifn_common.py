"""Shared pieces for the IFN-plot fine-tuning experiment: chip loading, plot
footprint weights, province folds and checkpoint loading."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
UC1 = HERE.parents[1] / "use_case_1"
sys.path.insert(0, str(UC1))
sys.path.insert(0, str(HERE.parent))

DATA = Path("/media/HD/Data/agb_spain_ifn")
CHIPS = DATA / "chips"
CKPT_DIR = Path("/media/HD/checkpoints/AI4DIS/case1")
BACKBONES = ["convnextv2", "efficientnet", "efficientnetv2", "mobilenetv4", "resnetrs101"]
MODES = ["early", "late"]
CACHE_PX = 112
# IFN_CACHE lets runs read a copy of the cache on a faster disk (the default lives on the HDD).
CACHE = Path(os.environ.get("IFN_CACHE", DATA / f"cache_{CACHE_PX}"))
PX = CACHE_PX  # model input size used everywhere (see build_chip_cache.py)
PLOT_RADIUS_M = 25.0  # IFN4 largest concentric subplot (trees >= 42.5 cm; most of the biomass)
N_FOLDS = 5


_cache = {}


def _cached():
    """Memory maps opened lazily, once per (DataLoader worker) process."""
    if not _cache:
        _cache["s2"] = np.load(CACHE / "s2.npy", mmap_mode="r")
        _cache["s1"] = np.load(CACHE / "s1.npy", mmap_mode="r")
        _cache["missing"] = np.load(CACHE / "missing.npy")
        _cache["cld"] = np.load(CACHE / "cld.npy")
        ids = pd.read_csv(CACHE / "ids.csv").chip_id
        _cache["index"] = dict(zip(ids, range(len(ids))))
    return _cache


def load_chip(chip_id: str, max_cloud: float | None = None):
    """[12, 15, PX, PX] raw (unnormalised) features in use_case_1 channel order, and the missing mask."""
    c = _cached()
    i = c["index"][chip_id]
    s1 = np.array(c["s1"][i], dtype=np.float32)
    s1[s1 < -9000] = 0.0  # nodata (-9999, stored as float16 -> -10000), zeroed as dataset.read_imgs
    s2 = np.array(c["s2"][i], dtype=np.float32)
    z = {"cld": c["cld"][i]}
    missing = c["missing"][i].copy()
    if max_cloud is not None:
        missing |= np.nan_to_num(z["cld"], nan=100.0) > max_cloud
        s2[missing] = 0.0
    if missing.all():  # keep the least cloudy month rather than an all-missing input
        missing[np.nanargmin(np.nan_to_num(z["cld"], nan=1e9))] = False
    return np.concatenate([s1, s2], axis=1), missing


def disk_weights(size: int = PX, cy: float | None = None, cx: float | None = None,
                 radius_m: float = PLOT_RADIUS_M) -> np.ndarray:
    """Normalised weights of the pixels whose centres fall within the plot disk.
    Default centre = chip centre, which is the corner between pixels size/2-1 and size/2."""
    cy = size / 2 if cy is None else cy
    cx = size / 2 if cx is None else cx
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    w = (((yy - cy) ** 2 + (xx - cx) ** 2) * 100.0 <= radius_m**2).astype(np.float32)
    return w / w.sum()


def plots(available_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(DATA / "plots.csv", dtype={"province_code": str})
    df["province_code"] = df.province_code.str.zfill(2)
    if available_only:
        df = df[df.chip_id.isin(set(pd.read_csv(CACHE / "ids.csv").chip_id))]
    df = df.merge(province_folds(), on="province_code")
    return df.reset_index(drop=True)


def province_folds() -> pd.DataFrame:
    """Provinces assigned to N_FOLDS folds, greedily balancing the number of plots (all plots, fixed)."""
    allp = pd.read_csv(DATA / "plots.csv", dtype={"province_code": str})
    sizes = allp.province_code.str.zfill(2).value_counts()
    load, rows = [0] * N_FOLDS, []
    for prov, n in sizes.items():
        f = int(np.argmin(load))
        load[f] += n
        rows.append((prov, f))
    return pd.DataFrame(rows, columns=["province_code", "fold"])


def configure(backbone: str):
    from inference import Config
    Config.backbone = backbone
    if "efficientnet" in backbone or "resnet" in backbone:
        Config.out_indices, Config.dec_channels = [0, 1, 2, 3, 4], [256, 160, 128, 64, 32]
    elif "convnext" in backbone or "swin" in backbone:
        Config.out_indices, Config.dec_channels = [0, 1, 2, 3], [256, 128, 64, 32]
    else:
        Config.out_indices, Config.dec_channels = [0, 1, 2, 3, 4], [256, 128, 64, 32, 16]
    return Config()


def load_model(name: str, device="cuda", init: str = "finland") -> torch.nn.Module:
    """The bare UnetVFLOW (no Lightning wrapper).
    init='finland': the BioMassters-trained weights (default, as in the paper).
    init='imagenet': same architecture, timm ImageNet encoder, fresh attention/decoder/head.
    init='random': same architecture, everything freshly initialised."""
    from models import UnetVFLOW
    backbone, mode = name.split("_")
    if init != "finland":
        cfg = configure(backbone)
        cfg.pretrained = init == "imagenet"
        return UnetVFLOW(cfg, mode=mode).to(device)
    from utils import CustomModel
    lm = CustomModel.load_from_checkpoint(CKPT_DIR / f"{name}.ckpt", model=UnetVFLOW(configure(backbone), mode=mode),
                                          batch_size=1, map_location="cpu")
    return lm.model.to(device)


def metrics(pred, obs) -> dict:
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    e = pred - obs
    ss = ((obs - obs.mean()) ** 2).sum()
    return {"n": len(obs), "bias": e.mean(), "rmse": np.sqrt((e**2).mean()), "mae": np.abs(e).mean(),
            "r2": 1 - (e**2).sum() / ss, "pearson_r": np.corrcoef(pred, obs)[0, 1],
            "rel_rmse_pct": 100 * np.sqrt((e**2).mean()) / obs.mean(), "obs_mean": obs.mean(), "pred_mean": pred.mean()}
