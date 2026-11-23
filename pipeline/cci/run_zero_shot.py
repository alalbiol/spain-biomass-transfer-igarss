"""
Zero-shot inference of the Finland-trained AGB checkpoints on the Spain
validation chips (M4 in papers/paper1-case1-IGARSS/paper_plan.md).

Lean replacement for use_case_1/inference.py for batch runs: one process, all
requested checkpoints, no per-chip PNG plotting, resumable (skips existing
outputs). Predictions are per-pixel AGBM (Mg/ha) at 10 m, 256x256, saved as
float32 .npy under <out_dir>/preds/<backbone>_<mode>/<chip_id>.npy.

Inputs are fed RAW (no normalization). Verified on Spain chips: with the
`--norm` flag shown in use_case_1/INF_README.md these checkpoints output a
near-constant map (chip-mean std 0.1 Mg/ha across chips), without it they track
ESA CCI AGB (r~0.85 over 30 chips). Consistent with use_case_1/dataset.py:
DS.__getitem__ feeds the training safetensors to the model unnormalized.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

UC1 = Path(__file__).resolve().parents[1] / "use_case_1"
sys.path.insert(0, str(UC1))

from dataset import read_imgs  # noqa: E402
from inference import Config  # noqa: E402
from models import UnetVFLOW  # noqa: E402
from utils import CustomModel  # noqa: E402

BACKBONES = ["convnextv2", "efficientnet", "efficientnetv2", "mobilenetv4", "resnetrs101"]
MODES = ["early", "late"]


def configure(backbone):
    Config.backbone = backbone
    if "efficientnet" in backbone or "resnet" in backbone:
        Config.out_indices = [0, 1, 2, 3, 4]
        Config.dec_channels = [256, 160, 128, 64, 32]
    elif "convnext" in backbone or "swin" in backbone:
        Config.out_indices = [0, 1, 2, 3]
        Config.dec_channels = [256, 128, 64, 32]
    else:
        Config.out_indices = [0, 1, 2, 3, 4]
        Config.dec_channels = [256, 128, 64, 32, 16]
    return Config()


class ChipDS(torch.utils.data.Dataset):
    def __init__(self, chip_ids, data_dir):
        self.chip_ids = chip_ids
        self.data_dir = Path(data_dir)

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, i):
        imgs, missing = read_imgs(self.chip_ids[i], self.data_dir, norm=False)
        mask = torch.zeros(12, dtype=torch.bool)
        mask[missing] = True
        return torch.from_numpy(imgs).float(), mask, i


def main():
    p = argparse.ArgumentParser(description="Zero-shot AGB inference on the Spain validation chips")
    p.add_argument("--data_dir", default="/media/HD/Data/agb_spain/validation")
    p.add_argument("--metafile", default="/media/HD/Data/agb_spain/validation/metadata.csv")
    p.add_argument("--model_dir", default="/media/HD/checkpoints/AI4DIS/case1")
    p.add_argument("--out_dir", default="/media/HD/Data/agb_spain/zero_shot")
    p.add_argument("--backbones", default=",".join(BACKBONES))
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--limit", type=int, default=0, help="Only the first N chips (0 = all)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.metafile)
    chip_ids = df.chip_id.tolist()
    if args.limit:
        chip_ids = chip_ids[: args.limit]
    loader = torch.utils.data.DataLoader(
        ChipDS(chip_ids, args.data_dir), batch_size=1, shuffle=False, num_workers=args.workers, prefetch_factor=4
    )

    for backbone in args.backbones.split(","):
        for mode in args.modes.split(","):
            name = f"{backbone}_{mode}"
            ckpt = Path(args.model_dir) / f"{name}.ckpt"
            out = Path(args.out_dir) / "preds" / name
            out.mkdir(parents=True, exist_ok=True)
            todo = [c for c in chip_ids if not (out / f"{c}.npy").exists()]
            if not todo:
                print(f"[{name}] all {len(chip_ids)} chips already done, skipping", flush=True)
                continue
            if not ckpt.exists():
                print(f"[{name}] checkpoint missing: {ckpt}", flush=True)
                continue

            model = UnetVFLOW(configure(backbone), mode=mode)
            lm = CustomModel.load_from_checkpoint(ckpt, model=model, batch_size=1).eval().to(device)
            t0, done = time.time(), 0
            todo_set = set(todo)
            for imgs, mask, i in loader:
                cid = chip_ids[int(i)]
                if cid not in todo_set:
                    continue
                with torch.no_grad():
                    pred = lm(imgs.to(device), mask.to(device)).squeeze().cpu().numpy().astype("float32")
                np.save(out / f"{cid}.npy", pred)
                done += 1
                if done % 50 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"[{name}] {done}/{len(todo)} chips ({el/done:.2f}s/chip, ~{el/done*(len(todo)-done)/60:.1f} min left)", flush=True)
            del lm, model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
