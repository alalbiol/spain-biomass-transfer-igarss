"""
Zero-shot predictions of the 10 Finland checkpoints on the IFN plot chips.

Saves, per checkpoint, the full 128x128 prediction map (float16) of every chip
in <DATA>/zero_shot/<name>.npz (resumable: chips already present are skipped)
and the 25 m plot-disk mean in <DATA>/zero_shot/plot_preds.csv.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from ifn_common import BACKBONES, DATA, MODES, disk_weights, load_chip, load_model, plots

OUT = DATA / "zero_shot"


class ChipDS(torch.utils.data.Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x, miss = load_chip(self.ids[i])
        return torch.from_numpy(x), torch.from_numpy(miss), i


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(f"{b}_{m}" for b in BACKBONES for m in MODES))
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("medium")
    df = plots()
    w = torch.from_numpy(disk_weights())
    rows = {}
    for name in args.models.split(","):
        f = OUT / f"{name}.npz"
        maps = dict(np.load(f)) if f.exists() else {}
        todo = [c for c in df.chip_id if c not in maps]
        if todo:
            model = load_model(name).eval()
            dl = torch.utils.data.DataLoader(ChipDS(todo), batch_size=args.batch, num_workers=6)
            for x, miss, idx in dl:
                with torch.no_grad():
                    pred = model(x.cuda(), miss.cuda().clone()).float().squeeze(1).cpu().numpy()
                for k, i in enumerate(idx.tolist()):
                    maps[todo[i]] = pred[k].astype(np.float16)
            np.savez(f, **maps)
            del model
            torch.cuda.empty_cache()
        rows[name] = {c: float((maps[c].astype(np.float32) * w.numpy()).sum()) for c in df.chip_id}
        print(f"{name}: {len(todo)} new, {len(df)} total", flush=True)
    out = df[["chip_id"]].copy()
    for name, d in rows.items():
        out[name] = out.chip_id.map(d)
    prev = OUT / "plot_preds.csv"
    if prev.exists():
        old = pd.read_csv(prev)
        old = old[~old.chip_id.isin(out.chip_id)]
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(prev, index=False)


if __name__ == "__main__":
    main()
