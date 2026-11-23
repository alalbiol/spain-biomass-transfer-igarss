"""
Pack the per-plot chip .npz files into one contiguous memory-mappable cache,
centre-cropped to CACHE_PX (112 px = 1.12 km, plot centre still at the chip
centre corner).

Why: the 6k compressed .npz files on the HDD (written concurrently by the
exporter, hence fragmented) read at ~5 chips/s with random access -- the GPU sat
idle and one fine-tuning run would take ~14 h. One contiguous ~28 GB file reads
sequentially and fits in the page cache. 112 px still leaves room for the
random 96 px training crops; the 128 px crop test (export_ifn_chips.py
docstring) shows predictions at this context size track the full 256 px chip.

Layout under <DATA>/cache_112/: s2.u16 [N,12,11,P,P] uint16, s1.f16 [N,12,4,P,P]
float16 (dB, -9999 nodata kept), missing.npy [N,12], cld.npy [N,12], ids.csv.
"""

import time

import numpy as np
import pandas as pd

from ifn_common import CACHE, CACHE_PX, CHIPS, DATA


def main():
    ids = pd.read_csv(DATA / "plots.csv").chip_id
    ids = [c for c in ids if (CHIPS / f"{c}.npz").exists()]
    ids.sort(key=lambda c: (CHIPS / f"{c}.npz").stat().st_mtime)  # export order ~ on-disk order: fewer seeks
    n, P = len(ids), CACHE_PX
    o = (128 - P) // 2
    CACHE.mkdir(parents=True, exist_ok=True)
    s2 = np.lib.format.open_memmap(CACHE / "s2.npy", mode="w+", dtype=np.uint16, shape=(n, 12, 11, P, P))
    s1 = np.lib.format.open_memmap(CACHE / "s1.npy", mode="w+", dtype=np.float16, shape=(n, 12, 4, P, P))
    missing = np.zeros((n, 12), bool)
    cld = np.zeros((n, 12), np.float32)
    t0 = time.time()
    for i, c in enumerate(ids):
        z = np.load(CHIPS / f"{c}.npz")
        s2[i] = z["s2"][..., o:o + P, o:o + P]
        s1[i] = z["s1"][..., o:o + P, o:o + P].astype(np.float16)
        missing[i], cld[i] = z["missing"], z["cld"]
        if (i + 1) % 250 == 0:
            el = time.time() - t0
            print(f"{i+1}/{n} {(i+1)/el:.1f} chips/s ~{(n-i-1)/((i+1)/el)/60:.0f} min left", flush=True)
    s2.flush()
    s1.flush()
    np.save(CACHE / "missing.npy", missing)
    np.save(CACHE / "cld.npy", cld)
    pd.DataFrame({"chip_id": ids}).to_csv(CACHE / "ids.csv", index=False)
    print(f"done: {n} chips in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
