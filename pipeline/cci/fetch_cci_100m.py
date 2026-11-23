"""
Fetch the ESA CCI AGB (2021) reference at its ~100 m native support for every
Spain validation chip, on a grid aligned to the chip's own 10 m grid.

Each chip is 256x256 px at 10 m (2560 m). We take the chip's exact origin from
its GeoTIFF and request a 25x25 grid of 100 m blocks (2500 m; the last 6 px
(60 m) of the chip are dropped so 10x10 model pixels pool cleanly into one
block). Bands: AGB (Mg/ha) and SD (CCI's per-pixel uncertainty). Masked CCI
pixels are written as -9999.

Resampling: CCI's native grid is EPSG:4326 at ~0.00089 deg, i.e. ~100 m N-S but
only ~65-75 m E-W at Spanish latitudes. Earth Engine resamples it to our
EPSG:3035 100 m grid with nearest neighbour, so each block carries the value of
the CCI cell under the block centre (a point sample, not an area mean) -- a
small support mismatch against the 10x10-pixel mean of the model prediction.

Output: <out_dir>/<chip_id>.tif, float32, shape (25, 25, 2) [AGB, SD], EPSG:3035.
"""

import os
import argparse
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import pandas as pd
import rasterio
import requests

N_BLOCKS = 25
BLOCK_M = 100
NODATA = -9999


def chip_origin(data_dir, chip_id):
    for m in range(12):
        f = data_dir / f"{chip_id}_S1_{m:02d}.tif"
        if f.exists():
            with rasterio.open(f) as s:
                assert s.crs.to_string() == "EPSG:3035", s.crs
                assert abs(s.transform.a - 10) < 1e-6 and abs(s.transform.e + 10) < 1e-6
                return s.transform.c, s.transform.f
    raise FileNotFoundError(f"no S1 tif for {chip_id}")


def fetch_one(chip_id, data_dir, out_dir, image, retries=5):
    out_path = out_dir / f"{chip_id}.tif"
    if out_path.exists():
        return chip_id, "skipped"
    left, top = chip_origin(data_dir, chip_id)
    span = N_BLOCKS * BLOCK_M
    # Inset by 1 m: the chip origin has a sub-mm fractional part, and a region
    # edge sitting exactly on a block boundary made Earth Engine return an extra
    # row (grid shifted 100 m north) or column for ~1 in 4 chips.
    region = ee.Geometry.Rectangle(
        [left + 1, top - span + 1, left + span - 1, top - 1], proj="EPSG:3035", geodesic=False
    )
    params = {
        "region": region,
        "crs": "EPSG:3035",
        "crs_transform": [BLOCK_M, 0, left, 0, -BLOCK_M, top],
        "format": "GEO_TIFF",
    }
    err = None
    for attempt in range(retries):
        try:
            resp = requests.get(image.getDownloadURL(params), timeout=120)
            resp.raise_for_status()
            content = resp.content
            if content[:2] == b"PK":
                zf = zipfile.ZipFile(io.BytesIO(content))
                content = zf.read(zf.namelist()[0])
            with rasterio.MemoryFile(content) as mf, mf.open() as ds:
                ok = (ds.count, ds.height, ds.width) == (2, N_BLOCKS, N_BLOCKS)
                ok = ok and abs(ds.transform.c - left) < 1 and abs(ds.transform.f - top) < 1
                if not ok:
                    raise ValueError(f"bad grid {(ds.count, ds.height, ds.width)} origin {(ds.transform.c, ds.transform.f)}")
            out_path.write_bytes(content)
            return chip_id, "ok"
        except Exception as exc:  # noqa: BLE001
            err = exc
            time.sleep(10 * 2**attempt if "429" in str(exc) else 5 * (attempt + 1))
    return chip_id, f"FAILED: {err}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/media/HD/Data/agb_spain/validation")
    p.add_argument("--out_dir", default="/media/HD/Data/agb_spain/cci_100m")
    p.add_argument("--year", type=int, default=2021)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ee.Initialize(project=os.environ["EE_PROJECT"])

    coll = ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
    image = (
        ee.Image(coll.filterDate(ee.Date.fromYMD(args.year, 1, 1), ee.Date.fromYMD(args.year + 1, 1, 1)).first())
        .select(["AGB", "SD"])
        .unmask(NODATA)
        .toFloat()
    )

    chip_ids = pd.read_csv(data_dir / "metadata.csv").chip_id.tolist()
    if args.limit:
        chip_ids = chip_ids[: args.limit]

    n_ok = n_skip = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(fetch_one, c, data_dir, out_dir, image) for c in chip_ids]
        for i, f in enumerate(as_completed(futs), 1):
            cid, status = f.result()
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                failed.append((cid, status))
            if i % 50 == 0 or i == len(chip_ids):
                print(f"[{i}/{len(chip_ids)}] ok={n_ok} skipped={n_skip} failed={len(failed)}", flush=True)
    for cid, st in failed:
        print(f"  {cid}: {st}")


if __name__ == "__main__":
    main()
