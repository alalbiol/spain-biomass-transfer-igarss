"""
Export 128x128 px (1.28 km, 10 m, EPSG:3035) Sentinel-1/2 chips centred on the
IFN4 plots written by build_ifn_plot_labels.py, one year of monthly frames in
the image year of each plot's survey.

Recipe = use_case_1_spain/export_bm_style_chips.py (BioMassters style):
  * S2: per month, the single least-cloudy COPERNICUS/S2_SR_HARMONIZED scene
    that covers the whole chip; cloud channel = s2cloudless probability x 2.55.
    A month with no full-coverage scene is missing (zeros + missing flag);
  * S1: per month, median of IW GRD per orbit direction, bands
    VV_asc, VH_asc, VV_desc, VH_desc in dB, -9999 where no pass.
Both sensors are downloaded as ONE multi-band image per chip (all months
stacked) instead of 24 requests, which is what makes thousands of chips
practical. 128 px instead of 256: on 75 centre windows of the Spain validation
chips the 128 px crop reproduces the full-chip prediction with r = 0.99.

Output per plot: <out_dir>/<chip_id>.npz with
  s2  uint16  [12, 11, 128, 128]   (raw L2A DN, CLP 0-255)
  s1  float32 [12, 4, 128, 128]    (dB, -9999 nodata)
  missing bool [12]                 (no S2 scene that month)
  cld float32 [12]                  (chosen scene's chip-mean cloud prob, NaN if missing)
The plot centre is the corner shared by pixels 63 and 64 in both axes.
"""

import argparse
import io
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import requests
import rasterio
from pyproj import Transformer

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]
PX = 128
HALF_M = PX * 10 / 2
TO_LAEA = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


def month_selection(year, month, geom):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(start, end)
    cp = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(geom).filterDate(start, end)
    joined = ee.ImageCollection(
        ee.Join.saveFirst("cp").apply(s2, cp, ee.Filter.equals(leftField="system:index", rightField="system:index"))
    ).filter(ee.Filter.notNull(["cp"]))

    def score(im):
        im = ee.Image(im)
        prob = ee.Image(im.get("cp")).select("probability")
        cld = prob.reduceRegion(ee.Reducer.mean(), geom, 20, maxPixels=1e8).get("probability")
        valid = im.select("B2").mask().reduceRegion(ee.Reducer.mean(), geom, 20, maxPixels=1e8).get("B2")
        return im.set({"chip_cld": cld, "chip_valid": valid})

    scored = joined.map(score).filter(ee.Filter.gte("chip_valid", 0.999)).filter(ee.Filter.notNull(["chip_cld"]))
    n = scored.size()
    best = ee.Image(scored.sort("chip_cld").first())
    return ee.Dictionary({
        "n": n,
        "cld": ee.Algorithms.If(n.gt(0), best.get("chip_cld"), None),
        "idx": ee.Algorithms.If(n.gt(0), best.get("system:index"), None),
    })


def s2_scene(idx):
    im = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{idx}")
    clp = ee.Image(f"COPERNICUS/S2_CLOUD_PROBABILITY/{idx}").select("probability").multiply(2.55).rename("CLP")
    return im.select(S2_BANDS).addBands(clp).unmask(0, sameFootprint=False).toFloat()


def s1_month(year, month, geom):
    start = ee.Date.fromYMD(year, month, 1)
    coll = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom).filterDate(start, start.advance(1, "month"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    nodata = ee.Image.constant([-9999, -9999]).rename(S1_BANDS).toFloat()

    def pass_median(direction):
        sub = coll.filter(ee.Filter.eq("orbitProperties_pass", direction)).select(S1_BANDS)
        med = sub.median().unmask(-9999, sameFootprint=False).toFloat()
        return ee.Image(ee.Algorithms.If(sub.size().gt(0), med, nodata))

    return ee.Image.cat([pass_median("ASCENDING"), pass_median("DESCENDING")])


def download(image, geom):
    params = {"region": geom, "dimensions": f"{PX}x{PX}", "crs": "EPSG:3035", "format": "GEO_TIFF"}
    resp = requests.get(image.getDownloadURL(params), timeout=300)
    resp.raise_for_status()
    content = resp.content
    if content[:2] == b"PK":
        zf = zipfile.ZipFile(io.BytesIO(content))
        content = zf.read(zf.namelist()[0])
    with rasterio.io.MemoryFile(content) as mf, mf.open() as ds:
        return ds.read()


def export_chip(row, out_dir, retries=5):
    out = out_dir / f"{row.chip_id}.npz"
    if out.exists():
        return "skip"
    cx, cy = TO_LAEA.transform(row.lon, row.lat)
    geom = ee.Geometry.Rectangle([cx - HALF_M, cy - HALF_M, cx + HALF_M, cy + HALF_M], proj="EPSG:3035", geodesic=False)
    year = int(row.image_year)
    err = None
    for attempt in range(retries):
        try:
            sel = ee.List([month_selection(year, m, geom) for m in range(1, 13)]).getInfo()
            have = [m for m, d in enumerate(sel) if d["n"] > 0]
            s2 = np.zeros((12, 11, PX, PX), dtype=np.uint16)
            if have:
                arr = download(ee.Image.cat([s2_scene(sel[m]["idx"]) for m in have]), geom)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).clip(0, 65535)
                s2[have] = arr.reshape(len(have), 11, PX, PX).astype(np.uint16)
            s1 = download(ee.Image.cat([s1_month(year, m, geom) for m in range(1, 13)]), geom)
            s1 = np.nan_to_num(s1.reshape(12, 4, PX, PX).astype(np.float32), nan=-9999, posinf=-9999, neginf=-9999)
            missing = np.array([d["n"] == 0 for d in sel])
            cld = np.array([np.nan if d["cld"] is None else d["cld"] for d in sel], dtype=np.float32)
            tmp = out.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, s2=s2, s1=s1, missing=missing, cld=cld)
            tmp.replace(out)
            return "ok"
        except Exception as exc:  # noqa: BLE001
            err = exc
            time.sleep(10 * 2**attempt if "429" in str(exc) else 5 * (attempt + 1))
    return f"FAILED {str(err)[:150]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plots", default="/media/HD/Data/agb_spain_ifn/plots.csv")
    p.add_argument("--out_dir", default="/media/HD/Data/agb_spain_ifn/chips")
    p.add_argument("--subset", default=None, help="CSV with a chip_id column restricting the export")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_fail_streak", type=int, default=10)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ee.Initialize(project=os.environ["EE_PROJECT"])
    df = pd.read_csv(args.plots)
    if args.subset:
        df = df[df.chip_id.isin(set(pd.read_csv(args.subset).chip_id))]
    # fixed shuffle: any prefix of the export is a random sample of the plots
    df = df.sample(frac=1.0, random_state=0)
    todo = [r for r in df.itertuples() if not (out_dir / f"{r.chip_id}.npz").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(df)} plots, {len(todo)} to export", flush=True)
    lock, stop = threading.Lock(), threading.Event()
    state = {"n": 0, "fail": 0, "streak": 0, "t0": time.time()}

    def work(r):
        if stop.is_set():
            return
        res = export_chip(r, out_dir)
        with lock:
            state["n"] += 1
            bad = res.startswith("FAILED")
            state["fail"] += bad
            state["streak"] = state["streak"] + 1 if bad else 0
            if bad:
                print(f"{r.chip_id} {res}", flush=True)
            if state["streak"] >= args.max_fail_streak:
                stop.set()
            if state["n"] % 25 == 0 or state["n"] == len(todo):
                el = time.time() - state["t0"]
                print(f"[{state['n']}/{len(todo)}] failed={state['fail']} {el/state['n']:.2f}s/chip "
                      f"~{el/state['n']*(len(todo)-state['n'])/60:.0f} min left", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))
    if stop.is_set():
        print("stopped: too many consecutive failures (quota?) -- re-run to resume", flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
