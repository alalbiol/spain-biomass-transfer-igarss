"""
Re-export the Sentinel-2 side of a stratified SUBSET of the Spain validation
chips with a BioMassters-style recipe, to separate pipeline mismatch from true
domain shift in the zero-shot error.

Differences from export_validation_set.py (all deliberate):
  * one scene per month = the least cloudy over the chip, instead of the median
    of every scene (BioMassters: "best image for each month", not a mean);
  * cloud channel = s2cloudless probability (COPERNICUS/S2_CLOUD_PROBABILITY,
    0-100) x 2.55 -> 0-255, i.e. like Sentinel Hub's CLP band that
    use_case_1/dataset.py normalizes by 255 (s2_max[10] = 255), instead of
    Sen2Cor's MSK_CLDPRB (0-100, and often ~0 on visibly cloudy scenes);
  * a month with no scene covering the whole chip is left MISSING (no file), so
    the model's missing-month handling can skip it, as with BioMassters.
S1 is not re-exported: the existing S1 files are symlinked into the new folder.
(BioMassters uses a monthly mean for S1, we have per-orbit medians -- small,
unverified difference, left as is.)

Selection score = chip-mean s2cloudless probability. The chosen score and scene
count are logged per chip-month so cloud-threshold variants can be built later
from the same download (--variant_out / --max_cloud) without re-downloading.
"""

import argparse
import io
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import pandas as pd
import requests

from fetch_cci_100m import chip_origin

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
CHIP_PIXELS = "256x256"


def pick_subset(meta, per_stratum, seed):
    parts = []
    for _, g in meta.groupby(["forest_type", "agb_bin"]):
        parts.append(g.sample(n=min(per_stratum, len(g)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def month_selection(year, month, geom):
    """Server-side: number of full-coverage scenes, chosen scene id and its chip-mean cloud probability."""
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
    return ee.Dictionary(
        {
            "n": n,
            "cld": ee.Algorithms.If(n.gt(0), best.get("chip_cld"), None),
            "idx": ee.Algorithms.If(n.gt(0), best.get("system:index"), None),
        }
    )


def scene_image(idx, geom):
    im = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{idx}")
    clp = ee.Image(f"COPERNICUS/S2_CLOUD_PROBABILITY/{idx}").select("probability").multiply(2.55).rename("CLP")
    # unmask(0, sameFootprint=False): scenes are required to cover >=99.9% of the
    # chip, but the remainder can be genuinely outside the scene's footprint (an
    # orbit/tile-seam sliver), not just cloud-masked. Plain unmask(0) only fills
    # masked pixels WITHIN the image's existing footprint and leaves true
    # outside-footprint pixels as nodata (-inf in the exported GeoTIFF) -- found
    # this by tracing 3/500 chips whose predictions came out 100% non-finite back
    # to a ~21-pixel corner with -inf on every band. sameFootprint=False pads the
    # image to be fully defined everywhere the region asks for.
    return im.select(S2_BANDS).addBands(clp).unmask(0, sameFootprint=False).toFloat().clip(geom)


def export_chip(chip_id, data_dir, out_dir, year, retries=5):
    left, top = chip_origin(data_dir, chip_id)
    geom = ee.Geometry.Rectangle([left, top - 2560, left + 2560, top], proj="EPSG:3035", geodesic=False)
    sel, err = None, None
    for attempt in range(retries):
        try:
            sel = ee.List([month_selection(year, m, geom) for m in range(1, 13)]).getInfo()
            break
        except Exception as exc:  # noqa: BLE001
            err = exc
            time.sleep(10 * 2**attempt if "429" in str(exc) else 5 * (attempt + 1))
    rows = []
    if sel is None:
        return [{"chip_id": chip_id, "month": m, "n_scenes": -1, "chosen_cld": None, "error": str(err)[:120]} for m in range(12)]
    for m, d in enumerate(sel):
        row = {"chip_id": chip_id, "month": m, "n_scenes": d["n"], "chosen_cld": d["cld"], "scene": d["idx"]}
        path = out_dir / f"{chip_id}_S2_{m:02d}.tif"
        if d["n"] > 0 and not path.exists():
            for attempt in range(retries):
                try:
                    params = {"region": geom, "dimensions": CHIP_PIXELS, "crs": "EPSG:3035", "format": "GEO_TIFF"}
                    resp = requests.get(scene_image(d["idx"], geom).getDownloadURL(params), timeout=180)
                    resp.raise_for_status()
                    content = resp.content
                    if content[:2] == b"PK":
                        zf = zipfile.ZipFile(io.BytesIO(content))
                        content = zf.read(zf.namelist()[0])
                    path.write_bytes(content)
                    break
                except Exception as exc:  # noqa: BLE001
                    row["error"] = str(exc)[:120]
                    time.sleep(10 * 2**attempt if "429" in str(exc) else 5 * (attempt + 1))
            else:
                row["n_scenes"] = -1
        rows.append(row)
    for m in range(12):  # S1: reuse existing files
        link = out_dir / f"{chip_id}_S1_{m:02d}.tif"
        src = data_dir / f"{chip_id}_S1_{m:02d}.tif"
        if src.exists() and not link.exists():
            os.symlink(src, link)
    return rows


def make_variant(out_dir, variant_out, max_cloud):
    log = pd.read_csv(out_dir / "best_scene_log.csv")
    variant_out.mkdir(parents=True, exist_ok=True)
    dropped = 0
    for f in out_dir.glob("*.tif"):
        cid, sensor, month = f.stem.rsplit("_", 2)
        link = variant_out / f.name
        if sensor == "S2":
            row = log[(log.chip_id == cid) & (log.month == int(month))]
            if len(row) and row.chosen_cld.iloc[0] > max_cloud:
                dropped += 1
                continue
        if not link.exists():
            os.symlink(f.resolve(), link)
    print(f"variant {variant_out}: dropped {dropped} S2 months with chip cloud prob > {max_cloud}")


def chip_done(chip_id, log, out_dir):
    rows = log[log.chip_id == chip_id]
    if len(rows) != 12 or (rows.n_scenes == -1).any():
        return False
    return all((out_dir / f"{chip_id}_S2_{m:02d}.tif").exists() for m in rows[rows.n_scenes > 0].month)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/media/HD/Data/agb_spain/validation")
    p.add_argument("--out_dir", default="/media/HD/Data/agb_spain/validation_bm")
    p.add_argument("--all_chips", action="store_true", help="Every chip in metadata.csv (default: stratified subset)")
    p.add_argument("--per_stratum", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--year", type=int, default=2021)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_fail_streak", type=int, default=6,
                   help="Stop (exit code 3) after this many consecutive chips fail -- likely quota exhaustion. "
                        "Progress is saved per chip; just re-run to resume.")
    p.add_argument("--variant_out", default=None)
    p.add_argument("--max_cloud", type=float, default=40.0)
    p.add_argument("--limit", type=int, default=0, help="Only process the first N chips still to do (testing)")
    args = p.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    if args.variant_out:
        make_variant(out_dir, Path(args.variant_out), args.max_cloud)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ee.Initialize(project=os.environ["EE_PROJECT"])

    meta = pd.read_csv(data_dir / "metadata.csv")
    if args.all_chips:
        sub = meta
        sub.to_csv(out_dir / "metadata_full.csv", index=False)  # metadata_subset.csv is left untouched
    else:
        sub = pick_subset(meta, args.per_stratum, args.seed)
        sub.to_csv(out_dir / "metadata_subset.csv", index=False)
    print(f"target: {len(sub)} chips", flush=True)

    log_path = out_dir / "best_scene_log.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=["chip_id", "month", "n_scenes", "chosen_cld", "scene"])
    todo = [c for c in sub.chip_id if not chip_done(c, log, out_dir)]
    print(f"already complete: {len(sub) - len(todo)}; to do: {len(todo)}", flush=True)
    if args.limit:
        todo = todo[: args.limit]

    lock, stop = threading.Lock(), threading.Event()
    state = {"streak": 0, "ok": 0, "failed": 0, "log": log}

    def work(cid):
        if stop.is_set():
            return None
        rows = export_chip(cid, data_dir, out_dir, args.year)
        failed = any(r["n_scenes"] == -1 for r in rows)
        with lock:
            cur = state["log"]
            state["log"] = pd.concat([cur[cur.chip_id != cid], pd.DataFrame(rows)], ignore_index=True)
            tmp = log_path.with_suffix(".tmp")
            state["log"].to_csv(tmp, index=False)
            tmp.replace(log_path)
            state["streak"] = state["streak"] + 1 if failed else 0
            state["failed" if failed else "ok"] += 1
            if state["streak"] >= args.max_fail_streak:
                stop.set()
            print(f"[{state['ok'] + state['failed']}/{len(todo)}] {cid} {'FAILED' if failed else 'ok'} "
                  f"(ok={state['ok']} failed={state['failed']} streak={state['streak']})", flush=True)
        return failed

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    log = state["log"]
    done = sum(chip_done(c, log, out_dir) for c in sub.chip_id)
    print(f"complete chips: {done}/{len(sub)}; months with no full-coverage scene: {(log.n_scenes == 0).sum()}", flush=True)
    if stop.is_set():
        print(f"STOPPED: {args.max_fail_streak} consecutive chips failed (quota or rate limit?). Re-run to resume.", flush=True)
        raise SystemExit(3)
    raise SystemExit(0 if done == len(sub) else 2)


if __name__ == "__main__":
    main()
