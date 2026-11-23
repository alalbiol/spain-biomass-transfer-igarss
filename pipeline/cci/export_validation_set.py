"""
Build a Spain validation set for the boreal-trained AGB ensemble (use_case_1).

For each sampled Spanish forest parcel this script exports:
  - {chip_id}_S1_{month:02d}.tif  (4 bands: VV_asc, VH_asc, VV_desc, VH_desc, dB)
  - {chip_id}_S2_{month:02d}.tif  (11 bands: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12,MSK_CLDPRB)
  - a metadata.csv row (chip_id, lon/lat, agb_mean label, forest_type, agb_bin, ...)

Chips are 256x256 px at 10 m (2.56 km side), matching use_case_1/dataset.py's
IMG_SIZE so the existing trained checkpoints can run on them directly without
retraining. Imagery is downloaded synchronously via Image.getDownloadURL
(no Google Drive export/polling step) since a few hundred to a couple
thousand chips fits comfortably within per-request size limits.

See use_case_1_spain/context/research_plan.md for the full design rationale
(grid size choice, output-pooling to 100 m for evaluation, forest-type
stratification proxy, etc).
"""

import argparse
import csv
import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import requests

REQUIRED_SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/devstorage.full_control",
]

GRID_SIZE_M = 2560  # 256 px * 10 m -> matches use_case_1/dataset.py IMG_SIZE
CHIP_PIXELS = "256x256"
PROJ_EPSG = "EPSG:3035"

S1_BANDS = ["VV", "VH"]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "MSK_CLDPRB"]

# CORINE Land Cover codes used as a forest-type stratification proxy in the
# absence of a confirmed GEE asset for Spain's species-level Mapa Forestal.
CORINE_FOREST_CLASSES = {311: "broadleaved", 312: "coniferous", 313: "mixed"}


def parse_args():
    p = argparse.ArgumentParser(description="Export a Spain AGB validation set (parcels + S1/S2 chips)")
    p.add_argument("--country", type=str, default="Spain")
    p.add_argument("--agb_year", type=int, default=2021, help="Must have good S1+S2 coverage (>=2015)")
    p.add_argument("--n_target", type=int, default=50, help="Target number of chips to export this run")
    p.add_argument("--oversample_factor", type=int, default=4, help="Raw candidate pool = n_target * factor")
    p.add_argument("--forest_fraction_threshold", type=float, default=0.60)
    p.add_argument("--min_agb_threshold", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="/media/HD/Data/agb_spain/validation")
    p.add_argument("--skip_imagery", action="store_true", help="Only sample parcels + write metadata, no chip downloads")
    p.add_argument("--workers", type=int, default=8, help="Concurrent chip exports (each chip = 24 sequential HTTP downloads, so this is the main lever on wall-clock time)")
    return p.parse_args()


def resolve_project_id():
    import subprocess

    env_project = os.getenv("EE_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if env_project:
        return env_project
    result = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True)
    project = result.stdout.strip()
    if result.returncode == 0 and project and project != "(unset)":
        return project
    return None


def init_ee():
    project_id = resolve_project_id()
    if not project_id:
        raise RuntimeError("No GCP project resolved. Set EE_PROJECT or `gcloud config set project <id>`.")
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate(scopes=REQUIRED_SCOPES, auth_mode="localhost")
        ee.Initialize(project=project_id)
    print(f"Earth Engine initialized (project={project_id})", flush=True)


def sample_candidate_parcels(country_geom, agb_image, forest_mask, corine_img, args, n_samples):
    # NOTE: `proj` (atScale) is only valid for raster reduceResolution/reproject
    # calls below. Geometry.transform()/Rectangle() must use `proj_plain` (no
    # atScale) - atScale rescales coordinate transforms into grid-index units
    # (e.g. meters/2560), which silently produced ~1000x-too-large squares
    # and made filterBounds() match almost the entire global collection when
    # this was first tried. Verified empirically before relying on it here.
    proj = ee.Projection(PROJ_EPSG).atScale(GRID_SIZE_M)
    proj_plain = ee.Projection(PROJ_EPSG)

    # maxPixels must cover (GRID_SIZE_M / native_resolution)^2 input pixels
    # per output pixel: ESA CCI AGB and CORINE are ~100 m native (~676 needed
    # for a 2560 m cell), WorldCover is 10 m native (~65536 needed) - use the
    # EE-allowed max (65536) throughout rather than tuning per-source.
    MAX_REDUCE_PIXELS = 65536
    agb_at_grid = (
        agb_image.select("AGB").rename("agb_mean")
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=MAX_REDUCE_PIXELS)
        .reproject(crs=proj)
    )
    forest_at_grid = (
        forest_mask.rename("forest_fraction")
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=MAX_REDUCE_PIXELS)
        .reproject(crs=proj)
    )
    forest_type_at_grid = (
        corine_img.rename("forest_type_code")
        .reduceResolution(reducer=ee.Reducer.mode(), maxPixels=MAX_REDUCE_PIXELS)
        .reproject(crs=proj)
    )

    base_grid = agb_at_grid.addBands(forest_at_grid).addBands(forest_type_at_grid)

    samples = base_grid.sample(
        region=country_geom,
        scale=GRID_SIZE_M,
        projection=proj,
        geometries=True,
        tileScale=8,
        numPixels=n_samples,
        seed=args.seed,
        dropNulls=True,
    )

    half = GRID_SIZE_M / 2

    def point_to_square(feat):
        # .sample() always returns point geometries in EPSG:4326 lon/lat,
        # regardless of the `projection` passed to sample() - must reproject
        # to real meters (proj_plain) before building a meter-sized square.
        projected_pt = feat.geometry().transform(proj_plain, 1)
        coords = projected_pt.coordinates()
        x = ee.Number(coords.get(0))
        y = ee.Number(coords.get(1))
        sq = ee.Geometry.Rectangle(
            [x.subtract(half), y.subtract(half), x.add(half), y.add(half)], proj=proj_plain, geodesic=False
        )
        return feat.setGeometry(sq)

    squares = samples.map(point_to_square)

    selected = squares.filter(ee.Filter.gte("forest_fraction", args.forest_fraction_threshold)).filter(
        ee.Filter.gte("agb_mean", args.min_agb_threshold)
    )

    def add_bins(feat):
        agb = ee.Number(feat.get("agb_mean"))
        agb_bin = ee.String(
            ee.Algorithms.If(agb.lt(50), "low", ee.Algorithms.If(agb.lt(120), "medium", "high"))
        )
        code = ee.Number(feat.get("forest_type_code"))
        forest_type = ee.String(
            ee.Algorithms.If(
                code.eq(311), "broadleaved",
                ee.Algorithms.If(code.eq(312), "coniferous",
                ee.Algorithms.If(code.eq(313), "mixed", "other")),
            )
        )
        # geometry is in EPSG:3035 meters at this point; reproject centroid
        # back to lon/lat for human-readable metadata.
        centroid = feat.geometry().centroid(1).transform("EPSG:4326", 1)
        coords = centroid.coordinates()
        return feat.set(
            {
                "agb_bin": agb_bin,
                "forest_type": forest_type,
                "lon": coords.get(0),
                "lat": coords.get(1),
                "country": args.country,
                "agb_year": args.agb_year,
                "grid_size_m": GRID_SIZE_M,
            }
        )

    # "other" means the CORINE mode-per-grid-cell landed on a non-forest
    # class even though the forest_fraction (mean-based) passed the
    # threshold - a mixed-pixel edge case, not a meaningful stratum. Drop it.
    return selected.map(add_bins).filter(ee.Filter.neq("forest_type", "other"))


def stratified_subset(features, n_target, seed, max_stratum_multiplier=3):
    """Balance across (forest_type x agb_bin) strata, python-side, from an
    oversampled candidate pool already pulled locally via getInfo().

    Uses capped round-robin rather than "fill shortfall from whatever's
    left": the latter lets the most abundant stratum (e.g. coniferous/
    medium-AGB, common in Spain) swamp the sample and starve rare strata
    of a meaningful comparison in the per-forest-type error analysis
    (paper M6). Round-robin caps any single stratum at
    `max_stratum_multiplier` times the equal-share target, so the result
    may land under n_target if rare strata are scarce - that's reported
    (pool sizes returned), not silently backfilled away.
    """
    import random
    from collections import defaultdict

    by_stratum = defaultdict(list)
    for f in features:
        props = f["properties"]
        key = (props.get("forest_type", "other"), props.get("agb_bin", "low"))
        by_stratum[key].append(f)

    rng = random.Random(seed)
    for v in by_stratum.values():
        rng.shuffle(v)

    n_strata = max(len(by_stratum), 1)
    per_stratum = max(n_target // n_strata, 1)
    cap = per_stratum * max_stratum_multiplier

    keys = list(by_stratum.keys())
    rng.shuffle(keys)
    cursors = {k: 0 for k in keys}
    chosen = []
    while len(chosen) < n_target:
        progressed = False
        for k in keys:
            if len(chosen) >= n_target:
                break
            if cursors[k] >= cap or cursors[k] >= len(by_stratum[k]):
                continue
            chosen.append(by_stratum[k][cursors[k]])
            cursors[k] += 1
            progressed = True
        if not progressed:
            break  # every stratum either capped or exhausted

    return chosen, {k: len(v) for k, v in by_stratum.items()}


def month_bounds(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    return start, end


def s1_month_image(year, month, geom):
    start, end = month_bounds(year, month)
    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    # A pass with no acquisition that month (seen in Finland, not in Spain) is
    # filled with -9999, BioMassters' S1 nodata value, which dataset.read_imgs
    # zeroes out -- instead of failing on an image with no bands.
    nodata = ee.Image.constant([-9999, -9999]).rename(S1_BANDS).toFloat()

    def pass_median(direction):
        sub = coll.filter(ee.Filter.eq("orbitProperties_pass", direction)).select(S1_BANDS)
        # unmask(sameFootprint=False): a pass can exist but cover only part of
        # the chip; without it the uncovered part exports as -inf.
        med = sub.median().unmask(-9999, sameFootprint=False).toFloat()
        return ee.Image(ee.Algorithms.If(sub.size().gt(0), med, nodata))

    asc = pass_median("ASCENDING")
    desc = pass_median("DESCENDING")
    img = ee.Image.cat(
        [asc.select(["VV"], ["VV_asc"]), asc.select(["VH"], ["VH_asc"]),
         desc.select(["VV"], ["VV_desc"]), desc.select(["VH"], ["VH_desc"])]
    )
    return img.clip(geom)


def s2_month_image(year, month, geom):
    start, end = month_bounds(year, month)
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(start, end)
        .select(S2_BANDS)
    )
    return coll.median().clip(geom)


def download_image(image, geom, out_path, retries=3):
    params = {"region": geom, "dimensions": CHIP_PIXELS, "crs": PROJ_EPSG, "format": "GEO_TIFF"}
    for attempt in range(retries):
        try:
            url = image.getDownloadURL(params)
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            content = resp.content
            if content[:2] == b"PK":  # zipped response, unwrap single tif
                zf = zipfile.ZipFile(io.BytesIO(content))
                names = zf.namelist()
                content = zf.read(names[0])
            with open(out_path, "wb") as f:
                f.write(content)
            return True
        except Exception as exc:
            wait = 5 * (attempt + 1)
            print(f"  retry {attempt+1}/{retries} for {out_path.name}: {exc} (waiting {wait}s)", flush=True)
            time.sleep(wait)
    return False


def export_chip_imagery(chip_id, geom, year, out_dir):
    ok_months = []
    for month in range(1, 13):
        s1_path = out_dir / f"{chip_id}_S1_{month-1:02d}.tif"
        if not s1_path.exists():
            s1_img = s1_month_image(year, month, geom)
            download_image(s1_img, geom, s1_path)

        s2_path = out_dir / f"{chip_id}_S2_{month-1:02d}.tif"
        if not s2_path.exists():
            s2_img = s2_month_image(year, month, geom)
            has_s2 = s2_img.bandNames().size().gt(0).getInfo()
            if has_s2:
                download_image(s2_img, geom, s2_path)
            else:
                ok_months.append(month - 1)
    return ok_months


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_ee()

    print("Loading input layers...", flush=True)
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    country_geom = countries.filter(ee.Filter.eq("country_na", args.country)).geometry()

    agb_collection = ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
    agb_image = ee.Image(
        agb_collection.filterDate(
            ee.Date.fromYMD(args.agb_year, 1, 1), ee.Date.fromYMD(args.agb_year + 1, 1, 1)
        ).first()
    )

    # Forest mask derived from CORINE (100 m, same native resolution as ESA
    # CCI AGB) rather than WorldCover (10 m): aggregating a 10 m mask up to
    # a 2560 m grid cell needs ~98800 input pixels per output pixel once
    # EPSG:3035 reprojection distortion is accounted for, which exceeds
    # Earth Engine's reduceResolution maxPixels hard cap (65536). CORINE
    # avoids the cross-resolution mismatch entirely and doubles as the
    # forest-type stratification layer.
    corine_img = ee.Image("COPERNICUS/CORINE/V20/100m/2018").select("landcover")
    forest_mask = corine_img.eq(311).Or(corine_img.eq(312)).Or(corine_img.eq(313)).rename("forest")

    n_samples = args.n_target * args.oversample_factor
    print(f"Sampling {n_samples} candidate parcels (target after filtering/stratifying: {args.n_target})...", flush=True)
    candidates_fc = sample_candidate_parcels(country_geom, agb_image, forest_mask, corine_img, args, n_samples)
    candidates = candidates_fc.getInfo()["features"]
    print(f"Candidates passing forest/AGB filters: {len(candidates)}", flush=True)

    chosen, strata_counts = stratified_subset(candidates, args.n_target, args.seed)
    print(f"Selected {len(chosen)} chips. Strata pool sizes: {strata_counts}", flush=True)

    meta_path = out_dir / "metadata.csv"
    write_header = not meta_path.exists()

    def process_chip(i, feat):
        props = feat["properties"]
        # Derived from coordinates, not a run-order index: an index-based id
        # (e.g. "0000") would get reassigned to a different real-world
        # parcel across runs with different sampling parameters, and the
        # per-month download resume check (skip if file exists) would then
        # silently splice imagery from two different locations into one
        # "chip". Lon/lat make the id stable and self-describing regardless
        # of run history.
        lat_s = f"{props['lat']:.4f}".replace("-", "S").replace(".", "p")
        lon_s = f"{props['lon']:.4f}".replace("-", "W").replace(".", "p")
        prefix = {"Spain": "ES", "Finland": "FI"}.get(args.country, args.country[:2].upper())
        chip_id = f"{prefix}_{args.agb_year}_{lat_s}_{lon_s}"
        # feat["geometry"] carries its own embedded "crs" (EPSG:3035,
        # verified via round-trip test) - don't pass an extra proj arg.
        geom = ee.Geometry(feat["geometry"])
        if not args.skip_imagery:
            export_chip_imagery(chip_id, geom, args.agb_year, out_dir)
        return chip_id, props

    with open(meta_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["chip_id", "lon", "lat", "agb_mean", "forest_fraction", "forest_type", "agb_bin", "agb_year", "grid_size_m"]
            )

        done = 0
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_chip, i, feat): i for i, feat in enumerate(chosen)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    chip_id, props = fut.result()
                except Exception as exc:
                    print(f"[FAILED] chip index {i}: {exc}", flush=True)
                    continue
                done += 1
                elapsed = time.time() - t_start
                rate = elapsed / done
                remaining = (len(chosen) - done) * rate / max(args.workers, 1)
                print(f"[{done}/{len(chosen)}] {chip_id} lon={props['lon']:.4f} lat={props['lat']:.4f} "
                      f"agb={props['agb_mean']:.1f} type={props['forest_type']} "
                      f"(elapsed={elapsed/60:.1f}m, est. remaining~{remaining/60:.1f}m)", flush=True)
                writer.writerow(
                    [chip_id, props["lon"], props["lat"], props["agb_mean"], props["forest_fraction"],
                     props["forest_type"], props["agb_bin"], args.agb_year, GRID_SIZE_M]
                )
                f.flush()

    print(f"Done. Metadata at {meta_path}", flush=True)


if __name__ == "__main__":
    main()
