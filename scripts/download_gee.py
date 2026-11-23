"""Download the study's Sentinel-1/2 inputs from Google Earth Engine.

    export EE_PROJECT=<your Earth Engine-enabled Cloud project>
    python scripts/download_gee.py chips --dataset spain --recipe best --limit 2 --out-dir downloads/spain_best
    python scripts/download_gee.py ifn --limit 2 --out-dir downloads/ifn

`chips` exports the 256 x 256 (2.56 km) chips of the CCI comparison as monthly GeoTIFFs, on
the original EPSG:3035 grid of each chip (data/cci/<dataset>/chips.csv):
  <chip_id>_S1_<mm>.tif  VV/VH ascending and descending, dB, per-orbit monthly median (-9999 = no pass)
  <chip_id>_S2_<mm>.tif  ten L2A bands plus a cloud band:
      recipe median: pixelwise monthly median of all scenes, Sen2Cor MSK_CLDPRB (0-100)
      recipe best:   the least-cloudy scene covering the whole chip, s2cloudless probability x 2.55
                     (0-255); a month without such a scene has no S2 file
Recipe best re-uses the scene chosen in the study (best_scene_months.csv) unless --reselect is
given. <mm> is the month index 00-11 of the chip year (2021).

`ifn` exports the 128 x 128 (1.28 km) chip centred on each inventory plot (data/ifn/plots.csv),
for the plot's image year, with recipe best, as one compressed array file per plot:
  <chip_id>.npz  s2 uint16 [12, 11, 128, 128], s1 float32 [12, 4, 128, 128], missing bool [12], cld float32 [12]

The image definitions are those of the original exports (pipeline/cci/export_validation_set.py,
export_bm_style_chips.py and pipeline/ifn/export_ifn_chips.py). Earth Engine collections are
occasionally reprocessed, so a new download can differ slightly from the original inputs.
"""
import argparse
import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('DATA_DIR', ROOT / 'data'))
CRS = 'EPSG:3035'
S1_BANDS = ['VV', 'VH']
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']


# ---------------------------------------------------------------- image definitions
def s1_month(year, month, geom):
    """Per-orbit monthly median of Sentinel-1 IW GRD VV/VH (dB); -9999 for a missing orbit."""
    start = ee.Date.fromYMD(year, month, 1)
    coll = (ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(geom).filterDate(start, start.advance(1, 'month'))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    nodata = ee.Image.constant([-9999, -9999]).rename(S1_BANDS).toFloat()

    def pass_median(direction):
        sub = coll.filter(ee.Filter.eq('orbitProperties_pass', direction)).select(S1_BANDS)
        med = sub.median().unmask(-9999, sameFootprint=False).toFloat()
        return ee.Image(ee.Algorithms.If(sub.size().gt(0), med, nodata))
    asc, desc = pass_median('ASCENDING'), pass_median('DESCENDING')
    return ee.Image.cat([asc.select(['VV'], ['VV_asc']), asc.select(['VH'], ['VH_asc']),
                         desc.select(['VV'], ['VV_desc']), desc.select(['VH'], ['VH_desc'])])


def s2_median(year, month, geom):
    """Recipe median: pixelwise median of all L2A scenes in the month, with Sen2Cor MSK_CLDPRB."""
    start = ee.Date.fromYMD(year, month, 1)
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(geom)
            .filterDate(start, start.advance(1, 'month')).select(S2_BANDS + ['MSK_CLDPRB']).median())


def s2_best_selection(year, month, geom):
    """Recipe best: scenes covering the whole chip, ranked by chip-mean s2cloudless probability."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(geom).filterDate(start, end)
    cp = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterBounds(geom).filterDate(start, end)
    joined = ee.ImageCollection(ee.Join.saveFirst('cp').apply(
        s2, cp, ee.Filter.equals(leftField='system:index', rightField='system:index'))).filter(ee.Filter.notNull(['cp']))

    def score(im):
        im = ee.Image(im)
        prob = ee.Image(im.get('cp')).select('probability')
        cld = prob.reduceRegion(ee.Reducer.mean(), geom, 20, maxPixels=1e8).get('probability')
        valid = im.select('B2').mask().reduceRegion(ee.Reducer.mean(), geom, 20, maxPixels=1e8).get('B2')
        return im.set({'chip_cld': cld, 'chip_valid': valid})
    scored = joined.map(score).filter(ee.Filter.gte('chip_valid', 0.999)).filter(ee.Filter.notNull(['chip_cld']))
    n = scored.size()
    best = ee.Image(scored.sort('chip_cld').first())
    return ee.Dictionary({'n': n, 'cld': ee.Algorithms.If(n.gt(0), best.get('chip_cld'), None),
                          'idx': ee.Algorithms.If(n.gt(0), best.get('system:index'), None)})


def s2_scene(idx):
    """One L2A scene with its s2cloudless probability scaled to 0-255 (band CLP)."""
    im = ee.Image(f'COPERNICUS/S2_SR_HARMONIZED/{idx}')
    clp = ee.Image(f'COPERNICUS/S2_CLOUD_PROBABILITY/{idx}').select('probability').multiply(2.55).rename('CLP')
    return im.select(S2_BANDS).addBands(clp).unmask(0, sameFootprint=False).toFloat()


# ---------------------------------------------------------------- download helpers
def fetch(image, geom, pixels, retries=5):
    """GeoTIFF bytes of `image` over `geom` on a pixels x pixels grid in EPSG:3035."""
    params = {'region': geom, 'dimensions': f'{pixels}x{pixels}', 'crs': CRS, 'format': 'GEO_TIFF'}
    for attempt in range(retries):
        try:
            resp = requests.get(image.getDownloadURL(params), timeout=300)
            resp.raise_for_status()
            content = resp.content
            if content[:2] == b'PK':  # zipped response holding one GeoTIFF
                zf = zipfile.ZipFile(io.BytesIO(content))
                content = zf.read(zf.namelist()[0])
            return content
        except Exception as exc:  # noqa: BLE001 - rate limits and transient HTTP errors
            if attempt == retries - 1:
                raise
            wait = 10 * 2 ** attempt if '429' in str(exc) else 5 * (attempt + 1)
            print(f'  retry {attempt + 1}/{retries}: {str(exc)[:100]} (waiting {wait}s)', flush=True)
            time.sleep(wait)


def with_retries(fn, retries=5):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(10 * 2 ** attempt if '429' in str(exc) else 5 * (attempt + 1))


# ---------------------------------------------------------------- chips of the CCI comparison
def export_cci_chip(chip, recipe, scenes, out_dir, year, reselect):
    geom = ee.Geometry.Rectangle([chip.left_3035, chip.top_3035 - 2560, chip.left_3035 + 2560, chip.top_3035],
                                 proj=CRS, geodesic=False)
    for m in range(12):
        path = out_dir / f'{chip.chip_id}_S1_{m:02d}.tif'
        if not path.exists():
            path.write_bytes(fetch(s1_month(year, m + 1, geom).clip(geom), geom, 256))
    if recipe == 'median':
        for m in range(12):
            path = out_dir / f'{chip.chip_id}_S2_{m:02d}.tif'
            image = s2_median(year, m + 1, geom)
            if not path.exists() and with_retries(lambda: image.bandNames().size().gt(0).getInfo()):
                path.write_bytes(fetch(image.clip(geom), geom, 256))
        return
    if reselect or scenes is None:
        sel = with_retries(lambda: ee.List([s2_best_selection(year, m, geom) for m in range(1, 13)]).getInfo())
        chosen = {m: d['idx'] for m, d in enumerate(sel) if d['n'] > 0}
    else:
        chosen = {r.month: r.scene for r in scenes.itertuples() if r.n_scenes > 0}
    for m, idx in chosen.items():
        path = out_dir / f'{chip.chip_id}_S2_{m:02d}.tif'
        if not path.exists():
            path.write_bytes(fetch(s2_scene(idx).clip(geom), geom, 256))


def run_chips(args):
    chips = pd.read_csv(DATA / 'cci' / args.dataset / 'chips.csv')
    if args.chips:
        chips = chips[chips.chip_id.isin(args.chips)]
    if args.limit:
        chips = chips.head(args.limit)
    months = None
    if args.recipe == 'best' and not args.reselect:
        months = pd.read_csv(DATA / 'cci' / args.dataset / 'best_scene_months.csv')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def work(chip):
        scenes = None if months is None else months[months.chip_id == chip.chip_id]
        export_cci_chip(chip, args.recipe, scenes, args.out_dir, int(chip.agb_year), args.reselect)
        print(f'{chip.chip_id}: done', flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, chips.itertuples()))


# ---------------------------------------------------------------- inventory plots
def export_ifn_chip(plot, out_dir, px=128):
    import rasterio
    from pyproj import Transformer
    out = out_dir / f'{plot.chip_id}.npz'
    if out.exists():
        return
    cx, cy = Transformer.from_crs('EPSG:4326', CRS, always_xy=True).transform(plot.lon, plot.lat)
    half = px * 10 / 2  # the plot centre is the corner shared by pixels 63 and 64
    geom = ee.Geometry.Rectangle([cx - half, cy - half, cx + half, cy + half], proj=CRS, geodesic=False)
    year = int(plot.image_year)

    def read(content):
        with rasterio.io.MemoryFile(content) as mf, mf.open() as ds:
            return ds.read()
    sel = with_retries(lambda: ee.List([s2_best_selection(year, m, geom) for m in range(1, 13)]).getInfo())
    have = [m for m, d in enumerate(sel) if d['n'] > 0]
    s2 = np.zeros((12, 11, px, px), dtype=np.uint16)
    if have:
        arr = read(fetch(ee.Image.cat([s2_scene(sel[m]['idx']) for m in have]), geom, px))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).clip(0, 65535)
        s2[have] = arr.reshape(len(have), 11, px, px).astype(np.uint16)
    s1_img = ee.Image.cat([s1_month(year, m, geom).select(['VV_asc', 'VH_asc', 'VV_desc', 'VH_desc'])
                           for m in range(1, 13)])
    s1 = read(fetch(s1_img, geom, px))
    s1 = np.nan_to_num(s1.reshape(12, 4, px, px).astype(np.float32), nan=-9999, posinf=-9999, neginf=-9999)
    missing = np.array([d['n'] == 0 for d in sel])
    cld = np.array([np.nan if d['cld'] is None else d['cld'] for d in sel], dtype=np.float32)
    tmp = out.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, s2=s2, s1=s1, missing=missing, cld=cld)
    tmp.replace(out)


def run_ifn(args):
    plots = pd.read_csv(args.plots)
    if args.chips:
        plots = plots[plots.chip_id.isin(args.chips)]
    if args.limit:
        plots = plots.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def work(plot):
        export_ifn_chip(plot, args.out_dir)
        print(f'{plot.chip_id}: done', flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, plots.itertuples()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--project', default=os.environ.get('EE_PROJECT'),
                        help='Earth Engine-enabled Google Cloud project (default: $EE_PROJECT)')
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('chips', help='2.56-km chips of the CCI comparison (monthly GeoTIFFs)')
    c.add_argument('--dataset', choices=['spain', 'finland_control'], default='spain')
    c.add_argument('--recipe', choices=['median', 'best'], required=True)
    c.add_argument('--reselect', action='store_true', help='recipe best: choose the scenes again instead of re-using the study\'s')
    i = sub.add_parser('ifn', help='1.28-km chips centred on the inventory plots (one .npz per plot)')
    i.add_argument('--plots', type=Path, default=DATA / 'ifn/plots.csv')
    for p in (c, i):
        p.add_argument('--out-dir', type=Path, required=True)
        p.add_argument('--chips', nargs='+', help='only these chip_id values')
        p.add_argument('--limit', type=int, default=0, help='only the first N chips (testing)')
        p.add_argument('--workers', type=int, default=4, help='parallel chips (Earth Engine rate limits apply)')
    args = parser.parse_args()
    if not args.project:
        parser.error('set EE_PROJECT or pass --project (see GEE.md)')
    ee.Initialize(project=args.project)
    run_chips(args) if args.command == 'chips' else run_ifn(args)


if __name__ == '__main__':
    main()
