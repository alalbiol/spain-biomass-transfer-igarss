"""
ESA CCI AGB (v5 collection in the sat-io community catalogue, ~100 m) sampled at
every IFN plot centre, in the CCI year closest to the plot's image year
(the collection's latest year is used for later surveys). A reference only:
CCI is itself a model product, here compared with the same field labels as the
Sentinel model. Output: <DATA>/cci_at_plots.csv (chip_id, cci_year, cci_agb, cci_sd).
"""

import os

import ee
import pandas as pd

from ifn_common import DATA


def main():
    ee.Initialize(project=os.environ["EE_PROJECT"])
    df = pd.read_csv(DATA / "plots.csv")
    coll = ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
    years = sorted({int(d[:4]) for d in coll.aggregate_array("system:time_start").map(
        lambda t: ee.Date(t).format("YYYY")).getInfo()})
    print("CCI years available:", years)
    out = []
    for year, g in df.groupby("image_year"):
        cy = min(years, key=lambda y: abs(y - year))
        img = ee.Image(coll.filterDate(ee.Date.fromYMD(cy, 1, 1), ee.Date.fromYMD(cy + 1, 1, 1)).first())
        for i in range(0, len(g), 1000):
            part = g.iloc[i:i + 1000]
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"chip_id": r.chip_id})
                                       for r in part.itertuples()])
            res = img.sampleRegions(fc, scale=100, geometries=False).getInfo()["features"]
            for f in res:
                pr = f["properties"]
                out.append({"chip_id": pr["chip_id"], "cci_year": cy, "cci_agb": pr.get("AGB"), "cci_sd": pr.get("SD")})
        print(year, "->", cy, len(g), flush=True)
    pd.DataFrame(out).to_csv(DATA / "cci_at_plots.csv", index=False)
    print(len(out), "plots sampled")


if __name__ == "__main__":
    main()
