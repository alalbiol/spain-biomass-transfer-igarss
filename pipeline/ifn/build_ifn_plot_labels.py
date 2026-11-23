"""
Plot-level AGB labels for Spanish pine stands from IFN4 field measurements,
for fine-tuning / evaluating the Finland-trained Sentinel AGB model.

Only IFN4 provinces surveyed in the Sentinel-2 L2A era are used (2016-2023):
Albacete, Ciudad Real, Cuenca, Guadalajara, Toledo, Avila, Salamanca, Zamora,
Extremadura (Badajoz + Caceres), Cadiz, Cordoba, Jaen, Malaga. All of these
report ETRS89 UTM coordinates (IFN4 field-database manual, CoorX/CoorY; ED50
is only used in Galicia, Madrid, Cataluna, ... which are outside the window).
CoorX/CoorY is the plot centre *before* fieldwork, so the real centre can be
off by GPS error (new/relocated plots) or by more for re-found IFN3 plots
(class A1). `plot_class` is kept so that can be checked.

Plot AGB (Mg/ha, dry matter, above ground) includes every tree >= 7.5 cm in the
concentric plot (5/10/15/25 m radii by diameter class; PCMayores) plus pine
"pies menores" 2.5-7.5 cm (PCRegenera CatDes 4, 5 m subplot, d = 5 cm), since
the satellite sees all of them:
  * the five Iberian pines: their own Ruiz-Peinado et al. (2011) equations
    (spain_biomass_equations.py, the same ones used in the paper-4 pine work);
  * other conifers: the P. pinaster equation (proxy);
  * broadleaves: the Q. ilex equation of Ruiz-Peinado et al. (2012) (proxy).
Plots are kept when the five pines hold >= MIN_PINE_SHARE of the basal area,
so the proxy equations touch at most a minority of the biomass.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "use_case_1_simulator/agb_forecast_analysis"))
from ifn_remeasurement_validation import _mdb, _num, trees_per_ha  # noqa: E402
from spain_biomass_equations import (  # noqa: E402
    pinus_halepensis_biomass, pinus_nigra_biomass, pinus_pinaster_biomass, pinus_pinea_biomass,
    pinus_sylvestris_biomass, quercus_ilex_biomass,
)

IFN4_DIR = Path("/media/HD/Data/ifn4")
# database -> nothing else needed; survey year is read per plot from PCParcelas.FechaIni
DATABASES = [
    "Ifn4_Albacete", "Ifn4_CiudadReal", "Ifn4p_Cuenca", "Ifn4_Guadalajara", "Ifn4_Toledo",
    "Ifn4p05-avila", "Ifn4p37-salamanca", "Ifn4p49-zamora", "Ifn4_Extremadura",
    "Ifn4p11", "Ifn4p14", "ifn4p23", "Ifn4p29",
]
PINES = {21: ("Pinus sylvestris", pinus_sylvestris_biomass), 23: ("Pinus pinea", pinus_pinea_biomass),
         24: ("Pinus halepensis", pinus_halepensis_biomass), 25: ("Pinus nigra", pinus_nigra_biomass),
         26: ("Pinus pinaster", pinus_pinaster_biomass)}
# IFN species codes 1-39 are conifers (Pinus, Abies, Juniperus, Cupressus, ...); the rest broadleaves.
CONIFER_MAX_CODE = 39
MIN_PINE_SHARE = 0.8
SMALL_TREE_D_CM = 5.0
SMALL_TREE_RADIUS_M = 5.0
UTM_EPSG = {29: 25829, 30: 25830, 31: 25831}


def tree_agb_kg(code: int, d: float, h: float) -> float:
    if code in PINES:
        return PINES[code][1](d, h).aboveground_kg
    if 0 < code <= CONIFER_MAX_CODE:
        return pinus_pinaster_biomass(d, h).aboveground_kg
    return quercus_ilex_biomass(d, h).aboveground_kg


def key(df: pd.DataFrame, cla: str) -> pd.Series:
    return df.Provincia.astype(str).str.strip().str.zfill(2) + "_" + df.Estadillo.str.strip() + "_" + df[cla].str.strip()


def province_plots(db: str) -> pd.DataFrame:
    path = str(IFN4_DIR / f"{db}.accdb")
    parc = _mdb(path, "PCParcelas")
    parc["plot_id"] = key(parc, "Cla") + parc.Subclase.fillna("").astype(str).str.strip()
    parc["loc_id"] = key(parc, "Cla")
    parc["date"] = pd.to_datetime(parc.FechaIni.str[:8], format="%m/%d/%y", errors="coerce")
    coords = _mdb(path, "PCDatosMap")
    coords["loc_id"] = key(coords, "Clase")
    coords = coords.drop_duplicates("loc_id", keep=False)  # ambiguous locations dropped
    coords["x"], coords["y"], coords["huso"] = _num(coords.CoorX), _num(coords.CoorY), _num(coords.Huso)
    parc = parc.merge(coords[["loc_id", "x", "y", "huso"]], on="loc_id", how="inner")

    t = _mdb(path, "PCMayores")
    t["plot_id"] = key(t, "Cla") + t.Subclase.fillna("").astype(str).str.strip()
    t["code"] = _num(t.Especie)
    t["d"] = (_num(t.Dn1) + _num(t.Dn2)) / 20.0
    t["h"] = _num(t.Ht)
    t = t[(t.d >= 7.5) & (t.h > 1.3) & t.code.notna()].copy()
    t["w"] = trees_per_ha(t.d.to_numpy())
    t["ba"] = t.w * math.pi * (t.d / 200.0) ** 2
    t["agb"] = [tree_agb_kg(int(c), d, h) for c, d, h in zip(t.code, t.d, t.h)]
    t["agb"] = t.agb * t.w / 1000.0  # Mg/ha contribution
    t["is_pine"] = t.code.isin(list(PINES))

    r = _mdb(path, "PCRegenera")
    r["plot_id"] = key(r, "Cla") + r.Subclase.fillna("").astype(str).str.strip()
    r["code"], r["n"], r["h"] = _num(r.Especie), _num(r.NumPies), _num(r.Hm) / 10.0
    r = r[(_num(r.CatDes) == 4) & r.code.isin(list(PINES)) & (r.n > 0)].copy()
    r["h"] = r.h.where(r.h > 1.3, 3.0)
    small_w = 10000.0 / (math.pi * SMALL_TREE_RADIUS_M**2)
    r["agb"] = [PINES[int(c)][1](SMALL_TREE_D_CM, h).aboveground_kg * n * small_w / 1000.0
                for c, h, n in zip(r.code, r.h, r.n)]

    g = t.groupby("plot_id")
    plot = pd.DataFrame({
        "agb_large": g.agb.sum(), "ba_total": g.ba.sum(),
        "ba_pine": t[t.is_pine].groupby("plot_id").ba.sum(),
        "n_trees_measured": g.size(), "trees_per_ha": g.w.sum(),
        "h_mean": t.groupby("plot_id").apply(lambda s: np.average(s.h, weights=s.w), include_groups=False),
    }).fillna({"ba_pine": 0.0})
    pine_ba = t[t.is_pine].groupby(["plot_id", "code"]).ba.sum().reset_index()
    main = pine_ba.sort_values("ba", ascending=False).drop_duplicates("plot_id").set_index("plot_id").code
    plot["main_pine"] = main.map(lambda c: PINES[int(c)][0])
    plot["agb_small"] = r.groupby("plot_id").agb.sum()
    plot = plot.fillna({"agb_small": 0.0})
    plot["agb_mg_ha"] = plot.agb_large + plot.agb_small
    plot["pine_ba_share"] = plot.ba_pine / plot.ba_total
    plot = plot[plot.pine_ba_share >= MIN_PINE_SHARE].reset_index()

    out = plot.merge(parc[["plot_id", "Cla", "Subclase", "date", "x", "y", "huso"]], on="plot_id", how="inner")
    out["plot_class"] = out.Cla.str.strip() + out.Subclase.fillna("").astype(str).str.strip()
    out["database"] = db
    out["province_code"] = out.plot_id.str[:2]
    return out.drop(columns=["Cla", "Subclase"])


def add_lonlat(df: pd.DataFrame) -> pd.DataFrame:
    from pyproj import Transformer
    df = df.copy()
    df["lon"], df["lat"] = np.nan, np.nan
    for huso, epsg in UTM_EPSG.items():
        m = df.huso == huso
        if m.any():
            tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
            df.loc[m, "lon"], df.loc[m, "lat"] = tr.transform(df.x[m].to_numpy(), df.y[m].to_numpy())
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="/media/HD/Data/agb_spain_ifn/plots.csv")
    args = p.parse_args()
    parts = []
    for db in DATABASES:
        part = province_plots(db)
        print(f"{db}: {len(part)} pine-dominated plots", flush=True)
        parts.append(part)
    df = add_lonlat(pd.concat(parts, ignore_index=True))
    df = df[df.lon.notna() & df.date.notna()]
    df["survey_year"] = df.date.dt.year
    # S2 L2A (COPERNICUS/S2_SR_HARMONIZED) starts 2017-03 in Europe: 2016 surveys use 2017 imagery.
    df["image_year"] = df.survey_year.clip(lower=2017)
    df["chip_id"] = "IFN4_" + df.plot_id.str.replace(r"[^0-9A-Za-z_]", "", regex=True)
    df = df.drop_duplicates("chip_id")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} plots -> {args.out}")
    print(df.groupby("main_pine").agb_mg_ha.describe().round(1).to_string())
    print(df.groupby("province_code").size().to_string())


if __name__ == "__main__":
    main()
