"""Draw the README figures from data/ (docs/figures/*.png).

    python -m pip install -r requirements-figures.txt
    python scripts/make_readme_figures.py

study_overview.png: where the Spanish CCI chips, the inventory plots and the Finnish control
chips are. example_chip.png: one Spanish chip, its imagery, CCI and the two recipes' maps.
Basemaps: Natural Earth (public domain), docs/basemap/.
"""
import os
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('DATA_DIR', ROOT / 'data'))
OUT = ROOT / 'docs/figures'
CRS = 'EPSG:3035'
# Categorical slots 1-3 of the reference palette (validated for scatter/map use).
FOREST = {'broadleaved': '#2a78d6', 'coniferous': '#eb6834', 'mixed': '#1baf7a'}
LAND, BORDER, INK, MUTED = '#ecebe7', '#b9b7b0', '#0b0b0b', '#52514e'
EXAMPLE = 'ES_2021_40p7800_W4p0495'

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'savefig.facecolor': 'white'})


def basemap(ax, countries, xlim, ylim, provinces=None):
    countries.plot(ax=ax, color=LAND, edgecolor=BORDER, linewidth=0.6, zorder=0)
    if provinces is not None:
        provinces.boundary.plot(ax=ax, color=BORDER, linewidth=0.3, zorder=1)
    ax.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
    ax.set_facecolor('white')
    for s in ax.spines.values():
        s.set_visible(False)


def to_points(lon, lat):
    return gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs='EPSG:4326').to_crs(CRS)


def forest_legend(ax, chips):
    handles = [Line2D([], [], marker='s', linestyle='none', markersize=6, markerfacecolor=c, markeredgecolor='white',
                      label=f'{t} ({(chips.forest_type == t).sum()})') for t, c in FOREST.items()]
    ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=8, title='CORINE forest type',
              title_fontsize=8, alignment='left')


def overview():
    countries = gpd.read_file(ROOT / 'docs/basemap/countries.geojson').to_crs(CRS)
    provinces = gpd.read_file(ROOT / 'docs/basemap/spain_provinces.geojson').to_crs(CRS)
    spain_chips = pd.read_csv(DATA / 'cci/spain/chips.csv')
    fin_chips = pd.read_csv(DATA / 'cci/finland_control/chips.csv')
    plots = pd.read_csv(DATA / 'ifn/plots.csv', dtype={'province_code': str})
    iberia = dict(xlim=(2.62e6, 3.82e6), ylim=(1.55e6, 2.50e6))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={'width_ratios': [1, 1, 0.62]}, layout='constrained')
    ax = axes[0]
    basemap(ax, countries, **iberia, provinces=provinces)
    pts = to_points(spain_chips.lon, spain_chips.lat)
    for t, c in FOREST.items():
        sel = (spain_chips.forest_type == t).to_numpy()
        ax.scatter(pts.x[sel], pts.y[sel], s=11, marker='s', color=c, edgecolor='white', linewidth=0.4, zorder=3)
    forest_legend(ax, spain_chips)
    ax.set_title('a  500 chips scored against ESA CCI Biomass')

    ax = axes[1]
    basemap(ax, countries, **iberia, provinces=provinces)
    pts = to_points(plots.lon, plots.lat)
    order = np.argsort(plots.agb_mg_ha.to_numpy())
    sc = ax.scatter(pts.x.to_numpy()[order], pts.y.to_numpy()[order], c=plots.agb_mg_ha.to_numpy()[order], s=2.2,
                    cmap='Greens', vmin=0, vmax=200, linewidth=0, zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01, extend='max')
    cb.set_label('Plot biomass (Mg/ha)')
    cb.outline.set_visible(False)
    ax.set_title(f'b  {len(plots):,} IFN4 pine plots, {plots.province_code.nunique()} provinces')

    ax = axes[2]
    pts = to_points(fin_chips.lon, fin_chips.lat)
    pad = 6e4
    basemap(ax, countries, xlim=(pts.x.min() - 2 * pad, pts.x.max() + pad), ylim=(pts.y.min() - 3 * pad, pts.y.max() + pad))
    for t, c in FOREST.items():
        sel = (fin_chips.forest_type == t).to_numpy()
        ax.scatter(pts.x[sel], pts.y[sel], s=11, marker='s', color=c, edgecolor='white', linewidth=0.4, zorder=3)
    forest_legend(ax, fin_chips)
    ax.set_title(f'c  {len(fin_chips)} Finnish control chips')
    fig.savefig(OUT / 'study_overview.png', dpi=170)
    plt.close(fig)


def example_chip():
    chips = pd.read_csv(DATA / 'cci/spain/chips.csv').set_index('chip_id')
    cci = np.load(DATA / 'cci/spain/cci_100m.npz')
    cci_grid = cci['agb'][list(cci['chip_id']).index(EXAMPLE)]
    with rasterio.open(DATA / f'cci/spain/example_inputs/best/{EXAMPLE}_S2_06.tif') as src:
        rgb = np.dstack([src.read(b) for b in (3, 2, 1)]).astype(float)  # B4, B3, B2
    rgb = np.clip(rgb / 2500, 0, 1) ** 0.8

    def late_ensemble(recipe):
        with rasterio.open(DATA / f'cci/spain/predictions_10m/{recipe}/{EXAMPLE}.tif') as src:
            idx = [i + 1 for i, d in enumerate(src.descriptions) if d.endswith('_late')]
            return src.read(idx).mean(axis=0)

    maps = {r: late_ensemble(r) for r in ('median', 'best')}
    valid = np.isfinite(cci_grid)

    def block_rmse(m):
        blocks = m[:250, :250].reshape(25, 10, 25, 10).mean(axis=(1, 3))
        return np.sqrt(np.mean((blocks[valid] - cci_grid[valid]) ** 2))

    ext = (0, 2.56, 0, 2.56)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.75), layout='constrained')
    axes[0].imshow(rgb, extent=ext)
    axes[0].set_title('Sentinel-2, July 2021')
    cmap = plt.get_cmap('Greens').with_extremes(bad='#c3c2b7')
    kw = dict(cmap=cmap, vmin=0, vmax=300)
    # The 25 x 25 blocks cover the first 250 x 250 pixels; the analysis drops the last 6.
    axes[1].imshow(np.where(valid, cci_grid, np.nan), extent=(0, 2.5, 0.06, 2.56), **kw)
    axes[1].set(xlim=(0, 2.56), ylim=(0, 2.56))
    axes[1].set_title('ESA CCI Biomass 2021 (100 m)')
    axes[1].text(0.03, 0.03, 'gray: no CCI value', transform=axes[1].transAxes, fontsize=8, color=INK,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))
    titles = {'median': 'Monthly median (M)', 'best': 'Least-cloudy scene (B)'}
    for ax, (recipe, m) in zip(axes[2:], maps.items()):
        im = ax.imshow(m, extent=ext, **kw)
        ax.set_title(titles[recipe])
        ax.text(0.03, 0.03, f'RMSE vs CCI {block_rmse(m):.0f} Mg/ha', transform=ax.transAxes, fontsize=8,
                color=INK, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))
    for ax in axes:
        ax.set(xticks=[0, 1, 2], yticks=[0, 1, 2])
        ax.set_xlabel('km')
    cb = fig.colorbar(im, ax=axes[1:], shrink=0.85, pad=0.01, extend='max')
    cb.set_label('Biomass (Mg/ha)')
    cb.outline.set_visible(False)
    c = chips.loc[EXAMPLE]
    fig.suptitle(f'{EXAMPLE} ({c.forest_type}, {c.lat:.2f}° N {abs(c.lon):.2f}° W): late-fusion ensemble at 10 m',
                 x=0.01, ha='left', fontsize=10, color=MUTED)
    fig.savefig(OUT / 'example_chip.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    overview()
    example_chip()
    print('Wrote', *sorted(p.name for p in OUT.glob('*.png')))
