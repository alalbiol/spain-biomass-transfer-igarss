"""Maintainer tool: copy the original Sentinel-1/2 inputs of a few Spanish chips into data/.

data/cci/spain/example_inputs/<recipe>/<chip_id>_S{1,2}_<mm>.tif, exactly as exported in the
study (both recipes share the Sentinel-1 files). They let readers run the models on real
inputs and compare their own Earth Engine downloads (GEE.md).
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPAIN = Path('/media/HD/Data/agb_spain')
CHIPS = ['ES_2021_40p7800_W4p0495',  # coniferous, high biomass (Sierra de Guadarrama)
         'ES_2021_42p7594_W2p5491']  # broadleaved, high biomass
SOURCES = {'median': SPAIN / 'validation', 'best': SPAIN / 'validation_bm'}


def main():
    for recipe, src in SOURCES.items():
        out = ROOT / 'data/cci/spain/example_inputs' / recipe
        out.mkdir(parents=True, exist_ok=False)
        for chip in CHIPS:
            files = sorted(src.glob(f'{chip}_S[12]_*.tif'))
            assert len(files) == 24, (recipe, chip, len(files))
            for f in files:
                shutil.copyfile(f.resolve(), out / f.name)  # resolves the shared S1 links
        print(recipe, len(list(out.iterdir())), 'files')


if __name__ == '__main__':
    main()
