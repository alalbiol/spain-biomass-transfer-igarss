"""Recompute every reported result from data/ and check it against expected/.

    python scripts/reproduce.py            # all analyses (about one minute on a laptop CPU)
    python scripts/reproduce.py --check    # only compare existing outputs/ with expected/

Exits non-zero if any value differs by more than 1e-9 (relative) from the reference.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('OUTPUT_DIR', ROOT / 'outputs'))
DATA = Path(os.environ.get('DATA_DIR', ROOT / 'data'))
# Order matters: ifn_adaptation.py writes the plot table read by the next two.
STEPS = ['ifn_adaptation.py', 'ifn_pretraining.py', 'ifn_baselines.py',
         'cci_blocks.py', 'cci_recipes.py', 'finland_benchmark.py']
NOT_RECOMPUTED = {'finland_benchmark_10m.csv'}  # Table I: needs GPU inference, see REPRODUCING.md
TOL = 1e-9


def close(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return a == b
    if a is None or b is None:
        return a is b
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)


def flatten(obj, prefix=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f'{prefix}/{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f'{prefix}[{i}]')
    else:
        yield prefix, obj


def compare(name):
    ref, new = ROOT / 'expected' / name, OUT / name
    if not new.exists():
        return [f'{name}: missing from {OUT}']
    if name.endswith('.json'):
        a, b = dict(flatten(json.loads(ref.read_text()))), dict(flatten(json.loads(new.read_text())))
    else:
        ta, tb = pd.read_csv(ref), pd.read_csv(new)
        if list(ta.columns) != list(tb.columns) or len(ta) != len(tb):
            return [f'{name}: different columns or row count']
        a = {(i, c): v for c in ta.columns for i, v in enumerate(ta[c].tolist())}
        b = {(i, c): v for c in tb.columns for i, v in enumerate(tb[c].tolist())}
    if a.keys() != b.keys():
        return [f'{name}: different fields']
    return [f'{name} {k}: expected {a[k]}, got {b[k]}' for k in a if not close(a[k], b[k])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--check', action='store_true', help='compare existing outputs only')
    args = parser.parse_args()
    if not args.check:
        if not (DATA / 'ifn').is_dir() or not (DATA / 'cci').is_dir():
            sys.exit(f'{DATA} must contain ifn/ and cci/: download the EOTDL snapshot (see README.md).')
        for step in STEPS:
            print(f'== {step}', flush=True)
            subprocess.run([sys.executable, str(ROOT / 'analysis' / step)], check=True, cwd=ROOT / 'analysis')
    names = sorted(p.name for p in (ROOT / 'expected').iterdir() if p.name not in NOT_RECOMPUTED)
    problems = [msg for name in names for msg in compare(name)]
    for msg in problems[:20]:
        print('MISMATCH', msg)
    if problems:
        sys.exit(f'{len(problems)} values differ from expected/.')
    print(f'All {len(names)} reference files reproduced (tolerance {TOL:g}).')


if __name__ == '__main__':
    main()
