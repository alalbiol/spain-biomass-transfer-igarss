"""Shared paths and metrics for the analysis scripts."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('DATA_DIR', ROOT / 'data'))  # downloaded from EOTDL
IFN = DATA / 'ifn'
CCI = DATA / 'cci'
OUT = Path(os.environ.get('OUTPUT_DIR', ROOT / 'outputs'))


def read(path):
    return pd.read_csv(path, dtype={'province_code': str})


def metrics(y, p):
    e = p - y
    return dict(n=int(len(y)), bias=float(e.mean()), rmse=float(np.sqrt(np.mean(e * e))),
                mae=float(np.abs(e).mean()),
                r2=float(1 - np.sum(e * e) / np.sum((y - y.mean()) ** 2)), r=float(np.corrcoef(y, p)[0, 1]))
