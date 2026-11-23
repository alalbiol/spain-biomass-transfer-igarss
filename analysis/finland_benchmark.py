"""BioMassters test set (2773 chips) pooled to 100 m against the pooled lidar-derived labels
(Section IV-B: ensemble R2 0.944 / RMSE 13.50 early, 0.873 / 20.35 late).

Port of the post-inference part of the original run_finland_pooled_benchmark.py. Predictions
and labels were pooled 10 x 10 (last 6 px dropped) and saved as float32; they are read back
in float64. Writes outputs/finland_benchmark_100m.csv.

Table I (native 10 m) is not recomputed here: it needs the 122 GB BioMassters test features
and GPU inference with the released checkpoints. See REPRODUCING.md.
"""
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

from common import CCI, OUT

BACKBONES = ['convnextv2', 'efficientnet', 'efficientnetv2', 'mobilenetv4', 'resnetrs101']
EPS, OVER, UNDER, SSIM_WIN, SSIM_RANGE = 10.0, 2.0, 0.5, 5, 500.0


def metrics(P, R, ssim_scores):
    err = P - R
    sst = np.sum((R - R.mean()) ** 2)
    return {'n_blocks': len(R), 'rmse': np.sqrt((err ** 2).mean()), 'mae': np.abs(err).mean(),
            'r2': 1 - np.sum(err ** 2) / sst if sst > 0 else np.nan, 'pearson_r': np.corrcoef(P, R)[0, 1],
            'frac_over': np.mean(P > OVER * np.maximum(R, EPS)), 'frac_under': np.mean((R >= EPS) & (P < UNDER * R)),
            'sci_100m': np.nanmean(ssim_scores)}


def score(P, R):
    ssim = [structural_similarity(R[i], P[i], data_range=SSIM_RANGE, win_size=SSIM_WIN) for i in range(len(R))]
    return metrics(P.reshape(-1), R.reshape(-1), ssim)


def main():
    z = np.load(CCI / 'finland_benchmark/pooled_100m.npz')
    models = list(z['models'])
    pred = {m: z['pred'][i].astype('float64') for i, m in enumerate(models)}
    R = z['ground_truth'].astype('float64')
    rows = []
    for b in BACKBONES:
        for mode in ['early', 'late']:
            rows.append({'model': b, 'mode': mode, **score(pred[f'{b}_{mode}'], R)})
    for mode in ['early', 'late']:
        members = np.stack([pred[f'{b}_{mode}'] for b in BACKBONES])
        P = members.mean(axis=0)
        m = score(P, R)
        m['uncertainty_ratio'] = float(members.std(axis=0).mean() / P.mean())
        rows.append({'model': 'ensemble', 'mode': mode, **m})
        print(f"ensemble {mode}: R2 {m['r2']:.3f} RMSE {m['rmse']:.2f}")
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'finland_benchmark_100m.csv', index=False)


if __name__ == '__main__':
    main()
