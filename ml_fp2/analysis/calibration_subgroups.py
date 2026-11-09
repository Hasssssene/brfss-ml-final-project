"""Calibration + subgroup analysis on the OOF predictions (report sections 10.5/11.1).

- Brier score + reliability curve (10 quantile bins) -> figures/fig09_calibration.png
- OOF AUC by SEXVAR and _RACE subgroups (demographic-association limitation)
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = r'C:\Users\Hassene\Documents\Machine learning final project\ml_fp2'
CKPT = ROOT + r'\data\interim'

from sklearn.calibration import calibration_curve         # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

oof = pd.read_parquet(CKPT + r'\oof_v1.parquet')['oof_prob']
y = pd.read_parquet(CKPT + r'\target_v1.parquet')['y']
X = pd.read_parquet(CKPT + r'\clean_v1.parquet')

# ---- calibration ----
brier = brier_score_loss(y, oof)
frac_pos, mean_pred = calibration_curve(y, oof, n_bins=10, strategy='quantile')
bin_counts, _ = np.histogram(oof, bins=np.quantile(oof, np.linspace(0, 1, 11)))
ece = float(np.average(np.abs(frac_pos - mean_pred), weights=bin_counts))
print(f'Brier score : {brier:.6f}')
print(f'ECE (10 quantile bins): {ece:.4f}')
print(f'OOF mean predicted P: {oof.mean():.4f} | actual rate: {y.mean():.4f}')
for mp, fp in zip(mean_pred, frac_pos):
    print(f'  predicted {mp:.3f} -> observed {fp:.3f}')

fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.plot(mean_pred, frac_pos, 'o-', color='#4C72B0', label='tuned LGBM (OOF)')
ax.plot([0, 1], [0, 1], '--', color='#888888', label='perfectly calibrated')
ax.set_xlabel('Mean predicted P(Fair/Poor)')
ax.set_ylabel('Observed fraction Fair/Poor')
ax.set_title(f'Score reliability (OOF, 437,532 respondents)\nBrier {brier:.4f} - ECE {ece:.3f}')
ax.legend(frameon=False, fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(ROOT + r'\figures\fig09_calibration.png', dpi=150)
print('saved figures/fig09_calibration.png')

# ---- subgroup AUC ----
print('\nOOF AUC by subgroup:')
for col, labels in [('SEXVAR', {1.0: 'male', 2.0: 'female'}),
                    ('_RACE', {1.0: 'white', 2.0: 'black', 3.0: 'am-indian',
                               4.0: 'asian', 5.0: 'pacific', 6.0: 'other',
                               7.0: 'multiracial', 8.0: 'hispanic'})]:
    g = X[col]
    for v, name in labels.items():
        mask = (g == v).to_numpy()
        if mask.sum() and y[mask].nunique() == 2:
            print(f'  {col}={name:<11s} n={mask.sum():>7,} '
                  f'({100*mask.mean():4.1f}%)  AUC {roc_auc_score(y[mask], oof[mask]):.4f}')
