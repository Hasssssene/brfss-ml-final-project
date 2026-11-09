"""EXP-023 — isolation test of the FLSHTMY3 fix (report section 11.1).

Recode the 777777 (don't-know) sentinel to NaN on the otherwise-frozen
matrix, re-run the tuned LightGBM (EXP-020 config) on the canonical folds,
log the result as EXP-023, and print the delta vs EXP-020.
The main pipeline is NOT changed: this measures the wart's impact.
"""
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = r'C:\Users\Hassene\Documents\Machine learning final project\ml_fp2'
CKPT = ROOT + r'\data\interim'

import lightgbm as lgb                                   # noqa: E402
from sklearn.metrics import roc_auc_score                # noqa: E402
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE, N_SPLITS = 42, 10
X = pd.read_parquet(CKPT + r'\clean_v1.parquet')
y = pd.read_parquet(CKPT + r'\target_v1.parquet')['y']
nom_cols = pd.read_csv(CKPT + r'\whitelist_v1.csv').query("role == 'NOM'")['col'].tolist()
assert len(nom_cols) == 54

# -- the fix: one recode, nothing else --
X2 = X.copy()
X2['FLSHTMY3'] = X2['FLSHTMY3'].replace(777777.0, np.nan)
Xd2 = pd.get_dummies(X2, columns=nom_cols, dummy_na=True)
print(f'design: fixed matrix {Xd2.shape} (was 322; FLSHTMY3_777777.0 merged into _nan)')

tuned = dict(num_leaves=31, n_estimators=500, min_child_samples=200,
             learning_rate=0.03)
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
folds = []
for i, (tr, te) in enumerate(cv.split(Xd2, y), 1):
    m = lgb.LGBMClassifier(**tuned, n_jobs=-1, verbose=-1,
                           random_state=RANDOM_STATE).fit(Xd2.iloc[tr], y.iloc[tr])
    a = roc_auc_score(y.iloc[te], m.predict_proba(Xd2.iloc[te])[:, 1])
    folds.append(a)
    print(f'fold {i:2d}: {a:.6f}', flush=True)
scores = np.array(folds)

log = pd.read_csv(ROOT + r'\experiments.csv')
ref = log[log.exp_id == 'EXP-020'].iloc[-1]
print(f'\nEXP-023 fixed-matrix mean: {scores.mean():.6f} +/- {scores.std():.6f}')
print(f'EXP-020 logged mean      : {float(ref.mean_auc):.6f}')
print(f'delta                    : {scores.mean() - float(ref.mean_auc):+.6f}')

row = {'exp_id': 'EXP-023', 'timestamp': datetime.now().isoformat(timespec='seconds'),
       'model': 'LGBM-tuned', 'params': 'FLSHTMY3 777777->NaN (wart fix test)',
       'notes': 'isolation test of the documented FLSHTMY3 wart'}
row.update({f'fold_{i+1:02d}': round(s, 6) for i, s in enumerate(scores)})
row['mean_auc'] = round(scores.mean(), 6)
row['std_auc'] = round(scores.std(), 6)
log = pd.concat([log, pd.DataFrame([row])]).drop_duplicates('exp_id', keep='last')
log.to_csv(ROOT + r'\experiments.csv', index=False)
print('logged as EXP-023')
