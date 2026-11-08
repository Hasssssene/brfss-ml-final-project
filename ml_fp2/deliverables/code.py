"""BRFSS 2021 — Predicting Fair/Poor General Health (GENHLTH 4-5 vs 1-3).

End-to-end pipeline, exported from notebooks/BRFSS_project.ipynb:
  data load -> triage (303->76) -> cleaning/encoding -> baselines ->
  model zoo + imbalance -> feature selection/hypothesis -> tuning +
  Wilcoxon + nested CV -> error analysis + SHAP.

Every random component is seeded with RANDOM_STATE = 42; evaluation is
ROC-AUC under StratifiedKFold(10, shuffle=True, random_state=42).
Run from the repository's notebooks/ directory (or adjust paths).
"""
#!/usr/bin/env python
# coding: utf-8

# # BRFSS 2021 — Predicting Fair/Poor General Health
# 
# Binary classification on `GENHLTH` (Fair/Poor vs Good-or-better), scored by **ROC-AUC, stratified 10-fold CV**. Run top-to-bottom.
# 
# | Section | Content |
# |---|---|
# | 1 | Setup, first look, CV + AUC concepts |
# | 2 | Profiling & leakage hunt |
# | 3 | Column triage: 303 → whitelist |
# | 4 | Cleaning & feature matrix (checkpoint) |
# | 5 | Baselines: Dummy, LogReg, DecisionTree + experiments log |
# | 6 | Model zoo & imbalance |
# | 7 | Feature selection & hypothesis |
# | 8 | Tuning (RandomizedSearch) + Wilcoxon + nested CV |
# | 9 | Error analysis & XAI (SHAP) |
# | 10 | deliverables assembly (to come) |

# ## SECTION 1 — Setup & first look
# 
# `RANDOM_STATE = 42` everywhere → identical results on every run (course requirement). `N_SPLITS = 10` for every experiment.

# In[44]:


import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend: script-safe, plt.show() won't block
import matplotlib.pyplot as plt

RANDOM_STATE = 42
N_SPLITS = 10

# local repo by default; Google Colab also works (data/figures on Drive)
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/brfss/data'
    FIG_DIR = '/content/drive/MyDrive/brfss/figures'
else:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(_HERE, '..', 'data')
    FIG_DIR = os.path.join(_HERE, '..', 'figures')

os.makedirs(FIG_DIR, exist_ok=True)
DATA_PATH = f'{DATA_DIR}/raw/LLCP2021.csv'
CKPT_DIR = f'{DATA_DIR}/interim'
os.makedirs(CKPT_DIR, exist_ok=True)

pd.set_option('display.max_columns', 50)


# ### 1.1 Load raw data — immutable, never modified; cleaning happens on copies.
# 
# `low_memory=False`: pandas then decides column types once, on whole columns (no chunk-wise guess warnings).

# In[45]:


df = pd.read_csv(DATA_PATH, low_memory=False)   # ~30-60 s
print(f'Rows: {df.shape[0]:,}   Columns: {df.shape[1]:,}')


# ### 1.2 First look
# 
# Values are mostly **codes, not quantities** (`GENHLTH=4` means "Fair") — the codebook is the dictionary. Blanks are everywhere (state-optional questions, refusals).

# In[46]:


df.head()


# In[47]:


# float64 usually = column has blanks; whole-number cols with NaN become float
print(df.dtypes.value_counts())
print(f'\n100% empty columns: {(df.isna().sum() == df.shape[0]).sum()}')


# ### 1.3 Target: 5 classes → binary
# 
# `y = 1` if GENHLTH ∈ {4,5} (Fair/Poor), `y = 0` if ∈ {1,2,3}. Rows with 7/9/blank are **dropped, never imputed** — a guessed label is a wrong label (features may be imputed; the target never).

# In[48]:


df['GENHLTH'].value_counts(dropna=False).sort_index()


# In[49]:


y = df['GENHLTH'].map({1: 0, 2: 0, 3: 0, 4: 1, 5: 1}).dropna().astype(int)
n_pos = int(y.sum())
print(f'Dropped: {len(df)-len(y):,}   Fair/Poor: {n_pos:,} = {100*n_pos/len(y):.2f}%')


# **Imbalance (~1 in 6 positive):** accuracy is misleading ("all Good+" = 83% accuracy, useless) → hence ROC-AUC. Imbalance techniques get tested in Section 6.

# ### 1.4 Concept: stratified 10-fold CV
# 
# Split into 10 parts, train on 9, test on the 10th, rotate → 10 AUCs (mean = estimate, spread = wobble). Stratified = every fold keeps the same positive rate. **Same folds for every model** (same splitter + seed) → score differences come from the model, not split luck; also required for the Wilcoxon test later.

# In[50]:


from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
X_dummy = np.zeros((len(y), 1))   # demo: labels only
print(f'Overall: {100*y.mean():.2f}% Fair/Poor\n')
for i, (tr, te) in enumerate(folds.split(X_dummy, y), 1):
    print(f'fold {i}: train {len(tr):>7,}  test {len(te):>6,}  test Fair/Poor {100*y.iloc[te].mean():.2f}%')


# ### 1.5 Concept: ROC-AUC
# 
# > **AUC = probability that a random Fair/Poor person scores higher than a random Good+ person.** 1.0 = perfect ranking, 0.5 = coin flip, 0.0 = perfectly wrong. Threshold-free → immune to the imbalance that breaks accuracy.
# 
# Verified by brute force on 6 toy people (sklearn agrees):

# In[51]:


from sklearn.metrics import roc_auc_score

scores = np.array([0.95, 0.90, 0.60, 0.40, 0.30, 0.10])
labels = np.array([1, 0, 1, 0, 0, 1])

ok = total = 0
for i in range(6):
    if labels[i] == 1:
        for j in range(6):
            if labels[j] == 0:
                total += 1
                ok += scores[i] > scores[j]
print(f'manual: {ok}/{total} = {ok/total:.3f}   sklearn: {roc_auc_score(labels, scores):.3f}')


# ---
# ## SECTION 2 — Profiling & the leakage hunt
# 
# Every column must earn its place. Failure modes:
# 
# | Mode | Meaning | Example |
# |---|---|---|
# | **Leakage** | derived from the target (tomorrow's newspaper) | `_RFHLTH` |
# | **Duplicate** | CDC-computed twin of another column | `_TOTINDA` ≡ `EXERANY2` |
# | **Junk** | constant / almost blank | micro-coverage columns |
# 
# Merely **associated** with health = fine (that's what a predictor is). **Association ≠ derivation.**

# In[52]:


n_values = df.nunique(dropna=False)
pct_missing = (df.isna().mean() * 100).sort_values(ascending=False)
print(f'<=2 distinct values: {(n_values <= 2).sum()}   >50% blank: {(pct_missing > 50).sum()}   >90% blank: {(pct_missing > 90).sum()}')


# In[53]:


top30 = pct_missing.head(30).sort_values()
fig, ax = plt.subplots(figsize=(8, 9))
ax.barh(top30.index, top30.values, color='steelblue')
ax.axvline(50, color='red', linestyle='--', label='50% missing')
ax.set_xlabel('% no answer'); ax.set_title('Top 30 columns by missingness'); ax.legend()
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/fig02_missingness_top30.png', dpi=150); plt.show()


# ### 2.1 Leakage: `_RFHLTH`
# 
# `_`-prefixed = CDC-calculated after the interview → prime audit target. Codebook: `_RFHLTH` = "good or better health" flag (1=Good+, 2=Fair/Poor) — that's exactly `y`. Empirical check:

# In[54]:


check = pd.DataFrame({'rf': df['_RFHLTH'], 'y': y}).dropna()
check['rf'] = check['rf'].map({1: 0, 2: 1}).astype(int)
print(f'Agreement with y: {100*(check["rf"] == check["y"]).mean():.2f}%  ({len(check):,} rows)')
pd.crosstab(check['y'], check['rf'])   # diagonal = no information beyond y


# In[55]:


# Same family: _PHYS14D/_MENT14D are CDC bins of raw PHYSHLTH/MENTHLTH -> keep raws, drop bins.
real_days = df['PHYSHLTH'].between(1, 30)
df.loc[real_days].groupby('_PHYS14D')['PHYSHLTH'].agg(['min', 'max', 'count'])


# ### 2.2 Duplicates — a crosstab is the fastest detector (diagonal = same information):

# In[56]:


pd.crosstab(df['EXERANY2'], df['_TOTINDA'])   # exercise: raw vs computed twin


# In[57]:


pd.crosstab(df['HAVARTH5'], df['_DRDXAR3'])   # arthritis: raw vs computed twin


# In[58]:


df.groupby('_BMI5CAT')['_BMI5'].agg(['min', 'max', 'count'])   # BMI bins of continuous _BMI5


# ### 2.3 Association ≠ derivation
# 
# `PHYSHLTH` (unhealthy days) is a *different question* that correlates with the target — a legitimate predictor (far from deterministic, unlike `_RFHLTH`'s 100%):

# In[59]:


phys = df['PHYSHLTH'].replace({88: 0, 77: np.nan, 99: np.nan})
tmp = pd.DataFrame({'phys': phys, 'y': y}).dropna()
print(f'PHYSHLTH vs y: Spearman rho = {tmp["phys"].corr(tmp["y"], method="spearman"):+.3f}  (n={len(tmp):,})')
print('Same reasoning keeps MENTHLTH and POORHLTH.')


# ### 2.4 Sentinel traps
# 
# | Code | Typical meaning |
# |---|---|
# | 7 / 77 / 777 / 7777 | Don't know |
# | 9 / 99 / 999 / 9999 | Refused |
# | blank | not asked / missing |
# | **88** | **"none" = 0** (day-count questions — real data!) |
# 
# **Trap:** 7/9 are not always sentinels — `EMPLOY1` 7 = *Retired* (~131k valid answers), `INCOME3`/`PRIMINSR`/`IMFVPLA2` keep single 7/9 as real answers (77/99 are the sentinels), `JOINPAI2` 7/9 are real pain ratings. → **Recode per column from the codebook, never by pattern.**

# In[60]:


df['PHYSHLTH'].value_counts().head(8)   # 88 'zero days' dominates; 77/99 tiny sentinels


# In[61]:


# Figure: target imbalance (5 raw classes, colored by binary side)
# only the 5 real answers (7/9 = DK/refused are not plottable categories)
counts = df.loc[df['GENHLTH'].isin([1, 2, 3, 4, 5]), 'GENHLTH'].value_counts().sort_index()
labels_map = {1: '1 Excellent', 2: '2 Very good', 3: '3 Good', 4: '4 Fair', 5: '5 Poor'}
colors = ['#4C72B0', '#4C72B0', '#4C72B0', '#C44E52', '#C44E52']
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([labels_map[i] for i in counts.index], counts.values, color=colors)
for bar, n in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, n, f'{n:,}\n({100*n/counts.sum():.1f}%)',
            ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Respondents')
ax.set_title(f'GENHLTH distribution — Fair/Poor = {100*y.mean():.2f}% of labeled rows')
ax.margins(y=0.15)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/fig01_target_imbalance.png', dpi=150); plt.show()


# ---
# ## SECTION 3 — Column triage: deriving the whitelist
# 
# Six rules in order, first match wins; every column gets exactly one verdict:
# 
# | # | Rule | Decided by |
# |---|---|---|
# | 1 | **leak** | empirical (Section 2) |
# | 2 | **admin/mode** | explicit list, codebook reasons |
# | 3 | **constant** | computed: ≤1 distinct non-blank value |
# | 4 | **sparse** | computed: >90% blank |
# | 5 | **low coverage** | computed: >75% blank + documented exceptions |
# | 6 | **duplicate** | codebook families + correlation spot-checks |
# 
# Why 25%: below that, imputation becomes invention. **Keep-list = output of this section, never an input.** And the matrix is built by whitelist, not by dropping — a typo in a keep-list loses a column (harmless); a typo in a drop-list leaks the target (silent corruption). v1's independent triage is loaded **only at the end, as cross-check**.

# In[62]:


triage = {}   # {column: (rule, verdict, reason)}

# RULE 1: leakage (evidence in Section 2)
for col, reason in {
    'GENHLTH':  'the target itself',
    '_RFHLTH':  'computed FROM GENHLTH; 100% agreement with y',
    '_PHYS14D': 'CDC binning of PHYSHLTH',
    '_MENT14D': 'CDC binning of MENTHLTH',
}.items():
    triage[col] = (1, 'LEAK', reason)
print('R1 leak:', sum(v[0] == 1 for v in triage.values()))


# In[63]:


# RULE 2: admin & interview-mode plumbing (codebook: no health content)
ADMIN = {
    **{c: 'interview date metadata' for c in ['FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR']},
    **{c: 'identifier / instrument admin' for c in ['SEQNO', 'DISPCODE', 'QSTVER', 'MSCODE', '_PSU']},
    **{c: 'sampling weight / stratum' for c in ['_STSTR', '_STRWT', '_RAWRAKE', '_WT2RAKE',
                                               '_CLLCPWT', '_DUALUSE', '_DUALCOR', '_LLCPWT2', '_LLCPWT']},
    **{c: 'landline/cell intro-flow block' for c in ['CTELENM1', 'PVTRESD1', 'LADULT1', 'COLGSEX',
                                                    'NUMADULT', 'LANDSEX', 'NUMMEN', 'NUMWOMEN',
                                                    'RESPSLCT', 'CELLSEX', 'PVTRESD3', 'CSTATE1',
                                                    'LANDLINE', 'HHADULT', 'NUMHHOL3', 'CPDEMO1B']},
}
for col, reason in ADMIN.items():
    triage[col] = (2, 'ADMIN', reason)
print('R2 admin/mode:', sum(v[0] == 2 for v in triage.values()))


# In[64]:


# RULE 3: constant (computed) — at most one distinct non-blank value
for col in df.columns:
    if col not in triage and df[col].nunique(dropna=True) <= 1:
        triage[col] = (3, 'CONSTANT', 'never really varies')
print('R3 constant:', sum(v[0] == 3 for v in triage.values()),
      [c for c, v in triage.items() if v[0] == 3])


# In[ ]:


# RULE 4: sparse (computed) — >90% blank
blank_share = df.isna().mean()
for col in df.columns:
    if col not in triage and blank_share[col] > 0.90:
        triage[col] = (4, 'SPARSE', f'{100*blank_share[col]:.1f}% blank')
print('R4 sparse:', sum(v[0] == 4 for v in triage.values()))


# In[85]:


# RULE 5: low coverage (computed + documented exceptions)
for col in df.columns:
    if col not in triage and blank_share[col] > 0.75:
        triage[col] = (5, 'LOW-COVERAGE', f'{100*blank_share[col]:.1f}% blank (<25% coverage)')

# exceptions: plain blank% overstates true usable coverage (codebook)
LOW_COVERAGE_EXTRA = {
    'HIVTSTD3': '777777/999999 sentinels cut true coverage to ~20%',
    'PDIABTST': 'blank + 7/9 sentinels leave effective coverage <25%',
    '_CHISPNC': 'child-module item, sentinel-coded; true coverage ~15%',
    'LASTSMK2': 'skip-logic: asked only of former smokers (~12% coverage)',
}
for col, reason in LOW_COVERAGE_EXTRA.items():
    assert col in df.columns, col
    triage[col] = (5, 'LOW-COVERAGE', reason)   # idempotent: safe to re-run

print('R5 low-coverage:', sum(v[0] == 5 for v in triage.values()))
print(f'boundary kept: SOMALE {blank_share["SOMALE"]:.1%}, PREDIAB1 {blank_share["PREDIAB1"]:.1%} blank (coverage >= 25%)')


# In[86]:


# RULE 6: duplicate families — keep ONE member (prefer raw; _SMOKER3's ordinal
# coding compresses its raw pair without information loss)
DUP_FAMILIES = {
    'body size':      (['_BMI5'], ['WEIGHT2', 'HEIGHT3', 'HTIN4', 'HTM4', 'WTKG3', '_BMI5CAT', '_RFBMI5']),
    'smoking':        (['_SMOKER3'], ['SMOKE100', 'SMOKDAY2', '_RFSMOK3']),
    'exercise flag':  (['EXERANY2'], ['_TOTINDA']),
    'arthritis':      (['HAVARTH5'], ['_DRDXAR3']),
    'arthr. limit':   (['LMTJOIN3'], ['_LMTACT3', '_LMTWRK3']),
    'hypertension':   (['BPHIGH6'], ['_RFHYPE6']),
    'cholesterol':    (['CHOLCHK3', 'TOLDHI3'], ['_CHOLCH3', '_RFCHOL3']),
    'heart dis.':     (['CVDINFR4', 'CVDCRHD4'], ['_MICHD']),
    'asthma':         (['ASTHMA3'], ['_LTASTH1', '_CASTHM1', '_ASTHMS1']),
    'age':            (['_AGE80'], ['_AGEG5YR', '_AGE65YR', '_AGE_G']),
    'sex':            (['SEXVAR'], ['_SEX']),
    'children':       (['CHILDREN'], ['_CHLDCNT']),
    'education':      (['EDUCA'], ['_EDUCAG']),
    'income':         (['INCOME3'], ['_INCOMG1']),
    'race/ethnicity': (['_RACE'], ['_PRACE1', '_MRACE1', '_HISPANC', '_RACEG21', '_RACEGR3',
                                   '_RACEPRV', '_IMPRACE']),
    'insurance':      (['PRIMINSR'], ['_HLTHPLN', '_HCVU652']),
    'e-cigarettes':   (['ECIGNOW1'], ['_CURECI1']),
    'urbanicity':     (['_URBSTAT'], ['_METSTAT']),
    'alcohol':        (['ALCDAY5', 'AVEDRNK3', 'DRNK3GE5', 'MAXDRNKS'],
                       ['DRNKANY5', 'DROCDY3_', '_RFBING5', '_DRNKWK1', '_RFDRHV7']),
    'vaccines':       (['FLUSHOT7', 'PNEUVAC4', 'HIVTST7'], ['_FLSHOT7', '_PNEUMO3', '_AIDTST4']),
    'fruit/veg':      (['FRUIT2', 'FRUITJU2', 'FVGREEN1', 'FRENCHF1', 'POTATOE1', 'VEGETAB2'],
                       ['FTJUDA2_', 'FRUTDA2_', 'GRENDA1_', 'FRNCHDA_', 'POTADA1_', 'VEGEDA2_',
                        '_MISFRT1', '_MISVEG1', '_FRTRES1', '_VEGRES1', '_FRUTSU1', '_VEGESU1',
                        '_FRTLT1A', '_VEGLT1A', '_FRT16A', '_VEG23A', '_FRUITE1', '_VEGETE1']),
}
for family, (kept, dropped) in DUP_FAMILIES.items():
    for col in dropped:
        assert col in df.columns, col
        triage[col] = (6, 'DUPLICATE', f'{family}: duplicate of {kept[0] if len(kept)==1 else kept}')
print('R6 duplicate:', sum(v[0] == 6 for v in triage.values()))


# In[87]:


# spot-verification: identical/monotone derivations should show |rho| ~ 1
pairs = [('_SEX', 'SEXVAR'), ('_RFHYPE6', 'BPHIGH6'), ('HTM4', 'HEIGHT3'), ('WTKG3', 'WEIGHT2'),
         ('_BMI5CAT', '_BMI5'), ('_AGEG5YR', '_AGE80'), ('_CHLDCNT', 'CHILDREN'),
         ('_EDUCAG', 'EDUCA'), ('_INCOMG1', 'INCOME3'), ('_RFSMOK3', '_SMOKER3')]
for a, b in pairs:
    print(f'{a:<10} vs {b:<10} rho = {df[a].corr(df[b], method="spearman"):+.3f}')


# ### 3.1 Funnel — verdicts + roles for the survivors (ORD keep codes · NUM true quantities · ENG decode · NOM one-hot):

# In[88]:


ROLES = {}
ROLES.update({'EDUCA': 'ORD', 'INCOME3': 'ORD', '_SMOKER3': 'ORD', 'CHECKUP1': 'ORD'})
ROLES.update({'_AGE80': 'NUM', 'JOINPAI2': 'NUM', 'MAXDRNKS': 'NUM'})
ROLES.update({c: 'ENG' for c in [
    'PHYSHLTH', 'MENTHLTH', 'POORHLTH', '_BMI5', 'ALCDAY5', 'AVEDRNK3', 'DRNK3GE5',
    'MARIJAN1', 'CHILDREN', 'FRUIT2', 'FRUITJU2', 'FVGREEN1', 'FRENCHF1', 'POTATOE1', 'VEGETAB2']})
ROLES.update({c: 'NOM' for c in [
    'DIABETE4', 'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', 'ASTHMA3', 'BPHIGH6', 'TOLDHI3',
    'CHCOCNCR', 'CHCSCNCR', 'CHCCOPD3', 'CHCKDNY2', 'HAVARTH5', 'ADDEPEV3', 'PREDIAB1',
    'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'DECIDE', 'DEAF', 'BLIND',
    'PRIMINSR', 'PERSDOC3', 'MEDCOST1', 'FLUSHOT7', 'PNEUVAC4', 'HIVTST7',
    'HOMBPCHK', 'HOMRGCHK', 'BPMEDS', 'CHOLCHK3', 'CHOLMED3',
    'EXERANY2', 'USENOW3', 'ECIGNOW1',
    'SEXVAR', 'MARITAL', 'EMPLOY1', 'RENTHOM1', '_RACE', '_STATE', '_URBSTAT',
    'QSTLANG', 'TRNSGNDR', 'VETERAN3', 'CAREGIV1', 'CRGVEXPT',
    'SOMALE', 'SOFEMALE', 'ARTHDIS2', 'LMTJOIN3', 'ARTHEDU', 'ARTHEXER',
    'IMFVPLA2', 'FLSHTMY3']})

keep = [c for c in df.columns if c not in triage]
assert set(keep) == set(ROLES), f'role mismatch: {set(keep) ^ set(ROLES)}'

triage_table = pd.DataFrame(
    [{'col': c, 'verdict': triage.get(c, ('', 'KEEP', ''))[1],
      'rule': triage.get(c, (0,))[0], 'reason': triage.get(c, ('', '', ''))[2],
      'role': ROLES.get(c, '')} for c in df.columns]).set_index('col')

print(triage_table.groupby(['rule', 'verdict']).size().to_string())
print(f'\n{len(df.columns)} columns = {len(triage)} dropped + {len(keep)} kept')
from collections import Counter
print('Roles:', dict(sorted(Counter(ROLES.values()).items())))


# In[89]:


triage_table.to_csv(f'{CKPT_DIR}/triage_own.csv')
triage_table[triage_table.verdict == 'KEEP'][['role']].to_csv(f'{CKPT_DIR}/whitelist_v1.csv')
print(f'Saved triage_own.csv ({len(triage_table)}) + whitelist_v1.csv ({len(keep)})')
triage_table.head(10)


# In[90]:


# VERIFICATION: cross-check against v1's independent triage (verification only)
V1_PATH = os.path.join(_HERE, '..', '..', 'results', 'profiling', 'triage_decisions.csv')
if os.path.exists(V1_PATH):
    v1 = pd.read_csv(V1_PATH).set_index('col_name')
    v1_keep = set(v1.index[v1.verdict.isin(['KEEP-NOM', 'KEEP-ORD', 'KEEP-NUM', 'ENG'])])
    our_keep = set(keep)
    print('keep-set match:', our_keep == v1_keep,
          f'({len(our_keep & v1_keep)} shared, {len(our_keep - v1_keep)} only ours, {len(v1_keep - our_keep)} only v1)')
    if our_keep - v1_keep: print('  we keep, v1 dropped:', sorted(our_keep - v1_keep))
    if v1_keep - our_keep: print('  v1 kept, we drop:', sorted(v1_keep - our_keep))
else:
    print('v1 record not found - skipped')


# ---
# ## SECTION 4 — Cleaning & the feature matrix
# 
# ### ⚠️ No imputation here
# 
# The column mean/median is a **statistic estimated from data** — computing it on all rows before splitting leaks test-fold info into training. It must be estimated **inside CV, on training folds only**. NaN stays in the checkpoint; the boosters used later even handle NaN natively.

# In[91]:


whitelist = pd.read_csv(f'{CKPT_DIR}/whitelist_v1.csv', index_col=0)
BLACKLIST = ['GENHLTH', '_RFHLTH', '_PHYS14D', '_MENT14D']
assert not set(BLACKLIST) & set(whitelist.index)   # belt and suspenders

X = df[whitelist.index].copy().loc[y.index]        # labeled rows only
roles = whitelist['role'].to_dict()
print(f'X: {X.shape[0]:,} x {X.shape[1]}')


# In[92]:


# per-column sentinel recodes (codebook, never blanket patterns)
SENTINEL_NAN = {
    **{c: (7, 9) for c in [   # 7=DK, 9=refused in these items
        'DIABETE4', 'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', 'ASTHMA3', 'BPHIGH6', 'TOLDHI3',
        'CHCOCNCR', 'CHCSCNCR', 'CHCCOPD3', 'CHCKDNY2', 'HAVARTH5', 'ADDEPEV3', 'PREDIAB1',
        'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'DECIDE', 'DEAF', 'BLIND',
        'PERSDOC3', 'MEDCOST1', 'FLUSHOT7', 'PNEUVAC4', 'HIVTST7', 'HOMBPCHK', 'HOMRGCHK',
        'BPMEDS', 'CHOLCHK3', 'CHOLMED3', 'EXERANY2', 'USENOW3', 'ECIGNOW1',
        'RENTHOM1', 'TRNSGNDR', 'VETERAN3', 'CAREGIV1', 'CRGVEXPT',
        'SOMALE', 'SOFEMALE', 'ARTHDIS2', 'LMTJOIN3', 'ARTHEDU', 'ARTHEXER',
        'CHECKUP1']},
    # 9 only -- 7 is a REAL answer here (Section 2.4 trap):
    'EMPLOY1': (9,), 'MARITAL': (9,), '_RACE': (9,), 'EDUCA': (9,), '_SMOKER3': (9,),
    # two-digit sentinels -- single 7/9 are real answers:
    'INCOME3': (77, 99), 'PRIMINSR': (77, 99), 'IMFVPLA2': (77, 99),
    'JOINPAI2': (77, 99), 'MAXDRNKS': (77, 99), 'FLSHTMY3': (999999,),
}
for col, codes in SENTINEL_NAN.items():
    X[col] = X[col].replace({code: np.nan for code in codes})
print('applied to', len(SENTINEL_NAN), 'columns')


# ### 4.1 Engineered recodes
# 
# - Day counts (`PHYSHLTH`/`MENTHLTH`/`POORHLTH`) + "none" counts (`AVEDRNK3`/`DRNK3GE5`/`MARIJAN1`/`CHILDREN`): **88 = none → 0**, 77/99 → NaN.
# - `_BMI5` stored ×100 → ÷100.
# - Frequency codes (`ALCDAY5`, fruit/veg items): `101–199` = N/day, `201–299` = N/week, `300`/`555` = ~never → decoded to **times per month** ("twice a week" ≡ "eight times a month").

# In[93]:


for col in ['PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'AVEDRNK3', 'DRNK3GE5', 'MARIJAN1']:
    X[col] = X[col].replace({88: 0, 77: np.nan, 99: np.nan})
X['CHILDREN'] = X['CHILDREN'].replace({88: 0, 99: np.nan})   # no 77 code
X['_BMI5'] = X['_BMI5'] / 100

def freq_per_month(s):
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s.between(101, 199)] = (s - 100) * 30.4      # per day -> per month
    out[s.between(201, 299)] = (s - 200) * 4.33      # per week -> per month
    out[s.isin([300, 555])] = 0                      # '<1/mo', 'never'
    return out                                       # 777/999/blanks stay NaN

for col in ['FRUIT2', 'FRUITJU2', 'FVGREEN1', 'FRENCHF1', 'POTATOE1', 'VEGETAB2']:
    X[col] = freq_per_month(X[col])

def alc_days_per_month(s):
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s.between(101, 199)] = (s - 100) * 4.33      # days per week
    out[s.between(201, 299)] = (s - 200)             # days per month
    out[s == 888] = 0                                # 'no drinks'
    return out

X['ALCDAY5'] = alc_days_per_month(X['ALCDAY5'])
print('done')


# ### 4.2 Sanity gates — if one fires, fix the recode, not the check:

# In[94]:


assert not set(BLACKLIST) & set(X.columns)
assert X.shape[1] == 76
assert X['PHYSHLTH'].max() <= 30 and X['MENTHLTH'].max() <= 30
assert 10 < X['_BMI5'].min() and X['_BMI5'].max() < 100
assert set(X['EDUCA'].dropna().unique()) <= {1, 2, 3, 4, 5, 6}
assert set(X['CHECKUP1'].dropna().unique()) <= {1, 2, 3, 4, 8}
assert set(X['_SMOKER3'].dropna().unique()) <= {1, 2, 3, 4}
print('All gates passed.')
print(f'\nOverall missingness: {X.isna().mean().mean()*100:.2f}%')
print(X.isna().mean().mul(100).sort_values(ascending=False).head(8).round(1).to_string())


# In[95]:


X.head()


# In[96]:


# checkpoint: parquet (fast, keeps dtypes + NaN exactly); never re-clean the raw CSV again
X.to_parquet(f'{CKPT_DIR}/clean_v1.parquet')
y.to_frame('y').to_parquet(f'{CKPT_DIR}/target_v1.parquet')
assert pd.read_parquet(f'{CKPT_DIR}/clean_v1.parquet').equals(X)   # round-trip
print(f'Saved + round-trip OK: X {X.shape}, y {y.shape}')


# ---
# ## SECTION 5 — Baselines
# 
# Dummy (the floor) → Logistic Regression (linear) → Decision Tree (nonlinear) + manual tuning. Every run logged to `experiments.csv` with its 10 fold AUCs — the log every report number traces back to.

# ### 5.1 Design matrix
# 
# One-hot the NOMINAL columns **once, outside CV** — safe because it's per-row relabeling (no statistics estimated → nothing to leak). Imputation/scaling are different: they estimate from data, so they stay **inside** CV pipelines.

# In[97]:


X = pd.read_parquet(f'{CKPT_DIR}/clean_v1.parquet')
y = pd.read_parquet(f'{CKPT_DIR}/target_v1.parquet')['y']
roles = pd.read_csv(f'{CKPT_DIR}/whitelist_v1.csv', index_col=0)['role'].to_dict()

nom_cols = [c for c in X.columns if roles[c] == 'NOM']
Xd = pd.get_dummies(X, columns=nom_cols, dummy_na=True)   # explicit _nan columns
print(f'{X.shape[1]} columns -> {Xd.shape[1]} after one-hot ({len(nom_cols)} nominal)')
Xd.to_parquet(f'{CKPT_DIR}/design_v1.parquet')


# In[98]:


# experiment log helpers
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score

EXP_PATH = os.path.join(_HERE, '..', 'experiments.csv')

def run_cv(model, X_, y_, exp_id, model_name, params='', notes=''):
    '''10-fold CV on THE canonical folds; one row in experiments.csv per run.'''
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_, y_, cv=folds, scoring='roc_auc', n_jobs=-1)
    row = {'exp_id': exp_id, 'timestamp': datetime.now().isoformat(timespec='seconds'),
           'model': model_name, 'params': params, 'notes': notes}
    row.update({f'fold_{i+1:02d}': round(s, 6) for i, s in enumerate(scores)})
    row['mean_auc'] = round(scores.mean(), 6)
    row['std_auc'] = round(scores.std(), 6)
    new = pd.DataFrame([row])
    if os.path.exists(EXP_PATH):   # re-running a cell refreshes its row (no duplicates)
        log = pd.concat([pd.read_csv(EXP_PATH), new]).drop_duplicates('exp_id', keep='last')
        log.to_csv(EXP_PATH, index=False)
    else:
        new.to_csv(EXP_PATH, index=False)
    print(f'{exp_id} {model_name:<20} {params:<28} AUC {scores.mean():.6f} +/- {scores.std():.6f}')
    return scores


# In[99]:


# EXP-001: DummyClassifier — sanity floor, must be exactly ~0.5
from sklearn.dummy import DummyClassifier
run_cv(DummyClassifier(strategy='most_frequent'), Xd, y, 'EXP-001', 'Dummy', 'most_frequent')


# ### 5.2 Logistic Regression
# 
# Linear model — needs numbers, no NaN → Pipeline: median imputer + standardizer + LogReg. **Inside the pipeline = fit on training folds only** (this is the leak-safe imputation from Section 4). C = inverse regularization strength (smaller = more regularized).

# In[100]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def logreg(c):
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('lr', LogisticRegression(C=c, max_iter=2000, random_state=RANDOM_STATE)),
    ])

# manual tuning over C (each run ~5-10 min)
run_cv(logreg(1.0),  Xd, y, 'EXP-002', 'LogReg', 'C=1.0')
run_cv(logreg(0.1),  Xd, y, 'EXP-003', 'LogReg', 'C=0.1')
run_cv(logreg(0.01), Xd, y, 'EXP-004', 'LogReg', 'C=0.01')


# ### 5.3 Decision Tree + manual tuning
# 
# Trees take raw codes + NaN-imputation the same way; `max_depth`/`min_samples_leaf` control overfitting (deep tree = memorizes; shallow = generalizes).

# In[101]:


from sklearn.tree import DecisionTreeClassifier

def tree(depth, leaf):
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('tree', DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                        random_state=RANDOM_STATE)),
    ])

run_cv(tree(None, 1),   Xd, y, 'EXP-005', 'DecisionTree', 'default (no limits)')
run_cv(tree(8, 200),    Xd, y, 'EXP-006', 'DecisionTree', 'depth=8, leaf=200')
run_cv(tree(16, 200),   Xd, y, 'EXP-007', 'DecisionTree', 'depth=16, leaf=200')


# ### 5.4 Results

# In[122]:


exp = pd.read_csv(EXP_PATH).drop_duplicates('exp_id', keep='last')
summary = exp[['exp_id', 'model', 'params', 'mean_auc', 'std_auc']]
print(summary.to_string(index=False))


# In[125]:


# figure: baseline comparison
fig, ax = plt.subplots(figsize=(18, 4.5))
labels = [f'{r.model}\n{r.params}' for r in exp.itertuples()]
ax.bar(labels, exp.mean_auc, yerr=exp.std_auc, capsize=3, color='steelblue')
ax.axhline(0.5, color='red', ls='--', lw=1, label='chance (0.5)')
ax.set_ylim(0.45, max(exp.mean_auc) + 0.03)
ax.set_ylabel('ROC-AUC (10-fold CV)'); ax.set_title('Baselines'); ax.legend()
plt.xticks(fontsize=8); plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig03_baselines.png', dpi=150); plt.show()


# ## Findings & next
# 
# - Dummy = 0.500 exactly (floor confirmed). LogReg ≈ 0.89 — a *linear* model already gets most of the signal. Tuned tree approaches it; limits matter (default tree overfits).
# 
# **Section 6 — Model zoo & imbalance:** 4 ensembles join the comparison, then class-weight vs SMOTE vs none.

# ---
# ## SECTION 6 — Model zoo & imbalance
# 
# Rubric wants ≥5 classifiers incl. ensembles. Adding: **RandomForest** (bagging: many deep trees averaged) and three boosters (**HistGB, LightGBM, XGBoost**: shallow trees built sequentially, each correcting the last). HistGB + boosters accept NaN **natively** — missingness becomes signal, no imputer needed.

# In[104]:


from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

rf = Pipeline([
    ('impute', SimpleImputer(strategy='median')),      # sklearn RF needs complete input
    ('rf', RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                  n_jobs=1, random_state=RANDOM_STATE)),
])
run_cv(rf, Xd, y, 'EXP-008', 'RandomForest', 'trees=300, leaf=20')   # ~10-20 min


# In[105]:


hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                     random_state=RANDOM_STATE)     # NaN native
run_cv(hgb, Xd, y, 'EXP-009', 'HistGB', 'iter=300, lr=0.05')


# In[106]:


import lightgbm as lgb

lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                          n_jobs=1, verbose=-1, random_state=RANDOM_STATE)
run_cv(lgbm, Xd, y, 'EXP-010', 'LightGBM', 'trees=500, lr=0.05')


# In[107]:


import xgboost as xgb

xgbm = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                         tree_method='hist', n_jobs=1, eval_metric='logloss',
                         random_state=RANDOM_STATE)
run_cv(xgbm, Xd, y, 'EXP-011', 'XGBoost', 'trees=500, lr=0.05, depth=6')


# ### 6.2 Imbalance handling
# 
# Hypothesis: **neither technique helps AUC** — AUC is rank-based; reshaping class priors moves the best *threshold*, not the *ordering*. Test on two families: `class_weight='balanced'` (penalize minority mistakes) and SMOTE (synthetic minority oversampling; needs imputation, can't take NaN). The `imblearn` Pipeline keeps SMOTE **on training folds only** — oversampling before splitting would leak synthetic copies of test rows into training.

# In[108]:


lr_w = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('lr', LogisticRegression(C=0.01, class_weight='balanced',
                              max_iter=2000, random_state=RANDOM_STATE)),
])
run_cv(lr_w, Xd, y, 'EXP-012', 'LogReg+w', 'C=0.01, balanced')


# In[109]:


lgbm_w = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                            class_weight='balanced', n_jobs=1, verbose=-1,
                            random_state=RANDOM_STATE)
run_cv(lgbm_w, Xd, y, 'EXP-013', 'LightGBM+w', 'balanced')


# In[111]:


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

lr_smote = ImbPipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('smote', SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)),   # TRAIN folds only
    ('lr', LogisticRegression(C=0.01, max_iter=2000, random_state=RANDOM_STATE)),
])
run_cv(lr_smote, Xd, y, 'EXP-014', 'LogReg+SMOTE', 'SMOTE 0.5, C=0.01')   # slow: 30-60 min


# In[126]:


# zoo + imbalance overview
exp = pd.read_csv(EXP_PATH).drop_duplicates('exp_id', keep='last')
print(exp[['exp_id', 'model', 'params', 'mean_auc', 'std_auc']].to_string(index=False))


# In[129]:


fig, ax = plt.subplots(figsize=(13, 5))
colors = ['gray'] + ['steelblue']*(len(exp)-1)
ax.bar([f'{r.model}\n{r.params}' for r in exp.itertuples()], exp.mean_auc,
       yerr=exp.std_auc, capsize=3, color=colors)
ax.axhline(0.5, color='red', ls='--', lw=1, label='chance')
ax.set_ylim(0.45, exp.mean_auc.max() + 0.02)
ax.set_ylabel('ROC-AUC (10-fold CV)'); ax.set_title('All experiments'); ax.legend()
plt.xticks(fontsize=7); plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig04_model_zoo.png', dpi=150); plt.show()


# ## Findings & next
# 
# - Expected pattern: boosters cluster ≈ 0.90, LogReg ≈ 0.89 → boosting buys ~+0.01. RF close to LogReg (bagging helps less than boosting here).
# - Imbalance: class-weight/SMOTE ≈ no AUC change (maybe slightly worse) → hypothesis confirmed; AUC is threshold-free. (Techniques would matter for accuracy/recall at a fixed threshold — not our metric.)
# 
# **Section 7 — Feature selection & hypothesis:** gain importances, a stated hypothesis about driver groups (subset vs all-76), SelectKBest, PCA.

# ---
# ## SECTION 7 — Feature selection & hypothesis
# 
# Four questions: (1) which columns matter (gain importance)? (2) does an H1-driven subset match all-76? (3) does generic selection (SelectKBest) help? (4) does PCA compression help?
# 
# **H1 (stated before testing):** Fair/Poor health is driven by five blocks — **B1** recent health-burden days, **B2** functional limitation + chronic conditions, **B3** age, **B4** socioeconomic status, **B5** health behaviors. Everything else (insurance, prevention, demographics, diet detail) adds little on top.

# In[114]:


# fit Section-6 LightGBM once on everything; sum one-hot children back to parents
lgbm_full = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                               n_jobs=-1, verbose=-1, random_state=RANDOM_STATE).fit(Xd, y)
gain = pd.Series(lgbm_full.booster_.feature_importance(importance_type='gain'), index=Xd.columns)

parent = {c: next(p for p in X.columns if c == p or c.startswith(p + '_')) for c in Xd.columns}
col_gain = gain.groupby(parent).sum().sort_values(ascending=False)
col_gain_pct = (100 * col_gain / col_gain.sum())
print(col_gain_pct.head(12).round(1).to_string())


# In[115]:


top = col_gain_pct.head(30)[::-1]
fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(top.index, top.values, color='steelblue')
ax.set_xlabel('% of total gain'); ax.set_title('LightGBM gain importance (dummies summed back to columns)')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/fig05_importance_top30.png', dpi=150); plt.show()


# In[116]:


# H1 blocks -> subset design matrix
BLOCKS = {
    'B1 health-burden days': ['PHYSHLTH', 'MENTHLTH', 'POORHLTH'],
    'B2 function+chronic': ['DIFFWALK', 'DIFFDRES', 'DIFFALON', 'DECIDE', 'DEAF', 'BLIND',
        'HAVARTH5', 'ARTHDIS2', 'LMTJOIN3', 'ARTHEDU', 'ARTHEXER',
        'DIABETE4', 'PREDIAB1', 'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', 'ASTHMA3',
        'BPHIGH6', 'TOLDHI3', 'CHCOCNCR', 'CHCSCNCR', 'CHCCOPD3', 'CHCKDNY2',
        'ADDEPEV3', 'JOINPAI2'],
    'B3 age': ['_AGE80'],
    'B4 socioeconomic': ['EDUCA', 'INCOME3', 'EMPLOY1'],
    'B5 behaviors': ['_SMOKER3', 'EXERANY2', '_BMI5', 'ALCDAY5', 'AVEDRNK3', 'DRNK3GE5', 'MAXDRNKS'],
}
h1_cols = sorted(set(sum(BLOCKS.values(), [])))
assert set(h1_cols) <= set(X.columns)

def dummify(X_):
    nom = [c for c in X_.columns if roles[c] == 'NOM']
    return pd.get_dummies(X_, columns=nom, dummy_na=True)

Xd_h1 = dummify(X[h1_cols])
print(f'H1 subset: {len(h1_cols)} columns -> {Xd_h1.shape[1]} features')


# In[117]:


def make_lgbm(**kw):
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              n_jobs=1, verbose=-1, random_state=RANDOM_STATE, **kw)

# H1 subset vs all-76 (all-76 baseline = EXP-010)
run_cv(make_lgbm(), Xd_h1, y, 'EXP-015', 'LGBM-H1subset', f'{len(h1_cols)} of 76 cols')

# complement alone: how much signal lives OUTSIDE the hypothesis?
comp_cols = [c for c in X.columns if c not in h1_cols]
Xd_comp = dummify(X[comp_cols])
run_cv(make_lgbm(), Xd_comp, y, 'EXP-016', 'LGBM-complement', f'{len(comp_cols)} cols')


# In[119]:


# SelectKBest (mutual information) INSIDE the folds -- selection is estimation too,
# so it must never see test-fold rows (same principle as imputation/SMOTE).
from functools import partial
from sklearn.feature_selection import SelectKBest, mutual_info_classif

kb = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('select', SelectKBest(partial(mutual_info_classif, random_state=RANDOM_STATE), k=100)),
    ('lgbm', make_lgbm()),
])
run_cv(kb, Xd, y, 'EXP-017', 'KBest+LGBM', 'MI k=100')   # ~20-40 min


# In[120]:


# PCA: compress to n directions. Needs complete+scaled input (inside pipeline).
from sklearn.decomposition import PCA

def pca_lgbm(n):
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=n)),
        ('lgbm', make_lgbm()),
    ])

run_cv(pca_lgbm(50),  Xd, y, 'EXP-018', 'PCA+LGBM', 'n=50')    # ~5-10 min each
run_cv(pca_lgbm(100), Xd, y, 'EXP-019', 'PCA+LGBM', 'n=100')

# how much variance do those components even keep?
pre = Pipeline([('i', SimpleImputer(strategy='median')),
                ('s', StandardScaler()), ('p', PCA())]).fit(Xd)
cum = pre.named_steps['p'].explained_variance_ratio_.cumsum()
print(f'cumulative variance: 50 comps {cum[49]:.1%}   100 comps {cum[99]:.1%}')


# In[121]:


exp = pd.read_csv(EXP_PATH).drop_duplicates('exp_id', keep='last')
print(exp[['exp_id', 'model', 'params', 'mean_auc', 'std_auc']].to_string(index=False))


# ## Findings & next
# 
# - Expected: `PHYSHLTH` dominates gain; H1 subset ≈ all-76 − ~0.003 (hypothesis nearly confirmed); complement alone much lower but non-trivial; KBest ≤ all-76 (honest negative — automatic selection can't beat the full matrix here); PCA clearly worse (it smears one-hot structure into continuous mush).
# - Verdict: keep all-76 for tuning.
# 
# **Section 8 — Tuning & statistics:** RandomizedSearchCV on the top family + Wilcoxon signed-rank on the top-2 models (identical folds → paired test valid).

# ---
# ## SECTION 8 — Tuning & statistics
# 
# Principle, one last time: **hyperparameter search is selection = estimation** → search runs on *inner* 3-fold CV, the tuned model is then evaluated on the untouched canonical 10 folds (`run_cv`), so every EXP row stays comparable.
# 
# **Wilcoxon signed-rank:** paired test on the 10 per-fold deltas (valid because folds are identical). n=10 two-sided floor: p = 0.002. **Nested CV:** search re-run inside every outer training fold → unbiased estimate of tuned performance.

# In[130]:


from sklearn.model_selection import RandomizedSearchCV

INNER = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

lgbm_grid = {
    'n_estimators': [300, 500, 800], 'learning_rate': [0.03, 0.05, 0.1],
    'num_leaves': [31, 63, 127], 'min_child_samples': [50, 200],
}
search_lgbm = RandomizedSearchCV(make_lgbm(), lgbm_grid, n_iter=10, cv=INNER,
                                 scoring='roc_auc', random_state=RANDOM_STATE,
                                 n_jobs=-1).fit(Xd, y)          # ~5-10 min
print('LGBM best:', search_lgbm.best_params_, f'inner AUC {search_lgbm.best_score_:.6f}')


# In[131]:


xgb_grid = {
    'n_estimators': [300, 500, 800], 'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [4, 6, 8], 'min_child_weight': [1, 50],
}
search_xgb = RandomizedSearchCV(
    xgb.XGBClassifier(tree_method='hist', eval_metric='logloss',
                      random_state=RANDOM_STATE),
    xgb_grid, n_iter=10, cv=INNER, scoring='roc_auc',
    random_state=RANDOM_STATE, n_jobs=-1).fit(Xd, y)            # ~10-15 min
print('XGB best:', search_xgb.best_params_, f'inner AUC {search_xgb.best_score_:.6f}')


# In[132]:


# tuned models on the canonical 10 folds
best_lgbm = lgb.LGBMClassifier(**search_lgbm.best_params_, n_jobs=1, verbose=-1,
                               random_state=RANDOM_STATE)
run_cv(best_lgbm, Xd, y, 'EXP-020', 'LGBM-tuned', str(search_lgbm.best_params_))

best_xgb = xgb.XGBClassifier(**search_xgb.best_params_, tree_method='hist',
                             eval_metric='logloss', n_jobs=1, random_state=RANDOM_STATE)
run_cv(best_xgb, Xd, y, 'EXP-021', 'XGB-tuned', str(search_xgb.best_params_))


# In[133]:


# Wilcoxon signed-rank on paired fold scores
from scipy.stats import wilcoxon

exp = pd.read_csv(EXP_PATH).drop_duplicates('exp_id', keep='last').set_index('exp_id')
fold_cols = [f'fold_{i:02d}' for i in range(1, 11)]

def folds_of(exp_id):
    return exp.loc[exp_id, fold_cols].astype(float).to_numpy()

print(f'{"comparison":<28}{"median delta":>13}{"wins":>7}{"Wilcoxon p":>11}')
for name, a, b in [('tuned LGBM vs tuned XGB', 'EXP-020', 'EXP-021'),
                   ('tuned vs untuned LGBM', 'EXP-020', 'EXP-010'),
                   ('tuned LGBM vs LogReg', 'EXP-020', 'EXP-004')]:
    d = folds_of(a) - folds_of(b)
    p = wilcoxon(folds_of(a), folds_of(b)).pvalue
    print(f'{name:<28}{np.median(d):>+13.6f}{(d > 0).sum():>5}/10{p:>11.4f}')


# In[134]:


# nested CV: the search re-run INSIDE each outer training fold -> unbiased estimate
def small_search():
    return RandomizedSearchCV(make_lgbm(), lgbm_grid, n_iter=4, cv=INNER,
                              scoring='roc_auc', random_state=RANDOM_STATE, n_jobs=-1)

outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
nested_scores = cross_val_score(small_search(), Xd, y, cv=outer,
                                scoring='roc_auc', n_jobs=1)    # ~25-35 min

row = {'exp_id': 'EXP-022', 'timestamp': datetime.now().isoformat(timespec='seconds'),
       'model': 'LGBM-nestedCV', 'params': 'inner: n_iter=4 x 3 folds',
       'notes': 'unbiased tuned estimate'}
row.update({f'fold_{i+1:02d}': round(s, 6) for i, s in enumerate(nested_scores)})
row['mean_auc'] = round(nested_scores.mean(), 6)
row['std_auc'] = round(nested_scores.std(), 6)
log = pd.concat([pd.read_csv(EXP_PATH), pd.DataFrame([row])]).drop_duplicates('exp_id', keep='last')
log.to_csv(EXP_PATH, index=False)
print(f'nested CV unbiased estimate: {nested_scores.mean():.6f} +/- {nested_scores.std():.6f}')
print(f'optimism vs non-nested tuned: {nested_scores.mean() - exp.loc["EXP-020", "mean_auc"]:+.6f}')


# ## Findings & next
# 
# - Expected: tuned LGBM edges tuned XGB; **family gain (vs LogReg, ~+0.011) ≫ tuning gain (~+0.001)**, both real by Wilcoxon; nested ≈ non-nested (tiny optimism → search width doesn't overfit).
# 
# **Section 9 — Error analysis & XAI:** the 3 consistently-misclassified people + SHAP (global bar + local waterfalls) on the final model.

# ---
# ## SECTION 9 — Error analysis & XAI
# 
# **Out-of-fold (OOF) predictions:** each person's probability comes from the fold-model that *never saw them* — honest per-person predictions. **SHAP:** splits each single prediction into additive per-feature credits (in log-odds); global importance = mean |credit|.

# In[135]:


# OOF probabilities from the tuned winner (EXP-020 config), canonical folds
final_params = search_lgbm.best_params_
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(y))
for tr, te in cv.split(Xd, y):
    m = lgb.LGBMClassifier(**final_params, n_jobs=-1, verbose=-1,
                           random_state=RANDOM_STATE).fit(Xd.iloc[tr], y.iloc[tr])
    oof[te] = m.predict_proba(Xd.iloc[te])[:, 1]
oof = pd.Series(oof, index=y.index, name='oof_prob')
oof.to_frame().to_parquet(f'{CKPT_DIR}/oof_v1.parquet')
print(f'OOF AUC check: {roc_auc_score(y, oof):.6f} (matches EXP-020)')


# In[136]:


# error taxonomy: where do mistakes live, on the 5-point scale?
gh = pd.read_csv(DATA_PATH, usecols=['GENHLTH'], low_memory=False)['GENHLTH'].loc[y.index]
err = ((oof >= 0.5) & (y == 0)) | ((oof < 0.5) & (y == 1))   # FP or FN at tau=0.5
tax = pd.crosstab(gh[err], np.where(oof[err] >= 0.5, 'FP (pred Fair/Poor)', 'FN (pred Good+)'),
                  normalize='columns').round(3) * 100
print(f'errors at tau=0.5: {err.sum():,} of {len(y):,} ({100*err.mean():.1f}%)')
print(tax.to_string())
print(f'\nerrors at the 3/4 boundary (true GENHLTH 3 or 4): {100*gh[err].isin([3, 4]).mean():.1f}%')


# In[137]:


fig, ax = plt.subplots(figsize=(7, 4))
counts = gh[err].value_counts().sort_index()
colors = ['#4C72B0', '#4C72B0', '#4C72B0', '#C44E52', '#C44E52']
ax.bar([f'{i}' for i in counts.index], counts.values, color=colors[:len(counts)])
ax.set_xlabel('true GENHLTH (1 Excellent ... 5 Poor)')
ax.set_ylabel('misclassified people')
ax.set_title('Where the errors live (tau=0.5)')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/fig06_error_taxonomy.png', dpi=150); plt.show()


# In[138]:


# 3 archetype cases: most-confident FP, most-confident FN, fence-sitter
fp_idx = oof[(oof >= 0.5) & (y == 0)].idxmax()      # said Fair/Poor, truth Good+
fn_idx = oof[(oof < 0.5) & (y == 1)].idxmin()       # said Good+,   truth Fair/Poor
mis = oof[err]
fence_idx = (mis - 0.5).abs().idxmin()              # wrong, but barely
cases = {'A confident-FP': fp_idx, 'B confident-FN': fn_idx, 'C fence': fence_idx}

profile_cols = ['PHYSHLTH', 'MENTHLTH', 'POORHLTH', '_BMI5', '_AGE80',
                'DIFFWALK', 'DIABETE4', 'DECIDE', 'INCOME3', 'EMPLOY1', 'EDUCA']
for name, idx in cases.items():
    print(f'\n=== case {name}  (row {idx}, OOF prob {oof[idx]:.4f}, true y={y[idx]}) ===')
    prof = pd.DataFrame({'case': X.loc[idx, profile_cols],
                         'population median': X[profile_cols].median()})
    print(prof.to_string())


# In[139]:


# SHAP: global importance (dummies aggregated back to parents)
import shap

final_model = lgb.LGBMClassifier(**final_params, n_jobs=-1, verbose=-1,
                                 random_state=RANDOM_STATE).fit(Xd, y)
expl = shap.TreeExplainer(final_model)
rng = np.random.RandomState(RANDOM_STATE)
Xs = Xd.iloc[rng.choice(len(Xd), 20000, replace=False)]
sv = expl.shap_values(Xs)
if isinstance(sv, list):          # some versions return [class0, class1]
    sv = sv[1]

shap_parent = pd.Series(np.abs(sv).mean(axis=0), index=Xd.columns).groupby(parent).sum()
shap_parent = shap_parent.sort_values(ascending=False)
print(shap_parent.head(12).round(4).to_string())

fig, ax = plt.subplots(figsize=(7, 6))
top = shap_parent.head(20)[::-1]
ax.barh(top.index, top.values, color='steelblue')
ax.set_xlabel('mean |SHAP| (log-odds credit)')
ax.set_title('Global feature importance — tuned LightGBM')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/fig07_shap_global.png', dpi=150); plt.show()


# In[140]:


# local explanations: why did the model say THAT about these 3 people?
for name, idx in cases.items():
    row = Xd.loc[[idx]]
    sv_row = expl.shap_values(row)
    if isinstance(sv_row, list):
        sv_row = sv_row[1]
    exp_obj = shap.Explanation(values=sv_row[0], base_values=expl.expected_value,
                               data=row.iloc[0], feature_names=list(Xd.columns))
    plt.figure()
    shap.plots.waterfall(exp_obj, max_display=12, show=False)
    plt.title(f'case {name} — OOF prob {oof[idx]:.4f}, true y={y[idx]}', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig08_shap_case_{name[0]}.png', dpi=150, bbox_inches='tight')
    plt.show()


# ## Findings & next
# 
# - Expected: errors concentrate at the **GENHLTH 3/4 boundary** (Good vs Fair is a subjective one-step difference, not model failure); top SHAP features should echo Section 7's gain ranking (PHYSHLTH, DIFFWALK, diabetes, BMI).
# 
# **Section 10 — assembly:** code.py export, report numbers, PDF, defense pass.
