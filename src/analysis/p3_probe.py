"""P3 EDA probe — verify shapes, target counts, missingness and Spearman associations.

Throwaway analysis snippet (data-analyst, P3). Prints facts used to style the
report figures; no files written.
"""
import pandas as pd
import numpy as np

from src.config import DATA_INTERIM

X = pd.read_parquet(DATA_INTERIM / "clean_v1.parquet")
T = pd.read_parquet(DATA_INTERIM / "target_v1.parquet")

print("clean_v1:", X.shape, "| dtypes:", X.dtypes.unique().tolist())
print("target_v1 cols:", T.columns.tolist(), T.dtypes.to_dict())

valid = T["valid"].to_numpy()
print(f"valid rows: {valid.sum()} / {len(T)}  dropped: {(~valid).sum()}")

y = T.loc[valid, "y"]
raw = T.loc[valid, "genhlth_raw"]
print("\nGENHLTH raw classes (valid rows):")
print(raw.value_counts().sort_index())
print((raw.value_counts(normalize=True).sort_index() * 100).round(3))

print("\nbinary target:")
print(y.value_counts().sort_index())
print((y.value_counts(normalize=True).sort_index() * 100).round(4))

Xv = X.loc[valid]
miss = (Xv.isna().mean() * 100).sort_values(ascending=False)
print("\nmissing% top 12:\n", miss.head(12).round(2).to_string())
print("missing% bottom 5:\n", miss.tail(5).round(3).to_string())
print(f"n>50%: {(miss > 50).sum()}  n>25%: {(miss > 25).sum()}  n==0%: {(miss == 0).sum()}")
print(f"median missing%: {miss.median():.3f}")

# ---- Spearman association with binary target (pairwise complete / NaN-aware) ----
corr_y = Xv.corrwith(y.astype(float), method="spearman")
corr_abs = corr_y.reindex(corr_y.abs().sort_values(ascending=False).index)
print("\nTop 15 |rho| vs y:")
print(corr_abs.head(15).round(4).to_string())
print("\nBottom 5 |rho| vs y:")
print(corr_abs.tail(5).round(4).to_string())
n_nan = corr_y.isna().sum()
print(f"columns with NaN rho: {n_nan}")

# specific checks
for c in ["PHYSHLTH", "MENTHLTH", "POORHLTH", "DIFFWALK", "DIFFALON", "DIFFDRES",
          "DECIDE", "INCOME3", "EDUCA", "_AGE80", "EMPLOY1", "JOINPAI2", "EXERANY2"]:
    print(f"rho[{c:9s}] = {corr_y[c]:+.4f}")

# ---- cluster internal correlations (pairwise complete) ----
clusters = {
    "disability": ["DIFFWALK", "DIFFALON", "DIFFDRES", "DECIDE"],
    "health-days": ["PHYSHLTH", "MENTHLTH", "POORHLTH"],
    "socio-demo": ["INCOME3", "EDUCA", "_AGE80", "EMPLOY1"],
}
for name, cols in clusters.items():
    cm = Xv[cols].corr(method="spearman")
    vals = [cm.iloc[i, j] for i in range(len(cols)) for j in range(i + 1, len(cols))]
    print(f"\n{name} pairwise rho: min {min(vals):+.3f} max {max(vals):+.3f}")
    print(cm.round(3).to_string())

# ---- distribution oddities ----
age = Xv["_AGE80"]
print(f"\n_AGE80 == 80: {(age == 80).sum()} rows ({(age == 80).mean() * 100:.2f}%), max: {age.max()}")
for c in ["PHYSHLTH", "MENTHLTH"]:
    v = Xv[c]
    print(f"{c} == 0: {(v == 0).sum()} ({(v == 0).mean() * 100:.2f}%), mean y=0 vs y=1:",
          round(v[y == 0].mean(), 3), round(v[y == 1].mean(), 3))
fv = Xv["FRUIT2"]
print(f"FRUIT2 == 0.5 (code 300): {(fv == 0.5).sum()}, == 0: {(fv == 0).sum()}, max: {fv.max()}")

print("\nDIFFWALK rate P(=1|y):", Xv.loc[y == 0, "DIFFWALK"].eq(1).mean().round(4),
      Xv.loc[y == 1, "DIFFWALK"].eq(1).mean().round(4))
print("\n_SMOKER3 shares by class:")
print(pd.crosstab(Xv["_SMOKER3"], y, normalize="columns").round(4))
print("\nINCOME3 median by class:", Xv.loc[y == 0, "INCOME3"].median(), Xv.loc[y == 1, "INCOME3"].median())
print("_AGE80 mean by class:", round(Xv.loc[y == 0, "_AGE80"].mean(), 3), round(Xv.loc[y == 1, "_AGE80"].mean(), 3))
