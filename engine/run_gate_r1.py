"""Real-data Gate R1 runner — XAUUSD 1m, Target A (barrier hit).

Statistical protocol (frozen PREREG):
- Walk-forward, purged + embargoed (train samples whose label crosses the fold
  boundary are dropped; a forward embargo after each test fold).
- DAY-CLUSTERED inference: AUC per calendar day, then mean/SE/t over independent days
  (N = days, NOT the 6.8M bars — bar-level is pseudo-replication).
- Effect size reported, not just significance (huge N => tiny effects are significant).
- Both-halves check + winsorize robustness.
- Deflation counts all prior trials (>= 20) — report but note the gate is academic.

Cost is a SECONDARY report (known operational killer); the gate tests predictive power.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path(__file__).resolve().parent.parent / "data"
FEATS = ["mom_s", "mom_m", "mom_l", "trend_eff_N", "breakout", "kalman_slope", "vol_gate"]


def load():
    X = pd.read_parquet(DATA / "features.parquet")
    y = pd.read_parquet(DATA / "targetA.parquet")["barrier_hit"]
    # align
    df = X.join(y, how="inner").dropna()
    return df[FEATS], df["barrier_hit"], df.index


def day_auc(y_true, y_score, index):
    """Day-clustered AUC: one AUC per calendar day, return per-day array."""
    from sklearn.metrics import roc_auc_score
    idx = pd.DatetimeIndex(index)
    days = {}
    for day, positions in pd.Series(np.arange(len(idx)), index=idx).groupby(idx.normalize()):
        pos = positions.index
        yt = y_true.loc[pos]
        ys = y_score.loc[pos]
        if yt.nunique() < 2:
            continue
        try:
            days[day] = roc_auc_score(yt, ys)
        except ValueError:
            continue
    return pd.Series(days)


def winsorize(x, q=0.10):
    lo, hi = np.quantile(x, q), np.quantile(x, 1 - q)
    return np.clip(x, lo, hi)


def main():
    X, y, idx = load()
    n = len(X)
    print(f"loaded {n} aligned (feature, label) rows")

    # Walk-forward folds: expanding train, forward test blocks. 5 folds.
    n_folds = 5
    boundaries = np.linspace(0, n, n_folds + 1, dtype=int)
    embargo = 60  # bars, >= H
    from sklearn.linear_model import LogisticRegression

    all_day_auc = []
    fold_aucs = []
    for f in range(1, n_folds):
        test_start, test_end = boundaries[f], boundaries[f + 1]
        # train = [0, test_start - embargo)  (purge: label of train bar t reaches t+H;
        # drop the last H bars of train so no train label crosses test_start)
        train_end = test_start - max(embargo, 1)
        Xtr = X.iloc[:train_end]
        ytr = y.iloc[:train_end]
        Xte = X.iloc[test_start:test_end]
        yte = y.iloc[test_start:test_end]
        if ytr.nunique() < 2 or yte.nunique() < 2 or len(Xtr) < 10000:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        yscore = pd.Series(clf.predict_proba(Xte)[:, 1], index=Xte.index)
        da = day_auc(yte, yscore, Xte.index)
        all_day_auc.append(da)
        from sklearn.metrics import roc_auc_score
        fold_aucs.append(roc_auc_score(yte, yscore))
        print(f"fold {f}: train {len(Xtr):,} test {len(Xte):,} | overall AUC {fold_aucs[-1]:.4f} | n_days {len(da)}")

    if not all_day_auc:
        print("NO FOLDS — abort")
        return
    dap = pd.concat(all_day_auc)
    n_days = len(dap)
    mean_auc = dap.mean()
    se = dap.std(ddof=1) / np.sqrt(n_days)
    tstat = (mean_auc - 0.5) / se
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(abs(tstat), n_days - 1))

    print("\n=== Gate R1 — day-clustered OOS AUC (N = independent days) ===")
    print(f"n_days={n_days} | mean day-AUC={mean_auc:.4f} | SE={se:.4f} | t={tstat:.3f} | p={p:.4g}")
    print(f"overall per-fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")

    # both halves
    half = len(dap) // 2
    h1, h2 = dap.iloc[:half].mean(), dap.iloc[half:].mean()
    print(f"first-half mean day-AUC {h1:.4f} | second-half {h2:.4f}")

    # winsorize
    w10 = winsorize(dap.values, 0.10)
    t_w = (w10.mean() - 0.5) / (w10.std(ddof=1) / np.sqrt(len(w10)))
    print(f"winsorized(10%) t={t_w:.3f} (broad signal if t ~holds/rises)")

    # trivial benchmark: majority class -> predict always the more common label
    base = y.mean()
    print(f"trivial-benchmark (always predict majority, base rate {base:.3f}) AUC = 0.500")
    print(f"gate criterion: day-clustered CI excludes 0.5 & both halves positive & winsorize holds")
    print(f"VERDICT: {'PASS' if (tstat>2 and h1>0.5 and h2>0.5 and t_w>2) else 'FAIL/null'}")


if __name__ == "__main__":
    main()
