"""AUC comparison: base REGIME features vs enhanced (+fracdiff +session) at 1h.

Question: does the edge-strengthening feature set raise OOS AUC enough to matter?
Walk-forward logistic, same protocol. If enhanced AUC >> base, meta-labeling on the
stronger edge may clear the cost hurdle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent / "data"

BASE_FEATS = ["mom_s", "mom_m", "mom_l", "trend_eff_N", "breakout", "kalman_slope", "vol_gate"]
ENH_FEATS = BASE_FEATS + ["fracdiff_price", "fracdiff_ret", "session"]
CONF = dict(mom_windows=(1, 5, 24), eff=5, breakout=24, atr=5, vol=24, H=4)


def load(df, cols):
    from engine.enhanced_features import build_enhanced_features
    from engine.features import atr
    from engine.target import barrier_hit
    a = atr(df, 5)
    f = build_enhanced_features(df)
    lab = barrier_hit(df, a, H=4, up_mult=2.0, down_mult=-1.0)
    # align to the COMMON index (dropna on features, then label), same length
    common = f.dropna().index.intersection(lab.dropna().index)
    X = f.loc[common, cols]
    y = lab.loc[common]
    return X, y


def walkforward_auc(X, y, n_folds=4, embargo=5):
    n = len(X)
    bounds = np.linspace(0, n, n_folds + 1, dtype=int)
    aucs = []
    for f in range(1, n_folds):
        ts, te = bounds[f], bounds[f + 1]
        Xtr, ytr = X.iloc[:ts - embargo], y.iloc[:ts - embargo]
        Xte, yte = X.iloc[ts:te], y.iloc[ts:te]
        if ytr.nunique() < 2 or yte.nunique() < 2 or len(Xtr) < 1000:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    return aucs


def month_block_ci(y_true, y_score, n_boot=800, max_per_month=2000):
    rng = np.random.default_rng(0)
    didx = pd.DatetimeIndex(y_score.index)
    months = didx.to_period("M").unique()
    obs_by_month = {m: didx[didx.to_period("M") == m] for m in months}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.choice(months, size=len(months), replace=True)
        pieces = []
        for m in sel:
            o = obs_by_month[m]
            if len(o) > max_per_month:
                o = rng.choice(o, size=max_per_month, replace=False)
            pieces.append(o)
        idx = np.concatenate(pieces)
        yt, ys = y_true.loc[idx], y_score.loc[idx]
        if yt.nunique() < 2:
            draws[b] = np.nan; continue
        try:
            draws[b] = roc_auc_score(yt, ys)
        except ValueError:
            draws[b] = np.nan
    draws = draws[np.isfinite(draws)]
    return np.quantile(draws, 0.025), np.quantile(draws, 0.975), draws.mean()


def main():
    df = pd.read_parquet(DATA / "xau_1h.parquet")
    df.columns = [str(c).lower() for c in df.columns]
    print("=== AUC comparison at 1h: base vs enhanced (+fracdiff +session) ===")
    for name, cols in [("BASE (7)", BASE_FEATS), ("ENHANCED (10)", ENH_FEATS)]:
        X, y = load(df, cols)
        aucs = walkforward_auc(X, y)
        # recompute pooled OOS for CI
        n = len(X)
        bounds = np.linspace(0, n, 5, dtype=int)
        yt_all, ys_all = [], []
        for f in range(1, 4):
            ts, te = bounds[f], bounds[f + 1]
            Xtr, ytr = X.iloc[:ts - 5], y.iloc[:ts - 5]
            Xte, yte = X.iloc[ts:te], y.iloc[ts:te]
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Xtr, ytr)
            ys_all.append(pd.Series(clf.predict_proba(Xte)[:, 1], index=Xte.index))
            yt_all.append(yte)
        yt = pd.concat(yt_all); ys = pd.concat(ys_all)
        lo, hi, m = month_block_ci(yt, ys)
        print(f"  {name:>12}: folds {[f'{a:.3f}' for a in aucs]} | mean {np.mean(aucs):.4f} "
              f"| month-block CI [{lo:.4f},{hi:.4f}]")
        # feature columns actually used
        print(f"    n_features={len(cols)} | n rows {len(X):,}")


if __name__ == "__main__":
    main()
