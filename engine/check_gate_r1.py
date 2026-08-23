"""Honesty check on Gate R1 — is the day-clustered PASS real?

Two concerns to resolve:
1. Effect decays across folds (0.533->0.507). Is the most recent fold (4) null?
2. Day-clustered SE is inflated by 1m label autocorrelation (H=60 overlap within and
   across days). Re-estimate with BLOCK bootstrap where the independent unit is a
   MONTH (much larger than the H=60 overlap), and report a month-block SE / CI.
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
FEATS = ["mom_s", "mom_m", "mom_l", "trend_eff_N", "breakout", "kalman_slope", "vol_gate"]


def load():
    X = pd.read_parquet(DATA / "features.parquet")
    y = pd.read_parquet(DATA / "targetA.parquet")["barrier_hit"]
    df = X.join(y, how="inner").dropna()
    return df[FEATS], df["barrier_hit"], df.index


def fold_scores(X, y, n_folds=5, embargo=60):
    n = len(X)
    boundaries = np.linspace(0, n, n_folds + 1, dtype=int)
    results = []
    for f in range(1, n_folds):
        ts, te = boundaries[f], boundaries[f + 1]
        train_end = ts - embargo
        Xtr, ytr = X.iloc[:train_end], y.iloc[:train_end]
        Xte, yte = X.iloc[ts:te], y.iloc[ts:te]
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        yscore = clf.predict_proba(Xte)[:, 1]
        results.append({
            "fold": f, "train_end": train_end, "test_start": ts,
            "yte": yte, "yscore": pd.Series(yscore, index=Xte.index),
            "auc": roc_auc_score(yte, yscore),
        })
    return results


def month_block_ci(y_true, y_score, index, n_boot=1000, alpha=0.05, max_per_month=2000):
    """Block bootstrap over MONTHS as independent units. AUC re-estimated per draw.

    Sub-samples up to max_per_month rows per month per draw so each AUC computation is
    bounded (valid: the bootstrap is about the resampling distribution, and the CI
    width is driven by month-to-month variation, not by using every row).
    """
    rng = np.random.default_rng(0)
    didx = pd.DatetimeIndex(index)
    months = didx.to_period("M").unique()
    obs_by_month = {m: didx[didx.to_period("M") == m] for m in months}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.choice(months, size=len(months), replace=True)
        pieces = []
        for m in sel:
            obs = obs_by_month[m]
            if len(obs) > max_per_month:
                obs = rng.choice(obs, size=max_per_month, replace=False)
            pieces.append(obs)
        idx = np.concatenate(pieces)
        yt = y_true.loc[idx]
        ys = y_score.loc[idx]
        if yt.nunique() < 2:
            draws[b] = np.nan
            continue
        try:
            draws[b] = roc_auc_score(yt, ys)
        except ValueError:
            draws[b] = np.nan
    draws = draws[np.isfinite(draws)]
    lo, hi = np.quantile(draws, alpha / 2), np.quantile(draws, 1 - alpha / 2)
    return lo, hi, draws.mean()


def main():
    X, y, idx = load()
    print("=== fold-level AUC (overall, bar-level) ===")
    res = fold_scores(X, y)
    for r in res:
        print(f"  fold {r['fold']}: overall AUC {r['auc']:.4f}")
    first, last = res[0], res[-1]
    print(f"\n=== Fold 1 (oldest) vs Fold 4 (most recent) ===")
    for r in [first, last]:
        lo, hi, m = month_block_ci(r["yte"], r["yscore"], r["yscore"].index, n_boot=1000)
        print(f"  fold {r['fold']}: overall AUC {r['auc']:.4f} | month-block CI [{lo:.4f},{hi:.4f}] mean {m:.4f}")
        # does month-block CI exclude 0.5?
        print(f"    month-block CI excludes 0.5: {'YES' if lo > 0.5 else 'NO (null)'}")

    # full-sample month-block CI
    print("\n=== ALL folds pooled, month-block CI ===")
    yt_all = pd.concat([r["yte"] for r in res])
    ys_all = pd.concat([r["yscore"] for r in res])
    lo, hi, m = month_block_ci(yt_all, ys_all, ys_all.index, n_boot=2000)
    print(f"pooled overall AUC {roc_auc_score(yt_all, ys_all):.4f} | month-block CI [{lo:.4f},{hi:.4f}] mean {m:.4f}")
    print(f"month-block CI excludes 0.5: {'YES' if lo > 0.5 else 'NO (null)'}")


if __name__ == "__main__":
    main()
