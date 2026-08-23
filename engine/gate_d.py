"""A-D pre-real-data gate for REGIME Gate R1 (simulation-first).

D. Synthetic controls (decoupled, valid):
  - POSITIVE: drift-switching series. Blocks alternate positive-drift / negative-drift /
    zero-drift. During drift blocks the barrier outcome (hit +2ATR before -1ATR) is
    genuinely directional-biased, and momentum + trend_eff features capture it. The
    pipeline MUST recover this (AUC >> 0.5). Injection is AR(1)-style persistence
    (same-sign runs), per taxonomy pitfall 23 (constant drift is scale-invariant).
  - NEGATIVE: identical vol/regime structure but ZERO drift everywhere -> no
    directional barrier edge -> pipeline must stay silent (AUC ~ 0.5).

This validates the PIPELINE (machinery correctly wired to recover a known edge),
NOT whether real XAU has an edge. Feature params are frozen in PREREG, not tuned here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.features import build_features
from engine.target import barrier_hit


def _barrier_proxy(df: pd.DataFrame, w: int = 20) -> pd.Series:
    """Crude causal ATR proxy from OHLC (high-low range mean)."""
    rng = df["high"] - df["low"]
    return rng.rolling(w).mean()


def make_regime_series(n=200000, seed=0, with_drift=True, vol_calm=0.0006, vol_chop=0.0016, block=2000):
    """Regime-switching series with AR(1) persistence.

    Blocks of `block` bars alternate calm / choppy vol. When with_drift=True, half the
    blocks carry a persistent same-sign drift (AR(1) momentum) so trend/momentum
    features genuinely predict the barrier direction; when False, drift=0 everywhere
    (pure noise with the same vol structure -> null).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="min")
    nblocks = n // block + 1
    vol = np.empty(n)
    drift = np.zeros(n)
    for b in range(nblocks):
        lo = b * block
        hi = min((b + 1) * block, n)
        calm = (b % 2 == 0)
        vol[lo:hi] = vol_calm if calm else vol_chop
        if with_drift and b % 4 == 0:          # persistent up-trend block
            drift[lo:hi] = 3.0 * vol_calm
        elif with_drift and b % 4 == 1:        # persistent down-trend block
            drift[lo:hi] = -3.0 * vol_calm
        # blocks 2,3: no drift (chop / calm)
    # AR(1) persistence in the shock: ret_t = drift + phi*ret_{t-1} + sigma*eps
    phi = 0.05
    eps = rng.normal(0.0, 1.0, n)
    rets = np.empty(n)
    rets[0] = drift[0] + vol[0] * eps[0]
    for i in range(1, n):
        rets[i] = drift[i] + phi * rets[i - 1] + vol[i] * eps[i]
    close = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"close": close}, index=idx)
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.abs(rng.normal(0, 0.0002, n)))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.abs(rng.normal(0, 0.0002, n)))
    return df


def run_synthetic(seed, with_drift, n=200000, H=60):
    df = make_regime_series(n=n, seed=seed, with_drift=with_drift)
    a = _barrier_proxy(df)
    f = build_features(
        df,
        mom_windows=(60, 300, 1440),
        eff_window=120,
        breakout_window=240,
        atr_window=20,
        vol_window=1440,
    )
    lab = barrier_hit(df, a, H=H, up_mult=2.0, down_mult=-1.0)

    X = f.dropna()
    y = lab.reindex(X.index)
    mask = y.notna()
    X, y = X[mask], y[mask]
    if len(X) < 1000 or y.nunique() < 2:
        return {"auc": np.nan, "n": len(X), "n_pos": int((y == 1).sum())}

    split = int(0.7 * len(X))
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    if yte.nunique() < 2:
        return {"auc": np.nan, "n": len(Xte), "n_pos": int((yte == 1).sum())}
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    return {"auc": float(auc), "n": len(Xte), "n_pos": int((yte == 1).sum())}


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("=== Gate D: synthetic positive/negative controls (logistic on frozen features) ===")
    pos = [run_synthetic(seed=s, with_drift=True) for s in range(3)]
    neg = [run_synthetic(seed=s, with_drift=False) for s in range(3)]
    pos_auc = np.mean([p["auc"] for p in pos if np.isfinite(p["auc"])])
    neg_auc = np.mean([p["auc"] for p in neg if np.isfinite(p["auc"])])
    pos_str = ", ".join(f"{p['auc']:.3f}" for p in pos)
    neg_str = ", ".join(f"{p['auc']:.3f}" for p in neg)
    print(f"positive AUC mean: {pos_auc:.3f} (expect >> 0.5)  per-seed: [{pos_str}]")
    print(f"negative AUC mean: {neg_auc:.3f} (expect ~ 0.5)  per-seed: [{neg_str}]")
    print(f"elapsed {time.time()-t0:.1f}s")
