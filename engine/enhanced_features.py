"""Edge-strengthening features for REGIME Gate R1 — AFML-style, library-first.

Two literature-backed boosts (per the owner's "raise the primary edge" decision):
1. Fractional differentiation (Lopez de Prado AFML Ch.5) — stationary-but-memory
   preserving features, d in (0,1). NO maintained PyPI lib exists (`fracdiff` not on
   PyPI; `mfe` 0.0.4 is a placeholder) => implemented here with scipy, documented per
   the use-existing-libraries rule. The AFML fracdiff is a weighted finite-difference
   of order d with binomial weights w_k = -w_{k-1} * (d-k+1)/k.
2. Session features (London+NY hours) — causal (hour of day only).

All causal (data <= t). A-D gate must still pass before real-data use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fracdiff_weights(d: float, n: int) -> np.ndarray:
    """Binomial weights w_k = (-1)^k * C(d, k), k=0..n-1. Causal, from AFML Ch.5."""
    w = np.ones(n)
    for k in range(1, n):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


def fracdiff(series: pd.Series, d: float = 0.4, window: int = 60) -> pd.Series:
    """Fractionally differentiate `series` to order d (stationary, memory-preserving).

    Fixed-window method (AFML Ch.5): out[t] = sum_{k=0}^{W-1} w_k * x[t-k], with
    binomial weights w_k = (-1)^k C(d,k). Causal (data <= t only). A FIXED window W
    is essential — the uncapped weights decay too slowly (d<0.5) and would NaN the
    whole series.
    """
    x = series.to_numpy(dtype=float)
    n = len(x)
    w = fracdiff_weights(d, window)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        out[t] = float(np.dot(w, x[t - window + 1:t + 1][::-1]))
    return pd.Series(out, index=series.index)


def session_london_ny(idx) -> pd.Series:
    """1 if hour in [7,16] UTC (approx London+NY overlap), else 0. Causal."""
    h = idx.hour
    return pd.Series(((h >= 7) & (h <= 16)).astype(float), index=idx)


def build_enhanced_features(df, d_mom=0.4, d_price=0.3):
    """REGIME features + fracdiff(log close) + fracdiff returns + session."""
    from engine.features import atr, build_features
    close = df["close"]
    base = build_features(df, mom_windows=(1, 5, 24), eff_window=5, breakout_window=24,
                          atr_window=5, vol_window=24)
    logp = np.log(close)
    fdiff_price = fracdiff(logp, d=d_price)
    fdiff_ret = fracdiff(close.pct_change().fillna(0.0), d=d_mom)
    sess = session_london_ny(df.index)
    out = base.copy()
    out["fracdiff_price"] = fdiff_price
    out["fracdiff_ret"] = fdiff_ret
    out["session"] = sess
    return out
