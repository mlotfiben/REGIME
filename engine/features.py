"""Causal OHLC features for the REGIME baseline.

LIBRARY-FIRST (mandatory rule):
- Kalman slope: filterpy.KalmanFilter, causal filter-only (.predict/.update).
  NO RTS smoother (two-sided -> lookahead -> forbidden).
- Momentum / efficiency / breakout / vol: numpy + pandas rolling/ewm. No hand-rolled
  indicator math.

Causality: every feature at bar t uses data <= t only. A feature is shifted so it
never sees the label bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter


def kalman_slope(log_price: pd.Series, dim: int = 2, R: float = 1.0) -> pd.Series:
    """Causal constant-velocity Kalman filter on log-price via filterpy.

    Returns the filtered velocity (slope) at each bar, aligned to the input index.
    Uses filter-only predict/update (no smoother). Standard causal library call.
    """
    x = np.asarray(log_price.values, dtype=float)
    n = len(x)
    kf = KalmanFilter(dim_x=dim, dim_z=1)
    # state = [level, velocity]; constant-velocity transition
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.P *= 1000.0
    kf.R = R
    kf.Q = np.array([[0.001, 0.0], [0.0, 0.001]])
    kf.x = np.array([[x[0]], [0.0]])
    slopes = np.empty(n)
    slopes[0] = 0.0
    for i in range(1, n):
        kf.predict()
        kf.update(x[i])
        slopes[i] = kf.x[1, 0]
    return pd.Series(slopes, index=log_price.index)


def momentum_norm(close: pd.Series, window: int) -> pd.Series:
    """Normalized return over `window` bars. Uses pandas pct_change (causal).

    The normalizing std uses at least 10 bars (a 1-bar momentum window is valid but
    rolling(1).std() is NaN, which would kill the feature at coarse timeframes).
    """
    ret = close.pct_change(window)
    std = close.pct_change().rolling(max(int(window), 10)).std()
    return ret / std


def trend_efficiency(close: pd.Series, n: int) -> pd.Series:
    """Trend efficiency = |close[t]-close[t-N]| / sum|close[i]-close[i-1]| over window.

    Net move / total path. High = trend, low = chop. Causal (uses data <= t only).
    """
    net_move = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return net_move / path


def breakout_state(close: pd.Series, n: int) -> pd.Series:
    """Close relative to prior N-bar high/low. Causal (rolling, data <= t)."""
    hi = close.rolling(n).max().shift(1)
    lo = close.rolling(n).min().shift(1)
    rng = (hi - lo).replace(0.0, np.nan)
    return (close - lo) / rng


def atr(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Average True Range via pandas rolling (library-first). Causal."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def vol_regime(vol: pd.Series, window: int = 200) -> pd.Series:
    """Forecast vol relative to its own rolling history. Causal."""
    return vol / vol.rolling(window).median()


def build_features(
    df: pd.DataFrame,
    mom_windows=(4, 12, 48),
    eff_window: int = 20,
    breakout_window: int = 20,
    atr_window: int = 20,
    vol_window: int = 200,
) -> pd.DataFrame:
    """Assemble all causal REGIME features on an OHLC DataFrame.

    All features are causal. Result is aligned to df's index; leading rows with
    insufficient history are NaN. The no-lookahead invariant: perturbing any future
    bar leaves every past row of the output bit-identical.
    """
    close = df["close"]
    logp = np.log(close)
    out = pd.DataFrame(index=df.index)
    for tag, w in zip("sml", mom_windows):
        out[f"mom_{tag}"] = momentum_norm(close, int(w))
    out["trend_eff_N"] = trend_efficiency(close, int(eff_window))
    out["breakout"] = breakout_state(close, int(breakout_window))
    out["kalman_slope"] = kalman_slope(logp)
    out["vol_gate"] = vol_regime(atr(df, int(atr_window)), int(vol_window))
    # causal: shift so feature at t never sees the label bar t+1..t+H
    out = out.shift(1)
    return out
