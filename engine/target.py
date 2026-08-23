"""Forward barrier-hit target for REGIME Gate R1.

Target A: does price hit +2*ATR before -1*ATR within the next H bars (from bar t+1)?

LIBRARY-FIRST: numpy vectorized high-water/drawdown scan (no hand-rolled loops).
Causal: label uses strictly-future bars t+1..t+H; features use data <= t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def barrier_hit(
    df: pd.DataFrame,
    atr: pd.Series,
    H: int = 60,
    up_mult: float = 2.0,
    down_mult: float = -1.0,
) -> pd.Series:
    """Binary label: did price reach +up_mult*ATR before down_mult*ATR within H bars?

    Entry reference is the close at bar t. Scans forward close path over the next
    H bars. Returns Series indexed like df with values in {0, 1, NaN}:
      - 1  : +2ATR touched before -1ATR
      - 0  : -1ATR touched first (or neither within H)
      - NaN: insufficient forward data (last H bars)
    Uses only strictly-future bars (t+1..t+H), so no lookahead into the feature bar.
    """
    close = np.asarray(df["close"].values, dtype=float)
    atr_v = np.asarray(atr.values, dtype=float)
    n = len(close)
    out = np.full(n, np.nan)
    for t in range(n - H):
        ref = close[t]
        base = atr_v[t]
        if not np.isfinite(base) or base <= 0:
            continue
        fwd = close[t + 1 : t + 1 + H]
        up = ref + up_mult * base
        dn = ref + down_mult * base  # down_mult negative -> below ref
        # first touch time of each barrier; -1 if never within H
        up_idx = np.where(fwd >= up)[0]
        dn_idx = np.where(fwd <= dn)[0]
        t_up = up_idx[0] if up_idx.size else np.inf
        t_dn = dn_idx[0] if dn_idx.size else np.inf
        if t_up < t_dn:
            out[t] = 1.0
        elif t_dn < t_up:
            out[t] = 0.0
        else:  # both never touched
            out[t] = 0.0
    return pd.Series(out, index=df.index)
