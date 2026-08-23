"""Tests for REGIME Target A (barrier hit) — causal, library-first."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.target import barrier_hit


def make_df(n=300):
    idx = pd.date_range("2020-01-01", periods=n, freq="min")
    close = pd.Series(100.0 + np.arange(n) * 0.0, index=idx)  # flat by default
    return pd.DataFrame({"close": close}, index=idx)


def test_flat_no_move_never_hits_barrier():
    df = make_df()
    atr = pd.Series(1.0, index=df.index)
    lab = barrier_hit(df, atr, H=20, up_mult=2.0, down_mult=-1.0).dropna()
    assert (lab == 0).all()  # no forward move -> no +2ATR touch


def test_up_trend_hits_up_barrier():
    df = make_df()
    # strong up move: +2 each minute for 20 bars
    df["close"] = 100.0 + pd.Series(np.arange(len(df)), index=df.index) * 2.0
    atr = pd.Series(1.0, index=df.index)
    lab = barrier_hit(df, atr, H=20, up_mult=2.0, down_mult=-1.0)
    # early bars: +2ATR=2 reached within a few bars -> label 1
    assert lab.iloc[5] == 1.0


def test_imbalance_reasonable():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=20000, freq="min")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    df = pd.DataFrame({"close": close}, index=idx)
    atr = pd.Series(np.full(len(idx), 0.1), index=idx)
    lab = barrier_hit(df, atr, H=60, up_mult=2.0, down_mult=-1.0).dropna()
    # barrier targets are not degenerate
    assert 0.05 < lab.mean() < 0.95


def test_label_uses_only_future_bars_no_lookahead():
    """Changing a bar BEFORE t must not change label at t (label uses t+1..t+H)."""
    df1 = make_df(300)
    df2 = make_df(300).copy()
    atr = pd.Series(1.0, index=df1.index)
    lab1 = barrier_hit(df1, atr, H=20, up_mult=2.0, down_mult=-1.0)
    # perturb an EARLY bar; labels at all later t must be unchanged
    df2["close"].iloc[0] = 999.0
    lab2 = barrier_hit(df2, atr, H=20, up_mult=2.0, down_mult=-1.0)
    pd.testing.assert_series_equal(lab1.iloc[1:], lab2.iloc[1:])
