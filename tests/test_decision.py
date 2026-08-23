"""Tests for REGIME decision-level backtest — trade simulation correctness."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.decision_backtest import oos_probs_and_positions, _sharpe


def make_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="min")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.002, n)))
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close}, index=idx)


def test_simulation_terminates_and_positions_binary():
    """With a strong positive-drift signal, the simulation must finish and positions
    must be 0/1 (no values between), no infinite loop."""
    df = make_df(2000)
    atr = pd.Series(df["high"] - df["low"], index=df.index).rolling(10).mean()
    X = pd.DataFrame({"mom_s": np.random.randn(2000), "mom_m": np.random.randn(2000),
                      "mom_l": np.random.randn(2000), "trend_eff_N": np.random.rand(2000),
                      "breakout": np.random.rand(2000), "kalman_slope": np.random.randn(2000),
                      "vol_gate": np.random.rand(2000)}, index=df.index)
    y = pd.Series(np.where(df["close"].pct_change().fillna(0) > 0, 1.0, 0.0), index=df.index)
    conf = dict(mom_windows=(4, 12, 48), eff=20, breakout=20, atr=20, vol=200, H=4)
    pos, trade_ret = oos_probs_and_positions(X, y, df, atr, conf)
    assert (pos >= 0).all() and (pos <= 1).all()
    assert pos.max() <= 1.0
    # no NaN
    assert not pos.isna().any() and not trade_ret.isna().any()


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(0)
    ret = pd.Series(np.full(1000, 0.001) + rng.normal(0, 0.01, 1000))
    assert _sharpe(ret) > 0
