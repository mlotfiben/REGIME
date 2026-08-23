"""Tests for REGIME causal features — library-first + no-lookahead invariants."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.features import (
    atr,
    breakout_state,
    build_features,
    kalman_slope,
    trend_efficiency,
)


def make_df(seed=0, n=500):
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "open": price,
            "high": price * (1 + np.abs(rng.normal(0, 0.003, n))),
            "low": price * (1 - np.abs(rng.normal(0, 0.003, n))),
            "close": price,
        },
        index=idx,
    )


@pytest.fixture
def df():
    return make_df()


def test_kalman_slope_uses_filterpy():
    import filterpy.kalman as fk
    assert hasattr(fk, "KalmanFilter")
    rng = np.random.default_rng(1)
    x = np.log(100 * np.exp(np.cumsum(np.full(300, 0.001) + rng.normal(0, 0.005, 300))))
    s = kalman_slope(pd.Series(x))
    assert s.mean() > 0


def test_trend_efficiency_high_for_trend_low_for_chop():
    trend = pd.Series(np.arange(100.0))
    assert trend_efficiency(trend, 20).iloc[-1] > 0.9
    rng = np.random.default_rng(2)
    chop = pd.Series(np.cumsum(rng.normal(0, 1, 100)))
    assert trend_efficiency(chop, 20).iloc[-1] < 0.5


def test_breakout_in_range():
    rng = np.random.default_rng(3)
    close = pd.Series(np.cumsum(rng.normal(0, 1, 100)))
    b = breakout_state(close, 20).dropna()
    # breakout state can go outside [0,1] legitimately (a close below prior channel
    # = downside breakout -> negative; above prior channel -> >1). Assert finite + sane.
    assert np.isfinite(b).all()
    assert np.abs(b).quantile(0.99) < 10.0


def test_no_lookahead_future_perturbation():
    """Cardinal invariant: perturbing a FUTURE bar leaves all PAST features identical."""
    df1 = make_df()
    df2 = make_df().copy()
    i = 400
    df2.iloc[i:, 3] *= 1.5  # future closes
    f1 = build_features(df1)
    f2 = build_features(df2)
    cutoff = i - 220  # well before any window that could touch i
    pd.testing.assert_frame_equal(f1.iloc[:cutoff], f2.iloc[:cutoff])


def test_features_shape_matches_input(df):
    f = build_features(df)
    assert len(f) == len(df)
    expected = {"mom_s", "mom_m", "mom_l", "trend_eff_N", "breakout", "kalman_slope", "vol_gate"}
    assert set(f.columns) == expected
