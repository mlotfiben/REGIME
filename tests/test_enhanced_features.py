"""Tests for REGIME enhanced features — fracdiff correctness + causality."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.enhanced_features import fracdiff, fracdiff_weights, session_london_ny


def test_fracdiff_weights_d0_identity():
    w = fracdiff_weights(0.0, 5)
    assert np.allclose(w, [1, 0, 0, 0, 0])


def test_fracdiff_weights_d1():
    # d=1 -> [1, -1, 0, 0, ...]
    w = fracdiff_weights(1.0, 5)
    assert np.allclose(w[:3], [1, -1, 0])


def test_fracdiff_d0_is_original():
    s = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 1, 100)))
    out = fracdiff(s, d=0.0)
    # d=0 -> original (allow warmup NaNs)
    valid = out.dropna()
    assert np.allclose(valid.values, s.iloc[-len(valid):].values)


def test_fracdiff_stationarity_reduces():
    """On a random walk (non-stationary), fracdiff(d>0) reduces autocorrelation."""
    rng = np.random.default_rng(1)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 2000)))
    raw_acf = np.abs(x.autocorr(1))
    fd = fracdiff(x, d=0.4).dropna()
    fd_acf = np.abs(fd.autocorr(1))
    assert fd_acf < raw_acf  # memory-preserving but more stationary


def test_fracdiff_causal_no_future():
    """Perturbing a future value leaves past fracdiff values bit-identical."""
    rng = np.random.default_rng(2)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 200)))
    a = fracdiff(x, d=0.4)
    x2 = x.copy()
    x2.iloc[150:] *= 2.0
    b = fracdiff(x2, d=0.4)
    assert np.allclose(a.iloc[:100].values, b.iloc[:100].values, equal_nan=True)


def test_session_london_ny():
    idx = pd.date_range("2021-01-01", periods=48, freq="h")
    s = session_london_ny(idx)
    assert s.iloc[8] == 1.0   # 09:00 UTC -> London open, in [7,16]
    assert s.iloc[0] == 0.0   # 00:00 UTC -> off
