"""Data loading for REGIME — library-first, read-only.

Loads the shared H1 cache (~/INFLECTION/data/xau_h1.parquet). Never re-resample
raw 1m here; the cache is the target-frequency input.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

H1_CACHE = Path(os.environ.get("REGIME_H1_CACHE", "~/INFLECTION/data/xau_h1.parquet")).expanduser()


def load_h1(cache: Path = H1_CACHE) -> pd.DataFrame:
    """Load XAUUSD H1 bars. Returns df indexed by timestamp with o/h/l/c columns."""
    if not cache.exists():
        raise FileNotFoundError(f"H1 cache not found: {cache}")
    df = pd.read_parquet(cache)
    df.columns = [str(c).lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def bar_count(df: pd.DataFrame) -> int:
    return int(len(df))
