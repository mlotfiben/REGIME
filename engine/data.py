"""Data loading for REGIME — library-first, read-only.

Reads the RAW 1m XAU CSV once and caches to parquet. Never re-parse the CSV per run.
The raw CSV is never modified/deleted.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

RAW_CSV = Path(os.environ.get("REGIME_RAW_1M", "~/machL/MarketPressure/data/raw/XAU_1m_data.csv")).expanduser()
CACHE = Path(__file__).resolve().parent.parent / "data" / "xau_1m.parquet"


def load_1m(csv: Path = RAW_CSV, cache: Path = CACHE, force: bool = False) -> pd.DataFrame:
    """Load XAUUSD 1m bars. Caches to parquet after first parse of the CSV."""
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
    else:
        if not csv.exists():
            raise FileNotFoundError(f"Raw 1m CSV not found: {csv}")
        df = pd.read_csv(csv)
        df.columns = [str(c).lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def bar_count(df: pd.DataFrame) -> int:
    return int(len(df))
