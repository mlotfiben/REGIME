"""Build + cache REGIME features and Target-A labels to parquet (one-time).

Run: python engine/build_cache.py
Outputs data/features.parquet + data/targetA.parquet. The walk-forward reads these,
never recomputes the 250s Kalman feature pass.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.data import load_1m
from engine.features import atr, build_features
from engine.target import barrier_hit

OUT = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    t0 = time.time()
    df = load_1m()
    print(f"loaded {len(df)} bars in {time.time()-t0:.1f}s")
    a = atr(df, window=20)
    f = build_features(
        df,
        mom_windows=(60, 300, 1440),   # 1h/5h/1d at 1m
        eff_window=120,
        breakout_window=240,
        atr_window=20,
        vol_window=1440,
    )
    print(f"features built in {time.time()-t0:.1f}s")
    lab = barrier_hit(df, a, H=60, up_mult=2.0, down_mult=-1.0)
    f.to_parquet(OUT / "features.parquet")
    pdlab = lab.to_frame(name="barrier_hit")
    pdlab.to_parquet(OUT / "targetA.parquet")
    print(f"wrote {OUT/'features.parquet'} and {OUT/'targetA.parquet'} in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
