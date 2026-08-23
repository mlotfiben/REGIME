"""Honesty check: does the REGIME feature set beat a trivial momentum baseline?

The gate compares AUC vs 0.5 (majority class). That's too weak — the program's
methodology requires benchmarking against a simple known baseline. If mom_s alone
(or persistence) matches the full 7-feature set, then "regime → barrier" adds nothing
beyond plain momentum, and the 0.63 AUC is not a new edge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent / "data"

TF_CONF = {
    "1h": dict(mom_windows=(4, 12, 48), eff=20, breakout=20, atr=20, vol=200, H=4),
    "4h": dict(mom_windows=(4, 12, 48), eff=20, breakout=20, atr=20, vol=200, H=4),
}


def load(df, conf):
    from engine.features import atr, build_features
    from engine.target import barrier_hit
    a = atr(df, conf["atr"])
    f = build_features(df, mom_windows=conf["mom_windows"], eff_window=conf["eff"],
                       breakout_window=conf["breakout"], atr_window=conf["atr"],
                       vol_window=conf["vol"])
    lab = barrier_hit(df, a, H=conf["H"], up_mult=2.0, down_mult=-1.0)
    X = f.dropna()
    y = lab.reindex(X.index)
    m = y.notna()
    return X[m], y[m]


def walkforward_auc(X, y, cols, n_folds=4, embargo=5):
    n = len(X)
    bounds = np.linspace(0, n, n_folds + 1, dtype=int)
    aucs = []
    for f in range(1, n_folds):
        ts, te = bounds[f], bounds[f + 1]
        Xtr, ytr = X.iloc[:ts - embargo][cols], y.iloc[:ts - embargo]
        Xte, yte = X.iloc[ts:te][cols], y.iloc[ts:te]
        if ytr.nunique() < 2 or yte.nunique() < 2 or len(Xtr) < 1000:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    return aucs


def main(tf):
    conf = TF_CONF[tf]
    df = pd.read_parquet(DATA / f"xau_{tf}.parquet")
    df.columns = [str(c).lower() for c in df.columns]
    X, y = load(df, conf)
    print(f"=== {tf}: full feature set vs momentum-only vs persistence baselines ===")
    full = ["mom_s", "mom_m", "mom_l", "trend_eff_N", "breakout", "kalman_slope", "vol_gate"]
    mom = ["mom_s", "mom_m", "mom_l"]
    eff = ["trend_eff_N", "breakout", "kalman_slope"]
    a_full = walkforward_auc(X, y, full)
    a_mom = walkforward_auc(X, y, mom)
    a_eff = walkforward_auc(X, y, eff)
    print(f"  FULL 7-feature set: AUC {np.mean(a_full):.4f}  folds {[f'{a:.3f}' for a in a_full]}")
    print(f"  MOMENTUM (3)       : AUC {np.mean(a_mom):.4f}  folds {[f'{a:.3f}' for a in a_mom]}")
    print(f"  EFF/CHANNEL/KALMAN : AUC {np.mean(a_eff):.4f}  folds {[f'{a:.3f}' for a in a_eff]}")


if __name__ == "__main__":
    tf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    main(tf)
